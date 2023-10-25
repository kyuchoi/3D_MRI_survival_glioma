#%%
import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import glob
import random
from tqdm import tqdm
from datetime import datetime
import numpy as np
import nibabel as nib
import pandas as pd

import math
from torchsummary import summary as summary

import torch # For building the networks 
import torch.nn as nn
import torch.nn.functional as F

import monai
from monai.networks.nets import *

from utils import *
from attention_models import *

from adamp import AdamP

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import argparse

from medcam import medcam
from medcam import *

import matplotlib
import matplotlib.pyplot as plt
import cv2

from skimage.transform import resize
from scipy import ndimage

#%%

'''
20221031: TO DO LIST
1. augment 해서 높아진 valid loss 62에서 낮추기 --> overfit 이었고, augment 종류 줄여서 해결됨
2. 예전 코드 scheduler 가져와보기
3. run_fold: for 문으로 돌리기
4. c-index 나오면 grad-CAM + curve 그리기: 일단 train,valid c-index라도 잘 나오는지 확인해보기 (안되면 5-fCV 결과만 낼 수도 있으니)
5. label correction 확인: 1.5
6. 안되면 아예 loss를 c-index 로 써보기
7. 안되면 DeepHit (원래 pycox 코드로 A6000 달아서 돌리면 됨) 쓰거나 seg 말고 resize 로 돌려보기: DeepHit 쓰면 좋은데 문제는 그러려면 nnet에서 y_train transform 하는 과정을 pycox 처럼 다르게 해야 함 
8. 안되면 augmentation 여러 가지 써보기: 현재 crop 64 + gaussian만 사용중
9. grad-CAM + calibration plot 등 원래 pycox 코드 뒷단에 있던 것들 추가하기
'''

#%%

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Deep survival GBL: image only', add_help=add_help)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--test_gpu_id', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=123456) # 12347541
    parser.add_argument('--spec_patho', type=str, default='all') # 'GBL' # 
    parser.add_argument('--spec_duration', type=str, default='1yr') # 'OS' # 
    parser.add_argument('--spec_event', type=str, default='death') # 'death' # 
    parser.add_argument('--ext_dataset_name', type=str, default='SNUH') # 'TCGA' # 
    parser.add_argument('--dataset_list', nargs='+', default=['UCSF','UPenn','TCGA','severance'], help='selected_training_datasets') # ,'TCGA'
    parser.add_argument('--remove_idh_mut', default=False, type=str2bool)
    parser.add_argument('--save_grad_cam', default=False, type=str2bool)
    parser.add_argument('--biopsy_exclusion', default=False, type=str2bool)
    # parser.add_argument('--remove_idh_mut', action='store_true', help='for subgroup analysis of IDH-wt, removing IDH-mut')
    # parser.add_argument('--sequence', nargs='+', default=['flair','t1ce','t2','t1'], help='selected_MR_sequences') # usage: python arg.py -l 1234 2345 3456 4567
    return parser

main_args = get_args_parser().parse_args()

#%%
# class get_args_parser(object):
#   def __init__(self):

#     self.gpu_id = 0
#     self.test_gpu_id = 1
#     self.epochs = 1 # 200 # 
#     self.seed = 123456
#     self.spec_patho = 'all' # 'GBL' # 
#     self.spec_duration = '1yr' # 'OS' # 
#     self.spec_event = 'death' # 'prog' # 
#     self.ext_dataset_name = 'SNUH' # 'severance'
#     self.dataset_list = ['UCSF','UPenn','TCGA','severance'] # ['SNUH','TCGA'] # 'UPenn', # NO PFS in UPenn # 순서 중요: SNUH > UPenn > TCGA or severance 순서
#     self.remove_idh_mut = False # True # 
#     self.save_grad_cam = False # True # 
#     self.biopsy_exclusion = False # True # 
    
# main_args = get_args_parser()

#%%

args = config()

breaks, n_intervals = get_n_intervals(fixed_interval_width = False) # True #
args.n_intervals = n_intervals

gpu_id = main_args.gpu_id

args.dataset_name = '_'.join(main_args.dataset_list)
print(f'Train dataset_name:{args.dataset_name}')

now = datetime.now()
exp_time = now.strftime("%Y_%m_%d_%H_%M")

DL_score_dir = os.path.join(args.exp_dir, 'DL_features', f'{exp_time}_{args.dataset_name}_ext_{main_args.ext_dataset_name}')
os.makedirs(DL_score_dir, exist_ok = True)
args.DL_score_dir = DL_score_dir

attention_map_dir = os.path.join(args.exp_dir, 'attention_maps', f'{exp_time}_{args.dataset_name}_ext_{main_args.ext_dataset_name}')
os.makedirs(attention_map_dir, exist_ok = True)
args.attention_map_dir = attention_map_dir

exp_path = os.path.join(args.exp_dir, f'{exp_time}_{args.dataset_name}_ext_{main_args.ext_dataset_name}')
os.makedirs(exp_path, exist_ok=True)
print_args(main_args, exp_path)

print(f'Training on GPU {gpu_id}')
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

#%%
os.environ['MKL_THREADING_LAYER'] = 'GNU' # in Linux, I had to write a script to call "export MKL_THREADING_LAYER=GNU" (which sets that environment variable) each time I activate the virtual environment, and a counter script to undo that change upon exiting the environment.
set_seed(main_args.seed)
print(f'Setting seed:{main_args.seed}')
#%%
to_np = lambda x: x.detach().cpu().numpy()
to_cuda = lambda x: torch.from_numpy(x).float().device()

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
torch.cuda.empty_cache()
# print(get_dir(DATA_DIR))

#%%

get_label_path = lambda dataset: os.path.join(args.label_dir, f'{dataset}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}.csv') #  'manual_labels', #
get_target = lambda df: (np.array(df.index.values, dtype=str), # int: not working
                          np.array(df[f'duration_{main_args.spec_event}'].tolist(), dtype=int), 
                          np.array(df[f'event_{main_args.spec_event}'].tolist(), dtype=int))

df = save_label_dataset_list(main_args, args)
ext_df = save_label_ext_dataset(main_args, args)

#%%

combine_img(main_args, args)

#%%
def make_kfold_df_proc_labels(args, dataset_name, remove_idh_mut = False, fixed_interval_width = 0):
  
  df = pd.read_csv(get_label_path(dataset_name), dtype='string')
  df = df.set_index('ID')
  df = df.sort_index(ascending=True)
  df = df[args.data_key_list]
  print(f'df.index.values:{len(df.index.values)}')
  
  if remove_idh_mut:
    condition = df.IDH.astype(int) == 0 # 1 means mut, not 0 
    filtered_ID = df[condition].index.tolist() 
    df = df.loc[sorted(filtered_ID),:]
    print(f'after removing idh mutation; df.index.values:{len(df.index.values)}')
  
  if '_' in dataset_name:
    list_dataset = dataset_name.split('_')
    print(f'list_dataset: {list_dataset}')
    
    comm_list = []
    for split_dataset in list_dataset:
      print(f'split_dataset: {split_dataset}')
      img_dir = os.path.join(args.data_dir, split_dataset, f'{args.compart_name}_BraTS')
      split_comm_list = [elem for elem in get_dir(img_dir) if elem in df.index.values]
      print(f'split_img_label_comm_list:{len(split_comm_list)}')
      comm_list.extend(split_comm_list)
    
  else:  
    img_dir = os.path.join(args.data_dir, dataset_name, f'{args.compart_name}_BraTS') 
    comm_list = [elem for elem in get_dir(img_dir) if elem in df.index.values]
  
  print(f'img_label_comm_list:{len(comm_list)}')
  print(f'dataset_name:{dataset_name}, {len(comm_list)}') # SNUH_UPenn, 1113
  
  df = df.loc[sorted(comm_list)] #.astype(int) 

  print(f'{dataset_name} df.shape: {df.shape}') # (1113, 8) 

  ID, duration, event = get_target(df)
  
  kfold = add_kfold_to_df(df, args, main_args.seed)
  
  breaks, _ = get_n_intervals(fixed_interval_width)

  proc_labels = make_surv_array(duration, event, breaks)
  df_proc_labels = pd.DataFrame(proc_labels)

  df_proc_labels['ID'] = ID
  df_proc_labels['kfold'] = kfold
  df_proc_labels = df_proc_labels.set_index('ID')
  
  proc_label_path = os.path.join(args.proc_label_dir, f'{dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_proc_labels.csv')
  df_proc_labels.to_csv(proc_label_path)
  
  return df_proc_labels, event, duration
#%%

df_proc_labels_train, event_train, duration_train = make_kfold_df_proc_labels(args, f'{args.dataset_name}', remove_idh_mut = main_args.remove_idh_mut)
df_proc_labels_test, event_test, duration_test = make_kfold_df_proc_labels(args, f'{main_args.ext_dataset_name}', remove_idh_mut = main_args.remove_idh_mut)

# print(f'df_proc_labels_train: {df_proc_labels_train.shape}') # (1113, 39)
# print(f'df_proc_labels_test: {df_proc_labels_test.shape}') # (66, 39)

#%%

print(f'train transform:')
args.train_transform = get_transform(args, f'{args.dataset_name}')
print(f'valid transform:')
args.valid_transform = get_transform(args, f'{args.dataset_name}')
print(f'test transform:')
test_transform = get_transform(args, f'{main_args.ext_dataset_name}')

# if AUGMENT_TRANSFORM:
#   train_transform = tio.Compose(transforms+aug_transforms)
# else:
#   train_transform = tio.Compose(transforms)
# valid_transform = tio.Compose(transforms)


#%%

# train_data = SurvDataset(dataset_name='SNUH_UPenn', transforms=train_transform)

# training_set = tio.SubjectsDataset(
#     training_subjects, transform=training_transform)

# train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True, shuffle=True)
# batch = next(iter(train_loader))
# print(f'img:{batch[0].shape}') # torch.Size([16, 4, 120, 120, 78])
# print(f'label:{batch[1].shape}') # torch.Size([16, 38])

#%%

''' ref: 
https://nittaku.tistory.com/111
https://nomalcy.tistory.com/330 
https://dbwp031.tistory.com/26 
https://discuss.pytorch.org/t/how-to-add-a-layer-to-an-existing-neural-network/30129/3 
https://sensibilityit.tistory.com/511
https://artiiicy.tistory.com/61
https://thewayaboutme.tistory.com/384
https://stackoverflow.com/questions/55875279/how-to-get-an-output-dimension-for-each-layer-of-the-neural-network-in-pytorch
'''
if args.net_architect == 'SEResNext50':
  base_model = monai.networks.nets.SEResNext50(spatial_dims=3, in_channels=4, num_classes=args.n_intervals)
elif args.net_architect == 'DenseNet':
  base_model = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=4, out_channels=args.n_intervals)
elif args.net_architect == 'resnet50_cbam':
  base_model = resnet50_cbam(num_classes=args.n_intervals)
# print(next(base_model.parameters()).device)

# last_removed_base_model = nn.Sequential(*(list(base_model.children())[:-1]))
# print(last_removed_base_model)
model = CustomNetwork(args, base_model = base_model).to(device)
# summary(model, (4, 120, 120, 78), device='cuda')

base_optimizer = AdamP
optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, weight_decay=args.weight_decay) # https://stackoverflow.com/questions/43779500/pytorch-network-parameters-missing-1-required-positional-argument-self
criterion = nnet_loss # TaylorCrossEntropyLoss(n=2,smoothing=0.2)
scheduler = fetch_scheduler(optimizer)

if not main_args.save_grad_cam:
  model, history = run_fold(df_proc_labels_train, args, model, criterion, optimizer, scheduler, device=device, fold=0, num_epochs=main_args.epochs)

# %%

test_gpu_id = main_args.test_gpu_id # int(main_args.gpu_id + 1) # 
print(f'Testing on GPU {test_gpu_id}')

test_device = torch.device(test_gpu_id)

model = CustomNetwork(args, base_model = base_model).to(test_device)
model = load_ckpt(args, model)

proc_label_path_test = os.path.join(args.proc_label_dir, f'{main_args.ext_dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_proc_labels.csv')
df_proc_labels_test = pd.read_csv(proc_label_path_test, dtype='string')
df_proc_labels_test = df_proc_labels_test.set_index('ID')

test_data = SurvDataset(df = df_proc_labels_test, args = args, dataset_name=f'{main_args.ext_dataset_name}', transforms=test_transform, aug_transform=False)
test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=4, pin_memory=True, shuffle=False) # args.batch_size
# test_batch = next(iter(test_loader))
# model(test_batch[0].to(device))
#%%
df_DL_score_test = df_proc_labels_test.copy()
df_DL_score_test.drop(columns=df_DL_score_test.columns,inplace=True) # index 열 제외하고 모두 삭제

for i in np.arange(n_intervals):
  df_DL_score_test.insert(int(i), f'MRI{i+1}', '')
df_DL_score_test.insert(n_intervals, 'oneyr_survs_test', '')

oneyr_survs_test = []
for subj_num, (inputs,labels) in enumerate(test_loader):
  model.eval()
  inputs = inputs.to(test_device)
  labels = labels.to(test_device)

  y_pred = model(inputs) # torch.Size([4, 19])
  print(f'y_pred:{y_pred}')
  print(f'labels:{labels}')
  print(f'subj_num:{subj_num}')
    
  ''' evaluate c-index 
  ref:
  https://lifelines.readthedocs.io/en/latest/lifelines.utils.html
  https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=cjh226&logNo=221380929786
  '''

  halflife=365.*2
  breaks=-np.log(1-np.arange(0.0,0.96,0.05))*halflife/np.log(2) 
  # breaks=np.arange(0.,365.*5,365./8)
  y_pred_np = to_np(y_pred)

  # ref for DeepSurv: https://github.com/havakv/pycox/issues/58 
  # oneyr_surv_test = 1 - np.cumprod(y_pred_np[:,0:np.nonzero(breaks>365)[0][0]], axis=1)[:,-1] # NOT WORKING
  cumprod = np.cumprod(y_pred_np[:,0:np.nonzero(breaks>365)[0][0]], axis=1)
  oneyr_surv_test = cumprod[:,-1]
  print(f'oneyr_surv_test: {oneyr_surv_test}')
  # ref for picking mid term in 형진샘 논문: https://pubs.rsna.org/doi/full/10.1148/radiol.2020192764
  ### "The output layer of the DLPM had six neurons, corresponding to the following intervals: 0–300, 300–600, 600–900, 900–1200, 1200–1800, and 1800–2400 days. The final outputs were the conditional probabilities of surviving in these intervals. In our study, continuous DLPM output was defined empirically with the following equation: [1 − cumulative survival probability (cumulative product) to the third interval]."
  
  DL_scores = []
  for n_interval in np.arange(1, n_intervals+1):
    DL_score = np.cumprod(y_pred_np[:,0:n_interval], axis=1)[:,-1][0]
    # print(f'DL_score_{n_interval}th_term_oneyr_surv_test:{DL_score}')  
    DL_scores.append(DL_score)
  DL_scores.append(oneyr_surv_test[0])
  # print(f'DL_scores:{len(DL_scores)}') # 19+1=20
  df_DL_score_test.loc[df_DL_score_test.index[subj_num]] = DL_scores
  oneyr_survs_test.extend(oneyr_surv_test)

print(f'df_DL_score_test.shape:{df_DL_score_test.shape}')
DL_score_path = os.path.join(DL_score_dir, f'{main_args.ext_dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_DL_score_s{main_args.seed}.csv')
df_DL_score_test.to_csv(DL_score_path)

print(len(oneyr_survs_test)) # 132
oneyr_survs_test = np.array(oneyr_survs_test)

#%%
print(f'duration_test.shape:{duration_test.shape}')
print(f'oneyr_survs_test.shape:{oneyr_survs_test.shape}')
print(f'event_test.shape:{event_test.shape}')

original_c_index, ci_lower, ci_upper = bootstrap_cindex(duration_test, oneyr_survs_test, event_test)

print(f'Original C-index for valid: {original_c_index:.4f}')
print(f'95% CI for C-index for valid: ({ci_lower:.4f}, {ci_upper:.4f})')

score_test = get_BS(event_test, duration_test, oneyr_survs_test)

# %%

''' grad CAM 
ref: https://github.com/MECLabTUDA/M3d-Cam

'''
plt.switch_backend('agg')

test_img_path='/mnt/hdd3/mskim/GBL/data/SNUH/resized_BraTS/73116251/'
seqs = []
for seq in ['t1','t2','flair','t1ce']:
  seq=nib.load(os.path.join(test_img_path, f'{seq}_resized.nii.gz')).get_fdata() # _seg # _cropped
  print(f'seq range:min {seq.min()}-max {seq.max()}')
  # print(seq.shape)
  # torch.cat([x[sequence][tio.DATA] for sequence in self.SEQUENCE], axis=0)
  seqs.append(seq)

x = np.stack(seqs, axis=0)
x = torch.from_numpy(x)
x = torch.unsqueeze(x, axis=0)
print(f'x.shape:{x.shape}') # torch.Size([1, 4, 120, 120, 78])

z_slice_num = int(x.shape[-1]//2) # 39 # AXL
y_slice_num = int(x.shape[-2]//2) # 60 # SAG
x_slice_num = int(x.shape[-3]//2) # 60 # COR

# x=x.to(test_device).float()
# model = model.eval()
# output = model(x)
# print(f'output:{output}')
# print(f'output.shape:{output.shape}') # torch.Size([1, 19])

seq_idx_dict = {'t1':0, 't2':1, 't1ce':2, 'flair':3}
selected_seq = 't1ce'
selected_seq_idx = seq_idx_dict[selected_seq]
print(f'selected_seq:{selected_seq}, {selected_seq_idx}')

#%%

print(f'args.attention_map_dir:{args.attention_map_dir}')

slice_3d = lambda x: x[selected_seq_idx,:,:,:]

if main_args.save_grad_cam:
  cam_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
  cam_model = medcam.inject(model, backend='gcampp', output_dir="attention_maps", save_maps=True)

  superimposed_imgs = []
  cam_model.eval()

  for subj_num, batch in enumerate(cam_loader):
    batch = batch[0].to(test_device) # cuda()
    output = cam_model(batch)
    cam=cam_model.get_attention_map()
    print(type(cam)) # numpy array 
    print(f'cam.shape:{cam.shape}') # (1,1,4,4,3) # summary(model, (4, 120, 120, 78), device='cuda') 하면 나오는 shape이 (1,1,4,4,3) 임.
    print(f'input.shape:{batch.shape}') # torch.Size([1, 4, 120, 120, 78])

    subj_id = df_DL_score_test.index[subj_num]
    print(f'subj_id:{subj_id}')

    img_4d = batch.squeeze().cpu().numpy()
    img_3d = slice_3d(img_4d)

    img_3d_scaled = min_max_norm(img_3d)

    result_3d = cam.squeeze()

    print(f'img_3d.shape:{img_3d.shape}') # (120, 120, 78)
    print(f'result_3d.shape:{result_3d.shape}') # (4, 4, 3)
    
    superimposed_img_3d, result_3d_resized = superimpose_img(img_3d_scaled, result_3d)
    print(f'superimposed_img_3d.shape:{superimposed_img_3d.shape}')
    print(f'result_3d_resized.shape:{result_3d_resized.shape}')
    # print(np.min(superimposed_img_3d), np.max(superimposed_img_3d)) # 0 - 255

    # ''' axl '''
    # img_2d = img_3d[:,:,z_slice_num]
    # superimposed_img_2d = superimposed_img_3d[:,:,z_slice_num]
    # result_2d_resized = result_3d_resized[:,:,z_slice_num]
    # # result_2d = result_3d[:,:,result_slice_num]

    # rot_result_2d_resized = ndimage.rotate(result_2d_resized, rot_degree)
    # rot_img_2d = ndimage.rotate(img_2d, rot_degree)

    # plt.imshow(rot_result_2d_resized, alpha = 0.9, cmap='jet') #'Spectral')
    # plt.imshow(rot_img_2d, alpha = 0.5, cmap='gray') # https://rk1993.tistory.com/278
    # # plt.show()

    # plt_saved_loc = os.path.join(args.attention_map_dir, 'grad_CAM_2d_axl')
    # os.makedirs(plt_saved_loc, exist_ok=True)
    # plt.savefig(os.path.join(plt_saved_loc, f'Grad_CAM_heatmap_{subj_id}_axl.jpg'), dpi=300)

    # ''' sag '''
    # img_2d = img_3d[x_slice_num,:,:]
    # superimposed_img_2d = superimposed_img_3d[x_slice_num,:,:]
    # result_2d_resized = result_3d_resized[x_slice_num,:,:]
    # # result_2d = result_3d[:,:,result_slice_num]

    # rot_result_2d_resized = ndimage.rotate(result_2d_resized, rot_degree)
    # rot_img_2d = ndimage.rotate(img_2d, rot_degree)

    # plt.imshow(rot_result_2d_resized, alpha = 0.9, cmap='jet') #'Spectral')
    # plt.imshow(rot_img_2d, alpha = 0.5, cmap='gray') # https://rk1993.tistory.com/278
    # # plt.show()

    # plt_saved_loc = os.path.join(args.attention_map_dir, 'grad_CAM_2d_sag')
    # os.makedirs(plt_saved_loc, exist_ok=True)
    # plt.savefig(os.path.join(plt_saved_loc, f'Grad_CAM_heatmap_{subj_id}_sag.jpg'), dpi=300)

    # ''' cor '''
    # img_2d = img_3d[:,y_slice_num,:]
    # superimposed_img_2d = superimposed_img_3d[:,y_slice_num,:]
    # result_2d_resized = result_3d_resized[:,y_slice_num,:]
    # # result_2d = result_3d[:,:,result_slice_num]

    # rot_result_2d_resized = ndimage.rotate(result_2d_resized, rot_degree)
    # rot_img_2d = ndimage.rotate(img_2d, rot_degree)

    # plt.imshow(rot_result_2d_resized, alpha = 0.9, cmap='jet') #'Spectral')
    # plt.imshow(rot_img_2d, alpha = 0.5, cmap='gray') # https://rk1993.tistory.com/278
    # # plt.show()

    # plt_saved_loc = os.path.join(args.attention_map_dir, 'grad_CAM_2d_cor')
    # os.makedirs(plt_saved_loc, exist_ok=True)
    # plt.savefig(os.path.join(plt_saved_loc, f'Grad_CAM_heatmap_{subj_id}_cor.jpg'), dpi=300)
    
    plt_saved_loc_3d = os.path.join(attention_map_dir, 'grad_CAM_3d')
    os.makedirs(plt_saved_loc_3d, exist_ok=True)

    matplotlib.rcParams['animation.embed_limit'] = 500 # 500 MB 까지 용량 상한 늘려줌 의미
    ani_html = plot_slices_superimposed(superimposed_img_3d)
   
    # save as html
    animation_html_path = os.path.join(plt_saved_loc_3d, f'Grad_CAM_heatmap_{subj_id}_3d.html')
    with open(animation_html_path, 'w') as f:
      f.write(ani_html)
     
    # save as gif: TOO LARGE file size!!
    # animation_gif_path = os.path.join(plt_saved_loc_3d, f'Grad_CAM_heatmap_{subj_id}_3d.gif')   
    # ani_gif.save(animation_gif_path, writer='imagemagick', fps=3, dpi=300)
    
    # plot_slices_superimposed(superimposed_img_3d)
    # plt.savefig(os.path.join(plt_saved_loc_3d, f'Grad_CAM_heatmap_{subj_id}_3d.jpg'), dpi=300)
    
    plt.close()

#%%

''' n=104 test set에 대한 3D attention map 이 /mnt/hdd2/kschoi/GBL/code/attention_maps/base_model.layer4/attention_map_${subj_num}_0_0.nii.gz 로 저장됨 
여기에 대해 https://github.com/MECLabTUDA/M3d-Cam 을 보고 bilinear, trilinear 든 resize를 통해 overlay하면 grad CAM을 얻을 수 있음 (참고: https://github.com/Project-MONAI/MONAI/blob/dev/monai/visualize/class_activation_maps.py 에서 upsampler 검색).
원래 구현해두었던 모든 code가 잘 안되는 게, 모델을 그대로 쓰지 않는 바람에 target_layer name이 모두 틀리게 됨.
'''

  # return grad_result

# #%%
# subj_num = 59
# test_image_path = f'/mnt/hdd2/kschoi/GBL/code/attention_maps/base_model.layer4/attention_map_{subj_num}_0_0.nii.gz'
# test_img = nib.load(test_image_path).get_fdata()

# superimpose_img(img, heatmap)

#%%

'''
# ref: https://docs.monai.io/en/stable/visualize.html?highlight=cam#monai.visualize.class_activation_maps.CAM

from monai.visualize import * # GradCAM

slice_num = 54 # int(x.shape[-1]//2) # (45, 38, 54) 
img=x[:,selected_seq_idx,:,:,slice_num] # get T1wCE image for cam
# img = np.concatenate([to_np(img), to_np(img), to_np(img)],axis=0)
img = to_np(img)
img = np.transpose(img,(1,2,0))
# print(f'img.shape:{img.shape}')
# img = np.squeeze(img, axis=-1)
img = img/np.max(img)
print(f'img.shape:{img.shape}') # (75, 90)
# img = np.uint8(img * 255)
print(f'img range:min {img.min()}-max {img.max()}')

if NET_ARCHITECT == 'DenseNet':
  cam = GradCAMpp(nn_module=net, target_layers='class_layers.relu') # For DenseNet, use target_layers 'class_layers.relu' with GradCAMpp
elif NET_ARCHITECT == 'SEResNext50':
  cam = GradCAM(nn_module=net, target_layers='layer4') # For SEResNext50, use target layers 'layer4'

result = cam(x)
result = result[:,0,:,:,slice_num]

아니면 https://www.kaggle.com/code/debarshichanda/gradcam-visualize-your-cnn 를 보고 해도 되나 grad CAM map 나온 결과가 별로 안 좋았음

'''

