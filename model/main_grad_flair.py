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

# def get_args_parser(add_help=True):
#     parser = argparse.ArgumentParser(description='Deep survival GBL: image only', add_help=add_help)
#     parser.add_argument('--gpu_id', type=int, default=0)
#     parser.add_argument('--test_gpu_id', type=int, default=1)
#     parser.add_argument('--epochs', type=int, default=200)
#     parser.add_argument('--seed', type=int, default=123456) # 12347541
#     parser.add_argument('--spec_patho', type=str, default='all') # 'GBL' # 
#     parser.add_argument('--spec_duration', type=str, default='1yr') # 'OS' # 
#     parser.add_argument('--spec_event', type=str, default='death') # 'death' # 
#     parser.add_argument('--ext_dataset_name', type=str, default='SNUH') # 'TCGA' # 
#     parser.add_argument('--dataset_list', nargs='+', default=['UCSF','UPenn','TCGA','severance'], help='selected_training_datasets') # ,'TCGA'
#     parser.add_argument('--remove_idh_mut', default=False, type=str2bool)
#     parser.add_argument('--save_grad_cam', default=False, type=str2bool)
#     parser.add_argument('--selected_seq', type=str, default='flair') # 't1ce' # 
#     parser.add_argument('--get_3d_grad_cam', default=False, type=str2bool)
#     parser.add_argument('--biopsy_exclusion', default=False, type=str2bool)
#     # parser.add_argument('--remove_idh_mut', action='store_true', help='for subgroup analysis of IDH-wt, removing IDH-mut')
#     # parser.add_argument('--sequence', nargs='+', default=['flair','t1ce','t2','t1'], help='selected_MR_sequences') # usage: python arg.py -l 1234 2345 3456 4567
#     return parser

# main_args = get_args_parser().parse_args()

#%%
class get_args_parser(object):
  def __init__(self):

    self.gpu_id = 1
    self.test_gpu_id = 1
    self.epochs = 1 # 200 # 
    self.seed = 123456
    self.spec_patho = 'all' # 'GBL' # 
    self.spec_duration = '1yr' # 'OS' # 
    self.spec_event = 'death' # 'prog' # 
    self.ext_dataset_name = 'SNUH' # 'severance' # 
    self.dataset_list = ['UCSF','UPenn','TCGA','severance'] # ['UCSF','UPenn','TCGA'] # 
    self.remove_idh_mut = False # True # 
    self.save_grad_cam = True # False # 
    self.get_3d_grad_cam = False # True # 
    self.biopsy_exclusion = False # True # 
    self.hide_ticks = True # False # 
    
main_args = get_args_parser()

# 아래 slice_num 몇으로 할 지 구하려면 밑에 plt.xticks([]) 눈금 없애는 거 지우고 그거 보고 하기
# x_slice_num, y_slice_num, z_slice_num = 50, 50, 35 # SAG, COR, AXL # for right (severance) # 75, 50, 35 # for left (SNUH)
x_slice_num, y_slice_num, z_slice_num = 75, 50, 43 # SAG, COR, AXL # for left SNUH

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
# print_args(main_args, exp_path)

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

#%%

print(f'train transform:')
args.train_transform = get_transform(args, f'{args.dataset_name}')
print(f'valid transform:')
args.valid_transform = get_transform(args, f'{args.dataset_name}')
print(f'test transform:')
test_transform = get_transform(args, f'{main_args.ext_dataset_name}')

#%%

if args.net_architect == 'SEResNext50':
  base_model = monai.networks.nets.SEResNext50(spatial_dims=3, in_channels=4, num_classes=args.n_intervals)
elif args.net_architect == 'DenseNet':
  base_model = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=4, out_channels=args.n_intervals)
elif args.net_architect == 'resnet50_cbam':
  base_model = resnet50_cbam(num_classes=args.n_intervals)

model = CustomNetwork(args, base_model = base_model).to(device)

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

# %%

''' grad CAM 
ref: https://github.com/MECLabTUDA/M3d-Cam

'''
# plt.switch_backend('agg')

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

seq_idx_dict = {'t1':0, 't2':1, 't1ce':2, 'flair':3}

#%%
print(f'args.attention_map_dir:{args.attention_map_dir}')

class SurvDataset_grad(nn.Module):
  def __init__(self, args, dataset_name, transforms=None, aug_transform=False):
    self.dataset_name = dataset_name
    
    self.args = args
    self.img_dir = os.path.join(args.data_dir, self.dataset_name, f'{args.compart_name}_grad') # 'SNUH_UPenn_TCGA_severance'
    self.transforms = transforms
    self.aug_transform = aug_transform

    self.crop_size = 64
    self.crop = RandSpatialCrop(roi_size=(self.crop_size, self.crop_size, self.crop_size), random_size=False)
    
    self.gaussian = RandGaussianNoise(prob=0.3)
    
    self.compart_name = args.compart_name
  
  def concat_seq_img(self, x):
      return torch.cat([x[sequence][tio.DATA] for sequence in self.args.sequence], axis=0)
  
  def __len__(self):
    return len(os.listdir(self.img_dir)) # 결국 df 로 index 를 하기 때문에 dataset의 길이도 len(df): df를 train_df, val_df 넣는것에 따라 dataset이 train_set, val_set이 됨.

  def augment(self, img):  
    img = self.crop(img)
    img = self.gaussian(img)
  
    return img

  def __getitem__(self, idx):
    if type(idx) is not int:
      raise ValueError(f"Need `index` to be `int`. Got {type(idx)}.")

    ID = os.listdir(self.img_dir)[idx]
    # kfold = self.df['kfold'][idx]
    # print(f'kfold:{kfold}')
    self.subj_id = str(ID)
    # print(f'subj_id:{str(ID)}') # UPENN-GBM-00427_11
    subj_img_dir = os.path.join(self.img_dir, str(ID))
    # print(f'IMG_DIR:{IMG_DIR}')
        
    subject = tio.Subject(
        t1=tio.ScalarImage(os.path.join(subj_img_dir, f't1_{self.compart_name}.nii.gz')), # t1_seg.nii.gz
        t2=tio.ScalarImage(os.path.join(subj_img_dir, f't2_{self.compart_name}.nii.gz')), 
        t1ce=tio.ScalarImage(os.path.join(subj_img_dir, f't1ce_{self.compart_name}.nii.gz')), 
        flair=tio.ScalarImage(os.path.join(subj_img_dir, f'flair_{self.compart_name}.nii.gz'))
        )   
    
    if self.transforms:
      subject = self.transforms(subject)
    
    img = self.concat_seq_img(subject)
    
    if self.aug_transform:
      img = self.augment(img)
    
    return img

grad_data = SurvDataset_grad(args = args, dataset_name=f'{main_args.ext_dataset_name}', transforms=test_transform, aug_transform=False)

rot_degree = 90

if main_args.save_grad_cam:
  with torch.no_grad():
    cam_loader = DataLoader(dataset=grad_data, batch_size=1, shuffle=False)
    cam_model = medcam.inject(model, backend='gcampp', output_dir="attention_maps", save_maps=True)

    superimposed_imgs = []
    cam_model.eval()

    for subj_num, batch in enumerate(cam_loader):
      batch = batch.to(test_device) # original: batch[0].to(test_device) # ValueError: expected 5D input (got 4D input)
      subj_id = os.listdir(grad_data.img_dir)[subj_num]
      print(f'subj_id:{subj_id}')

      output = cam_model(batch)
      cam=cam_model.get_attention_map()
      print(type(cam)) # numpy array 
      print(f'cam.shape:{cam.shape}') # (1,1,4,4,3) # summary(model, (4, 120, 120, 78), device='cuda') 하면 나오는 shape이 (1,1,4,4,3) 임.
      print(f'input.shape:{batch.shape}') # torch.Size([1, 4, 120, 120, 78])

      img_4d = batch.squeeze().cpu().numpy()
      
      img_3d_t1ce_scaled = min_max_norm(img_4d[2,:,:,:])
      img_3d_flair_scaled = min_max_norm(img_4d[3,:,:,:])
      
      result_3d = cam.squeeze()

      print(f'img_3d_t1ce_scaled.shape:{img_3d_t1ce_scaled.shape}') # (120, 120, 78)
      print(f'img_3d_flair_scaled.shape:{img_3d_flair_scaled.shape}') # (120, 120, 78)
      print(f'result_3d.shape:{result_3d.shape}') # (4, 4, 3)
      
      superimposed_img_3d_t1ce, result_3d_resized = superimpose_img(img_3d_t1ce_scaled, result_3d)
      superimposed_img_3d_flair, result_3d_resized = superimpose_img(img_3d_flair_scaled, result_3d)
      
      print(f'superimposed_img_3d_t1ce.shape:{superimposed_img_3d_t1ce.shape}')
      print(f'superimposed_img_3d_flair.shape:{superimposed_img_3d_flair.shape}')
      print(f'result_3d_resized.shape:{result_3d_resized.shape}')
      # print(np.min(superimposed_img_3d), np.max(superimposed_img_3d)) # 0 - 255

      ''' axl '''
      for seq_idx, (img_3d, superimposed_img_3d) in enumerate([(img_3d_t1ce_scaled, superimposed_img_3d_t1ce), (img_3d_flair_scaled, superimposed_img_3d_flair)]):

        seq = ['t1ce', 'flair'][seq_idx]

        img_2d = img_3d[:,:,z_slice_num]
        superimposed_img_2d = superimposed_img_3d[:,:,z_slice_num]
        result_2d_resized = result_3d_resized[:,:,z_slice_num]
        # result_2d = result_3d[:,:,result_slice_num]

        rot_result_2d_resized = ndimage.rotate(result_2d_resized, rot_degree)
        rot_img_2d = ndimage.rotate(img_2d, rot_degree)
        
        # plot rot_img_2d and rot_result_2d_resized using subplot(1,2,1)
        plt_saved_loc = os.path.join(args.attention_map_dir, 'grad_CAM_2d_axl')
        os.makedirs(plt_saved_loc, exist_ok=True)
        
        # plt.subplot(1,2,1)
        if main_args.hide_ticks:
          plt.xticks([])
          plt.yticks([])
        plt.imshow(rot_img_2d, cmap='gray')
        plt.savefig(os.path.join(plt_saved_loc, f'{subj_id}_{seq}_axl.jpg'), dpi=300)
        plt.show()

        # plt.subplot(1,2,2)
        if main_args.hide_ticks:
          plt.xticks([])
          plt.yticks([])
        plt.imshow(rot_result_2d_resized, alpha = 0.9, cmap='jet') #'Spectral')
        plt.imshow(rot_img_2d, alpha = 0.5, cmap='gray') # https://rk1993.tistory.com/278
        plt.savefig(os.path.join(plt_saved_loc, f'Grad_CAM_{subj_id}_{seq}_axl.jpg'), dpi=300)
        plt.show()

        ''' sag ''' 
        img_2d = img_3d[x_slice_num,:,:]
        superimposed_img_2d = superimposed_img_3d[x_slice_num,:,:]
        result_2d_resized = result_3d_resized[x_slice_num,:,:]
        # result_2d = result_3d[:,:,result_slice_num]

        rot_result_2d_resized = ndimage.rotate(result_2d_resized, rot_degree)
        rot_img_2d = ndimage.rotate(img_2d, rot_degree)

        plt_saved_loc = os.path.join(args.attention_map_dir, 'grad_CAM_2d_sag')
        os.makedirs(plt_saved_loc, exist_ok=True)

        # plt.subplot(1,2,1)
        if main_args.hide_ticks:
          plt.xticks([])
          plt.yticks([])
        plt.imshow(rot_img_2d, cmap='gray')
        plt.savefig(os.path.join(plt_saved_loc, f'{subj_id}_{seq}_sag.jpg'), dpi=300)
        plt.show()
        
        # plt.subplot(1,2,2)
        if main_args.hide_ticks:
          plt.xticks([])
          plt.yticks([])
        plt.imshow(rot_result_2d_resized, alpha = 0.9, cmap='jet') #'Spectral')
        plt.imshow(rot_img_2d, alpha = 0.5, cmap='gray') # https://rk1993.tistory.com/278        
        plt.savefig(os.path.join(plt_saved_loc, f'Grad_CAM_{subj_id}_{seq}_sag.jpg'), dpi=300)
        plt.show()

        ''' cor '''
        img_2d = img_3d[:,y_slice_num,:]
        superimposed_img_2d = superimposed_img_3d[:,y_slice_num,:]
        result_2d_resized = result_3d_resized[:,y_slice_num,:]
        # result_2d = result_3d[:,:,result_slice_num]

        rot_result_2d_resized = ndimage.rotate(result_2d_resized, rot_degree)
        rot_img_2d = ndimage.rotate(img_2d, rot_degree)
        
        plt_saved_loc = os.path.join(args.attention_map_dir, 'grad_CAM_2d_cor')
        os.makedirs(plt_saved_loc, exist_ok=True)
        
        # plt.subplot(1,2,1)
        if main_args.hide_ticks:
          plt.xticks([])
          plt.yticks([])
        plt.imshow(rot_img_2d, cmap='gray')
        plt.savefig(os.path.join(plt_saved_loc, f'{subj_id}_{seq}_cor.jpg'), dpi=300)
        plt.show()
        
        # plt.subplot(1,2,2)
        if main_args.hide_ticks:
          plt.xticks([])
          plt.yticks([])
        plt.imshow(rot_result_2d_resized, alpha = 0.9, cmap='jet') #'Spectral')
        plt.imshow(rot_img_2d, alpha = 0.5, cmap='gray') # https://rk1993.tistory.com/278
        plt.savefig(os.path.join(plt_saved_loc, f'Grad_CAM_{subj_id}_{seq}_cor.jpg'), dpi=300)
        plt.show()

      plt_saved_loc_3d = os.path.join(attention_map_dir, 'grad_CAM_3d')
      os.makedirs(plt_saved_loc_3d, exist_ok=True)

      if main_args.get_3d_grad_cam:
        # save as html
        matplotlib.rcParams['animation.embed_limit'] = 500 # 500 MB 까지 용량 상한 늘려줌 의미
        ani_html_t1ce = plot_slices_superimposed(superimposed_img_3d_t1ce, x_slice_num, y_slice_num, z_slice_num, use_midline=True)
        ani_html_flair = plot_slices_superimposed(superimposed_img_3d_flair, x_slice_num, y_slice_num, z_slice_num, use_midline=True)
      
        ani_html_t1ce_path = os.path.join(plt_saved_loc_3d, f'Grad_CAM_heatmap_{subj_id}_3d_t1ce.html')
        with open(ani_html_t1ce_path, 'w') as f:
          f.write(ani_html_t1ce)
        
        ani_html_flair_path = os.path.join(plt_saved_loc_3d, f'Grad_CAM_heatmap_{subj_id}_3d_flair.html')
        with open(ani_html_flair_path, 'w') as f:
          f.write(ani_html_flair)
      
      plt.close()