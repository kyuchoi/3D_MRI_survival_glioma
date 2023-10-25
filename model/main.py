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

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Deep survival GBL: image only', add_help=add_help)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--test_gpu_id', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=123456) 
    parser.add_argument('--spec_patho', type=str, default='all') # 'GBL' 
    parser.add_argument('--spec_duration', type=str, default='1yr') 
    parser.add_argument('--spec_event', type=str, default='death') 
    parser.add_argument('--ext_dataset_name', type=str, default='SNUH') 
    parser.add_argument('--dataset_list', nargs='+', default=['UCSF','UPenn','TCGA','severance'], help='selected_training_datasets') # ,'TCGA'
    parser.add_argument('--remove_idh_mut', default=False, type=str2bool)
    parser.add_argument('--save_grad_cam', default=False, type=str2bool)
    parser.add_argument('--biopsy_exclusion', default=False, type=str2bool)
    return parser

main_args = get_args_parser().parse_args()

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

os.environ['MKL_THREADING_LAYER'] = 'GNU'
set_seed(main_args.seed)
print(f'Setting seed:{main_args.seed}')

to_np = lambda x: x.detach().cpu().numpy()
to_cuda = lambda x: torch.from_numpy(x).float().device()

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
torch.cuda.empty_cache()

get_label_path = lambda dataset: os.path.join(args.label_dir, f'{dataset}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}.csv') #  'manual_labels', #
get_target = lambda df: (np.array(df.index.values, dtype=str), # int: not working
                          np.array(df[f'duration_{main_args.spec_event}'].tolist(), dtype=int), 
                          np.array(df[f'event_{main_args.spec_event}'].tolist(), dtype=int))

df = save_label_dataset_list(main_args, args)
ext_df = save_label_ext_dataset(main_args, args)

combine_img(main_args, args)

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
  print(f'dataset_name:{dataset_name}, {len(comm_list)}')
  
  df = df.loc[sorted(comm_list)]

  print(f'{dataset_name} df.shape: {df.shape}')

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

df_proc_labels_train, event_train, duration_train = make_kfold_df_proc_labels(args, f'{args.dataset_name}', remove_idh_mut = main_args.remove_idh_mut)
df_proc_labels_test, event_test, duration_test = make_kfold_df_proc_labels(args, f'{main_args.ext_dataset_name}', remove_idh_mut = main_args.remove_idh_mut)

print(f'train transform:')
args.train_transform = get_transform(args, f'{args.dataset_name}')
print(f'valid transform:')
args.valid_transform = get_transform(args, f'{args.dataset_name}')
print(f'test transform:')
test_transform = get_transform(args, f'{main_args.ext_dataset_name}')

if args.net_architect == 'SEResNext50':
  base_model = monai.networks.nets.SEResNext50(spatial_dims=3, in_channels=4, num_classes=args.n_intervals)
elif args.net_architect == 'DenseNet':
  base_model = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=4, out_channels=args.n_intervals)
elif args.net_architect == 'resnet50_cbam':
  base_model = resnet50_cbam(num_classes=args.n_intervals)

model = CustomNetwork(args, base_model = base_model).to(device)

base_optimizer = AdamP
optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, weight_decay=args.weight_decay) # https://stackoverflow.com/questions/43779500/pytorch-network-parameters-missing-1-required-positional-argument-self
criterion = nnet_loss 
scheduler = fetch_scheduler(optimizer)

if not main_args.save_grad_cam:
  model, history = run_fold(df_proc_labels_train, args, model, criterion, optimizer, scheduler, device=device, fold=0, num_epochs=main_args.epochs)

test_gpu_id = main_args.test_gpu_id 
print(f'Testing on GPU {test_gpu_id}')

test_device = torch.device(test_gpu_id)

model = CustomNetwork(args, base_model = base_model).to(test_device)
model = load_ckpt(args, model)

proc_label_path_test = os.path.join(args.proc_label_dir, f'{main_args.ext_dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_proc_labels.csv')
df_proc_labels_test = pd.read_csv(proc_label_path_test, dtype='string')
df_proc_labels_test = df_proc_labels_test.set_index('ID')

test_data = SurvDataset(df = df_proc_labels_test, args = args, dataset_name=f'{main_args.ext_dataset_name}', transforms=test_transform, aug_transform=False)
test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=4, pin_memory=True, shuffle=False) 

df_DL_score_test = df_proc_labels_test.copy()
df_DL_score_test.drop(columns=df_DL_score_test.columns,inplace=True)

for i in np.arange(n_intervals):
  df_DL_score_test.insert(int(i), f'MRI{i+1}', '')
df_DL_score_test.insert(n_intervals, 'oneyr_survs_test', '')

oneyr_survs_test = []
for subj_num, (inputs,labels) in enumerate(test_loader):
  model.eval()
  inputs = inputs.to(test_device)
  labels = labels.to(test_device)

  y_pred = model(inputs) 

  halflife=365.*2
  breaks=-np.log(1-np.arange(0.0,0.96,0.05))*halflife/np.log(2) 
  y_pred_np = to_np(y_pred)

  cumprod = np.cumprod(y_pred_np[:,0:np.nonzero(breaks>365)[0][0]], axis=1)
  oneyr_surv_test = cumprod[:,-1]
  print(f'oneyr_surv_test: {oneyr_surv_test}')
  
  DL_scores = []
  for n_interval in np.arange(1, n_intervals+1):
    DL_score = np.cumprod(y_pred_np[:,0:n_interval], axis=1)[:,-1][0]

    DL_scores.append(DL_score)
  DL_scores.append(oneyr_surv_test[0])

  df_DL_score_test.loc[df_DL_score_test.index[subj_num]] = DL_scores
  oneyr_survs_test.extend(oneyr_surv_test)

print(f'df_DL_score_test.shape:{df_DL_score_test.shape}')
DL_score_path = os.path.join(DL_score_dir, f'{main_args.ext_dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_DL_score_s{main_args.seed}.csv')
df_DL_score_test.to_csv(DL_score_path)

print(len(oneyr_survs_test)) 
oneyr_survs_test = np.array(oneyr_survs_test)

#%%
print(f'duration_test.shape:{duration_test.shape}')
print(f'oneyr_survs_test.shape:{oneyr_survs_test.shape}')
print(f'event_test.shape:{event_test.shape}')

original_c_index, ci_lower, ci_upper = bootstrap_cindex(duration_test, oneyr_survs_test, event_test)

print(f'Original C-index for valid: {original_c_index:.4f}')
print(f'95% CI for C-index for valid: ({ci_lower:.4f}, {ci_upper:.4f})')

score_test = get_BS(event_test, duration_test, oneyr_survs_test)

plt.switch_backend('agg')

test_img_path='/mnt/hdd3/mskim/GBL/data/SNUH/resized_BraTS/73116251/'
seqs = []
for seq in ['t1','t2','flair','t1ce']:
  seq=nib.load(os.path.join(test_img_path, f'{seq}_resized.nii.gz')).get_fdata() 
  print(f'seq range:min {seq.min()}-max {seq.max()}')
  seqs.append(seq)

x = np.stack(seqs, axis=0)
x = torch.from_numpy(x)
x = torch.unsqueeze(x, axis=0)

z_slice_num = int(x.shape[-1]//2) 
y_slice_num = int(x.shape[-2]//2) 
x_slice_num = int(x.shape[-3]//2) 

seq_idx_dict = {'t1':0, 't2':1, 't1ce':2, 'flair':3}
selected_seq = 't1ce'
selected_seq_idx = seq_idx_dict[selected_seq]

slice_3d = lambda x: x[selected_seq_idx,:,:,:]

if main_args.save_grad_cam:
  cam_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
  cam_model = medcam.inject(model, backend='gcampp', output_dir="attention_maps", save_maps=True)

  superimposed_imgs = []
  cam_model.eval()

  for subj_num, batch in enumerate(cam_loader):
    batch = batch[0].to(test_device) 
    output = cam_model(batch)
    cam=cam_model.get_attention_map()
    subj_id = df_DL_score_test.index[subj_num]

    img_4d = batch.squeeze().cpu().numpy()
    img_3d = slice_3d(img_4d)

    img_3d_scaled = min_max_norm(img_3d)

    result_3d = cam.squeeze()
 
    superimposed_img_3d, result_3d_resized = superimpose_img(img_3d_scaled, result_3d)
      
    plt_saved_loc_3d = os.path.join(attention_map_dir, 'grad_CAM_3d')
    os.makedirs(plt_saved_loc_3d, exist_ok=True)

    matplotlib.rcParams['animation.embed_limit'] = 500 
    ani_html = plot_slices_superimposed(superimposed_img_3d)
   
    # save as html
    animation_html_path = os.path.join(plt_saved_loc_3d, f'Grad_CAM_heatmap_{subj_id}_3d.html')
    with open(animation_html_path, 'w') as f:
      f.write(ani_html)
    plt.close()
