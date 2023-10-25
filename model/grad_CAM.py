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

import matplotlib.pyplot as plt
import plotly.io as pio
from plotly.offline import plot

from skimage.transform import resize
from scipy import ndimage

from medcam import medcam

from utils import *
from attention_models import *

from adamp import AdamP

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.utils import resample
import argparse
import json

args = config()

_, n_intervals = get_n_intervals(fixed_interval_width = 0)
args.n_intervals = n_intervals

attention_map_dir=os.path.join(r'/mnt/hdd3/mskim/GBL/code/experiment/attention_maps', 'test_0527')
os.makedirs(attention_map_dir, exist_ok=True)

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_gpu_id', type=int, default=1)
    parser.add_argument('--spec_patho', type=str, default='all') # 'GBL' # 
    parser.add_argument('--spec_duration', type=str, default='1yr') # 'OS' # 
    parser.add_argument('--spec_event', type=str, default='death') # 'prog' # 
    parser.add_argument('--dataset_list', nargs='+', default=['UCSF','UPenn','TCGA','severance'], help='selected_training_datasets') # ,'TCGA'# ['SNUH','UPenn','TCGA']
    return parser

infer_args = get_args_parser().parse_args()
test_gpu_id = infer_args.test_gpu_id

args.dataset_name = '_'.join(infer_args.dataset_list)
print(f'Train dataset_name:{args.dataset_name}')

print(f'Using GPU {test_gpu_id}')
device = torch.device(f'cuda:{test_gpu_id}' if torch.cuda.is_available() else 'cpu')

valid_batch_size = 1

if args.net_architect == 'SEResNext50':
  base_model = monai.networks.nets.SEResNext50(spatial_dims=3, in_channels=4, num_classes=19)
elif args.net_architect == 'DenseNet':
  base_model = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=4, out_channels=19)
elif args.net_architect == 'resnet50_cbam':
  base_model = resnet50_cbam(num_classes=19)

model = CustomNetwork(args, base_model = base_model).to(device)
model = load_ckpt(args, model)

valid_df_path = os.path.join(args.proc_label_dir, f'valid_df_proc_labels_{args.dataset_name}.csv')
valid_df = pd.read_csv(valid_df_path, dtype='string')
valid_df = valid_df.set_index('ID')

print(f'valid_df.index:{valid_df.index}')
print(f'valid_df.shape:{valid_df.shape}')

valid_transform = get_transform(args, f'{args.dataset_name}')
#%%

to_np = lambda x: x.detach().cpu().numpy()
get_label_path = lambda dataset: os.path.join(args.label_dir, f'{dataset}_{infer_args.spec_duration}_{infer_args.spec_patho}_{infer_args.spec_event}.csv')

get_target = lambda df: (np.array(df.index.values, dtype=str), # int: not working
                          np.array(df[f'duration_{infer_args.spec_event}'].tolist(), dtype=int), 
                          np.array(df[f'event_{infer_args.spec_event}'].tolist(), dtype=int))

valid_data = SurvDataset(df = valid_df, args = args, dataset_name = f'{args.dataset_name}', transforms=valid_transform, aug_transform=False)
valid_loader = DataLoader(dataset=valid_data, batch_size=valid_batch_size, num_workers=4, pin_memory=True, shuffle=False)
valid_idx = valid_df.index.values

df = pd.read_csv(get_label_path(f'{args.dataset_name}'), dtype='string')
df = df.set_index('ID')
valid_df = df.loc[valid_idx]
_, duration_valid, event_valid = get_target(valid_df)

#%%
oneyr_survs_valid = []
for inputs,labels in valid_loader:
  model.eval()
  inputs = inputs.to(device)
  labels = labels.to(device)

  y_pred = model(inputs) # torch.Size([16, 19])
  print(f'y_pred:{y_pred}')
  print(f'labels:{labels}')
  
  halflife=365.*2
  breaks=-np.log(1-np.arange(0.0,0.96,0.05))*halflife/np.log(2) 
  # breaks=np.arange(0.,365.*5,365./8)
  y_pred_np = to_np(y_pred)
  
  oneyr_surv_valid = np.cumprod(y_pred_np[:,0:np.nonzero(breaks>365)[0][0]], axis=1)[:,-1]
  oneyr_survs_valid.extend(oneyr_surv_valid)

oneyr_survs_valid = np.array(oneyr_survs_valid)

original_c_index, ci_lower, ci_upper = bootstrap_cindex(duration_valid, oneyr_survs_valid, event_valid)

print(f'Original C-index for valid: {original_c_index:.4f}')
print(f'95% CI for C-index for valid: ({ci_lower:.4f}, {ci_upper:.4f})')

score_valid = get_BS(event_valid, duration_valid, oneyr_survs_valid)

''' grad CAM '''

plt.switch_backend('agg')

test_img_path='/mnt/hdd3/mskim/GBL/data/SNUH/resized_BraTS/73116251/'
seqs = []
for seq in ['t1','t2','flair','t1ce']:
  seq=nib.load(os.path.join(test_img_path, f'{seq}_resized.nii.gz')).get_fdata() # _seg # _cropped
  print(f'seq range:min {seq.min()}-max {seq.max()}')

  seqs.append(seq)

x = np.stack(seqs, axis=0)
x = torch.from_numpy(x)
x = torch.unsqueeze(x, axis=0)
print(f'x.shape:{x.shape}') 

z_slice_num = int(x.shape[-1]//2) 
y_slice_num = int(x.shape[-2]//2) 
x_slice_num = int(x.shape[-3]//2) 

seq_idx_dict = {'t1':0, 't2':1, 't1ce':2, 'flair':3}
selected_seq = 't1ce'
selected_seq_idx = seq_idx_dict[selected_seq]
print(f'selected_seq:{selected_seq}, {selected_seq_idx}')

slice_3d = lambda x: x[selected_seq_idx,:,:,:]

cam_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
cam_model = medcam.inject(model, backend='gcampp', output_dir="attention_maps", save_maps=True)

superimposed_imgs = []
cam_model.eval()

rot_degree = 90

for subj_num, batch in enumerate(cam_loader):
  batch = batch[0].to(device) # cuda()
  output = cam_model(batch)
  cam=cam_model.get_attention_map()
  print(type(cam)) # numpy array 
  print(f'cam.shape:{cam.shape}') 
  print(f'input.shape:{batch.shape}') 

  subj_id = valid_df.index[subj_num]
  print(f'subj_id:{subj_id}')

  img_4d = batch.squeeze().cpu().numpy()
  img_3d = slice_3d(img_4d)

  img_3d_scaled = min_max_norm(img_3d)

  result_3d = cam.squeeze()

  print(f'img_3d.shape:{img_3d.shape}') 
  print(f'result_3d.shape:{result_3d.shape}') 
  
  superimposed_img_3d, result_3d_resized = superimpose_img(img_3d_scaled, result_3d, alpha=0.3) 

  print(f'superimposed_img_3d.shape:{superimposed_img_3d.shape}')
  print(f'result_3d_resized.shape:{result_3d_resized.shape}')
  plt_saved_loc_3d = os.path.join(attention_map_dir, 'grad_CAM_3d')
  os.makedirs(plt_saved_loc_3d, exist_ok=True)
  
  plot_slices_superimposed(superimposed_img_3d)
  plt.savefig(os.path.join(plt_saved_loc_3d, f'Grad_CAM_heatmap_{subj_id}_3d.jpg'), dpi=300)
  plt.close()

  superimposed_imgs.append(superimposed_img_3d)
