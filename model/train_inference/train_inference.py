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

# import pycox
# from pycox.models import * # LogisticHazard, DeepHitSingle, PMF
# from pycox.evaluation import * # EvalSurv
# from pycox.utils import * # kaplan_meier
# from torchtuples.callbacks import Callback

from utils import *
from attention_models import *

from adamp import AdamP

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.utils import resample
import argparse
import json

''' 
USAGE: 
python train_inference.py --spec_patho all --dataset_list SNUH UPenn severance 
--> main.py 돌릴 때와 동일한 args 를 입력해줘야 함
'''

args = config()

_, n_intervals = get_n_intervals(fixed_interval_width = 0)
args.n_intervals = n_intervals

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--spec_patho', type=str, default='all') # 'GBL' # 
    parser.add_argument('--spec_duration', type=str, default='1yr') # 'OS' # 
    parser.add_argument('--spec_event', type=str, default='death') # 'prog' # 
    parser.add_argument('--dataset_list', nargs='+', default=['UCSF','UPenn','TCGA','severance'], help='selected_training_datasets') # ,'TCGA'# ['SNUH','UPenn','TCGA']
    return parser

infer_args = get_args_parser().parse_args()
gpu_id = infer_args.gpu_id

args.dataset_name = '_'.join(infer_args.dataset_list)
print(f'Train dataset_name:{args.dataset_name}')

print(f'Using GPU {gpu_id}')
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

train_batch_size = 1

if args.net_architect == 'SEResNext50':
  base_model = monai.networks.nets.SEResNext50(spatial_dims=3, in_channels=4, num_classes=19)
elif args.net_architect == 'DenseNet':
  base_model = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=4, out_channels=19)
elif args.net_architect == 'resnet50_cbam':
  base_model = resnet50_cbam(num_classes=19)

model = CustomNetwork(args, base_model = base_model).to(device)
model = load_ckpt(args, model)
#%%

train_df_path = os.path.join(args.proc_label_dir, f'train_df_proc_labels_{args.dataset_name}.csv')
train_df = pd.read_csv(train_df_path, dtype='string')
train_df = train_df.set_index('ID')

print(f'train_df.index:{train_df.index}')
print(f'train_df.shape:{train_df.shape}')

train_transform = get_transform(args, f'{args.dataset_name}')
#%%

to_np = lambda x: x.detach().cpu().numpy()
# get_label_path = lambda dataset_name: os.path.join(args.label_dir, f'{dataset_name}_labels{args.use_correct}.csv')
get_label_path = lambda dataset: os.path.join(args.label_dir, f'{dataset}_{infer_args.spec_duration}_{infer_args.spec_patho}_{infer_args.spec_event}.csv')

get_target = lambda df: (np.array(df.index.values, dtype=str), # int: not working
                          np.array(df[f'duration_{infer_args.spec_event}'].tolist(), dtype=int), 
                          np.array(df[f'event_{infer_args.spec_event}'].tolist(), dtype=int))

train_data = SurvDataset(df = train_df, args = args, dataset_name = f'{args.dataset_name}', transforms=train_transform, aug_transform=False)
train_loader = DataLoader(dataset=train_data, batch_size=train_batch_size, num_workers=4, pin_memory=True, shuffle=False)
train_idx = train_df.index.values

df = pd.read_csv(get_label_path(f'{args.dataset_name}'), dtype='string')
df = df.set_index('ID')
train_df = df.loc[train_idx]
_, duration_train, event_train = get_target(train_df)
#%%
oneyr_survs_train = []
for inputs,labels in train_loader:
  model.eval()
  inputs = inputs.to(device)
  labels = labels.to(device)

  y_pred = model(inputs) # torch.Size([16, 19])
  
  print(f'y_pred:{y_pred}')
  print(f'labels:{labels}')
  
  ''' evaluate c-index 
  ref:
  https://lifelines.readthedocs.io/en/latest/lifelines.utils.html
  https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=cjh226&logNo=221380929786
  '''

  halflife=365.*2
  breaks=-np.log(1-np.arange(0.0,0.96,0.05))*halflife/np.log(2) 
  y_pred_np = to_np(y_pred)
  oneyr_surv_train=np.cumprod(y_pred_np[:,0:np.nonzero(breaks>365)[0][0]], axis=1)[:,-1]
  print(f'oneyr_surv_train: {oneyr_surv_train}')
  oneyr_survs_train.extend(oneyr_surv_train)
# print(len(oneyr_survs)) # 66
oneyr_survs_train = np.array(oneyr_survs_train)

original_c_index, ci_lower, ci_upper = bootstrap_cindex(duration_train, oneyr_survs_train, event_train)

print(f'Original C-index for train: {original_c_index:.4f}')
print(f'95% CI for C-index for train: ({ci_lower:.4f}, {ci_upper:.4f})')

score_train = get_BS(event_train, duration_train, oneyr_survs_train)