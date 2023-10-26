#%% 

import os
import csv
import argparse
import pickle
from tqdm import tqdm
from datetime import datetime
from distutils.dir_util import copy_tree
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from collections import OrderedDict
from attention_models import resnet50_cbam, se_resnext50

from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler

import cv2
from monai.visualize import * # GradCAM

import monai
from monai.networks.nets import *
from monai.transforms import RandFlip, Rand3DElastic, RandAffine, RandGaussianNoise, AdjustContrast, RandSpatialCrop # Rand3DElastic

import torch # For building the networks 
import torch.nn as nn
import torch.nn.functional as F
import torchtuples as tt # Some useful functions
import torchio as tio
from torch.utils.data import *
import torchvision.models as models

from utils import *

import pycox
from pycox.models import * # LogisticHazard, DeepHitSingle, PMF
from pycox.evaluation import * # EvalSurv
from pycox.utils import * # kaplan_meier
from torchtuples.callbacks import Callback

start_time = datetime.now()

def get_args_parser(add_help=True):
  parser = argparse.ArgumentParser(description='SNUH GBM deep survival prediction model', add_help=add_help)
  parser.add_argument('--load_best_model', action='store_true', help='load_best_model')
  parser.add_argument('--load_best_dataset', action='store_true', help='make new dataset')
  parser.add_argument('--use_fullsize', action='store_true', help='not using half-resize')
  parser.add_argument('--augment_transform', action='store_true', help='augment_transform')
  parser.add_argument('--use_combined_net', action='store_true', help='use_combined_net')
  parser.add_argument('--use_onecycleLR', action='store_true', help='use_onecycleLR')
  parser.add_argument('--remove_idh_mut', action='store_true', help='for subgroup analysis of IDH-wt, removing IDH-mut')
  parser.add_argument('--selected_sequence', nargs='+', default=['flair','t1ce','t2','t1'], help='selected_MR_sequences')
  parser.add_argument('--clin_vars', nargs='+', default= ['age','MGMT','IDH','sex'], help='selected_clinical_variables')
  parser.add_argument('--use_logistic_hazard', action='store_true', help='use_logistic_hazard')
  parser.add_argument('--root_dir', type=str, default=r'/mnt/hdd2/kschoi/GBL')
  parser.add_argument('--net_architect', type=str, default='DenseNet')
  parser.add_argument('--resize_type', type=str, default='bhk')
  parser.add_argument('--int_dataset_name', type=str, default='SNUH_merged') 
  parser.add_argument('--ext_dataset_name', type=str, default='severance') 
  parser.add_argument('--gpu_id', type=int, default=3)
  parser.add_argument('--split_ratio', type=float, default=0.8) 
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--epochs', type=int, default=25)
  parser.add_argument('--num_durations', type=int, default=11)
  parser.add_argument('--seed', type=int, default=987654)
  parser.add_argument('--get_num_durations', action='store_true')
  
  return parser
 
def set_seed(seed):
    # random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = True
    
SEED = args.seed 
set_seed(SEED)

''' set experiment '''
MAKE_NEW_DATASET = not args.load_best_dataset # False
GET_NUM_DURATIONS = args.get_num_durations # False

LOAD_BEST_DATASET = args.load_best_dataset
LOAD_BEST_MODEL = args.load_best_model 

AUGMENT_TRANSFORM = args.augment_transform
REMOVE_LONG_DURATION = args.remove_long_duration
REMOVE_IDH_MUT = args.remove_idh_mut
REMOVE_NON_GBL = args.remove_non_GBL

USE_FULLSIZE = args.use_fullsize  
use_onecycleLR = args.use_onecycleLR
USE_LOGISTIC_HAZARD = args.use_logistic_hazard
DURATION_THRESHOLD = args.duration_threshold

NET_ARCHITECT = args.net_architect
print(f'Using {NET_ARCHITECT}')

comm_suffix = args.comm_suffix 
n_channels = 16
global HEURISTIC_NUM
HEURISTIC_NUM = 64 if NET_ARCHITECT == 'DenseNet' else 16

RESIZE_TYPE = args.resize_type
split = f'preprocessed_{RESIZE_TYPE}' if USE_FULLSIZE else f'seg_{RESIZE_TYPE}'
print(f'split:{split}')
split_RATIO = args.split_ratio 
print(f'split_RATIO:{split_RATIO}')
epochs = args.epochs 
batch_size = args.batch_size
print(f'batch_size:{batch_size}')
#%%
PATHOLOGY = args.pathology
print(f'PATHOLOGY:{PATHOLOGY}')

PROGRESS_DURATION = args.progress_duration
print(f'PROGRESS_DURATION:{PROGRESS_DURATION}')

#%% 
resize_dict = {'bhk': (150,180,150),  
              'BraTS': (240,240,155),
              'SNUH': (180,256,256)}

IMG_SIZE = resize_dict[RESIZE_TYPE] 
RESIZE = IMG_SIZE if USE_FULLSIZE else (round(IMG_SIZE[0]/2),round(IMG_SIZE[1]/2),round(IMG_SIZE[2]/2))
print(f'image size:{RESIZE}')

CLIN_VARS = args.clin_vars # ['age','MGMT','IDH','sex']
print(f'SELECTED_CLIN_VARS:{CLIN_VARS}')
NUM_CLIN_VARS = len(CLIN_VARS) # 4 

global SELECTED_SEQUENCE
SELECTED_SEQUENCE = args.selected_sequence
print(f'SELECTED_SEQUENCE:{SELECTED_SEQUENCE}')
in_features = len(SELECTED_SEQUENCE)

deepsurv_name = 'LH' if USE_LOGISTIC_HAZARD else 'DH' # LogisticHazard better loss curve (i.e. similar loss between train and val loss) than DeepHit 
combination = 'mri'
use_augment = 'aug' if AUGMENT_TRANSFORM else 'no_aug'
EXP_DESCRIP = f'{deepsurv_name}_seq{len(SELECTED_SEQUENCE)}_b{batch_size}_{combination}_{NET_ARCHITECT}_{use_augment}_s{SEED}_b{batch_size}_spl{split_RATIO}_e{epochs}_{PATHOLOGY}_{PROGRESS_DURATION}' # noCrop_noMask
EARLY_PROG = args.early_prog

''' set path '''
ROOT_DIR = args.root_dir
LABEL_DIR = os.path.join(ROOT_DIR, 'label', 'surv_labels')
DL_feature_DIR = os.path.join(LABEL_DIR, 'DL_features', 'best', f'{EXP_DESCRIP}')
os.makedirs(DL_feature_DIR, exist_ok=True)
INT_DATASET_NAME = args.int_dataset_name
DATA_DIR = os.path.join(ROOT_DIR, INT_DATASET_NAME) # 'SNUH_glioma' 
EXT_DATASET_NAME = args.ext_dataset_name
EXT_DATA_DIR = os.path.join(ROOT_DIR, EXT_DATASET_NAME) 
INPUT_DIR = os.path.join(DATA_DIR, split)
TARGET_DIR = os.path.join(DATA_DIR, 'experiment', EXP_DESCRIP)
BEST_TARGET_DIR = os.path.join(DATA_DIR, 'experiment', 'BEST_LH_seq4_n11_b8_noCrop_noMask_SEResNext50')
os.makedirs(TARGET_DIR, exist_ok = True)

''' save setting '''
args_dict = args.__dict__
config_path = os.path.join(TARGET_DIR, 'config.csv')

with open(config_path, 'w') as f:  
    writer = csv.writer(f)
    writer.writerow(['setting', 'value'])
    for k, v in args_dict.items():
      writer.writerow([k, v])

get_label_path = lambda dataset_name: os.path.join(LABEL_DIR, f'{dataset_name}_labels.csv') # f'{dataset_name}_{PATHOLOGY}_{PROGRESS_DURATION}_comm.csv') # f'{dataset_name}_all_1yr_comm.csv') # f'{dataset_name}_survival_curated_final.csv')
''' multi gpu is not working with torchtuples dataloader: instead USE CUDA_VISIBLE_DEVICES=2 python xxx.py, or as follows to use nohup, and do experiment parallel '''

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
torch.cuda.empty_cache()

convert_list2df = lambda idlist: pd.DataFrame(idlist, columns = ['ID'], dtype='string').set_index('ID')

def df_filter_halflife(dataset_name, split, remove_long_duration = REMOVE_LONG_DURATION, remove_idh_mut = REMOVE_IDH_MUT, remove_non_GBL = REMOVE_NON_GBL, exclusion = True): 
  
  data_dir = os.path.join(ROOT_DIR, dataset_name)
  idlist = sorted([ID for ID in os.listdir(os.path.join(data_dir, split)) if os.path.isdir(os.path.join(data_dir, split, ID))])
  n_sample = len(idlist)
  print(f'n_sample of {dataset_name}:{n_sample}')

  duration_list = []
  event_list = []
  
  df = pd.read_csv(get_label_path(dataset_name), index_col=0, dtype='string') # int: not working
  df = df.sort_index(ascending=True)
  df=df.astype({'event':int, 'duration':int})
  
  if remove_long_duration:
    print(f'before removing IDs with duration > {DURATION_THRESHOLD}: {len(idlist)}')
    condition = df.duration.astype(int) > DURATION_THRESHOLD
    filtered_ID = df[condition].index.tolist() 
    idlist = [ID for ID in idlist if (ID not in filtered_ID)]
    print(f'after removing IDs with duration > {DURATION_THRESHOLD}: {len(idlist)}')

  df_idlist = convert_list2df(idlist)
  # print(df_idlist.index)
  comm_idx = df.index.intersection(df_idlist.index)
  df_subj_ids=df.loc[comm_idx]
  # print(df_subj_ids.head())
  
  duration_list = np.array(df_subj_ids["duration"].values.tolist(), dtype=int)
  event_list = np.array(df_subj_ids["event"].values.tolist(), dtype=int)

  halflife1 = int(np.median(duration_list)) # 354 in SNUH vs 165 in TCGA
  halflife = halflife1 * 2
  print(f'halflife of {dataset_name}:{halflife}') 
  
  # df_subj_ids = df.loc[sorted(idlist),:]
  len_df_subj_ids = df_subj_ids.shape[0]
  print(f'len_df_subj_ids:{len_df_subj_ids}')
  
  print(f'df.shape:{df.shape}')

  return df_subj_ids, halflife, idlist

def df_transform_early(df, early_threshold = 185):

  df['event'] = [ 1 if duration < early_threshold and event == 1 else 0 for duration, event in zip(df['duration'], df['event'])]  
  return df
  
df, HALFLIFE, IDLIST = df_filter_halflife(INT_DATASET_NAME, split, REMOVE_IDH_MUT, REMOVE_NON_GBL, exclusion = False)
ext_df, EXT_HALFLIFE, EXT_IDLIST = df_filter_halflife(EXT_DATASET_NAME, split, REMOVE_IDH_MUT, REMOVE_NON_GBL, exclusion = False ) # True) # 

print(f'df.shape:{df.shape}')
print(f'ext_df.shape:{ext_df.shape}')

print(f'length of ID LIST:{len(IDLIST)}')
IDLIST = [ID for ID in IDLIST if ID not in EXT_IDLIST]
print(f'length of ID LIST after removing excluding SNUH_temporal list:{len(IDLIST)}')

#%%
print(f'df event sum:{df["event"].sum()}')
print(f'ext_df event sum:{ext_df["event"].sum()}')

df_early = df_transform_early(df) if EARLY_PROG else df
ext_df_early = df_transform_early(ext_df) if EARLY_PROG else ext_df

print(f'df_early event sum:{df_early["event"].sum()}')
print(f'ext_df_early event sum:{ext_df_early["event"].sum()}')

#%%
def calculate_num_durations(halflife):
  
    breaks=-np.log(1-np.arange(0.0,0.96,0.05))*halflife/np.log(2) 
    num_durations = len(breaks)-1 # tutorial: 20 -> our data: 19
    print(f"num_durations:{num_durations}")
    
    return num_durations
  
if GET_NUM_DURATIONS:
  # NUM_DURATIONS = max(calculate_num_durations(HALFLIFE), calculate_num_durations(EXT_HALFLIFE)) # np.min()는 에러 # min
  NUM_DURATIONS = int(np.average([calculate_num_durations(HALFLIFE), calculate_num_durations(EXT_HALFLIFE)]))
else:
  NUM_DURATIONS = args.num_durations
  
print(f'NUM_DURATIONS: {NUM_DURATIONS}')
out_features = NUM_DURATIONS

#%% 
''' transform labels for DeepHit, like labtransform in LogisticHazard '''
def set_labtrans(halflife, num_durations, use_logistic_hazard):
    
    if use_logistic_hazard:
      print("using logistic hazard")
      scheme = 'quantiles'
      labtrans = LogisticHazard.label_transform(num_durations, scheme)
      
    else:
      print("using DeepHitSingle")
      labtrans = DeepHitSingle.label_transform(num_durations)
    
    return labtrans

labtrans = set_labtrans(HALFLIFE, NUM_DURATIONS, USE_LOGISTIC_HAZARD)

get_target = lambda df: (np.array(df['duration'].tolist(), dtype=int),
                          np.array(df['event'].tolist(), dtype=int))

def get_labtrans(df, sub_id_list, labtrans):
  
    df_sub_IDs = df.loc[sorted(sub_id_list),:]
  
    print(f'get_labtrans length: {df_sub_IDs.shape[0]}')
    return labtrans.fit_transform(*get_target(df_sub_IDs))

def random_split(id_list, split_ratio):
  ''' df: dataframe for total dataset '''
  n_sample = len(id_list) 
  id_list = sorted(id_list)
  train_nums = np.random.choice(n_sample, size = int(split_ratio * n_sample), replace = False)
  print(f'train_nums:{len(train_nums)}')
  val_nums = [num for num in np.arange(n_sample) if num not in train_nums]
  
  return train_nums, val_nums

train_nums, val_nums = random_split(IDLIST, split_RATIO)
train_IDs, val_IDs = np.array(IDLIST)[train_nums], np.array(IDLIST)[val_nums]

landmarks_dir= os.path.join(ROOT_DIR, 'histograms', INT_DATASET_NAME)

t1_landmarks_path = os.path.join(landmarks_dir, 't1_histgram.npy')
t2_landmarks_path = os.path.join(landmarks_dir, 't2_histgram.npy')
t1ce_landmarks_path = os.path.join(landmarks_dir, 't1ce_histgram.npy')
flair_landmarks_path = os.path.join(landmarks_dir, 'flair_histgram.npy')

landmarks = {'t1': t1_landmarks_path, 
            't2': t2_landmarks_path, 
            't1ce': t1ce_landmarks_path, 
            'flair': flair_landmarks_path 
            }

transforms = [
    tio.HistogramStandardization(landmarks),
    tio.ZNormalization()
]

aug_transforms = [
    tio.RandomMotion(p=0.4),
    tio.RandomNoise(p=0.3),
    tio.RandomAffine(p=0.3),
]

if AUGMENT_TRANSFORM:
  train_transform = tio.Compose(transforms+aug_transforms)
else:
  train_transform = tio.Compose(transforms)
val_transform = tio.Compose(transforms)

to_np = lambda x: x.detach().cpu().numpy()
to_cuda = lambda x: torch.from_numpy(x).float().cuda() # to(device)


#%%
class make_subjs_dataset:
  ''' 
  make torchio Dataset using specified ID list (i.e. sub_IDLIST) 
  1) loading image data from subject directory, 
  2) and duration/event data from dataframe which is indexed by sub_IDLIST      
  '''
  def __init__(self, dataset_name, CLIN_VARS, SEQUENCE):
    self.dataset_name = dataset_name
    print(f'dataset_name:{self.dataset_name}')
    self.data_dir = os.path.join(ROOT_DIR, dataset_name)
    self.CLIN_VARS = CLIN_VARS
    self.SEQUENCE = SEQUENCE
    self.df = pd.read_csv(get_label_path(self.dataset_name), index_col=0, dtype='string') # int: not working

  def __call__(self, sub_IDLIST, transform_, suffix = "seg"): # seg # cropped
    subjects = []
    sub_IDLIST = sorted(sub_IDLIST)
    
    df_sub_id_list = convert_list2df(sub_IDLIST)
    sub_IDLIST = df_sub_id_list.index.intersection(self.df.index)
    
    for idx, ID in enumerate(sub_IDLIST):            
        IMG_DIR = os.path.join(self.data_dir, split, ID)
        
        subject = tio.Subject(
            t1=tio.ScalarImage(os.path.join(IMG_DIR, f't1_seg.nii.gz')), # _seg
            t2=tio.ScalarImage(os.path.join(IMG_DIR, f't2_seg.nii.gz')), 
            t1ce=tio.ScalarImage(os.path.join(IMG_DIR, f't1ce_seg.nii.gz')), 
            flair=tio.ScalarImage(os.path.join(IMG_DIR, f'flair_seg.nii.gz')), 
            mask=tio.LabelMap(os.path.join(IMG_DIR, 'WT_seg_resized.nii.gz')),
            ID=ID
            )   
        
        if suffix == "cropped":
          
          transform_ = tio.Compose([
            tio.ToCanonical(),
            tio.Resample('t2'),
            tio.CropOrPad(RESIZE,
                          mask_name = 'mask'
                          ),  
            tio.Resize([96, 96, 96])
          ])
            
          transform_subject = transform_(subject)
          # transform_subject.plot(reorient=False)
          
          transform_subject_path = os.path.join(IMG_DIR, 't1_cropped.nii.gz')
          transform_subject.t1.save(transform_subject_path)
          
          transform_subject_path = os.path.join(IMG_DIR, 't2_cropped.nii.gz')
          transform_subject.t2.save(transform_subject_path)
          # print(f'transform_subject.t2.shape:{transform_subject.t2.shape}') # (96,96,96)
          transform_subject_path = os.path.join(IMG_DIR, 't1ce_cropped.nii.gz')
          transform_subject.t1ce.save(transform_subject_path)
          
          transform_subject_path = os.path.join(IMG_DIR, 'flair_cropped.nii.gz')
          transform_subject.flair.save(transform_subject_path)
          
          transform_subject_path = os.path.join(IMG_DIR, 'WT_seg_cropped.nii.gz')
          transform_subject.mask.save(transform_subject_path)
          
          subjects.append(transform_subject)
          
        else:
          subjects.append(subject)
        
    dataset = tio.SubjectsDataset(subjects, transform=transform_)
    concat_seq_img = lambda x: torch.cat([x[sequence][tio.DATA] for sequence in self.SEQUENCE], axis=0)
    
    imgs = []  

    sub_IDLIST = sorted(sub_IDLIST)
    for idx, ID in enumerate(tqdm(sub_IDLIST)):
      
      img = concat_seq_img(dataset[idx])
      imgs.append(img)      
    
    images = torch.stack(imgs)
    duration, event = get_labtrans(self.df, sub_IDLIST, labtrans)    

    duration_tensor, event_tensor = torch.from_numpy(duration), torch.from_numpy(event) # tt.tuplefy(target[0], target[1]).to_tensor()
    
    duration_tensor = duration_tensor.long() 
    event_tensor = event_tensor.float()
    
    targets = (duration_tensor, event_tensor)
    
    return images, targets, sub_IDLIST

x_train, y_train, train_IDs = make_subjs_dataset(INT_DATASET_NAME, CLIN_VARS, SELECTED_SEQUENCE)(train_IDs, None, suffix = comm_suffix) # train_transform)
x_val, y_val, val_IDs = make_subjs_dataset(INT_DATASET_NAME, CLIN_VARS, SELECTED_SEQUENCE)(val_IDs, None, suffix = comm_suffix) # val_transform)
x_test, y_test, EXT_IDLIST = make_subjs_dataset(EXT_DATASET_NAME, CLIN_VARS, SELECTED_SEQUENCE)(EXT_IDLIST, None, suffix = comm_suffix) # val_transform)

IDLIST = sorted(train_IDs + val_IDs)
EXT_IDLIST = sorted(EXT_IDLIST)

x_val = x_val.float().cuda()
x_test = x_test.float().cuda()

class MakeDatasetSingle(Dataset):

  def __init__(self, imgs, duration, event):

    self.imgs = imgs
    self.duration, self.event = tt.tuplefy(duration, event).to_tensor()

    self.znorm = tio.ZNormalization()
    self.rescale = tio.RescaleIntensity(out_min_max=(-1, 1))
    
  def __len__(self):
    return len(self.imgs)
  
  def __getitem__(self, index):
    if type(index) is not int:
      raise ValueError(f"Need `index` to be `int`. Got {type(index)}.")

    img = self.imgs[index]
    img = self.znorm(img)
    
    print(f'img.mean:{img.mean()}')
    
    return img, (self.duration[index], self.event[index])

def collate_fn(batch):

  return tt.tuplefy(batch).stack()


class MakeDatasetSingleTrain(Dataset):

  def __init__(self, imgs, duration, event):

    self.imgs = imgs
    self.duration, self.event = tt.tuplefy(duration, event).to_tensor()

    self.znorm = tio.ZNormalization()
    self.rescale = tio.RescaleIntensity(out_min_max=(-1, 1))
    self.crop_size = 64
    self.crop = RandSpatialCrop(roi_size=(self.crop_size, self.crop_size, self.crop_size), random_size=False)
    
    self.rand_affiner = RandAffine(prob=0.9)
    self.rand_elastic = Rand3DElastic(prob=0.8, magnitude_range = [-1,1], sigma_range = [0,1])
    self.flipper1 = RandFlip(prob=0.5, spatial_axis=0)
    self.flipper2 = RandFlip(prob=0.5, spatial_axis=1)
    self.flipper3 = RandFlip(prob=0.5, spatial_axis=2)
    self.gaussian = RandGaussianNoise(prob=0.3)
    self.contrast = AdjustContrast(gamma=2)

  def __len__(self):
    return len(self.imgs)
  
  def augment(self, img):
    img = self.znorm(img)
    img = self.crop(img)
    img = self.rand_elastic(img)
    img = self.rand_affiner(img)
    img = self.flipper1(img)
    img = self.flipper2(img)
    img = self.flipper3(img)
    img = self.gaussian(img)
    
    return img

  def __getitem__(self, index):
    if type(index) is not int:
      raise ValueError(f"Need `index` to be `int`. Got {type(index)}.")

    img = self.imgs[index]
    augmented_img = self.augment(img)
    print(f'augmented_img.mean:{augmented_img.mean()}')  
    
    return augmented_img, (self.duration[index], self.event[index])

def collate_fn(batch):
  
  return tt.tuplefy(batch).stack()

train_dataset = MakeDatasetSingleTrain(x_train, *y_train)
val_dataset = MakeDatasetSingle(x_val, *y_val)
test_dataset = MakeDatasetSingle(x_test, *y_test)

#%%
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn = collate_fn) # NOT WORKING with torchtuples!!: num_workers=NUM_WORKERS)#, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, collate_fn = collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, collate_fn = collate_fn)

#%%
''' select model from monai '''
def get_back_net(net_architect, in_features, out_features, n_channels):
  if net_architect == 'customCNN':
    
    back_net = UNet_encoder(in_channels=in_features, num_classes=out_features, n_channels=n_channels)

  elif net_architect == 'SEResNext50':
    
    back_net = monai.networks.nets.SEResNext50(spatial_dims=3, in_channels=in_features, num_classes=out_features) # list(back_net.children())[-1].in_features = 2048
    
  elif net_architect == 'SEResNet50':
    
    back_net = monai.networks.nets.SEResNet50(spatial_dims=3, in_channels=in_features, num_classes=out_features) # list(back_net.children())[-1].in_features = 2048

  elif net_architect == 'DenseNet':
    
    back_net = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=in_features, out_channels=out_features) # init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), 
                                    # bn_size=4, act=('relu', {'inplace': True}), norm='batch', dropout_prob=0.0) 

  elif NET_ARCHITECT == 'se_resnext50':
    
    back_net = se_resnext50(num_classes=out_features) # list(back_net.children())[-1].in_features = 512

  elif NET_ARCHITECT == 'resnet50_cbam':
    
    back_net = resnet50_cbam(num_classes=out_features) # list(back_net.children())[-1].in_features = 512
    
  elif net_architect == 'EfficientNetBN':
    
    back_net = EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=in_features, num_classes=out_features) # list(back_net.children())[-2].in_features = 1280

  else:
    back_net = ResNet18Features3Way(in_channels=in_features, num_classes=out_features, num_hidden=64, image_size=RESIZE)
    
  return back_net

back_net = get_back_net(NET_ARCHITECT, in_features, out_features, n_channels)
back_net = back_net.cuda()

#%%
net = back_net
net = net.cuda()
print(f'net.is_cuda:{next(net.parameters()).is_cuda}')

#%% 
total_params = sum(p.numel() for p in net.parameters())
print(f'total_params: {total_params}')

class LRScheduler(Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self):
        self.scheduler.step()

''' onecycle LR prevents overfitting '''

if use_onecycleLR:
  
  torch_opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9) # 1e-3
  scheduler = torch.optim.lr_scheduler.OneCycleLR(torch_opt, max_lr=1e-3, div_factor=10, final_div_factor=1e3, epochs = epochs, steps_per_epoch = len(x_train) // batch_size, pct_start=0.1) # BEST max_lr 1e-3) # steps_per_epoch=len(train_dataloader), 
  
  if USE_LOGISTIC_HAZARD:
    model = LogisticHazard(net, torch_opt, duration_index=labtrans.cuts)
  else:
    model = DeepHitSingle(net, torch_opt, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
  callbacks = [LRScheduler(scheduler)]

else:
  ''' DeepHitSingle tutorial: /home/kschoi/Documents/GBM_survival/survival_code/pycox/examples/deephit.ipynb '''

  optimizer = tt.optim.AdamWR(lr=0.01, decoupled_weight_decay=0.01, cycle_eta_multiplier=0.8) # BEST

  if USE_LOGISTIC_HAZARD:
    model = LogisticHazard(net, optimizer, duration_index=labtrans.cuts)
  else:
    model = DeepHitSingle(net, optimizer, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts) # BEST
  
  callbacks = [tt.callbacks.EarlyStopping()]   
  lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=3)
  
  _ = lr_finder.plot()
  plt.savefig(os.path.join(TARGET_DIR, 'lr_finder_plot.png'))
  plt.show(block=False)
  plt.pause(1)
  plt.close()

  optimized_lr = lr_finder.get_best_lr()
  print(f'optimized_lr:{optimized_lr}')
  model.optimizer.set_lr(optimized_lr)

#%%
''' To train/save or load the saved model '''

def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = defaultdict(list)

    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','valid']:
            if(phase == 'train'):
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluation mode
            
            running_loss = 0.0
            running_corrects = 0.0
            
            # Iterate over data
            for inputs,labels in dataloaders[phase]:
                inputs = inputs.to(CFG.device)
                labels = labels.to(CFG.device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs,1)
                    loss = criterion(outputs, labels) # use this loss for any training statistics

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # first forward-backward pass
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        
                        # second forward-backward pass
                        criterion(model(inputs), labels).backward()
                        optimizer.second_step(zero_grad=True)


                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).double().item()

            
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects/dataset_sizes[phase]

            history[phase + ' loss'].append(epoch_loss)
            history[phase + ' acc'].append(epoch_acc)

            if phase == 'train' and scheduler != None:
                scheduler.step()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase=='valid' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                PATH = f"Fold{fold}_{best_acc}_epoch{epoch}.bin"
                torch.save(model.state_dict(), PATH)

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Accuracy ",best_acc)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def get_DL_score(input, id_list, dataframe, dataset):
  
  surv_ids = []
  surv_total = []
  id_list = sorted(id_list)
  for i in range(input.shape[0]):
    input=input.cuda().float()
    surv_prob = model.predict_surv(input[i].unsqueeze(0)) 
    surv_id = str(id_list[i])
    surv_prob_np = np.squeeze(to_np(surv_prob), axis=0)
    surv_total.append(surv_prob_np)
    surv_ids.append(surv_id)
  surv_total = np.stack(surv_total)
  print(f'surv_total: {surv_total.shape}')
  print(f'surv_ids: {surv_ids}')
  df_surv_total = pd.DataFrame(surv_total)
  df_surv_total['ID'] = surv_ids
  df_surv_total.columns = [f'MRI{i}' for i in range(1, NUM_DURATIONS+1)] + ['ID']
  df_surv_total = df_surv_total.set_index('ID')
  print(df_surv_total.head(5))
  surv_total_path = os.path.join(DATA_DIR, 'surv_predict_total.csv')
  df_surv_total.to_csv(surv_total_path)

  ''' combine predictions of MRI4 and MRI5, and labels together for selected IDs '''
  df = dataframe
  df_selected_label = df.loc[surv_ids]
  for i in range(1, NUM_DURATIONS+1):
    df_selected_label[f'MRI{i}'] = df_surv_total[f'MRI{i}']

  cut_feats = [1.0 - i * 0.1 for i in range(1, NUM_DURATIONS)]
  bin_feats = [f'MRI{i}'for i in range(1, NUM_DURATIONS+1)]
  
  for cut_feat, feat in zip(cut_feats, bin_feats):
    bin_feat = feat+"_bin"
    print(bin_feat, cut_feat)
    df_selected_label.loc[df_selected_label[feat] >= cut_feat, bin_feat] = 1
    df_selected_label.loc[df_selected_label[feat] < cut_feat, bin_feat] = 0

  print(df_selected_label.head(5))
  selected_label_total_path = os.path.join(DL_feature_DIR, f'{dataset}_{PATHOLOGY}_{PROGRESS_DURATION}_prediction_and_label_selected.csv')
  df_selected_label.to_csv(selected_label_total_path)

get_DL_score(x_test, id_list = EXT_IDLIST, dataframe = ext_df, dataset = EXT_DATASET_NAME)

from sklearn.calibration import calibration_curve
from lifelines import KaplanMeierFitter

def get_pred_obs_csv(x_test, y_test, NUM_DURATIONS, ext_df, EXT_IDLIST, save_filename, spec_time, root_path = TARGET_DIR):  
  surv_test = model.interpolate(1).predict_surv_df(x_test)

  kmf = KaplanMeierFitter()
  ext_df_select=ext_df.loc[sorted(EXT_IDLIST)].astype(str) # .astype(int) 
  kmf.fit(ext_df_select['duration'].values.tolist(),ext_df_select['event'].values.tolist())

  kmf_prediction = kmf.predict(surv_test.index.values)

  y_obs = y_test[1].numpy() 
  
  y_pred = 1 - surv_test[surv_test.index <= spec_time].iloc[-1].values 
  
  df_calib = pd.DataFrame(list(zip(y_obs, y_pred)), columns =['obs','pred'])
  df_calib.to_csv(os.path.join(root_path, f'df_calib_{save_filename}_{NUM_DURATIONS}_{spec_time}.csv'))
  
  return y_obs, y_pred, kmf_prediction

save_filename = f'{EXT_DATASET_NAME}_test'
surv_test = model.predict_surv_df(x_test)
spec_time = surv_test.index.values[4] 
print(f'spec_time:{spec_time}')

y_obs_test, y_pred_test, kmf_prediction_test = get_pred_obs_csv(x_test, y_test, NUM_DURATIONS, ext_df, EXT_IDLIST, f'{EXT_DATASET_NAME}_test', spec_time, root_path = TARGET_DIR)

surv_val = model.interpolate(1).predict_surv_df(x_val) # 10
df_val=df.loc[sorted(val_IDs)].astype(str) 
y_obs_val, y_pred_val, kmf_prediction_val = get_pred_obs_csv(x_val, y_val, NUM_DURATIONS, df_val, val_IDs, 'val', spec_time, root_path = TARGET_DIR)
df_calib = pd.DataFrame(list(zip(y_obs_test, y_pred_test)), columns =['obs','pred'])
df_calib.to_csv(os.path.join(TARGET_DIR, f'df_calib_{save_filename}_{NUM_DURATIONS}.csv'))

import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

get_surv_train = False 
if get_surv_train:
  surv_train = model.predict_surv_df(x_train) #.interpolate(1) # IDH-wt만 할 땐 interpolate(10) 아닐땐 interpolate(1).
  df_train=df.loc[sorted(train_IDs)].astype(str) 
  durations_train, events_train = get_target(df_train)
  ev_train = EvalSurv(surv_train, durations_train, events_train, censor_surv='km')

  y_obs_train, y_pred_train, kmf_prediction_train = get_pred_obs_csv(x_train, y_train, NUM_DURATIONS, df_train, train_IDs, 'train', spec_time)
  
  rf_y, rf_x = calibration_curve(y_obs_train, y_pred_train * 1.2, n_bins=20)
  
  fig, ax = plt.subplots()
  plt.plot(rf_x, rf_y, marker='o', linewidth=1, label='rf')

  # reference line, legends, and axis labels
  line = mlines.Line2D([0, 1], [0, 1], color='black')
  transform = ax.transAxes
  line.set_transform(transform)
  ax.add_line(line)
  fig.suptitle('Calibration plot for Titanic data')
  ax.set_xlabel('Predicted probability')
  ax.set_ylabel('True probability in each bin')
  plt.legend()
  plt.show()

  concordance_td_train = ev_train.concordance_td('antolini') # 'adj_antolini'
  print(f'concordance_td_train:{concordance_td_train}')
  print(f'ev_train.concordance_td():{ev_train.concordance_td()}')

  ibs_train, nbll_train = get_ibs(ev_train, durations_train)
  print(f'ibs_train:{ibs_train}')
  print(f'nbll_train:{nbll_train}')

  train_set_size = x_train.shape[0]
  train_rand_list = np.random.choice(train_set_size, size = int(train_set_size / 2))
  # print(f'n of train sample for curve: {len(train_rand_list)}')

  surv_train.iloc[:, train_rand_list].plot(drawstyle='steps-post', legend=False)
  plt.ylabel('S(t | x)')
  _ = plt.xlabel('Time')
  plt.savefig(os.path.join(TARGET_DIR, 'survival_curve_interpolate_train.png'))
  plt.show(block=False)
  plt.pause(1)
  plt.close()

#%%
testset_size = x_test.shape[0]
test_rand_list = np.random.choice(testset_size, size = int(testset_size / 2))
print(f'n of test sample for curve: {len(test_rand_list)}')
surv_test = model.interpolate(1).predict_surv_df(x_test)
surv_test.iloc[:, test_rand_list].mean(axis=1).plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

plt.savefig(os.path.join(TARGET_DIR, 'survival_curve_step_test.png'))
plt.show(block=False)
plt.pause(1)
plt.close()

''' for IBS '''
def get_ibs(ev, durations, grid_num=100):
  time_grid = np.linspace(durations.min(), durations.max(), grid_num)
  _ = ev.brier_score(time_grid).plot()
  ibs=ev.integrated_brier_score(time_grid)
  nbll=ev.integrated_nbll(time_grid)
  return ibs, nbll

surv_val = model.interpolate(1).predict_surv_df(x_val) # 10
df_val=df.loc[sorted(val_IDs)].astype(str) #.astype(int) 
durations_val, events_val = get_target(df_val)
ev_val = EvalSurv(surv_val, durations_val, events_val, censor_surv='km')

get_pred_obs_csv(x_val, y_val, NUM_DURATIONS, df_val, val_IDs, 'val', spec_time)

concordance_td_val = ev_val.concordance_td('antolini') # 'adj_antolini'
print(f'concordance_td_val:{concordance_td_val}')
print(f'ev_val.concordance_td():{ev_val.concordance_td()}')

ibs_val, nbll_val = get_ibs(ev_val, durations_val)
print(f'ibs_val:{ibs_val}')
print(f'nbll_val:{nbll_val}')

df_test=ext_df.loc[sorted(EXT_IDLIST)].astype(str) # .astype(int) 
durations_test, events_test = get_target(df_test)
ev_test = EvalSurv(surv_test, durations_test, events_test, censor_surv='km')

get_pred_obs_csv(x_test, y_test, NUM_DURATIONS, df_test, EXT_IDLIST, 'test', spec_time)

concordance_td_test = ev_test.concordance_td('antolini') # 'adj_antolini'
print(f'concordance_td_test:{concordance_td_test}')
print(f'ev_test.concordance_td():{ev_test.concordance_td()}')

ibs_test, nbll_test = get_ibs(ev_test, durations_test)
print(f'ibs_test:{ibs_test}')
print(f'nbll_test:{nbll_test}')

def superimpose_img(img, grad_cam_map):
  
  grad_heatmap = np.float32(grad_cam_map)# / 255
  grad_result = grad_heatmap + img
  grad_result = grad_result / np.max(grad_result)
  grad_result = np.uint8(255 * grad_result)

  return grad_result

use_grad_CAM = True 
use_ext_grad_CAM = False # True # 

if not use_ext_grad_CAM:
  test_subj_id = IDLIST[0] 
  test_img_path = os.path.join(DATA_DIR, 'seg_BraTS', test_subj_id)
else:
  test_subj_id = EXT_IDLIST[0] 
  test_img_path = os.path.join(EXT_DATA_DIR, 'seg_BraTS', test_subj_id)

seqs = []
for seq in SELECTED_SEQUENCE: # ['t1','t2','flair','t1ce']:
  seq=nib.load(os.path.join(test_img_path,f'{seq}_seg.nii.gz')).get_fdata() # _seg # _cropped
  print(f'seq range:min {seq.min()}-max {seq.max()}')
  seqs.append(seq)
  
x = np.stack(seqs, axis=0)
x = torch.from_numpy(x)
x = torch.unsqueeze(x, axis=0)
print(f'x.shape:{x.shape}')
x=x.cuda().float()
net = net.eval()
output = net(x)
print(f'output:{output}')
print(f'output.shape:{output.shape}')

seq_idx_dict = {'t1':0, 't2':1, 'flair':2, 't1ce':3}
selected_seq = 't1ce'
selected_seq_idx = 0 # seq_idx_dict[selected_seq]
print(f'selected_seq:{selected_seq}, {selected_seq_idx}')
#%%
if use_grad_CAM:

  jacob = False # True # 
  if not jacob:
    
    from monai.visualize import * # GradCAM
    
    slice_num = 54 
    img=x[:,selected_seq_idx,:,:,slice_num] 
    img = to_np(img)
    img = np.transpose(img,(1,2,0))
    
    img = img/np.max(img)
    print(f'img.shape:{img.shape}') 
    
    print(f'img range:min {img.min()}-max {img.max()}')
    
    occ_map = False 
    if not occ_map:
         
      if NET_ARCHITECT == 'DenseNet':
        cam = GradCAMpp(nn_module=net, target_layers='class_layers.relu') # For DenseNet, use target_layers 'class_layers.relu' with GradCAMpp
      elif NET_ARCHITECT == 'SEResNext50':
        cam = GradCAM(nn_module=net, target_layers='layer4') # For SEResNext50, use target layers 'layer4'

      result = cam(x)
      result = result[:,0,:,:,slice_num]
    
    else:
    
      occ_sens = OcclusionSensitivity(nn_module=net, n_batch=16, stride=3)
      occ_map, most_probable_class = occ_sens(x, b_box=[-1, -1, 1, 3, -1, -1, -1, -1])
      result = occ_map[...,0] 

      result = result[:,:,:,slice_num,0] # occlusion_map
    
    result = to_np(result)
    result = np.transpose(result,(1,2,0))

    result = result / np.max(result)
    print(f'result.shape:{result.shape}') 

    print(f'result range:min {result.min()}-max {result.max()}')
    superimposed_img = superimpose_img(img, result) # heatmap)
    plt.imshow(np.squeeze(result), alpha = 0.9, cmap='Spectral')
    hm = plt.imshow(img, alpha = 0.5, cmap='gray')
    plt.show()

    saved_loc = os.path.join(test_img_path, 'grad_CAM')
    os.makedirs(saved_loc, exist_ok=True)
    grad_heatmap = cv2.applyColorMap(np.uint8(255 * superimposed_img), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(saved_loc, 'Grad_CAM_heatmap.jpg'), grad_heatmap)
  else:
    
    from pytorch_grad_cam import * # GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
    from PIL import Image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    
    target_layers = [net.features[-2][-1].layers] 
    cam = GradCAMPlusPlus(model=net, target_layers=target_layers, use_cuda=True)
    targets = [ClassifierOutputTarget(1)]

    grayscale_cam = cam(input_tensor=x, targets=targets)
    print(f'grayscale_cam.shape:{grayscale_cam.shape}') 
  
    grayscale_cam = grayscale_cam[0, :]
    grayscale_cam = grayscale_cam[..., 0]
    print(f'grayscale_cam.shape:{grayscale_cam.shape}') 
    
    slice_num = 54 
    img=x[:,selected_seq_idx,:,:,slice_num] 
    img = to_np(img)
    print(f'img.shape:{img.shape}')  
    img = np.transpose(img,(2,1,0))
    # print(f'img.shape:{img.shape}')
    img = np.squeeze(img, axis=-1)
    img = img/np.max(img)
    print(f'img.shape:{img.shape}')  
    
    plt.imshow(np.squeeze(grayscale_cam), alpha = 0.9)
    hm = plt.imshow(img, alpha = 0.5, cmap='gray')
    plt.show()
    
    superimposed_img = superimpose_img(img, grayscale_cam)
    
    saved_loc = os.path.join(test_img_path, 'grad_CAM')
    os.makedirs(saved_loc, exist_ok=True)
    grad_heatmap = cv2.applyColorMap(np.uint8(255 * superimposed_img), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(saved_loc, 'Grad_CAM_heatmap_monai.jpg'), grad_heatmap)
