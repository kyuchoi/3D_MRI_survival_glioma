import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
from torch.optim import lr_scheduler
from torchsummary import summary as summary

from torch.utils.data import DataLoader, Dataset
import torchio as tio

from monai.transforms import RandFlip, Rand3DElastic, RandAffine, RandGaussianNoise, AdjustContrast, RandSpatialCrop # Rand3DElastic
from sklearn.model_selection import train_test_split, StratifiedKFold

import os
import glob
from distutils.dir_util import copy_tree
from tqdm import tqdm
import copy
from datetime import datetime
import cv2
from PIL import Image

import scipy.interpolate
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation

import plotly.graph_objects as go
from IPython.display import HTML

from skimage.transform import resize
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import random
import json

from lifelines.utils import concordance_index
from sklearn.utils import resample
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import brier_score

#%%

''' lambda '''
to_np = lambda x: x.detach().cpu().numpy()
to_cuda = lambda x: torch.from_numpy(x).float().device()

# get_dataset_name = lambda dataset_list: '_'.join(dataset_list)
get_dir = lambda directory: [dir for dir in os.listdir(directory) if os.path.isdir(os.path.join(directory, dir))]
convert_list2df = lambda idlist: pd.DataFrame(idlist, columns = ['ID'], dtype='string').set_index('ID')

#%%

def print_args(args, exp_path):
  args_path = os.path.join(exp_path,'commandline_args.txt')
  with open(args_path, 'w') as f:
      json.dump(args.__dict__, f, indent=2)

  with open(args_path, 'r') as f:
      args.__dict__ = json.load(f)

class config(object):
  
  def __init__(self):
    self.scheduler = 'CosineAnnealingLR'
    self.T_max = 10
    self.T_0 = 10
    self.lr = 1e-4 # ORIGINAL: 1e-4
    self.min_lr = 1e-6
    
    self.weight_decay = 1e-6
    # seed = 123456
    
    self.n_fold = 10 # 5
    self.smoothing = 0.2
    
    self.net_architect = 'SEResNext50' # 'DenseNet' # 'resnet50_cbam' # 
    self.compart_name = 'resized' # 'seg' # 
    # dataset_list = ['SNUH','UPenn','TCGA']
    
    # ext_dataset_name = 'severance' # 'TCGA' # 
    # spec_patho = 'all' # 'GBL' # 
    # spec_duration = '1yr' # 'OS' # 
    self.sequence = ['t1','t2','t1ce','flair']
    # self.spec_event = 'death' # 'prog' # 
    self.data_key_list = ['sex', 'age', 'IDH', 'MGMT', 'GBL', 'EOR', 'duration_death', 'event_death', 'duration_prog', 'event_prog', 'biopsy_exclusion']
    ### 'KPS': ONLY in UPenn, severance, NOT in SNUH, TCGA; 'prog': NOT in UPenn, TCGA, ONLY in SNUH, severance, TCGA; 'EOR': NOT in TCGA
    self.batch_size = 64
    # epochs = 200 # 300 #
       
    self.root_dir = r'/mnt/hdd3/mskim/GBL'
    self.data_dir = os.path.join(self.root_dir, 'data')
    self.label_dir = os.path.join(self.data_dir, 'label', 'surv_labels')
    self.proc_label_dir = os.path.join(self.label_dir, 'proc_labels')
    os.makedirs(self.proc_label_dir, exist_ok = True)
    
    self.exp_dir = os.path.join(self.root_dir, 'code', 'experiment')
    os.makedirs(self.exp_dir, exist_ok = True)

    # exp_descrip = f'nnet_b{batch_size}_{net_architect}_s{seed}_e{epochs}' 
    # target_dir = os.path.join(root_dir, 'experiment', exp_descrip)
    # os.makedirs(target_dir, exist_ok = True)
    
    self.dataset_name = ["SNUH"]
    # dataset_name = get_dataset_name(dataset_list)
    
args = config()
print(f'args:{args.__dict__}')

print(f'Using {args.net_architect}')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
# set_seed(args.seed)

#%%

''' ref: 
1) https://cumulu-s.tistory.com/41 -> 0-255 사이값이어야 컬러로 나온다 
2) https://stackoverflow.com/questions/65510388/superimposing-2d-heat-map-on-3d-image-with-transparency -> alpha 써서 superimpose 하는 법
'''

# min-max scaling of img and convert to uint8 into range of 0-255
def min_max_norm(img):
  
  img = img.astype(np.float32)
  img = (img-img.min())/(img.max()-img.min())
  img = (img*255).astype(np.uint8)
  img = np.stack((img,)*3, axis=-1)
  
  return img

def superimpose_img(img, heatmap, alpha = 0.3):
  
  # overlay heatmap with cmap jet over img of grayscale using cv2.addWeighted
  
  # print(f'range of heatmap: {np.min(heatmap)}-{np.max(heatmap)}') # 0.0 1.0
  # print(f'range of img: {np.min(img)}-{np.max(img)}') # 0.0 1.0
  grad_heatmap = resize(heatmap, (img.shape[0], img.shape[1], img.shape[2]))
  
  # print(f'range of grad_heatmap: {np.min(grad_heatmap)}-{np.max(grad_heatmap)}') # 0.0 1.0
  
  # thresholding heatmap: NOT working
  # threshold = 0.3
  # grad_heatmap_thr = np.where(grad_heatmap < threshold, 0, grad_heatmap)
  # grad_heatmap_thr = (grad_heatmap_thr - np.min(grad_heatmap_thr))/(np.max(grad_heatmap_thr)-np.min(grad_heatmap_thr))
  # print(f'range of grad_heatmap_thr: {np.min(grad_heatmap_thr)}-{np.max(grad_heatmap_thr)}') # 0.0 1.0

  cmap = plt.cm.jet
  grad_heatmap_rgb = cmap(grad_heatmap)
  grad_heatmap_rgb = grad_heatmap_rgb[...,:3]
  grad_heatmap_rgb = np.uint8(grad_heatmap_rgb * 255)

  grad_result = grad_heatmap_rgb * alpha + img * (1 - alpha) #.astype(np.uint8)
  grad_result = grad_result / np.max(grad_result)

  # print(f'range of grad_result: {np.min(grad_result)}-{np.max(grad_result)}') # 0.0 1.0
  # print(f'shape of grad_result: {grad_result.shape}')
  return grad_result, grad_heatmap

#%% Plot slices of the data at the given coordinates:
#  https://stackoverflow.com/questions/56688602/plotting-slices-in-3d-as-heatmap
#  https://github.com/matplotlib/matplotlib/issues/3919
#  https://community.plotly.com/t/is-it-possible-to-update-multiple-traces-with-different-value-arrays-in-one-go/31597

def plot_slices_superimposed(data, x_slice, y_slice, z_slice, use_midline = True):
    
    matplotlib.rcParams['animation.embed_limit'] = 500
    
    # get the x, y, z coordinates
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    z = np.arange(data.shape[2])
    print(f'x:{x}, y:{y}, z:{z}')

    if use_midline:
      xslice = data.shape[0] // 2 # specify
      yslice = data.shape[1] // 2 # specify
      zslice = data.shape[2] // 2 # specify
    else:
      xslice = x_slice
      yslice = y_slice
      zslice = z_slice

    print(f'xslice:{xslice}, yslice:{yslice}, zslice:{zslice}')

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define meshgrid
    x, y, z = np.mgrid[:data.shape[0], :data.shape[1], :data.shape[2]]
    
    # Take slices
    mask_x = np.abs(x - xslice) < 0.5
    mask_y = np.abs(y - yslice) < 0.5
    mask_z = np.abs(z - zslice) < 0.5
    mask = mask_x | mask_y | mask_z

    # Plot slices with alpha = 0.5 for some transparency
    scatter = ax.scatter(x[mask], y[mask], z[mask], c=data[mask], s=20, cmap = 'gray') # norm=norm, # , cmap = 'jet'

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Invert axis
    # ax.invert_xaxis()
    ax.invert_yaxis()
    # ax.invert_zaxis()
    
    # rotate the ax.scatter plot by 90 degrees:
    # https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
    # https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.view_init.html
    # https://matplotlib.org/stable/gallery/mplot3d/rotate_axes3d_sgskip.html 
    # https://codetorial.net/matplotlib/animation_funcanimation.html

    # ax.view_init(azim=300, elev=255, roll=300) # BEST: (300, 285) > (300, 255) > (300, 75)
    # plt.show()
    
    # make 2 rotations with different directions
    total_frames = 720  # for two rounds # 360 #

    def update(num):
        
        # Angles for first rotation
        final_azim_1 = 300
        final_elev_1 = 285

        # Angles for second rotation
        final_azim_2 = 600
        final_elev_2 = 570

        if num < total_frames / 2:
            azim = (final_azim_1 / (total_frames / 2)) * num
            elev = (final_elev_1 / (total_frames / 2)) * num
        else:
            azim = final_azim_1 + ((final_azim_2 - final_azim_1) / (total_frames / 2)) * (num - total_frames / 2)
            elev = final_elev_1 + ((final_elev_2 - final_elev_1) / (total_frames / 2)) * (num - total_frames / 2)

        ax.view_init(elev=elev, azim=azim)
        return scatter
    
    # # start with (300, 285) > (300, 255) > (300, 75)
    # def update(num):
    #   azim = (num + 300) % 360
    #   elev = (num + 285) % 360
    #   ax.view_init(elev=elev, azim=azim)
    #   return scatter

    # # original: start with (0,0,0)
    # def update(num):
    #   ax.view_init(elev=num, azim=num, roll=num)
    #   return scatter
    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, total_frames, 1), interval=60)
    html = ani.to_jshtml() # save as html 
    # return ani # save as gif
    return html#, ani
    
    # for angle in range(0, 360):
    #   ax.view_init(30, angle)
    #   plt.draw()
    #   plt.pause(.001)

# def plot_slices(data, heatmap):
#     # get the x, y, z coordinates
#     # print(f'data shape: {data.shape}')
#     # print(f'heatmap shape: {heatmap.shape}')

#     x = np.arange(data.shape[0])
#     y = np.arange(data.shape[1])
#     z = np.arange(data.shape[2])
#     print(f'x:{x}, y:{y}, z:{z}')

#     # Slice indices
#     xslice = data.shape[0] // 2 # specify
#     yslice = data.shape[1] // 2 # specify
#     zslice = data.shape[2] // 2 # specify
#     # print(f'xslice:{xslice}, yslice:{yslice}, zslice:{zslice}')

#     # Create a 3D plot
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     # Take slices
#     data_x = data[xslice, :, :]
#     data_y = data[:, yslice, :]
#     data_z = data[:, :, zslice]

#     # Define meshgrid
#     x, y, z = np.mgrid[:data.shape[0], :data.shape[1], :data.shape[2]]
    
#     # Take slices
#     mask_x = np.abs(x - xslice) < 0.5
#     mask_y = np.abs(y - yslice) < 0.5
#     mask_z = np.abs(z - zslice) < 0.5
#     mask = mask_x | mask_y | mask_z
    
#     # Normalize heatmap
#     norm_heatmap = plt.Normalize(heatmap.min(), heatmap.max())
    
#     # Overlay slices of heatmap data with alpha = 0.5 for some transparency
#     ax.scatter(x[mask], y[mask], z[mask], c=heatmap[mask], cmap='jet', s=30, alpha=0.9) # , norm=norm_heatmap #
    
#     # Normalize color map
#     norm = plt.Normalize(data.min(), data.max())
    
#     # Plot slices with alpha = 0.5 for some transparency
#     ax.scatter(x[mask], y[mask], z[mask], c=data[mask], cmap='gray', s=10, alpha=0.3) # norm=norm, 

#     # Set labels
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
    
#     # Invert axis
#     ax.invert_xaxis()
#     ax.invert_yaxis()
    
#     # rotate the ax.scatter plot by 90 degrees: 
#     # https://matplotlib.org/stable/gallery/mplot3d/rotate_axes3d_sgskip.html 
#     # https://seong6496.tistory.com/131
#     ax.view_init(azim=300, elev=75, roll=30) # BEST: 300, 75

#     # for angle in range(0, 360):
#     #   ax.view_init(30, angle)
#     #   plt.draw()
#     #   plt.pause(.001)



### plotly 코드로 바꾸기: https://wooiljeong.github.io/python/plotly_01/ 

def plotly_slices_superimposed(data):
    
    # get the x, y, z coordinates
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    z = np.arange(data.shape[2])
    print(f'x:{x}, y:{y}, z:{z}')

    # Slice indices
    xslice = data.shape[0] // 2 # specify
    yslice = data.shape[1] // 2 # specify
    zslice = data.shape[2] // 2 # specify
    print(f'xslice:{xslice}, yslice:{yslice}, zslice:{zslice}')

    # Define meshgrid
    x, y, z = np.mgrid[:data.shape[0], :data.shape[1], :data.shape[2]]
    
    # Take slices
    mask_x = np.abs(x - xslice) < 0.5
    mask_y = np.abs(y - yslice) < 0.5
    mask_z = np.abs(z - zslice) < 0.5
    mask = mask_x | mask_y | mask_z

    # Ensure there's data to plot
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x[mask].flatten(),
        y=y[mask].flatten(),
        z=z[mask].flatten(),
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.8,
            color=data[mask].flatten()
        )
    )])

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()

# def plotly_slices(volume):

#     # Assume 'volume' is your 3D MRI data array
#     # And 'heatmap' is your 3D heatmap data array
#     r, c, d, _ = volume.shape

#     # Slice MRI data at the middle of each dimension
#     x_slice_mri = volume[r//2, :, :]
#     y_slice_mri = volume[:, c//2, :]
#     z_slice_mri = volume[:, :, d//2]

#     # X slice surface for MRI data
#     trace1 = go.Surface(
#         z=(r//2) * np.ones((c, d)),
#         surfacecolor=x_slice_mri,
#         colorscale='Gray',
#         cmin=0, cmax=200,
#         colorbar=dict(thickness=20, ticklen=4)
#     )

#     # Y slice surface for MRI data
#     trace2 = go.Surface(
#         x=(c//2) * np.ones((r, d)),
#         surfacecolor=y_slice_mri,
#         colorscale='Gray',
#         cmin=0, cmax=200,
#         colorbar=dict(thickness=20, ticklen=4)
#     )

#     # Z slice surface for MRI data
#     trace3 = go.Surface(
#         y=(d//2) * np.ones((r, c)),
#         surfacecolor=z_slice_mri,
#         colorscale='Gray',
#         cmin=0, cmax=200,
#         colorbar=dict(thickness=20, ticklen=4)
#     )

#     fig = go.Figure(data=[trace1, trace2, trace3])

#     fig.update_layout(scene = 
#                     dict(xaxis = dict(nticks=4, range=[-2000,2000],),
#                         yaxis = dict(nticks=4, range=[-1500,1500],),
#                         zaxis = dict(nticks=4, range=[-660,10],),),
#                         title='vorticity', autosize=False,
#                         width=800, height=700,
#                         margin=dict(l=65, r=50, b=65, t=90))

#     fig.update_layout(scene = dict(
#                         xaxis_title='X [km]',
#                         yaxis_title='Y [km]',
#                         zaxis_title='Z [km]'),
#                         )


#     fig.show()

#     return fig

#%%

def get_n_intervals(fixed_interval_width = False):

  if fixed_interval_width:
    breaks=np.arange(0.,365.*5,365./8)
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    print(f'n_intervals: {n_intervals}') # 19
  else:
    halflife=365.*2
    breaks=-np.log(1-np.arange(0.0,0.96,0.05))*halflife/np.log(2) 
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    print(f'n_intervals: {n_intervals}') # 19

    return breaks, n_intervals

# _, n_intervals = get_n_intervals(fixed_interval_width = 0) # 1 #
# args.n_intervals = n_intervals

# get 95% confidence interval of concordance index using bootstrap
def bootstrap_cindex(time, prediction, event, n_iterations=1000):
    # Compute the original C-index
    original_c_index = concordance_index(time, prediction, event)

    # Initialize a list to store bootstrapped C-indexes
    bootstrap_c_indexes = []

    # Perform bootstrapping
    for i in range(n_iterations):
        # Resample with replacement
        resample_indices = resample(np.arange(len(time)), replace=True)
        time_sample = time[resample_indices]
        event_sample = event[resample_indices]
        prediction_sample = prediction[resample_indices]

        # Compute the C-index on the bootstrap sample
        c_index_sample = concordance_index(time_sample, prediction_sample, event_sample)

        bootstrap_c_indexes.append(c_index_sample)

    # Compute the 95% confidence interval for the C-index
    ci_lower = np.percentile(bootstrap_c_indexes, 2.5)
    ci_upper = np.percentile(bootstrap_c_indexes, 97.5)

    return original_c_index, ci_lower, ci_upper

#%%

''' copying and combining images of dataset_list '''

def combine_img(main_args, args):
  dataset_name = '_'.join(main_args.dataset_list)
  target_dataset_path = os.path.join(args.data_dir, dataset_name, f'{args.compart_name}_BraTS')
  os.makedirs(target_dataset_path, exist_ok=True)
  
  if len(os.listdir(target_dataset_path)) != 0:
    print(f"Already copyied images of {dataset_name} for training to {target_dataset_path} path")
  else:
    for dataset in main_args.dataset_list:
      print(f"copying images of {dataset} for training to {target_dataset_path} path")
      img_dataset_path = os.path.join(args.data_dir, dataset, f'{args.compart_name}_BraTS')
      for img_dir in tqdm(os.listdir(img_dataset_path)):
        img_dir_path = os.path.join(img_dataset_path, img_dir)
        print(f'img_dir_path:{img_dir_path}')
        os.makedirs(os.path.join(target_dataset_path, img_dir), exist_ok=True)
        copy_tree(img_dir_path, os.path.join(target_dataset_path, img_dir))

#%%

''' getting together multiple (i.e. SNUH, severance, UPenn) ${dataset}_OS_all.csv files into final csv indexing only 1) GBL vs all; and 2) 1yr vs OS, and save them into anoter .csv file '''

def save_label_dataset_list(main_args, args):
  
  df = pd.DataFrame()
  for dataset in main_args.dataset_list:
    print(f'dataset:{dataset} for training')
    df_dataset_path = os.path.join(args.label_dir, f'{dataset}_OS_all.csv')
    df_data = pd.read_csv(df_dataset_path, dtype='string') # , index_col=0, dtype='string') # int: not working
    df_data = df_data.set_index('ID')
    df_data = df_data.sort_index(ascending=True)
        
    df_dataset = df_data[args.data_key_list]
    print(f'df_dataset.shape:{df_dataset.shape}')
    # df_label_dataset_list = pd.merge(df_dataset, df_label_dataset_list, left_on='ID', right_index=True) # NOT WORKING
    df = pd.concat([df_dataset, df])
  print(f'df_label_dataset_list.shape:{df.shape}') # 
  # print(f'df.head:{df.head(10)}')

  dataset_name = '_'.join(main_args.dataset_list)
  
  # ref: https://wooono.tistory.com/293

  if main_args.spec_patho == 'GBL':
    print(f'filtering before GBL; {len(df.index.values)} cases')
    condition = df.GBL.astype(int) == 1 # 1 means mut, not 0 
    filtered_ID = df[condition].index.tolist() 
    df = df.loc[sorted(filtered_ID),:]
    print(f'filtering after GBL; {len(df.index.values)} cases')

  if main_args.biopsy_exclusion:
    print(f'filtering before biopsy_exclusion; {len(df.index.values)} cases')
    print(f'df.columns:{df.columns}')
    if "biopsy_exclusion" in df.columns:
      condition = df.biopsy_exclusion.astype(int) == 0 # 1 means biopsy exclusion, not 0 
      filtered_ID = df[condition].index.tolist() 
      df = df.loc[sorted(filtered_ID),:]
      print(f'filtering after biopsy_exclusion; {len(df.index.values)} cases')

  if main_args.spec_event == 'death':
    if main_args.spec_duration == '1yr':
        df = df.astype({'event_death': 'int'})
        print('events before 1yr:')
        print(df['event_death'].sum())
        df.loc[(df['event_death'] == 1) & (df['duration_death'].astype(int) > 365), 'event_death'] = 0
        print(f'events after 1yr:')
        print(df['event_death'].sum())
        
        # filtered_ID = df[condition].index.tolist() 
        # df = df.loc[sorted(filtered_ID),:]
    else:
        pass

  elif main_args.spec_event == 'prog':

    if main_args.spec_duration == '1yr':
        df = df.astype({'event_prog': 'int'})
        print('events before 1yr:')
        print(df['event_prog'].sum())
        df.loc[(df['event_prog'] == 1) & (df['duration_prog'].astype(int) > 365), 'event_prog'] = 0
        print(f'events after 1yr:')
        print(df['event_prog'].sum())
        
        # filtered_ID = df[condition].index.tolist() 
        # df = df.loc[sorted(filtered_ID),:]
    else:
        pass

  df_path = os.path.join(args.label_dir, f'{dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}.csv')
  df.to_csv(df_path)
  print(f'saving new label csv file for {dataset_name} at {df_path}') # 

  return df

def save_label_ext_dataset(main_args, args):
  
  ext_df = pd.DataFrame()
  
  print(f'dataset:{main_args.ext_dataset_name} for training')
  ext_df_dataset_path = os.path.join(args.label_dir, f'{main_args.ext_dataset_name}_OS_all.csv')
  ext_df_data = pd.read_csv(ext_df_dataset_path, dtype='string') # , index_col=0, dtype='string') # int: not working
  ext_df_data = ext_df_data.set_index('ID')
  ext_df_data = ext_df_data.sort_index(ascending=True)
    
  ext_df = ext_df_data[args.data_key_list]
  print(f'ext_df_dataset.shape:{ext_df.shape}')
  
  # ref: https://wooono.tistory.com/293

  if main_args.spec_patho == 'GBL':
    print(f'filtering before GBL; {len(ext_df.index.values)} cases')
    condition = ext_df.GBL.astype(int) == 1
    filtered_ID = ext_df[condition].index.tolist() 
    ext_df = ext_df.loc[sorted(filtered_ID),:]
    print(f'filtering after GBL; {len(ext_df.index.values)} cases')

  if main_args.biopsy_exclusion:
    print(f'filtering before biopsy_exclusion; {len(ext_df.index.values)} cases')
    if "biopsy_exclusion" in ext_df.columns:
      condition = ext_df.biopsy_exclusion.astype(int) == 0 # 1 means biopsy exclusion, not 0 
      filtered_ID = ext_df[condition].index.tolist() 
      ext_df = ext_df.loc[sorted(filtered_ID),:]
      print(f'filtering after biopsy_exclusion; {len(ext_df.index.values)} cases')

  if main_args.spec_event == 'death':
    if main_args.spec_duration == '1yr':
        ext_df = ext_df.astype({'event_death': 'int'})
        print('events before 1yr:')
        print(ext_df['event_death'].sum())
        ext_df.loc[(ext_df['event_death'] == 1) & (ext_df['duration_death'].astype(int) > 365), 'event_death'] = 0
        print(f'events after 1yr:')
        print(ext_df['event_death'].sum())
        
        # filtered_ID = ext_df[condition].index.tolist() 
        # ext_df = ext_df.loc[sorted(filtered_ID),:]
    else:
        pass

  elif main_args.spec_event == 'prog':

    if main_args.spec_duration == '1yr':
        ext_df = ext_df.astype({'event_prog': 'int'})
        print('events before 1yr:')
        print(ext_df['event_prog'].sum())
        ext_df.loc[(ext_df['event_prog'] == 1) & (ext_df['duration_prog'].astype(int) > 365), 'event_prog'] = 0
        print(f'events after 1yr:')
        print(ext_df['event_prog'].sum())
        
        # filtered_ID = ext_df[condition].index.tolist() 
        # ext_df = ext_df.loc[sorted(filtered_ID),:]
    else:
        pass

  ext_df_path = os.path.join(args.label_dir, f'{main_args.ext_dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}.csv')
  ext_df.to_csv(ext_df_path)
  print(f'saving new label csv file for external set {main_args.ext_dataset_name} at {ext_df_path}') # 

  return ext_df


#%%
''' criterion argument의 순서 (y_pred, y_true 여야 돌아가고, y_true, y_pred 면 차원 안 맞음)가 중요하고 보통 (output, target 또는 label) 순으로 선언되며, 이 경우는 utils.py의 train_model 에 criterion(output, label) 로 되어 있음. '''
def nnet_loss(y_pred, y_true, n_intervals = 19): 
  # print(f'y_true.shape:{y_true.shape}')
  # print(f'y_pred.shape:{y_pred.shape}')
  # print(f'mid_term of y_true.shape:{y_true[:, 0:n_intervals].shape}')
  # y_true_expand = torch.new_tensor(y_true, requires_grad=False)
  cens_uncens = 1. + y_true[:, 0:n_intervals] * (y_pred-1.)
  # print(f'cens_uncens.shape:{cens_uncens.shape}')
  uncens = 1. - y_true[:, n_intervals: 2 * n_intervals] * y_pred
  # print(f'uncens.shape:{uncens.shape}')
  loss = torch.sum(-torch.log(torch.clip(torch.cat((cens_uncens, uncens), dim=-1), torch.finfo(torch.float32).eps, None)), axis=-1)
  # print(f'loss:{loss.shape}')
  loss = loss.mean()
  # print(f'loss:{loss.shape}')
  return loss

def nnet_pred_surv(y_pred, breaks, fu_time):
#Predicted survival probability from Nnet-survival model
#Inputs are Numpy arrays.
#y_pred: Rectangular array, each individual's conditional probability of surviving each time interval
#breaks: Break-points for time intervals used for Nnet-survival model, starting with 0
#fu_time: Follow-up time point at which predictions are needed
#
#Returns: predicted survival probability for each individual at specified follow-up time
  y_pred=np.cumprod(y_pred, axis=1)
  pred_surv = []
  for i in range(y_pred.shape[0]):
    pred_surv.append(np.interp(fu_time,breaks[1:],y_pred[i,:]))
  return np.array(pred_surv)


#Example of finding model-predicted survival probability.
#Predicted survival prob. for first individual at follow-up time of 30 days:
# pred_surv = nnet_survival.nnet_pred_surv(model.predict_proba(x_train,verbose=0), breaks, 30)
# print(pred_surv[0])

def add_kfold_to_df(df, args, seed):
  
  ''' Create Folds 
  ref:
  1) https://www.kaggle.com/code/debarshichanda/seresnext50-but-with-attention 
  2) https://stackoverflow.com/questions/60883696/k-fold-cross-validation-using-dataloaders-in-pytorch
  '''  
  
  skf = StratifiedKFold(n_splits=args.n_fold, shuffle=True, random_state=seed)
  for fold, ( _, val_) in enumerate(skf.split(X=df, y=df.event_death)):
      # print(fold, val_)
      # print(df.index[val_])
      df.loc[df.index[val_] , "kfold"] = int(fold)
      
  df['kfold'] = df['kfold'].astype(int)
  kfold = df['kfold'].values
  
  return kfold

def random_split(id_list, split_ratio):
  ''' df: dataframe for total dataset '''
  n_sample = len(id_list) 
  id_list = sorted(id_list)
  train_nums = np.random.choice(n_sample, size = int(split_ratio * n_sample), replace = False)
  print(f'train_nums:{len(train_nums)}')
  val_nums = [num for num in np.arange(n_sample) if num not in train_nums]
  
  return train_nums, val_nums

# train_nums, val_nums = random_split(IDLIST, split_RATIO)
# train_IDs, val_IDs = np.array(IDLIST)[train_nums], np.array(IDLIST)[val_nums]

def make_surv_array(t,f,breaks):
  """Transforms censored survival data into vector format that can be used in Keras.
    Arguments
        t: Array of failure/censoring times.
        f: Censoring indicator. 1 if failed, 0 if censored.
        breaks: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
    Returns
        Two-dimensional array of survival data, dimensions are number of individuals X number of time intervals*2
  """
  n_samples=t.shape[0]
  n_intervals=len(breaks)-1
  timegap = breaks[1:] - breaks[:-1]
  breaks_midpoint = breaks[:-1] + 0.5*timegap
  y_train = np.zeros((n_samples,n_intervals*2))
  for i in range(n_samples):
    if f[i]: #if failed (not censored)
      y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks[1:]) #give credit for surviving each time interval where failure time >= upper limit
      if t[i]<breaks[-1]: #if failure time is greater than end of last time interval, no time interval will have failure marked
        y_train[i,n_intervals+np.where(t[i]<breaks[1:])[0][0]]=1 #mark failure at first bin where survival time < upper break-point
    else: #if censored
      y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks_midpoint) #if censored and lived more than half-way through interval, give credit for surviving the interval.
  return y_train

#%%

def fetch_scheduler(optimizer):
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr)
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=1, eta_min=args.min_lr)
    elif args.scheduler == None:
        return None
        
    return scheduler

#%%

def run_fold(df, args, model, criterion, optimizer, scheduler, device, fold, num_epochs=10):
    valid_df = df[df.kfold == fold]
    train_df = df[df.kfold != fold]
    
    '''
    df_proc_labels_test 를 그냥 .csv 로 저장하고 load 하는 방식으로 하기
    '''
    train_df_path = os.path.join(args.proc_label_dir, f'train_df_proc_labels_{args.dataset_name}.csv')
    train_df.to_csv(train_df_path)

    valid_df_path = os.path.join(args.proc_label_dir, f'valid_df_proc_labels_{args.dataset_name}.csv')
    valid_df.to_csv(valid_df_path)
    
    train_data = SurvDataset(df = train_df, args = args, dataset_name = f'{args.dataset_name}', transforms=args.train_transform, aug_transform=True) #True)
    valid_data = SurvDataset(df = valid_df, args = args, dataset_name = f'{args.dataset_name}', transforms=args.valid_transform, aug_transform=False)
  
    dataset_sizes = {
        'train' : len(train_data),
        'valid' : len(valid_data)
    }
    
    print(f'num of train_data: {len(train_data)}')
    print(f'num of valid_data: {len(valid_data)}')
    
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    
    dataloaders = {
        'train' : train_loader,
        'valid' : valid_loader
    }

    model, history = train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold)
    
    return model, history
#%%

''' MONAI transform 는 torchio 와 다르다: tutorials
  1. https://github.com/Project-MONAI/tutorials/blob/main/modules/postprocessing_transforms.ipynb
  2. https://github.com/Project-MONAI/MONAIBootcamp2020/blob/master/day1notebooks/lab1_transforms.ipynb
'''

def get_transform(args, dataset_name):
  
  if dataset_name == 'SNUH_UPenn_TCGA':
    landmark_dataset_name = 'SNUH_UPenn_TCGA' # train/valid/test=0.75/0.72/0.69 # 
    print(f'landmark_dataset_name:{landmark_dataset_name}')
  else:
    landmark_dataset_name = dataset_name
    print(f'landmark_dataset_name:{landmark_dataset_name}')
  
  landmarks_dir = os.path.join(args.data_dir, 'histograms', landmark_dataset_name)
  
  landmarks = {}
  for seq in args.sequence:
    
    seq_landmarks_path = os.path.join(landmarks_dir, f'{seq}_histgram.npy')
    landmarks[f'{seq}'] = seq_landmarks_path
  
  # print(f'landmarks:{list(landmarks.keys())}') # ['t1', 't2', 't1ce', 'flair']
    
  basic_transforms = [
      tio.HistogramStandardization(landmarks), 
      # tio.ZNormalization() # (masking_method=lambda x: x > 0) # x.mean() # # NOT working: RuntimeError: Standard deviation is 0 for masked values in image    
  ]

  basic_transform = tio.Compose(basic_transforms)
  # aug_transform = Compose(aug_transforms)
  
  print(f'transform for {dataset_name} was obtained')
  
  return basic_transform

#%%
def load_ckpt(args, model):
  ckpt_dir = os.path.join(os.getcwd(), 'saved_models', f'{args.net_architect}')
  os.makedirs(ckpt_dir, exist_ok = True)
  ckpt_list = glob.glob(f'{ckpt_dir}/*.pth') 
  ckpt_model = max(ckpt_list, key=os.path.getctime)
  print(f'latest_ckpt_model: {ckpt_model}') #'Fold0_3.1244475595619647_epoch23.pth'
  ckpt_path = os.path.join(ckpt_dir, ckpt_model) 
  model_dict = torch.load(ckpt_path, map_location='cuda') # NOT working when in utils.py: f'cuda:{gpu_id}'
  model.load_state_dict(model_dict)
  return model

class SurvDataset(nn.Module):
  def __init__(self, df, args, dataset_name, transforms=None, aug_transform=False): # ['27179925', '45163562', 'UPENN-GBM-00291_11', '42488471', 'UPENN-GBM-00410_11', '28802482']
    self.dataset_name = dataset_name
    self.df = df 
    self.args = args
    # print(self.df.shape) # (890, 39) # (223, 39)
    self.img_dir = os.path.join(args.data_dir, self.dataset_name, f'{args.compart_name}_BraTS') # 'SNUH_UPenn_TCGA_severance'
    self.transforms = transforms
    self.aug_transform = aug_transform

    self.znorm = tio.ZNormalization()
    self.rescale = tio.RescaleIntensity(out_min_max=(-1, 1))
    self.crop_size = 64
    self.crop = RandSpatialCrop(roi_size=(self.crop_size, self.crop_size, self.crop_size), random_size=False)
    
    # self.rand_affiner = RandAffine(prob=0.9, rotate_range=[-0.5,0.5], translate_range=[-7,7],scale_range= [-0.15,0.1], padding_mode='zeros')
    self.rand_affiner = RandAffine(prob=0.9)
    self.rand_elastic = Rand3DElastic(prob=0.8, magnitude_range = [-1,1], sigma_range = [0,1])
    self.flipper1 = RandFlip(prob=0.5, spatial_axis=0)
    self.flipper2 = RandFlip(prob=0.5, spatial_axis=1)
    self.flipper3 = RandFlip(prob=0.5, spatial_axis=2)
    self.gaussian = RandGaussianNoise(prob=0.3)
    self.contrast = AdjustContrast(gamma=2)
    self.compart_name = args.compart_name
  
  def concat_seq_img(self, x):
      return torch.cat([x[sequence][tio.DATA] for sequence in self.args.sequence], axis=0)
  
  def __len__(self):
    return len(self.df) # 결국 df 로 index 를 하기 때문에 dataset의 길이도 len(df): df를 train_df, val_df 넣는것에 따라 dataset이 train_set, val_set이 됨.

  def augment(self, img):
    # img = self.znorm(img)
    # img = self.rescale(img)
    img = self.crop(img)
    # img = self.rand_elastic(img)
    # img = self.rand_affiner(img)
    # img = self.flipper1(img)
    # img = self.flipper2(img)
    # img = self.flipper3(img)
    img = self.gaussian(img)
    # img = self.contrast(img)
    
    return img

  def __getitem__(self, idx):
    if type(idx) is not int:
      raise ValueError(f"Need `index` to be `int`. Got {type(idx)}.")

    ID = self.df.iloc[idx].name
    kfold = self.df['kfold'][idx]
    # print(f'kfold:{kfold}')
    # print(f'ID:{str(ID)}') # UPENN-GBM-00427_11
    subj_img_dir = os.path.join(self.img_dir, str(ID))
    # print(f'IMG_DIR:{IMG_DIR}')
        
    subject = tio.Subject(
        t1=tio.ScalarImage(os.path.join(subj_img_dir, f't1_{self.compart_name}.nii.gz')), # t1_seg.nii.gz
        t2=tio.ScalarImage(os.path.join(subj_img_dir, f't2_{self.compart_name}.nii.gz')), 
        t1ce=tio.ScalarImage(os.path.join(subj_img_dir, f't1ce_{self.compart_name}.nii.gz')), 
        flair=tio.ScalarImage(os.path.join(subj_img_dir, f'flair_{self.compart_name}.nii.gz')), 
        # mask=tio.LabelMap(os.path.join(subj_img_dir, 'WT_seg_resized.nii.gz')),
        # ID=ID
        )   
    
    if self.transforms:
      subject = self.transforms(subject)
    
    img = self.concat_seq_img(subject)
    # print(f'img loaded: {img.shape}') # torch.Size([4, 120, 120, 78])
    
    if self.aug_transform:
      # subject = self.transforms(subject)
      # subject.plot()
      
      # subject = self.augment(subject) # NOT working: monai는 subject 에선 안되고, image 에 대해서만 작동함. 따라서 이전에 내가 했던 것처럼 일단 전체 tio.Subjects()로 x_train, x_test 를 따로 만들어두고 거기서 dataloader 해오면서 augment 하기
      ''' https://torchio.readthedocs.io/quickstart.html#tutorials '''
      img = self.augment(img)
    # print(f'final input image shape: {img.shape}') # torch.Size([4, 120, 120, 78])
    
    proc_label_list = list(self.df[self.df.columns.difference(['ID', 'kfold'])].iloc[idx].values) # ID, kfold 는 제외
    proc_labels = [int(float(proc_label)) for proc_label in proc_label_list] # '1.0' -> 1: string -> float -> int
    proc_labels = torch.tensor(proc_labels)
    # print(f'proc_labels.shape:{proc_labels.shape}') # torch.Size([38])
    
    return img, proc_labels

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
https://auro-227.medium.com/writing-a-custom-layer-in-pytorch-14ab6ac94b77
''' 

class PropHazards(nn.Module):
  def __init__(self, size_in, size_out):#, device):
    super().__init__()
    
    # weights = torch.Tensor(size_out, size_in)
    # self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
    # bias = torch.Tensor(size_out)
    # self.bias = nn.Parameter(bias)
    # self.device = device
    self.linear = nn.Linear(size_in, size_out)#.to(device)
    self.flatten = nn.Flatten()#.to(device)
    # # initialize weights and biases
    # nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
    # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
    # bound = 1 / math.sqrt(fan_in)
    # nn.init.uniform_(self.bias, -bound, bound)  # bias init

  def forward(self, x):
    # print(f'before prophazard: {x.shape}')
    x = self.flatten(x)
    # print(f'after flatten: {x.shape}')
    # print(x.is_cuda) # true
    x = self.linear(x)
    # print(x.is_cuda) # true
    # print(f'after prophazard: {x.shape}')
    return torch.pow(torch.sigmoid(x), torch.exp(x)) #.float().to(device)

class CustomNetwork(nn.Module):
  def __init__(self, args, base_model):
    super().__init__()
    self.base_model = base_model
    self.model = vars(base_model)['_modules']
    self.output_dim = args.n_intervals # 19
    layers = []
    self.layer_name_list = [n for n in self.model][:-1]

    for name in self.layer_name_list:
      layers.append(self.model[name])
    self.layer1 = nn.ModuleList(layers)
    self.prophazards = PropHazards(2048, 19) # (size_in = args.last_size[args.net_architect], size_out = self.output_dim)
    
  def forward(self, x):
    for name in self.layer1:
    #   print(name)
      x = name(x)
    # print(x.size())
    last_size = x.view(x.size(0), -1).shape[-1]
    # print(f'last_size: {last_size}') # 2048
    device = x.device
    x = self.prophazards(x)
    # x = PropHazards(last_size, self.output_dim, device)(x)
    # x = self.flatten(x)
    # x = x.view(-1, 64*5*5) # Flatten layer
    return x

def get_output_shape(x, model):
  model.eval()
  x = model(x)
  return torch.tensor(x.shape[1:]).prod()#.cuda()

# import monai
# base_model = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=4, out_channels=19).cuda() 
# last_removed_base_model = nn.Sequential(*(list(base_model.children())[:-1]))
# # print(last_removed_base_model)
# last_size = get_output_shape(torch.rand(1, 4, 120, 120, 78).cuda(), last_removed_base_model)
# print(last_size)
# # args.last_size = last_size
# # torch.rand((4, 120, 120, 78)).cuda()
# # summary(base_model, (4, 120, 120, 78), device='cuda')
# model = CustomNetwork(args, base_model = base_model).cuda()
# summary(model, (4, 120, 120, 78), device='cuda')
#%%

''' Loss Function '''

class TaylorSoftmax(nn.Module):
    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n+1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out

class LabelSmoothingLoss(nn.Module): 
    def __init__(self, classes=5, smoothing=0.0, dim=-1): 
        super(LabelSmoothingLoss, self).__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls = classes 
        self.dim = dim 
    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred) 
            true_dist.fill_(self.smoothing / (self.cls - 1)) 
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    

class TaylorCrossEntropyLoss(nn.Module):
    def __init__(self, n=2, ignore_index=-1, reduction='mean', smoothing=0.2):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lab_smooth = LabelSmoothingLoss(args.num_classes, smoothing=smoothing)

    def forward(self, logits, labels):
        log_probs = self.taylor_softmax(logits).log()
        loss = self.lab_smooth(log_probs, labels)
        return loss

#%%
''' Training Function '''

def train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0
    best_loss = 100
    history = defaultdict(list)
    model = model.to(device)

    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','valid']:
            if(phase == 'train'):
                print(f'phase:{phase}')
                model.train() # Set model to training mode
            else:
                print(f'phase:{phase}')
                model.eval() # Set model to evaluation mode
            
            running_loss = 0.0
            # running_corrects = 0.0
            
            # Iterate over data
            for inputs,labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print(f'labels:{labels}')
                # print(f'labels.shape:{labels.shape}')

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(f'outputs:{outputs}')
                    # print(f'outputs.shape:{outputs.shape}')
                    # _, preds = torch.max(outputs,1)
                    # print(f'preds:{preds}')
                    # print(f'preds.shape:{preds.shape}')
                    loss = criterion(outputs, labels) # use this loss for any training statistics
                    # print(f'loss:{loss}')
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # first forward-backward pass
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        
                        # second forward-backward pass
                        criterion(model(inputs), labels).backward()
                        optimizer.second_step(zero_grad=True)


                running_loss += loss.item()*inputs.size(0)
                # print(f'running_loss: {running_loss}')
                # running_corrects += torch.sum(preds == labels.data).double().item()

            
            epoch_loss = running_loss/dataset_sizes[phase]
            # epoch_acc = running_corrects/dataset_sizes[phase]

            history[phase + ' loss'].append(epoch_loss)
            # history[phase + ' acc'].append(epoch_acc)

            if phase == 'train' and scheduler != None:
                scheduler.step()

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            
            # deep copy the model
            if phase=='valid' and epoch_loss <= best_loss: # epoch_acc >= best_acc:
                # best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                saved_model_path = os.path.join(os.getcwd(), 'saved_models', f'{args.net_architect}')
                os.makedirs(saved_model_path, exist_ok=True)
                PATH = os.path.join(saved_model_path, f"{datetime.now().strftime('%d_%m_%H_%m')}_{args.net_architect}_epoch{epoch}.pth")
                torch.save(model.state_dict(), PATH)

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    # print("Best Accuracy ",best_acc)
    print("Best Loss ",best_loss)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

#%%
''' Modified version declared in main.py '''
# def run_fold(model, criterion, optimizer, scheduler, device, fold, num_epochs=10):
#     valid_df = df[df.kfold == fold]
#     train_df = df[df.kfold != fold]
    
#     train_data = CassavaLeafDataset(TRAIN_DIR, train_df, transforms=data_transforms["train"])
#     valid_data = CassavaLeafDataset(TRAIN_DIR, valid_df, transforms=data_transforms["valid"])
    
#     dataset_sizes = {
#         'train' : len(train_data),
#         'valid' : len(valid_data)
#     }
    
#     train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
#     valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    
#     dataloaders = {
#         'train' : train_loader,
#         'valid' : valid_loader
#     }

#     model, history = train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold)
    
#     return model, history

#%%

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
#%%

''' grad CAM '''

''' BEST ref: https://www.kaggle.com/code/debarshichanda/gradcam-visualize-your-cnn '''

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)

        return target_activations, x

def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()
        
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

#%%

''' ref: 
Integrated BS: https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.integrated_brier_score.html 
BS: https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.brier_score.html

# from sksurv.datasets import load_gbsg2
# from sksurv.preprocessing import OneHotEncoder
# from sksurv.linear_model import CoxPHSurvivalAnalysis

X, y = load_gbsg2() # y = array([( True, 1814.), ( True, 2018.), ( True,  712.), ( True, 1807.), ... # y.shape = (686,)
X.loc[:, "tgrade"] = X.loc[:, "tgrade"].map(len).astype(int)
Xt = OneHotEncoder().fit_transform(X)

est = CoxPHSurvivalAnalysis(ties="efron").fit(Xt, y)

survs = est.predict_survival_function(Xt)
times = np.arange(365, 1826)
# preds = np.asarray([[fn(t) for t in times] for fn in survs]) # Integrated BS
preds = [fn(1825) for fn in survs] # BS

# print(f'preds:{preds}') 
# print(f'preds.shape:{preds.shape}') # IBS # (686, 1461) # N=686 # 1461 = 1826 - 365
print(f'len(preds):{len(preds)}') # BS # 686
# score = integrated_brier_score(y, y, preds, times)
times, score = brier_score(y, y, preds, 1825)
print(score)

'''

# BS at 1yr
def get_BS(event, duration, oneyr_survs, duration_set=365):
  y = [(evt, dur) for evt, dur in zip(np.asarray(event, dtype=bool), duration)]
  y = np.asarray(y, dtype=[('cens', bool), ('time', float)])
  times, score = brier_score(y, y, oneyr_survs, duration_set)
  print(f'BS score at {duration_set}:{score}')
  return score

#%%
''' NOT USING '''

def get_calibration_plot(model, x_train, y_train):
  surv_train = model.predict_surv_df(x_train) #.interpolate(1) # IDH-wt만 할 땐 interpolate(10) 아닐땐 interpolate(1).
  df_train=df.loc[sorted(train_IDs)].astype(str) 
  durations_train, events_train = get_target(df_train)
  ev_train = EvalSurv(surv_train, durations_train, events_train, censor_surv='km')

  y_obs_train, y_pred_train, kmf_prediction_train = get_pred_obs_csv(x_train, y_train, NUM_DURATIONS, df_train, train_IDs, 'train', spec_time)
  
  rf_y, rf_x = calibration_curve(y_obs_train, y_pred_train * 1.2, n_bins=20)
  
  fig, ax = plt.subplots()
  # only these two lines are calibration curves
  # plt.plot(logreg_x,logreg_y, marker='o', linewidth=1, label='logreg')
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
''' NOT USING '''

# from pytorch_grad_cam import * # GradCAMPlusPlus
# from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
# from PIL import Image
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# class GradCam:
#     def __init__(self, model, feature_module, target_layer_names, use_cuda):
#         self.model = model
#         self.feature_module = feature_module
#         self.model.eval()
#         self.cuda = use_cuda
#         if self.cuda:
#             self.model = model.cuda()

#         self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

#     def forward(self, input_img):
#         return self.model(input_img)

#     def __call__(self, input_img, target_category=None):
#         if self.cuda:
#             input_img = input_img.cuda()

#         features, output = self.extractor(input_img)

#         if target_category == None:
#             target_category = np.argmax(output.cpu().data.numpy())

#         one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
#         one_hot[0][target_category] = 1
#         one_hot = torch.from_numpy(one_hot).requires_grad_(True)
#         if self.cuda:
#             one_hot = one_hot.cuda()
        
#         one_hot = torch.sum(one_hot * output)

#         self.feature_module.zero_grad()
#         self.model.zero_grad()
#         one_hot.backward(retain_graph=True)

#         grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

#         target = features[-1]
#         target = target.cpu().data.numpy()[0, :]

#         weights = np.mean(grads_val, axis=(2, 3))[0, :]
#         cam = np.zeros(target.shape[1:], dtype=np.float32)

#         for i, w in enumerate(weights):
#             cam += w * target[i, :, :]

#         cam = np.maximum(cam, 0)
#         cam = cv2.resize(cam, input_img.shape[2:])
#         cam = cam - np.min(cam)
#         cam = cam / np.max(cam)
#         return cam

# grad_cam = GradCam(model=model, feature_module=model.SEResNeXtBottleneck, target_layer_names=["6"], use_cuda=True) 

# def get_DL_score(input, id_list, dataframe, dataset):
  
#   surv_ids = []
#   surv_total = []
#   id_list = sorted(id_list)
#   for i in range(input.shape[0]):
#     input=input.cuda().float()
#     surv_prob = model.predict_surv(input[i].unsqueeze(0)) #, c_total[i].unsqueeze(0))) 
#     surv_id = str(id_list[i])
#     # print(f'surv_id:{surv_id}')
#     # tensor([[0.9994, 0.6746, 0.4023, 0.2937, 0.2179, 0.1801, 0.1361, 0.1044, 0.0617, 0.0503]], device='cuda:0') 
#     surv_prob_np = np.squeeze(to_np(surv_prob), axis=0)
#     # print(f'surv_prob: {surv_prob}')
#     # print(f'surv_prob_np: {surv_prob_np}')
#     surv_total.append(surv_prob_np)
#     surv_ids.append(surv_id)
#   surv_total = np.stack(surv_total)
#   print(f'surv_total: {surv_total.shape}')
#   print(f'surv_ids: {surv_ids}')
#   df_surv_total = pd.DataFrame(surv_total)
#   df_surv_total['ID'] = surv_ids
#   df_surv_total.columns = [f'MRI{i}' for i in range(1, NUM_DURATIONS+1)] + ['ID']
#   df_surv_total = df_surv_total.set_index('ID')
#   print(df_surv_total.head(5))
#   surv_total_path = os.path.join(DATA_DIR, 'surv_predict_total.csv')
#   df_surv_total.to_csv(surv_total_path)

#   ''' combine predictions of MRI4 and MRI5, and labels together for selected IDs '''
#   df = dataframe
#   df_selected_label = df.loc[surv_ids]
#   for i in range(1, NUM_DURATIONS+1):
#     df_selected_label[f'MRI{i}'] = df_surv_total[f'MRI{i}']

#   cut_feats = [1.0 - i * 0.1 for i in range(1, NUM_DURATIONS)]
#   bin_feats = [f'MRI{i}'for i in range(1, NUM_DURATIONS+1)]
  
#   for cut_feat, feat in zip(cut_feats, bin_feats):
#     bin_feat = feat+"_bin"
#     print(bin_feat, cut_feat)
#     df_selected_label.loc[df_selected_label[feat] >= cut_feat, bin_feat] = 1
#     df_selected_label.loc[df_selected_label[feat] < cut_feat, bin_feat] = 0

#   print(df_selected_label.head(5))
#   selected_label_total_path = os.path.join(DL_feature_DIR, f'{dataset}_{PATHOLOGY}_{PROGRESS_DURATION}_prediction_and_label_selected.csv')
#   df_selected_label.to_csv(selected_label_total_path)
