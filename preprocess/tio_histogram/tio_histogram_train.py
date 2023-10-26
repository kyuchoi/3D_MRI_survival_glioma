#%%

import os
import numpy as np
import glob
import argparse
# from tqdm import tqdm
from datetime import datetime
import torchio as tio

start_time = datetime.now()
  
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='SNUH GBM deep survival prediction model', add_help=add_help)
    parser.add_argument('--root_dir', default=r'/mnt/hdd3/mskim/GBL/data') 
    parser.add_argument('--dataset_name', type=str, default='SNUH') # 'UPenn' # 'severance' #
    parser.add_argument('--resize_type', type=str, default='BraTS') 
    parser.add_argument('--dataset_list', nargs='+', default=[]) # Usage: python tio_histogram_train.py --dataset_list SNUH UPenn
    
    return parser

args = get_args_parser().parse_args()

ROOT_DIR = args.root_dir
RESIZE_TYPE = args.resize_type
DATASET_LIST = args.dataset_list
print(f'DATASET_LIST: {len(DATASET_LIST)}')

GET_DIR_LIST = lambda PATH: sorted([DIR for DIR in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, DIR))])

if len(DATASET_LIST) == 0:
    print('hey, you passed dataset_name only, not dataset_list as arguments')
    DATASET_NAME = args.dataset_name
    DATASET_LIST = [DATASET_NAME]
    print(f'DATASETS: {DATASET_NAME}')
    # print(f'DATASET_LIST: {DATASET_LIST}')
else:
    DATASET_NAME = '_'.join(DATASET_LIST)
    print(f'DATASETS: {DATASET_NAME}')
    # print(f'DATASET_LIST: {DATASET_LIST}')

OUTPUT_DIR = os.path.join(ROOT_DIR, 'histograms', DATASET_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'OUTPUT_DIR: {OUTPUT_DIR}')
#%%
total_t1_image_paths = []
for DATASET in DATASET_LIST:
    print(f'Processing {DATASET}')
    DATA_DIR = os.path.join(ROOT_DIR, DATASET)
    INPUT_DIR = os.path.join(DATA_DIR, f'resized_{RESIZE_TYPE}')

    t1_image_paths = sorted(glob.glob(f'{INPUT_DIR}/*/t1_resized.nii.gz'))
    total_t1_image_paths+=t1_image_paths
# print(f'total_t1_image_paths: {total_t1_image_paths}')
print(f'total_t1_image_paths: {len(total_t1_image_paths)}')

total_t2_image_paths = []
for DATASET in DATASET_LIST:
    print(f'Processing {DATASET}')
    DATA_DIR = os.path.join(ROOT_DIR, DATASET)
    INPUT_DIR = os.path.join(DATA_DIR, f'resized_{RESIZE_TYPE}')

    t2_image_paths = sorted(glob.glob(f'{INPUT_DIR}/*/t2_resized.nii.gz'))
    total_t2_image_paths+=t2_image_paths
print(f'total_t2_image_paths: {len(total_t2_image_paths)}')

total_flair_image_paths = []
for DATASET in DATASET_LIST:
    print(f'Processing {DATASET}')
    DATA_DIR = os.path.join(ROOT_DIR, DATASET)
    INPUT_DIR = os.path.join(DATA_DIR, f'resized_{RESIZE_TYPE}')

    flair_image_paths = sorted(glob.glob(f'{INPUT_DIR}/*/flair_resized.nii.gz'))
    total_flair_image_paths+=flair_image_paths
print(f'total_flair_image_paths: {len(total_flair_image_paths)}')

total_t1ce_image_paths = []
for DATASET in DATASET_LIST:
    print(f'Processing {DATASET}')
    DATA_DIR = os.path.join(ROOT_DIR, DATASET)
    INPUT_DIR = os.path.join(DATA_DIR, f'resized_{RESIZE_TYPE}')

    t1ce_image_paths = sorted(glob.glob(f'{INPUT_DIR}/*/t1ce_resized.nii.gz'))
    total_t1ce_image_paths+=t1ce_image_paths
print(f'total_t1ce_image_paths: {len(total_t1ce_image_paths)}')
    
#%%

t1_histogram_landmarks_path = os.path.join(OUTPUT_DIR, 't1_histgram.npy')
t1_landmarks = tio.HistogramStandardization.train(
    total_t1_image_paths,
    output_path=t1_histogram_landmarks_path
)
np.set_printoptions(suppress=True, precision=3)
print('\nTrained t1 landmarks:', t1_landmarks)
t1_landmarks_dict = {'t1': t1_landmarks}
t1_histogram_transform = tio.HistogramStandardization(t1_landmarks_dict)

#%%
t2_image_paths = sorted(glob.glob(f'{INPUT_DIR}/*/t2_resized.nii.gz'))
t2_histogram_landmarks_path = os.path.join(OUTPUT_DIR, 't2_histgram.npy')
t2_landmarks = tio.HistogramStandardization.train(
    total_t2_image_paths,
    output_path=t2_histogram_landmarks_path
)
np.set_printoptions(suppress=True, precision=3)
print('\nTrained t2 landmarks:', t2_landmarks)
t2_landmarks_dict = {'t2': t2_landmarks}
t2_histogram_transform = tio.HistogramStandardization(t2_landmarks_dict)

#%%
flair_image_paths = sorted(glob.glob(f'{INPUT_DIR}/*/flair_resized.nii.gz'))
flair_histogram_landmarks_path = os.path.join(OUTPUT_DIR, 'flair_histgram.npy')
flair_landmarks = tio.HistogramStandardization.train(
    total_flair_image_paths,
    output_path=flair_histogram_landmarks_path
)
np.set_printoptions(suppress=True, precision=3)
print('\nTrained flair landmarks:', flair_landmarks)
flair_landmarks_dict = {'flair': flair_landmarks}
flair_histogram_transform = tio.HistogramStandardization(flair_landmarks_dict)

#%%
t1ce_image_paths = sorted(glob.glob(f'{INPUT_DIR}/*/t1ce_resized.nii.gz'))
t1ce_histogram_landmarks_path = os.path.join(OUTPUT_DIR, 't1ce_histgram.npy')
t1ce_landmarks = tio.HistogramStandardization.train(
    total_t1ce_image_paths,
    output_path=t1ce_histogram_landmarks_path
)
np.set_printoptions(suppress=True, precision=3)
print('\nTrained t1ce landmarks:', t1ce_landmarks)
t1ce_landmarks_dict = {'t1ce': t1ce_landmarks}
t1ce_histogram_transform = tio.HistogramStandardization(t1ce_landmarks_dict)

#%%
end_time = datetime.now()
process_time = end_time - start_time

print(f'preprocessing time: {process_time}')
print('DONE')
