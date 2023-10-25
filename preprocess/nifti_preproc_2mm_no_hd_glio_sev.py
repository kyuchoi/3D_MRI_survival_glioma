#%%
import os
import numpy as np
import glob
import shutil
from tqdm import tqdm
from datetime import datetime
import torchio as tio
import argparse

#%%

start_time = datetime.now()

os.environ['MKL_THREADING_LAYER'] = 'GNU'

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='SNUH GBM dataset preprocessor', add_help=add_help)
    parser.add_argument('--fix_sequence', type=str, default='t1ce')
    parser.add_argument('--resize_type', type=str, default='BraTS') 
    parser.add_argument('--dataset_name', type=str, default= 'severance')
    parser.add_argument('--root_dir', default=r'/mnt/hdd3/mskim/GBL/data') 
    return parser

args = get_args_parser().parse_args()

DATASET_NAME = args.dataset_name
RESIZE_TYPE = args.resize_type

resize_dict = {'bhk': (150,180,150),  
              'BraTS': (240,240,155),
              'SNUH': (192,256,256)}

IMG_SIZE = resize_dict[RESIZE_TYPE] 
SEQUENCE = ['t1ce', 'flair', 't1', 't2'] 

ROOT_DIR = args.root_dir
DATA_DIR = os.path.join(ROOT_DIR, DATASET_NAME)
INPUT_DIR = os.path.join(DATA_DIR, 'no_hd_glio')
OUTPUT_DIR = os.path.join(DATA_DIR, f'resampled_{RESIZE_TYPE}') #_no_hd_glio')
RESIZE_DIR = os.path.join(DATA_DIR, f'resized_{RESIZE_TYPE}') #_no_hd_glio')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESIZE_DIR, exist_ok=True)

GET_DIR_LIST = lambda PATH: sorted([DIR for DIR in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, DIR))])
crop_z_bottom = 15 # better than 20 for SNUH_merged

#%%
''' get subjects with all sequences (i.e. t1, t2, t1ce, flair) before resampling '''


for subject in tqdm(GET_DIR_LIST(INPUT_DIR)):
    print(f'checking for complete sequences: {subject}')
    
    seq_count = 0
    for sequence in SEQUENCE: 
        bet_path = os.path.join(INPUT_DIR, subject, f'{sequence}.nii.gz')
        if os.path.isfile(bet_path):
            seq_count += 1 
    print(f'seq_count of {subject}:{seq_count}')
    
    incomplete_subjs=[]
    if seq_count == 4:    
        input_subj_path = os.path.join(INPUT_DIR, subject)

        complete_subj_path = os.path.join(OUTPUT_DIR, subject)
        os.makedirs(complete_subj_path, exist_ok=True)

    else:
        print(f'{subject}: no complete sequences')
        incomplete_subjs.append(subject)

# incomplete_subjs = np.array(incomplete_subjs)
# incomplete_path = os.path.join(DATA_DIR, 'incomplete_seq_subj_list.csv')
# print(f'Saving subject list without complete sequences')
# np.savetxt(incomplete_path, incomplete_subjs, delimiter=",")
    
#%%

''' RAS+ and 1mm isotropic resampling '''

transforms = [
        tio.ToCanonical(),
        tio.Resample(1),                        
        tio.Crop((0, 0, 0, 0, crop_z_bottom, 0)),
        tio.CropOrPad(IMG_SIZE)
    ]
transform = tio.Compose(transforms)

print(f'Complete sequences: total {len(GET_DIR_LIST(OUTPUT_DIR))} of {len(GET_DIR_LIST(INPUT_DIR))} subjects')
for subject in tqdm(GET_DIR_LIST(INPUT_DIR)):
    print(f'processing {subject}')
    
    for sequence in SEQUENCE:
        resampled_path = os.path.join(OUTPUT_DIR, subject, f'{sequence}_resampled.nii.gz')
        if not os.path.isfile(resampled_path):
            print(f'canonize and resampling {sequence}')
            try:
                image = tio.ScalarImage(os.path.join(INPUT_DIR, subject, f'{sequence}.nii.gz'))
                preprocessed = transform(image)
                preprocessed.save(resampled_path)
                # preprocessed.plot()
            except FileNotFoundError:
                print(f'No file named {sequence}.nii.gz in {subject}')
                pass
        else:
            print(f'{resampled_path} already exists')
    

#%%

resize_transforms = [
    tio.Resample(2),
    tio.CropOrPad((120, 120, 78)) # 이거 안해주면 (121, 120, 78) 이런 게 생김
]
resize_transform = tio.Compose(resize_transforms) 

for subject in tqdm(GET_DIR_LIST(OUTPUT_DIR)):
    print(f'Resizing {subject} in 2mm')
    
    os.makedirs(os.path.join(RESIZE_DIR, subject), exist_ok=True)        
    for sequence in SEQUENCE:
        try:
            print(f'resizing {sequence}')
            image = tio.ScalarImage(os.path.join(OUTPUT_DIR, subject, f'{sequence}_resampled.nii.gz'))
            
            resized_path = os.path.join(RESIZE_DIR, subject, f'{sequence}_resized.nii.gz')
            
            preprocessed = resize_transform(image)
            preprocessed.save(resized_path)
            # preprocessed.plot()

        except FileNotFoundError:
            print(f'No file named {sequence}_resampled.nii.gz in {OUTPUT_DIR}')
            pass    