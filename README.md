# 3D_MRI_survival_glioma
*Official repository* for *Added Prognostic Value of 3D Deep Learning-Derived Features from Preoperative MRI for Adult-type Diffuse Gliomas* (Neuro Oncol
. 2023 Oct 19:noad202. doi: 10.1093/neuonc/noad202. https://pubmed.ncbi.nlm.nih.gov/37855826/)

The developed 3-types based network (**Squeeze-and-Excitation ResNeXt**, **DenseNet**, **Resnet-Convolutional Block Attention Module**), *'CustomNet'* extracts spatial features from preoperative ```whole-brain MRI images```. It aims to quantitatively evaluate ```the additional value of these imaging features as prognostic factors```, independently from well-established clinical and pathological prognostic factors, for adult-type gliomas.


## Requirements
Python3 (Anaconda) with following packages:

    pytorch >= cuda 11.4 version

MUST install <https://pytorch.org/>

## Directories Structure

    3D_MRI_survival_glioma
    ├─3D grad CAM
    ├─model
    │  ├─attention_models
    │  ├─deep_surv_image_only_attention
    │  ├─grad_CAM
    │  ├─inference
    │  ├─main
    │  ├─train_inference
    │  └─utils
    └─preprocess
        ├─resample
        └─skull_strip_coreg

```3D grad CAM``` is the output of an 3D CNN Deep Learning Model From Preoperative MRI for Adult-type Diffuse Gliomas

## Docker images (images:tag)
1. snuhradaicon/gbl_surv_distribution:snuhradaicon (released version)  
  

2. snuhradaicon/gbl_surv_custom:snuhradaicon (custom version)
  

## Use (Docker)
#### if you want to use a released version,


First
    
    docker pull snuhradaicon/gbl_surv_distribution:snuhradaicon

Second

    docker run --gpus all -it snuhradaicon/gbl_surv_distribution:snuhradaicon
  

Third  
  

```3-1.``` Run main_copy.py

    python main_copy.py 
  

```3-2.``` Run train_inference_copy.py

    python train_inference_copy.py


**Dataset : UCSF, UPenn, TCGA**

```main_copy.py (input_args)``` : --gpu_id, --test_gpu_id, --epochs, --seed, --spec_patho (all,GBL), --spec_duration (OS,1yr), --spec_event (death), --ext_dataset_name, --dataset_list

```train_inference_copy.py (input_args)``` : --gpu_id, --spec_patho (all,GBL), --spec_duration (OS,1yr), --spec_event (death), --dataset_list (* you must pass same args when run main_copy.py)


#### if you want to use a custom version,  
  

First
    
    docker pull snuhradaicon/gbl_surv_custom:snuhradaicon
  

Second
    
    docker run --gpus all -it snuhradaicon/gbl_surv_custom:snuhradaicon
  

Third


```3-1.``` You must load your data in container.


```3-2.``` Have to load a label of data (duration,event) in container.  


Fourth


```4-1.``` Run skull_strip_coreg.sh

    sh skull_strip_coreg.sh [root_dir] [--input_dir] [input_dir]
    
* if you don't run a sh file, <https://github.com/NeuroAI-HD/HD-GLIO>


```4-2.``` Run resample.py

    python resample.py [--fix_sequence] [--resize_type] [--dataset_name] [--root_dir]
  
  
Fifth


```5-1.``` Run python main_copy.py

    python main_copy.py
 

```5-2.``` Run python train_inference_copy.py

    python train_inference_copy.py


```5-3.``` Run python inference_copy.py

    python inference_copy.py

**Dataset : UCSF, UPenn, TCGA**

```inference_copy.py (input args)``` : --test_gpu_id, --spec_patho (all,GBL), --spec_duration (OS,1yr), --spec_event (death), --dataset_list (* you must pass same args when run main_copy.py)
