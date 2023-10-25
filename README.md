# 3D_MRI_survival_glioma
Official repository for Added Prognostic Value of 3D Deep Learning-Derived Features from Preoperative MRI for Adult-type Diffuse Gliomas

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

'''3D grad CAM''' is applied to Preoperative MRI using 3D CNN Deep Learning Model for Adult-type Diffuse Gliomas

## Docker images (images:tag)
1. snuhradaicon/gbl_surv_distribution:snuhradaicon (released version)
2. snuhradaicon/gbl_surv_custom:snuhradaicon (custom version)

### Composition (File in Docker images)
1. main_copy.py, train_inference_copy.py, data(UCSF,TCGA,UPenn), utils_copy.py, attention_models.py,Preprocess (1 bash file, 1 python file)
2. main_copy.py, train_inference_copy.py, inference_copy.py, utils_copy.py, attention_models.py, data (UCSF,TCGA,UPenn), preprocess (1 bash file, 1 python file)

### Use (Docker images)
if you want to use a released version,

First
    
    docker pull snuhradaicon/gbl_surv_distribution:snuhradaicon

Second

    docker run --gpus all -it snuhradaicon/gbl_surv_distribution:snuhradaicon

third
 1. Run python main_copy.py 

    python main_copy.py 

 2. Run python train_inference_copy.py

    python train_inference_copy.py

* main_copy.py -> input_args : --gpu_id, --test_gpu_id, --epochs, --seed, --spec_patho (all,GBL), --spec_duration (OS,1yr), --spec_event (death), --ext_dataset_name, --dataset_list,

* train_inference_copy.py -> inpurt args : --gpu_id, --spec_patho (all,GBL), --spec_duration (OS,1yr), --spec_event (death), --dataset_list (* you must pass same args when run main_copy.py)


if you want to use a custom version,

First
    
    docker pull snuhradaicon/gbl_surv_custom:snuhradaicon

Second
    
    docker run --gpus all -it snuhradaicon/gbl_surv_custom:snuhradaicon

Third
 1. You must load data in container
 
 2. Preprocess
    2-1. Run skull_strip_coreg.sh

    sh skull_strip_coreg.sh [root_dir] [--input_dir] [input_dir]

    * input_arg : $1 Real root_dir $2 --input_dir (string) $3 your Real input_dir (Sub-directory of root_dir )
    
    * if you don't run a sh file, ref: <https://github.com/NeuroAI-HD/HD-GLIO>

    2-2. Run resample.py

    python resample.py [--fix_sequence] [--resize_type] [--dataset_name] [--root_dir]

    * input_arg : --fix_sequence, --resize_type(defalut:BraTS), --dataset_name, --root_dir
 
 3. Run python main_copy.py

    python main_copy.py
 
 4. Run python train_inference_copy.py
 
    python train_inference_copy.py

 5. Run python inference_copy.py

    python inference_copy.py

* main_copy.py -> input_args : --gpu_id, --test_gpu_id, --epochs, --seed, --spec_patho (all,GBL), --spec_duration (OS,1yr), --spec_event (death), --ext_dataset_name, --dataset_list,

* train_inference_copy.py -> inpurt args : --gpu_id, --spec_patho (all,GBL), --spec_duration (OS,1yr), --spec_event (death), --dataset_list (* you must pass same args when run main_copy.py)

* inference_copy.py -> inpurt args : --test_gpu_id, --spec_patho (all,GBL), --spec_duration (OS,1yr), --spec_event (death), --dataset_list (* you must pass same args when run main_copy.py)