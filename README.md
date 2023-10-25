# 3D_MRI_survival_glioma
Official repository for Added Prognostic Value of 3D Deep Learning-Derived Features from Preoperative MRI for Adult-type Diffuse Gliomas

# Docker hub (images:tag)
1. snuhradaicon/gbl_surv_distribution:snuhradaicon (released version)
2. snuhradaicon/gbl_surv_custom:snuhradaicon (custom version)

# Composition (File in images)
1. main_copy.py, train_inference_copy.py, data(UCSF,TCGA,UPenn), utils_copy.py, attention_models.py,Preprocess (1 bash file, 1 python file)
2. main_copy.py, train_inference_copy.py, inference_copy.py, utils_copy.py, attention_models.py, data (UCSF,TCGA,UPenn), preprocess (1 bash file, 1 python file)

if you want to use a released version,

First
 - docker pull snuhradaicon/gbl_surv_distribution:snuhradaicon

Second
 - docker run --gpus all -it snuhradaicon/gbl_surv_distribution:snuhradaicon

third
 1. python main_copy.py (ex : --gpu_id 0 --spec_patho all ...)
 2. python train_inference.py (ex : --gpu_id 0 --spec_patho all)

main_copy.py -> input_args : --gpu_id, --test_gpu_id, --epochs, --seed, --spec_patho (all,GBL), --spec_duration (OS,1yr), --spec_event (death), --ext_dataset_name, --dataset_list,

train_inference_copy.py -> inpurt args : --gpu_id, --spec_patho (all,GBL), --spec_duration (OS,1yr), --spec_event (death), --dataset_list (* you must pass same args when run main_copy.py)

if you want to use a custom version,

First
 - docker pull snuhradaicon/gbl_surv_custom:snuhradaicon

Second
 - docker run --gpus all -it snuhradaicon/gbl_surv_custom:snuhradaicon

Third
 1. You must load data in container
 
 2. Preprocess
    2-1. Run no_hd_glio_only_coreg_survival_severance_copy.sh
    * input_arg : $1 Real root_dir $2 --input_dir (string) $3 your Real input_dir (root_dir -r)
    * if you don't run a sh file, ref: https://github.com/NeuroAI-HD/HD-GLIO

    2-2. Run nifti_preproc_2mm_no_hd_glio_sev_copy.py
    * input_arg : --fix_sequence, --resize_type(defalut:BraTS), --dataset_name, --root_dir
 
 3. python main_copy.py (ex : --gpu_id 0 --spec_patho all ...)
 
 4. python train_inference_copy.py (ex : --gpu_id 0 --spec_patho all ...)
 
 5. python inference_copy.py (ex:--test_gpu_id 1 --spec_patho all ...)
 
 inference_copy.py -> inpurt args : --test_gpu_id, --spec_patho (all,GBL), --spec_duration (OS,1yr), --spec_event (death), --dataset_list (* you must pass same args when run main_copy.py)