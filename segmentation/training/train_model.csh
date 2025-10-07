#! /bin/bash
#BSUB -J seg_model
#BSUB -o train_out.%J
#BSUB -e train_err.%J
#BSUB -W 300
#BSUB -n 1
#BSUB -R span[hosts=1]
#BSUB -R rusage[mem=10G]
#BSUB -q gpu
#BSUB -gpu "num=1:mode=exclusive_process"
source ~/.bashrc

export APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_TRANSFORMERS_CACHE="/base/directory/segmentation/segmentation_gym/hf_cache_portable"

module load cuda/12.1
module load apptainer

# Define the project base directory to make binding easier
PROJECT_DIR="/base/directory/segmentation"

# Define the path to your Apptainer image
IMAGE_PATH="${PROJECT_DIR}/training/container/seg_gym_tf.sif"

# Define the name of the specific data directory you wish to use
DATA_DIR_NAME="template" ####---EDIT THIS LINE---####

# Define NPZ and configuration file directories
NPZ_DIR="${PROJECT_DIR}/data/${DATA_DIR_NAME}/fromDoodler/npz4gym"
CONFIG_DIR="${PROJECT_DIR}/data/${DATA_DIR_NAME}/config"

# Execute the container with the correct syntax
apptainer exec --nv \
    --bind ${PROJECT_DIR} \
    ${IMAGE_PATH} \
    python ${PROJECT_DIR}/segmentation_gym/train_model_script_no_tkinter.py \
    --train_data_dir ${NPZ_DIR}/train_data/train_npzs \
    --val_data_dir ${NPZ_DIR}/val_data/val_npzs \
    --config_file ${CONFIG_DIR}/segformer_example.json ####---EDIT THIS LINE---####
