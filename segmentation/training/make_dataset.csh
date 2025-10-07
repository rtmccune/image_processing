#! /bin/bash
#BSUB -J make_data
#BSUB -o data_gen_out.%J
#BSUB -e data_gen_err.%J
#BSUB -W 60
#BSUB -n 12
#BSUB -R span[hosts=1]
#BSUB -q gpu
#BSUB -gpu "num=1"
source ~/.bashrc

module load cuda/12.1
module load apptainer

# Define the project base directory to make binding easier
PROJECT_DIR="/base/directory/segmentation"

# Define the path to your Apptainer image
IMAGE_PATH="${PROJECT_DIR}/training/container/seg_gym_tf.sif"

# Define the name of the specific data directory you wish to use
DATA_DIR_NAME="template" ####---EDIT THIS LINE---####

CONFIG_DIR="${PROJECT_DIR}/data/${DATA_DIR_NAME}/config"

# Execute the container with the correct syntax
apptainer exec --nv \
    --bind ${PROJECT_DIR} \
    ${IMAGE_PATH} \
    python ${PROJECT_DIR}/segmentation_gym/make_dataset_no_tkinter.py \
    --output ${PROJECT_DIR}/data/${DATA_DIR_NAME}/fromDoodler/npz4gym \
    --label_dir ${PROJECT_DIR}/data/${DATA_DIR_NAME}/fromDoodler/labels \
    --image_dirs ${PROJECT_DIR}/data/${DATA_DIR_NAME}/fromDoodler/images \
    --config ${CONFIG_DIR}/segformer_example.json ###---EDIT THIS LINE---###
