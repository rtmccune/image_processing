# Image Processing

This repository contains all of the scripts and information required for processing imagery as part of the Sunny Day Flooding Project at NC State University and the University of North Carolina Chapel Hill. 

## Machine Learning

Machine learning implementations for segmentation of imagery are conducted with the Doodleverse library developed by Dan Buscombe and Evan Goldstein. This repository does not contain instructions on labeling imagery with Doodler or the training steps of Segmentation Gym. Rather, this is for deployment of a trained Segmentation Gym model for processing and quantification. 

The segmentation directory contains the 'data', 'segmentation_gym', and 'training' folders.

* 'data': Contains a 'template' folder which mimics the directory structure required for Segmentation Gym. This is where images and labels from Doodler will be stored along with a config file.
* 'segmentation_gym': Contains Segmentation Gym and the cached transformers weights.
* 'training': Contains the submission scripts for running the model predictions. This also contains the submission scripts for training the model in two steps. 'make_dataset' generates the npz files for Segmentation Gym while 'train_model' trains the model. We will use the 'predictions' file to generate predicted segmentations. All of these files are written to operate as submission scripts on an HPC system which uses LSF scheduling. 

