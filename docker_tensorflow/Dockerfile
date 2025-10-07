# Start from NVIDIA NGC TensorFlow container (TF 2.12 + CUDA 11.8 + cuDNN 8.6)
FROM nvcr.io/nvidia/tensorflow:23.05-tf2-py3

# Install extra dependencies with apt if needed (not many here)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        && rm -rf /var/lib/apt/lists/*

# Install conda-forge packages directly with pip or conda
# (since the base already comes with conda, but you can skip and just use pip)
RUN pip install --upgrade pip

# Install Python packages
RUN pip install \
      numpy \
      scipy \
      scikit-image \
      ipython \
      joblib \
      tqdm \
      pandas \
      natsort \
      matplotlib \
      doodleverse_utils \
      transformers==4.35.2

# Note:
# - cudatoolkit/cudnn/tensorflow are already correctly handled in this base image
# - You don’t need to install `cuda` or `cudatoolkit` again, since NGC images ship with the right runtime stack

# Default shell
CMD ["/bin/bash"]

