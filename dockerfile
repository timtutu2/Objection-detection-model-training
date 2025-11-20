# 1. base: official image with CUDA + cuDNN (use devel for compilation tools)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 2. basic tools (including build tools for Detectron2)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git sudo curl ca-certificates wget \
    build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 3. make python3 is python
RUN ln -s /usr/bin/python3 /usr/bin/python

# 4. create work directory
WORKDIR /workspace

# 5. install PyTorch (use official, for CUDA 12.x)
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5.5. install OpenCV and other dependencies needed for all frameworks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libopencv-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install opencv-python

# 6. clone YOLOv5
RUN git clone https://github.com/ultralytics/yolov5.git
WORKDIR /workspace/yolov5
RUN pip install -r requirements.txt

# 6.5. clone YOLOv7
WORKDIR /workspace
RUN git clone https://github.com/WongKinYiu/yolov7.git
WORKDIR /workspace/yolov7
RUN pip install -r requirements.txt

# 6.7. install Detectron2 for Faster R-CNN and other models
WORKDIR /workspace
# Install detectron2 dependencies first
RUN pip install fvcore cloudpickle omegaconf hydra-core black isort flake8 iopath pycocotools
# Build and install detectron2 from source (use --no-build-isolation to access torch)
RUN git clone https://github.com/facebookresearch/detectron2.git /workspace/detectron2_repo
WORKDIR /workspace/detectron2_repo
RUN python -m pip install -e . --no-build-isolation

# 6.9. create Detectron2 working directory
WORKDIR /workspace
RUN mkdir -p /workspace/detectron2_workspace

# 7. install wandb for training monitoring and logging
RUN pip install wandb

# 8. create a folder to mount volume for dataset
RUN mkdir -p /workspace/datasets

# 9. default directory is yolov5 folder 
WORKDIR /workspace/yolov5

# 10. default command: show help, avoid container exit immediately
CMD ["python", "train.py", "--help"]
