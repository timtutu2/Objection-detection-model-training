# 1. base: official image with CUDA + cuDNN (use devel for compilation tools)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 2. basic tools (including build tools for Detectron2)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git sudo curl ca-certificates wget \
    build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libopencv-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install opencv-python

RUN git clone https://github.com/ultralytics/yolov5.git
WORKDIR /workspace/yolov5
RUN pip install -r requirements.txt

WORKDIR /workspace
RUN git clone https://github.com/WongKinYiu/yolov7.git
WORKDIR /workspace/yolov7
RUN pip install -r requirements.txt

WORKDIR /workspace
RUN pip install fvcore cloudpickle omegaconf hydra-core black isort flake8 iopath pycocotools
RUN git clone https://github.com/facebookresearch/detectron2.git /workspace/detectron2_repo
WORKDIR /workspace/detectron2_repo
RUN python -m pip install -e . --no-build-isolation

WORKDIR /workspace
RUN mkdir -p /workspace/detectron2_workspace

RUN pip install wandb

RUN mkdir -p /workspace/datasets

WORKDIR /workspace/yolov5

CMD ["python", "train.py", "--help"]
