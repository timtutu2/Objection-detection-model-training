# 1. base: official image with CUDA + cuDNN
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 2. basic tools   
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git sudo curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 3. make python3 is python
RUN ln -s /usr/bin/python3 /usr/bin/python

# 4. create work directory
WORKDIR /workspace

# 5. install PyTorch (use official, for CUDA 12.x)
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 6. clone YOLOv5
RUN git clone https://github.com/ultralytics/yolov5.git
WORKDIR /workspace/yolov5

# 7. install YOLOv5 requirements
RUN pip install -r requirements.txt

# 8. create a folder to mount volume for dataset
RUN mkdir -p /workspace/datasets

# 9. default directory is this folder
WORKDIR /workspace/yolov5

# 10. default command: show help, avoid container exit immediately
CMD ["python", "train.py", "--help"]
