#!/bin/bash

# --- [1] Configurable variables ---
IMAGE_NAME="timtt/yolov5-nrp:latest"      # your docker image name
LOCAL_ROOT="$HOME/Desktop/ECE253/dataset/All"
CODE_PATH="$HOME/Desktop/ECE253/model_training"
CONTAINER_WORKDIR="/workspace"
DATASET_PATH="$LOCAL_ROOT/car_train_split"       # where your dataset lives

# --- [2] Build Docker image ---
echo ">>> Building Docker image: $IMAGE_NAME ..."
docker build -t $IMAGE_NAME .

# --- [3] Run container interactively with GPU + mounts ---
echo ">>> Launching container with mounted folders ..."
docker run --gpus all -it --rm \
    -v "$DATASET_PATH":"$CONTAINER_WORKDIR/datasets" \
    -v "$CODE_PATH":"$CONTAINER_WORKDIR/model_training" \
    -w "$CONTAINER_WORKDIR/yolov5" \
    $IMAGE_NAME bash

# short notes:
# - --gpus all   : enable all available GPUs
# - -it          : interactive shell
# - --rm         : auto remove container when you exit
# - -v ...       : mount host folders to container
# - -w ...       : start in /workspace/yolov5 directory
