# Model Training Repository

This repository is designed for training **YOLO5**, **YOLO7**, and **ResNet** object detection models.

## Prerequisites

### 1. Pull Docker Image

Before training, please pull the Docker image which contains pre-trained YOLO5 and YOLO7 models:

```bash
docker pull timttu/yolov5-nrp:v4
```

**Note:** The Docker image already includes pre-trained YOLO5 and YOLO7 models. If you don't want to train from scratch, you can directly use these pre-trained models.

### 2. Run Preprocessing Script

Before running any training scripts, you must first run the preprocessing script to download and prepare the datasets:

```bash
bash preprocessing.sh
```

This script will automatically download and extract:
- **Training Dataset**: `car_train_split` 
  - Source: [TODO: Add source description]
  - Google Drive: https://drive.google.com/uc?id=1qL-1PV1jvNDRF_yToKUGptVpvmZUfor1

- **Test Dataset**: `car_test`
  - Source: [TODO: Add source description]
  - Google Drive: https://drive.google.com/uc?id=1E5mqA18Dto2l0MjqERjQm3BSAPbcwoeF

The preprocessing script will skip downloading if the datasets already exist on your machine.

## Usage

### Training Models

[Training instructions will be added based on your scripts]

### Using Pre-trained Models

If you prefer to use the pre-trained models from the Docker image instead of training from scratch, you can directly load and use the YOLO5 and YOLO7 models included in the image.

## Project Structure

- `run_yolo5s.py` - YOLO5 training script
- `run_yolo7.py` - YOLO7 training script
- `test_yolo5s.py` - YOLO5 testing script
- `preprocessing.sh` - Dataset preparation script
- `Car_model_*.yaml` - Model configuration files
- `split_dataset/` - Dataset splitting utilities
- `job & pod/` - Kubernetes configuration files
- `YOLO5s_result/` - YOLO5 training results
- `YOLO7_result/` - YOLO7 training results

