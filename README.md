# Model Training Repository

This repository is designed for training **YOLO5**, **YOLO7**, and **ResNet** object detection models.

## Prerequisites

### 1. Pull Docker Image

Before training, please pull the Docker image which contains pre-trained YOLO5 and YOLO7 models:

```bash
docker pull timttu/yolov5-nrp:v6
```

**Note:** The Docker image already includes pre-trained YOLO5 and YOLO7 models. If you don't want to train from scratch, you can directly use these pre-trained models.

### 2. Run Preprocessing Script

Before running any training scripts, you must first run the preprocessing script to download and prepare the datasets:

```bash
bash preprocessing.sh
```

This script will automatically download and extract:
- **Training Dataset**: `car_train_split` 
  - Google Drive: https://drive.google.com/uc?id=1qL-1PV1jvNDRF_yToKUGptVpvmZUfor1

- **Test Dataset**: `car_test`
  - Google Drive: https://drive.google.com/uc?id=1E5mqA18Dto2l0MjqERjQm3BSAPbcwoeF

- **Fine-tune Dataset**: `finetune-all`
  - Google Drive: https://drive.google.com/uc?id=1pbHF8CLRT3vCcjA9fyFiDEqciqvOn0qM

The preprocessing.sh script will skip downloading if the datasets already exist on your machine.

### Using Pre-trained Models

After building the Docker image with the models included, you can use the pre-trained models for testing:

**Model Locations in Docker Container:**
- YOLOv5 models: `/workspace/yolov5/yolo5_best.pt` and `/workspace/yolov5/yolo5_fintune_best.pt`
- YOLOv7 model: `/workspace/yolov7/yolo7_best.pt`

**To run testing:**

1. Update the dataset path in `test/test_253_dataset.yaml` to point to your mounted dataset location
2. Run the test script:
   ```bash
   docker run --rm --gpus all \
     -v /path/to/your/datasets:/workspace/datasets \
     -v $(pwd)/test:/workspace/test \
     timttu/yolov5-nrp:v6 \
     python /workspace/test/test_253_local.py
   ```

3. Modify `test/test_253_dataset.yaml` as needed:
   - Change the `path:` field to match your dataset location
   - Adjust `train:` and `val:` paths if needed

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

