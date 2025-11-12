import sys
from pathlib import Path

# add yolov5 project to Python path
sys.path.append("/workspace/yolov5")

from train import run  # this time is import yolov5/train.py in run

if __name__ == "__main__":
    run(
        data="/workspace/model_training/Car_model.yaml",
        weights="yolov5s.pt",
        imgsz=640,
        epochs=50,
        batch_size=16,
        device="0",
        project="runs/train",
        name="car_house_yolov5s",
        exist_ok=True,
        workers=2,
    )
