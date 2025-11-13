import sys
from pathlib import Path

# add yolov5 project to Python path
sys.path.insert(0, "/workspace/yolov5")

from train import run  # import yolov5/train.py in run

if __name__ == "__main__":
    run(
        data="/pers_vol/yolov5-nrp/Car_model.yaml",
        weights="yolov5s.pt",
        imgsz=640,
        epochs=50,
        batch_size=16,
        device="0",
        project="/pers_vol/yolov5-nrp/runs_obj_det/train",
        name="car_yolov5s",
        exist_ok=True,
        workers=2,
    )
