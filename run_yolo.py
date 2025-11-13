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
        epochs=100,
        batch_size=48,  # 3 GPUs × 16 per GPU = 48 total
        device="0,1,2",  # Use 3 GPUs
        project="/pers_vol/yolov5-nrp/runs_obj_det/train",
        name="car_yolov5s_multi_gpu",
        exist_ok=True,
        workers=8,  # Increase workers for multi-GPU
    )
