import sys
import os
from pathlib import Path

# add yolov5 project to Python path
sys.path.insert(0, "/workspace/yolov5")

from train import run  # import yolov5/train.py in run

if __name__ == "__main__":
    # 检查 wandb API key 是否存在
    wandb_key = os.environ.get('WANDB_API_KEY')
    if wandb_key:
        print(f"✓ WANDB_API_KEY detected (length: {len(wandb_key)})")
        print("✓ Wandb logging will be enabled")
    else:
        print("✗ WANDB_API_KEY not found - wandb logging disabled")
    
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
        
        # Wandb integration - integrated wandb
        save_period=1,  # save checkpoint every 1 epoch (will save all .pt files)
    )
