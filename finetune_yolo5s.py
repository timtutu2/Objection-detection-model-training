import sys
import os
from pathlib import Path
import subprocess

print("Upgrade numpy to fix compatibility issue...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "numpy>=1.26.0", "-q"])
print("Numpy upgraded successfully.")

# add yolov5 project to Python path
sys.path.insert(0, "/workspace/yolov5")

from train import run  

if __name__ == "__main__":
    wandb_key = os.environ.get('WANDB_API_KEY')
    if wandb_key:
        print(f"✓ WANDB_API_KEY detected (length: {len(wandb_key)})")
        print("✓ Wandb logging will be enabled")
    else:
        print("✗ WANDB_API_KEY not found - wandb logging disabled")
    
    run(
        data="/pers_vol/yolov5-nrp/Car_model_yolo5.yaml",
        weights="/pers_vol/yolov5-nrp/yolo5_best.pt",
        imgsz=640,
        epochs=100,
        batch_size=48, 
        device="0,1,2",  # Use 3 GPUs
        project="/pers_vol/yolov5-nrp/runs_obj_det/finetune",
        name="car_yolov5s_finetune",
        exist_ok=True,
        workers=8,  # Increase workers for multi-GPU
        save_period=1,  
    )
