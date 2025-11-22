import sys
import os
from pathlib import Path

sys.path.insert(0, "/workspace/yolov7")

import subprocess

if __name__ == "__main__":
    # 禁用 wandb 日志记录
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_MODE'] = 'disabled'
    print("✓ Wandb logging is DISABLED")
    
    cmd = [
        "python", "/workspace/yolov7/train.py",
        "--data", "/pers_vol/yolov5-nrp/Car_model_yolo7.yaml",
        "--weights", "yolov7.pt",
        "--img", "640",
        "--epochs", "50",  
        "--batch-size", "48",  
        "--device", "0,1,2",  
        "--project", "/pers_vol/yolov5-nrp/runs_obj_det/yolov7_train",
        "--name", "car_yolov7_multi_gpu",
        "--exist-ok",
        "--workers", "4",
        "--cache-images",  # YOLOv7: 緩存圖像到內存
        "--hyp", "/workspace/yolov7/data/hyp.scratch.p5.yaml",
        "--save-period", "-1",  # 每个epoch只保存last.pt和best.pt
    ]  
    
    print(f"\nRunning command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, env=os.environ.copy())
    
    sys.exit(result.returncode)

