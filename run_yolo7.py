import sys
import os
from pathlib import Path

# 添加 yolov7 项目到 Python 路径
sys.path.insert(0, "/workspace/yolov7")

# YOLOv7 使用 argparse，我们需要通过命令行参数或者直接调用
import subprocess

if __name__ == "__main__":
    # 检查 wandb API key 是否存在
    wandb_key = os.environ.get('WANDB_API_KEY')
    if wandb_key:
        print(f"✓ WANDB_API_KEY detected (length: {len(wandb_key)})")
        print("✓ Wandb logging will be enabled")
    else:
        print("✗ WANDB_API_KEY not found - wandb logging disabled")
    
    # YOLOv7 训练命令
    cmd = [
        "python", "/workspace/yolov7/train.py",
        "--data", "/pers_vol/yolov5-nrp/Car_model.yaml",
        "--weights", "yolov7.pt",
        "--img", "640",
        "--epochs", "100",
        "--batch-size", "48",  # 3 GPUs × 16 per GPU = 48 total
        "--device", "0,1,2",  # 使用 3 个 GPU
        "--project", "/pers_vol/yolov5-nrp/runs_obj_det/train",
        "--name", "car_yolov7_multi_gpu",
        "--exist-ok",
        "--workers", "8",  # 增加 workers 以支持多 GPU
        "--hyp", "data/hyp.scratch.p5.yaml",  # YOLOv7 默认超参数文件
    ]
    
    # 如果有 wandb key，添加 wandb 相关参数
    if wandb_key:
        # YOLOv7 需要在训练脚本中启用 wandb
        os.environ['WANDB_MODE'] = 'run'  # 确保 wandb 启用
    
    print(f"\n执行命令: {' '.join(cmd)}\n")
    
    # 执行训练
    result = subprocess.run(cmd, env=os.environ.copy())
    
    # 返回训练进程的退出码
    sys.exit(result.returncode)

