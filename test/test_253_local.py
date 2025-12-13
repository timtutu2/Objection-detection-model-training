#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test dataset in local environment
test dataset structure:
    output_base_dir/
        ├── images/
        │   └── test/
        │       ├── IMG_0001.jpg
        │       ├── IMG_0002.jpg
        │       └── ...
        └── labels/
            └── test/
                ├── IMG_0001.txt
                ├── IMG_0002.txt
                └── ...

"""
import sys
import os
from pathlib import Path
from datetime import datetime
import logging
import numpy as np

# fix numpy version issue
try:

    if not hasattr(np, '_core'):
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
        sys.modules['numpy._core.umath'] = np.core.umath
        sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
except AttributeError:
    pass

def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('YOLOv5_Test_253')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

if __name__ == "__main__":
    # config
    weights_path = "/workspace/datasets/model_training/yolo5_fintune_best.pt"
    data_yaml = "/workspace/datasets/model_training/test/test_253_dataset.yaml"
    output_dir = "/workspace/datasets/model_training/test_dataset_low_light/CLANE/enh_aft/Q30"
    
    # test parameters
    batch_size = 16        
    img_size = 640        
    conf_threshold = 0.25 
    iou_threshold = 0.5  
    device = '0'          # use CPU, if GPU is available, change to '0'
    
    # ========== 检查依赖 ==========
    print("="*80)
    print("check YOLOv5 environment...")
    print("="*80)
    
    yolov5_paths = [
        "/workspace/yolov5",  
        str(Path.home() / "yolov5"),  
        "./yolov5", 
    ]
    
    yolov5_found = False
    for yolov5_path in yolov5_paths:
        if os.path.exists(yolov5_path):
            sys.path.insert(0, yolov5_path)
            print(f"✓ found YOLOv5: {yolov5_path}")
            yolov5_found = True
            break
    
    if not yolov5_found:
        print(" error: YOLOv5 not found")
        print("\nplease install YOLOv5:")
        sys.exit(1)
    
    try:
        from val import run
    except ImportError as e:
        print("\nplease ensure YOLOv5 dependencies are installed:")
        sys.exit(1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_name = f"253_test_{timestamp}"
    
    log_file = os.path.join(output_dir, test_name, "test_results.log")
    logger = setup_logger(log_file)
    
    logger.info("=" * 80)
    logger.info("YOLOv5 model test - 253 dataset")
    logger.info("=" * 80)
    logger.info(f"test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("-" * 80)
    logger.info("test parameters:")
    logger.info(f"  - batch size: {batch_size}")
    logger.info(f"  - image size: {img_size}")
    logger.info(f"  - confidence threshold: {conf_threshold}")
    logger.info(f"  - IOU threshold: {iou_threshold}")
    logger.info(f"  - device: {device}")
    logger.info("=" * 80)

    if not os.path.exists(weights_path):
        logger.error(f"error: weights file not found: {weights_path}")
        sys.exit(1)
    else:
        logger.info(f"found weights file")
    
    if not os.path.exists(data_yaml):
        logger.error(f"error: data configuration file not found: {data_yaml}")
        sys.exit(1)
    else:
        logger.info(f"found data configuration file")
    
    logger.info("-" * 80)
    logger.info("start running test...")
    logger.info("-" * 80)
    
    try:
        results = run(
            data=data_yaml,
            weights=weights_path,
            batch_size=batch_size,
            imgsz=img_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            task='val',
            device=device,
            workers=0,          
            project=output_dir,
            name=test_name,
            exist_ok=True,
            save_txt=True,      
            save_conf=True,     
            save_json=False,    
            verbose=True,
            plots=True,         
        )
        
        logger.info("-" * 80)
        logger.info("test results summary:")
        logger.info("-" * 80)
        
        if results:
            logger.info(f"test completed! results saved.")
            logger.info(f"detailed metrics please check: {output_dir}/{test_name}")
        
        logger.info("=" * 80)
        logger.info("test completed successfully!")
        logger.info("=" * 80)
        logger.info(f"all results files saved in: {output_dir}/{test_name}")
        logger.info("contains the following files:")
        logger.info("  - test_results.log : test log")
        logger.info("  - confusion_matrix.png : confusion matrix")
        logger.info("  - F1_curve.png : F1 curve")
        logger.info("  - P_curve.png : precision curve")
        logger.info("  - R_curve.png : recall curve")
        logger.info("  - PR_curve.png : PR curve")
        logger.info("  - labels.jpg : label statistics")
        logger.info("  - labels_correlogram.jpg : label correlogram")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"error occurred during test: {str(e)}")
        logger.error("=" * 80)
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

