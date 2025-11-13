import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# 添加 yolov5 项目到 Python 路径
sys.path.insert(0, "/workspace/yolov5")

from val import run  # 导入 yolov5/val.py 中的 run 函数

def setup_logger(log_file):
    """setup logger"""
    # create log directory
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # configure logger
    logger = logging.getLogger('YOLOv5_Test')
    logger.setLevel(logging.INFO)
    
    # file handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

if __name__ == "__main__":
    # ========== configure parameters ==========
    # model weights path
    weights_path = "/pers_vol/yolov5-nrp/runs_obj_det/train/car_yolov5s_multi_gpu/weights/best.pt"
    data_yaml = "/pers_vol/yolov5-nrp/Car_model_test.yaml"
    output_dir = "/pers_vol/yolov5-nrp/runs_obj_det/test"
    
    # test parameters
    batch_size = 48          # batch size
    img_size = 640          # image size
    conf_threshold = 0.001  # confidence threshold
    iou_threshold = 0.6     # NMS IOU threshold
    device = '0,1,2'            # GPU device
    
    # ========== start testing ==========
    # generate test name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_name = f"car_test_{timestamp}"
    
    # setup logger
    log_file = os.path.join(output_dir, test_name, "test_results.log")
    logger = setup_logger(log_file)
    
    # log test information
    logger.info("=" * 80)
    logger.info("YOLOv5 model testing started")
    logger.info("=" * 80)
    logger.info(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("-" * 80)
    logger.info("Test parameters:")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Image size: {img_size}")
    logger.info(f"  - Confidence threshold: {conf_threshold}")
    logger.info(f"  - IOU threshold: {iou_threshold}")
    logger.info(f"  - Device: GPU {device}")
    logger.info("=" * 80)
    
    # check if weight file exists
    if not os.path.exists(weights_path):
        logger.error(f"Error: weight file not found {weights_path}")
        sys.exit(1)
    else:
        logger.info(f"✓ Weight file found")
    
    # check if data configuration file exists
    if not os.path.exists(data_yaml):
        logger.error(f"Error: data configuration file not found {data_yaml}")
        sys.exit(1)
    else:
        logger.info(f"Data configuration file found")
    
    logger.info("-" * 80)
    logger.info("Running test...")
    logger.info("-" * 80)
    
    try:
        # run test
        results = run(
            data=data_yaml,                    # 数据配置文件
            weights=weights_path,              # use best.pt weights
            batch_size=batch_size,             # test batch size
            imgsz=img_size,                    # image size
            conf_thres=conf_threshold,         # confidence threshold
            iou_thres=iou_threshold,           # NMS IOU threshold
            task='val',                        # use validation set for testing
            device=device,                     # GPU device
            workers=8,                         # data loading threads
            project=output_dir,                # project directory
            name=test_name,                    # test name
            exist_ok=True,                     # allow overwrite
            save_txt=True,                     # save results to text file
            save_conf=True,                    # save confidence
            save_json=True,                    # save JSON format results
            verbose=True,                      # detailed output
            plots=True,                        # generate visualization plots
        )

        logger.info("-" * 80)
        logger.info("Test results summary:")
        logger.info("-" * 80)
        
        # YOLOv5 results is a tuple: (mp, mr, map50, map50-95, ...)
        if results:
            logger.info(f"Test completed! Results saved.")
            logger.info(f"Detailed metrics please check: {output_dir}/{test_name}")
        
        logger.info("=" * 80)
        logger.info("✓ Test completed successfully!")
        logger.info("=" * 80)
        logger.info(f"All results files saved in: {output_dir}/{test_name}")
        logger.info("Contains the following files:")
        logger.info("  - test_results.log : test log")
        logger.info("  - predictions/ : prediction results")
        logger.info("  - *.jpg : visualization plots")
        logger.info("  - results.txt : numerical results")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f" Error during test: {str(e)}")
        logger.error("=" * 80)
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

