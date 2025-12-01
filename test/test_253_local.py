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

# 兼容性修复：将 numpy._core 映射到 numpy.core (用于加载新版本 numpy 保存的模型)
# 这是因为 numpy 2.0+ 将 numpy.core 重命名为 numpy._core
try:
    # 如果是旧版本 numpy，需要创建这些映射
    if not hasattr(np, '_core'):
        sys.modules['numpy._core'] = np.core
        sys.modules['numpy._core.multiarray'] = np.core.multiarray
        sys.modules['numpy._core.umath'] = np.core.umath
        sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
except AttributeError:
    # 如果是新版本 numpy 2.0+，这些模块已经存在，不需要映射
    pass

def setup_logger(log_file):
    """设置日志记录器"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('YOLOv5_Test_253')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

if __name__ == "__main__":
    # config
    weights_path = "/workspace/datasets/model_training/yolo5_best.pt"
    data_yaml = "/workspace/datasets/model_training/test/test_253_dataset.yaml"
    output_dir = "/workspace/datasets/model_training/test_dataset_foggy/DCP/enh_bef/Q90"
    
    # 测试参数
    batch_size = 16          # 批次大小
    img_size = 640          # 图片大小
    conf_threshold = 0.25   # 置信度阈值 (0.25 是常用值)
    iou_threshold = 0.45    # NMS IOU 阈值
    device = '0'          # 使用 CPU，如果有 GPU 可以改为 '0'
    
    # ========== 检查依赖 ==========
    print("="*80)
    print("检查 YOLOv5 环境...")
    print("="*80)
    
    # 尝试导入 YOLOv5
    yolov5_paths = [
        "/workspace/yolov5",  # Docker 环境
        str(Path.home() / "yolov5"),  # 用户主目录
        "./yolov5",  # 当前目录
    ]
    
    yolov5_found = False
    for yolov5_path in yolov5_paths:
        if os.path.exists(yolov5_path):
            sys.path.insert(0, yolov5_path)
            print(f"✓ 找到 YOLOv5: {yolov5_path}")
            yolov5_found = True
            break
    
    if not yolov5_found:
        print("✗ 错误: 未找到 YOLOv5")
        print("\n请先安装 YOLOv5:")
        print("  git clone https://github.com/ultralytics/yolov5")
        print("  cd yolov5")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    try:
        from val import run
        print("✓ 成功导入 YOLOv5 val 模块")
    except ImportError as e:
        print(f"✗ 错误: 无法导入 YOLOv5 val 模块: {e}")
        print("\n请确保已安装 YOLOv5 的依赖:")
        print("  cd yolov5")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # ========== 开始测试 ==========
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_name = f"253_test_{timestamp}"
    
    log_file = os.path.join(output_dir, test_name, "test_results.log")
    logger = setup_logger(log_file)
    
    logger.info("=" * 80)
    logger.info("YOLOv5 模型测试 - 253 数据集")
    logger.info("=" * 80)
    logger.info(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("-" * 80)
    logger.info("测试参数:")
    logger.info(f"  - 批次大小: {batch_size}")
    logger.info(f"  - 图片大小: {img_size}")
    logger.info(f"  - 置信度阈值: {conf_threshold}")
    logger.info(f"  - IOU 阈值: {iou_threshold}")
    logger.info(f"  - 设备: {device}")
    logger.info("=" * 80)
    
    # 检查文件存在
    if not os.path.exists(weights_path):
        logger.error(f"错误: 权重文件不存在: {weights_path}")
        sys.exit(1)
    else:
        logger.info(f"✓ 找到权重文件")
    
    if not os.path.exists(data_yaml):
        logger.error(f"错误: 数据配置文件不存在: {data_yaml}")
        sys.exit(1)
    else:
        logger.info(f"✓ 找到数据配置文件")
    
    logger.info("-" * 80)
    logger.info("开始运行测试...")
    logger.info("-" * 80)
    
    try:
        # 运行测试
        results = run(
            data=data_yaml,
            weights=weights_path,
            batch_size=batch_size,
            imgsz=img_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            task='val',
            device=device,
            workers=0,          # 设置为 0 避免共享内存问题（Docker 环境）
            project=output_dir,
            name=test_name,
            exist_ok=True,
            save_txt=True,      # 保存预测结果到文本文件
            save_conf=True,     # 保存置信度
            save_json=False,    # 不保存 JSON (需要 COCO 格式)
            verbose=True,
            plots=True,         # 生成可视化图表
        )
        
        logger.info("-" * 80)
        logger.info("测试结果摘要:")
        logger.info("-" * 80)
        
        if results:
            logger.info(f"测试完成! 结果已保存.")
            logger.info(f"详细指标请查看: {output_dir}/{test_name}")
        
        logger.info("=" * 80)
        logger.info("✓ 测试成功完成!")
        logger.info("=" * 80)
        logger.info(f"所有结果文件保存在: {output_dir}/{test_name}")
        logger.info("包含以下文件:")
        logger.info("  - test_results.log : 测试日志")
        logger.info("  - confusion_matrix.png : 混淆矩阵")
        logger.info("  - F1_curve.png : F1 曲线")
        logger.info("  - P_curve.png : 精确率曲线")
        logger.info("  - R_curve.png : 召回率曲线")
        logger.info("  - PR_curve.png : PR 曲线")
        logger.info("  - labels.jpg : 标签统计")
        logger.info("  - labels_correlogram.jpg : 标签相关图")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"测试过程中出错: {str(e)}")
        logger.error("=" * 80)
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

