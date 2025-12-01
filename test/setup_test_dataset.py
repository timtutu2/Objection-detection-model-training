#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path

def setup_test_dataset(images_dir, labels_dir, output_base_dir):
    """
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
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_base_dir = Path(output_base_dir)
    
    # 创建目录结构
    test_images_dir = output_base_dir / 'images' / 'test'
    test_labels_dir = output_base_dir / 'labels' / 'test'
    
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"创建目录结构:")
    print(f"  图片目录: {test_images_dir}")
    print(f"  标签目录: {test_labels_dir}")
    print()
    
    # 复制图片文件
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.JPG'))
    print(f"找到 {len(image_files)} 个图片文件")
    
    copied_images = 0
    for img_file in image_files:
        dest = test_images_dir / img_file.name
        shutil.copy2(img_file, dest)
        copied_images += 1
    
    print(f"✓ 已复制 {copied_images} 个图片文件")
    
    # 复制标签文件
    label_files = list(labels_dir.glob('*.txt'))
    print(f"\n找到 {len(label_files)} 个标签文件")
    
    copied_labels = 0
    missing_labels = []
    
    for img_file in image_files:
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            dest = test_labels_dir / label_file.name
            shutil.copy2(label_file, dest)
            copied_labels += 1
        else:
            missing_labels.append(img_file.name)
    
    print(f"✓ 已复制 {copied_labels} 个标签文件")
    
    if missing_labels:
        print(f"\n⚠ 警告: {len(missing_labels)} 个图片没有对应的标签文件:")
        for img in missing_labels[:10]:  # 只显示前10个
            print(f"  - {img}")
        if len(missing_labels) > 10:
            print(f"  ... 还有 {len(missing_labels) - 10} 个")
    
    print(f"\n{'='*60}")
    print(f"测试数据集设置完成!")
    print(f"输出目录: {output_base_dir}")
    print(f"图片数量: {copied_images}")
    print(f"标签数量: {copied_labels}")
    print(f"{'='*60}")
    
    return output_base_dir

if __name__ == "__main__":
    # 配置参数
    images_directory = "/home/tim/Desktop/ECE253/Kuang_Ting_Tu(Tim)/DCP/253_resized_640"
    labels_directory = "/home/tim/Desktop/ECE253/Kuang_Ting_Tu(Tim)/DCP/253_resized_640_labels"
    output_directory = "/home/tim/Desktop/ECE253/model_training/test_dataset_253"
    
    print("="*60)
    print("YOLOv5 测试数据集设置工具")
    print("="*60)
    print(f"图片源目录: {images_directory}")
    print(f"标签源目录: {labels_directory}")
    print(f"输出目录: {output_directory}")
    print("="*60)
    print()
    
    # 执行设置
    setup_test_dataset(images_directory, labels_directory, output_directory)

