#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import xml.etree.ElementTree as ET
from pathlib import Path

def convert_bbox(size, box):
    """
    convert VOC format bounding box to YOLO format
    VOC: (xmin, ymin, xmax, ymax) - absolute coordinates
    YOLO: (x_center, y_center, width, height) - relative coordinates (0-1)
    """
    dw = 1.0 / size[0]  # 1/width
    dh = 1.0 / size[1]  # 1/height
    
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]
    
    x_center = x_center * dw
    y_center = y_center * dh
    width = width * dw
    height = height * dh
    
    return (x_center, y_center, width, height)

def convert_xml_to_yolo(xml_file, output_file, class_names):
    """
    convert single XML file to YOLO format
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    with open(output_file, 'w') as f:
        for obj in root.iter('object'):
            difficult = obj.find('difficult')
            if difficult is not None and int(difficult.text) == 1:
                continue
            
            cls_name = obj.find('name').text
            if cls_name not in class_names:
                print(f"Warning: Unknown class '{cls_name}' found in {xml_file}")
                continue
            
            cls_id = class_names.index(cls_name)
            
            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            ymin = float(xmlbox.find('ymin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymax = float(xmlbox.find('ymax').text)
            
            bbox = convert_bbox((w, h), (xmin, ymin, xmax, ymax))
            
            # write to file: class_id x_center y_center width height
            f.write(f"{cls_id} {' '.join([f'{x:.6f}' for x in bbox])}\n")

def batch_convert(xml_dir, output_dir, class_names):
    """
    batch convert all XML files in the directory
    """
    xml_dir = Path(xml_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    xml_files = list(xml_dir.glob('*.xml'))
    
    if not xml_files:
        print(f"Error: No XML files found in {xml_dir}")
        return
    
    print(f"Found {len(xml_files)} XML files")
    print(f"Starting conversion...\n")
    
    success_count = 0
    fail_count = 0
    
    for xml_file in xml_files:
        try:
            output_file = output_dir / f"{xml_file.stem}.txt"
            
            convert_xml_to_yolo(xml_file, output_file, class_names)
            
            print(f"converted: {xml_file.name} -> {output_file.name}")
            success_count += 1
            
        except Exception as e:
            print(f"conversion failed: {xml_file.name}, error: {str(e)}")
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion completed!")
    print(f"Success: {success_count}")
    print(f"Failure: {fail_count}")
    print(f"Total files converted: {success_count + fail_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    xml_directory = "/home/tim/Desktop/ECE253/Kuang_Ting_Tu(Tim)/DCP/253_resized_640"
    output_directory = "/home/tim/Desktop/ECE253/Kuang_Ting_Tu(Tim)/DCP/253_resized_640_labels"
    class_names = ['Car']
    
    print("="*60)
    print("Pascal VOC XML to YOLO format conversion tool")
    print("="*60)
    print(f"Input directory: {xml_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Classes: {class_names}")
    print("="*60)
    print()
    
    batch_convert(xml_directory, output_directory, class_names)

