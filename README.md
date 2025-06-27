# LR-Inst: A Lightweight and Robust Instance Segmentation Network for Apple Detection in Complex Orchard Environments

# Introduction
This project proposes a lightweight and robust instance segmentation network called “LR-Inst“, specifically designed for apple instance segmentation in complex orchard environments.

# Requirements
Python > 3.7
PyTorch >= 1.10
CUDA >= 11.3
MMCV >= 2.0.0
MMEngine >= 0.7.0
MMDetection >= 3.0.0

# Datasets
The dataset is organized in COCO format.
'''
dataset/
├── images/                  
│   ├── train/
│   │   ├── 0001.jpg
│   │   └── ...
│   └── val/
│       ├── 1001.jpg
│       └── ...
├── annotations/            
│   ├── instances_train.json
│   └── instances_val.json
'''

# Training
‘python tools/train.py configs/my_model/LR-Inst.py’

# Testing
'python tools/test.py configs/my_model/LR-Inst.py /path/to/checkpoint file'

