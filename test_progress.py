#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试进度条过滤效果
"""

import os
import torch
from ultralytics import YOLO

# 环境设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("🚀 开始快速测试训练...")

# 快速训练测试
model = YOLO("yolov8s.pt")
results = model.train(
    data="dataset_OnHands/data.yaml",
    epochs=1,
    batch=4,
    imgsz=320,
    device="cuda",
    workers=1,
    cache=False,
    name="progress-test",
    project="runs/train",
    verbose=True
)

print("✅ 测试训练完成!") 