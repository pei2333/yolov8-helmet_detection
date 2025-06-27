#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试自适应查找功能
"""

import sys
from pathlib import Path

def find_latest_weights(base_dir: str, pattern: str) -> str:
    """
    自适应查找最新的权重文件
    
    Args:
        base_dir (str): 基础目录
        pattern (str): 文件夹名称模式，如 'lw-yolo' 或 'yolov8-baseline'
        
    Returns:
        str: 找到的最新权重文件路径
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return ""
    
    # 查找匹配的文件夹
    matching_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and pattern in item.name:
            matching_dirs.append(item)
    
    if not matching_dirs:
        return ""
    
    # 按文件夹名称排序，取最新的（数字最大的）
    latest_dir = sorted(matching_dirs, key=lambda x: x.name)[-1]
    weights_path = latest_dir / "weights" / "best.pt"
    
    if weights_path.exists():
        return str(weights_path)
    else:
        return ""

def main():
    print("🔍 测试自适应查找功能")
    print("=" * 50)
    
    # 测试LW-YOLOv8权重查找
    print("1. 查找LW-YOLOv8权重:")
    lw_weights = find_latest_weights('runs/train', 'lw-yolo')
    if lw_weights:
        print(f"   ✅ 找到: {lw_weights}")
    else:
        print("   ❌ 未找到")
    
    # 测试基线YOLOv8权重查找
    print("\n2. 查找基线YOLOv8权重:")
    baseline_weights = find_latest_weights('runs/train', 'yolov8-baseline')
    if baseline_weights:
        print(f"   ✅ 找到: {baseline_weights}")
    else:
        print("   ❌ 未找到")
    
    # 列出所有训练目录
    print("\n3. 当前训练目录:")
    train_dir = Path('runs/train')
    if train_dir.exists():
        for item in train_dir.iterdir():
            if item.is_dir():
                print(f"   📁 {item.name}")
    else:
        print("   ❌ runs/train 目录不存在")
    
    print("\n" + "=" * 50)
    print("测试完成!")

if __name__ == '__main__':
    main() 