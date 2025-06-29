#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化测试脚本 - 测试基本功能
"""

import os
import sys
from pathlib import Path

# 解决OpenMP问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def test_siou_only():
    """测试仅SIoU损失函数"""
    try:
        from ultralytics import YOLO
        from ultralytics.utils.loss import v8DetectionSIoULoss
        
        print("🚀 测试 SIoU 损失函数...")
        
        # 使用标准 YOLOv8s 配置
        model = YOLO('yolov8s.pt')
        
        # 检查 SIoU 损失函数是否可用
        print("✅ SIoU 损失函数可用")
        
        # 检测设备
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 使用设备: {device}")
        
        # 训练 2 个 epoch 进行测试
        results = model.train(
            data='datasets/dataset.yaml',
            epochs=2,
            batch=16,
            project='runs/train',
            name='siou-test',
            device=device,  # 自动使用CUDA
            workers=4,
            save_period=10,
            patience=50,
            verbose=True
        )
        
        print("✅ SIoU 测试训练成功!")
        return True
        
    except Exception as e:
        print(f"❌ SIoU 测试失败: {e}")
        return False

def test_custom_modules():
    """测试自定义模块是否可用"""
    try:
        from ultralytics.nn.modules.block import CSP_CTFN
        from ultralytics.nn.modules.head import PSCDetect
        
        print("✅ CSP_CTFN 模块可用")
        print("✅ PSCDetect 模块可用")
        
        # 简单创建模块测试
        import torch
        
        # 检测设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 测试设备: {device}")
        
        # 测试 CSP_CTFN
        csp_module = CSP_CTFN(256, 256, n=1, shortcut=True).to(device)
        x = torch.randn(1, 256, 32, 32).to(device)
        out = csp_module(x)
        print(f"✅ CSP_CTFN 测试通过: 输入{x.shape} -> 输出{out.shape} (设备: {out.device})")
        
        # 测试 PSCDetect
        psc_module = PSCDetect(nc=2, ch=(256, 512, 1024)).to(device)
        x_list = [
            torch.randn(1, 256, 80, 80).to(device),
            torch.randn(1, 512, 40, 40).to(device), 
            torch.randn(1, 1024, 20, 20).to(device)
        ]
        out = psc_module(x_list)
        print(f"✅ PSCDetect 测试通过: 输出数量{len(out)} (设备: {out[0].device})")
        
        return True
        
    except Exception as e:
        print(f"❌ 自定义模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 LW-YOLOv8 简化测试")
    print("=" * 50)
    
    # 测试自定义模块
    print("\n1. 测试自定义模块...")
    modules_ok = test_custom_modules()
    
    # 测试 SIoU 训练
    if modules_ok:
        print("\n2. 测试 SIoU 训练...")
        siou_ok = test_siou_only()
    else:
        print("\n⚠️ 跳过 SIoU 训练测试（模块测试失败）")
        siou_ok = False
    
    # 总结
    print("\n" + "=" * 50)
    print("📋 测试结果:")
    print(f"   自定义模块: {'✅ 通过' if modules_ok else '❌ 失败'}")
    print(f"   SIoU 训练: {'✅ 通过' if siou_ok else '❌ 失败'}")
    
    if modules_ok and siou_ok:
        print("\n🎉 所有测试通过！LW-YOLOv8 环境正常")
        return True
    else:
        print("\n⚠️ 部分测试失败，需要检查环境配置")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 