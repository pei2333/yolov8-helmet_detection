#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LW-YOLOv8 训练对比脚本
OnHands安全帽检测数据集

功能：
1. 基线YOLOv8s模型训练
2. CSP-CTFN模块测试
3. PSC检测头测试
4. SIoU损失函数测试
5. 完整LW-YOLOv8模型训练

使用方法：
    python run_all.py [--epochs N] [--batch N] [--device cuda/cpu]

    python run_all.py --epochs 1 --device cuda --batch 16
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def setup_dataset():
    """确保数据集配置正确"""
    dataset_yaml = Path("dataset_OnHands/data.yaml")
    
    if not dataset_yaml.exists():
        print(f"❌ 数据集配置文件不存在: {dataset_yaml}")
        return None
        
    print(f"✅ 找到数据集配置: {dataset_yaml}")
    return str(dataset_yaml)

def train_model(name, config_file, dataset_path, epochs=10, batch=16, imgsz=640, device='cuda'):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"🚀 开始训练: {name}")
    print(f"📁 配置文件: {config_file}")
    print(f"📊 参数: epochs={epochs}, batch={batch}, imgsz={imgsz}")
    print(f"{'='*60}")
    
    try:
        # 清理显存
        if 'cuda' in device:
            import torch
            torch.cuda.empty_cache()
        
        # 加载模型
        if config_file == "yolov8s.pt":
            print("🔄 加载基线YOLOv8s模型...")
            model = YOLO("yolov8s.pt")
        else:
            # 检查配置文件是否存在
            config_path = Path(config_file)
            if not config_path.exists():
                print(f"❌ 配置文件不存在: {config_path}")
                return False
            
            print(f"🔄 加载自定义配置: {config_path.name}")
            model = YOLO(config_file)
        
        # 开始训练
        print("🚀 开始训练...")
        results = model.train(
            data=dataset_path,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            workers=1,           # 减少worker避免内存问题
            cache=False,         # 关闭缓存节省内存
            name=name,
            project='runs/train',
            patience=20,
            save=True,
            plots=True,
            verbose=True,
            amp=True             # 混合精度训练
        )
        
        print(f"✅ {name} 训练完成!")
        
        # 显示最终结果
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            if 'metrics/mAP50(B)' in metrics:
                mAP50 = metrics['metrics/mAP50(B)']
                print(f"📊 最终 mAP@0.5: {mAP50:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ {name} 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='LW-YOLOv8 训练对比')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("🎯 LW-YOLOv8 训练对比脚本")
    print(f"📊 训练参数: epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}")
    print(f"🖥️  设备: {args.device}")
    
    # 检查数据集
    dataset_path = setup_dataset()
    if not dataset_path:
        print("❌ 数据集检查失败!")
        return
    
    # 训练配置列表
    models = [
        ("baseline-yolov8s", "yolov8s.pt"),
        ("csp-ctfn-only", "ultralytics/cfg/models/v8/csp-ctfn-only.yaml"),
        ("psc-head-only", "ultralytics/cfg/models/v8/psc-head-only.yaml"), 
        ("siou-only", "ultralytics/cfg/models/v8/siou-only.yaml"),
        ("lw-yolov8-full", "ultralytics/cfg/models/v8/lw-yolov8-full.yaml")
    ]
    
    successful_trains = []
    failed_trains = []
    
    # 逐个训练模型
    for name, config in models:
        print(f"\n🔄 准备训练 {name}...")
        
        success = train_model(
            name=name,
            config_file=config,
            dataset_path=dataset_path,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device
        )
        
        if success:
            successful_trains.append(name)
        else:
            failed_trains.append(name)
    
    # 最终总结
    print(f"\n{'='*60}")
    print("📊 训练总结")
    print(f"{'='*60}")
    print(f"✅ 成功训练: {len(successful_trains)}")
    for name in successful_trains:
        print(f"   - {name}")
    
    if failed_trains:
        print(f"❌ 失败训练: {len(failed_trains)}")
        for name in failed_trains:
            print(f"   - {name}")
    
    print(f"\n📁 结果保存在: runs/train/")
    print("🔍 使用以下命令查看结果:")
    print("   python view_results.py")

if __name__ == "__main__":
    main() 