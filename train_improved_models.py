#!/usr/bin/env python3
"""
改进的LW-YOLOv8训练脚本
直接运行: python train_improved_models.py
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def train_single_model(model_name, config_or_weight, dataset_path, args):
    """训练单个模型"""
    print(f"\n{'='*60}")
    print(f"🚀 开始训练: {model_name}")
    print(f"📁 配置: {config_or_weight}")
    print(f"{'='*60}")
    
    try:
        # 清理显存
        import torch
        torch.cuda.empty_cache()
        
        # 加载模型
        model = YOLO(config_or_weight)
        
        # 训练参数
        train_args = {
            'data': dataset_path,
            'epochs': args.epochs,
            'batch': args.batch,
            'imgsz': args.imgsz,
            'device': args.device,
            'workers': 8,
            'cache': False,
            'name': model_name,
            'project': 'runs/train',
            'patience': 30,
            'save': True,
            'plots': True,
            'verbose': True,
            'amp': True,
            'exist_ok': args.resume,
        }
        
        # 根据模型类型调整超参数
        if 'improved' in model_name or 'lw-yolov8' in model_name:
            # 轻量化模型使用更保守的学习率
            train_args.update({
                'lr0': 0.005,
                'lrf': 0.01,
                'warmup_epochs': 5,
                'mosaic': 0.8,
                'mixup': 0.0,
                'scale': 0.3,
                'translate': 0.05,
                'close_mosaic': 20,
            })
        
        # 开始训练
        results = model.train(**train_args)
        
        print(f"✅ {model_name} 训练完成!")
        return True
        
    except Exception as e:
        print(f"❌ {model_name} 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='改进的LW-YOLOv8训练脚本')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['all', 'baseline', 'csp-ctfn', 'psc-head', 'siou', 
                               'improved-csp-ctfn', 'improved-psc-head', 'improved-full'],
                       help='要训练的模型')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch', type=int, default=32, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--resume', action='store_true', help='是否恢复训练')
    
    args = parser.parse_args()
    
    print("🎯 改进的LW-YOLOv8训练脚本")
    print(f"📊 参数: epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}")
    print(f"🖥️  设备: {args.device}")
    
    # 数据集路径
    dataset_path = "dataset_OnHands/data.yaml"
    
    # 检查数据集
    if not Path(dataset_path).exists():
        print(f"❌ 数据集配置文件不存在: {dataset_path}")
        return
    
    # 定义模型配置
    models = {
        'improved-full': ('improved-lw-yolov8', 'ultralytics/cfg/models/v8/improved-lw-yolov8.yaml'),
    }
    
    # 选择要训练的模型
    if args.model == 'all':
        # 训练所有改进模型
        selected_models = ['improved-csp-ctfn', 'improved-psc-head', 'improved-full']
    else:
        selected_models = [args.model]
    
    # 训练选定的模型
    results = {}
    for model_key in selected_models:
        if model_key in models:
            name, config = models[model_key]
            success = train_single_model(name, config, dataset_path, args)
            results[name] = success
        else:
            print(f"⚠️  未知的模型: {model_key}")
    
    # 总结
    print(f"\n{'='*60}")
    print("📊 训练总结")
    print(f"{'='*60}")
    
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{status} - {name}")
    
    print(f"\n📁 结果保存在: runs/train/")
    print("💡 提示: 使用 python view_results.py 查看结果对比")

if __name__ == "__main__":
    main()