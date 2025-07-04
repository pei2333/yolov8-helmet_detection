#!/usr/bin/env python3
"""
对比所有训练模型的结果
"""

import os
from pathlib import Path
import yaml
import pandas as pd

def get_model_results(train_dir):
    """获取训练结果"""
    results = {}
    
    # 读取args.yaml获取配置
    args_file = train_dir / 'args.yaml'
    if args_file.exists():
        with open(args_file, 'r') as f:
            args = yaml.safe_load(f)
            results['epochs'] = args.get('epochs', 0)
            results['batch'] = args.get('batch', 0)
    
    # 读取results.csv获取最佳结果
    results_file = train_dir / 'results.csv'
    if results_file.exists():
        df = pd.read_csv(results_file)
        if len(df) > 0:
            # 获取最佳mAP50的行
            best_idx = df['metrics/mAP50(B)'].idxmax()
            results['best_epoch'] = best_idx + 1
            results['mAP50'] = df.loc[best_idx, 'metrics/mAP50(B)']
            results['mAP50-95'] = df.loc[best_idx, 'metrics/mAP50-95(B)']
            results['precision'] = df.loc[best_idx, 'metrics/precision(B)']
            results['recall'] = df.loc[best_idx, 'metrics/recall(B)']
    
    # 检查模型文件
    best_pt = train_dir / 'weights' / 'best.pt'
    if best_pt.exists():
        results['model_size_mb'] = best_pt.stat().st_size / 1024 / 1024
    
    return results

def main():
    """主函数"""
    runs_dir = Path('runs/train')
    
    if not runs_dir.exists():
        print("❌ 找不到训练结果目录")
        return
    
    # 收集所有训练结果
    all_results = []
    
    for train_dir in sorted(runs_dir.iterdir()):
        if train_dir.is_dir():
            results = get_model_results(train_dir)
            if results:
                results['name'] = train_dir.name
                all_results.append(results)
    
    if not all_results:
        print("❌ 没有找到训练结果")
        return
    
    # 创建DataFrame并排序
    df = pd.DataFrame(all_results)
    
    # 按mAP50排序
    if 'mAP50' in df.columns:
        df = df.sort_values('mAP50', ascending=False)
    
    print("📊 LW-YOLOv8 模型训练结果对比")
    print("=" * 100)
    
    # 显示主要指标
    display_cols = ['name', 'epochs', 'best_epoch', 'mAP50', 'mAP50-95', 'precision', 'recall', 'model_size_mb']
    available_cols = [col for col in display_cols if col in df.columns]
    
    # 格式化显示
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    
    print(df[available_cols].to_string(index=False))
    
    print("\n" + "=" * 100)
    
    # 找出最佳模型
    if 'mAP50' in df.columns:
        best_model = df.iloc[0]
        print(f"\n🏆 最佳模型: {best_model['name']}")
        print(f"   mAP50: {best_model['mAP50']:.4f}")
        if 'mAP50-95' in best_model:
            print(f"   mAP50-95: {best_model['mAP50-95']:.4f}")
        if 'model_size_mb' in best_model:
            print(f"   模型大小: {best_model['model_size_mb']:.2f} MB")
    
    # 分析轻量化效果
    print("\n📈 轻量化分析:")
    baseline_models = df[df['name'].str.contains('baseline', case=False)]
    lw_models = df[df['name'].str.contains('lw-yolov8|csp-ctfn|psc-head|siou', case=False)]
    
    if len(baseline_models) > 0 and len(lw_models) > 0:
        baseline_best = baseline_models.iloc[0]
        print(f"\n基线模型 ({baseline_best['name']}):")
        print(f"  mAP50: {baseline_best.get('mAP50', 0):.4f}")
        print(f"  模型大小: {baseline_best.get('model_size_mb', 0):.2f} MB")
        
        print("\n轻量化模型:")
        for _, model in lw_models.iterrows():
            print(f"\n  {model['name']}:")
            print(f"    mAP50: {model.get('mAP50', 0):.4f}")
            print(f"    模型大小: {model.get('model_size_mb', 0):.2f} MB")
            if baseline_best.get('mAP50', 0) > 0:
                map_drop = (baseline_best['mAP50'] - model.get('mAP50', 0)) / baseline_best['mAP50'] * 100
                print(f"    性能下降: {map_drop:.2f}%")
            if baseline_best.get('model_size_mb', 0) > 0:
                size_reduction = (baseline_best['model_size_mb'] - model.get('model_size_mb', 0)) / baseline_best['model_size_mb'] * 100
                print(f"    模型压缩: {size_reduction:.2f}%")

if __name__ == "__main__":
    main()