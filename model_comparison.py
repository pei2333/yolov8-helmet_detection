#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LW-YOLOv8 模型对比脚本
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

# 解决OpenMP问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def find_latest_weight(pattern):
    """找到最新的训练权重文件"""
    train_dir = Path('runs/train')
    if not train_dir.exists():
        return None
    
    # 找到所有匹配的目录
    matching_dirs = list(train_dir.glob(f"{pattern}*"))
    if not matching_dirs:
        return None
    
    # 按修改时间排序，取最新的
    latest_dir = max(matching_dirs, key=lambda x: x.stat().st_mtime)
    weight_file = latest_dir / 'weights' / 'best.pt'
    
    return weight_file if weight_file.exists() else None

def get_model_info(model_path, model_name):
    """获取模型信息"""
    try:
        print(f"正在评估模型: {model_name}")
        model = YOLO(model_path)
        
        # 获取模型参数
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        # 获取模型大小
        model_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
        
        # 验证模型
        print(f"  验证中...")
        results = model.val(data='datasets/dataset.yaml', verbose=False)
        
        return {
            'Model': model_name,
            'Parameters': total_params,
            'Trainable_Params': trainable_params,
            'Model_Size_MB': round(model_size, 2),
            'mAP50': round(results.box.map50, 4),
            'mAP50-95': round(results.box.map, 4),
            'Precision': round(results.box.mp, 4),
            'Recall': round(results.box.mr, 4)
        }
    except Exception as e:
        print(f"❌ 错误评估模型 {model_name}: {e}")
        return None

def main():
    print("🔍 开始LW-YOLOv8模型对比分析...")
    print("=" * 60)
    
    # 收集所有训练好的模型权重
    weight_patterns = [
        ('Baseline YOLOv8s', 'baseline-yolov8s'),
        ('CSP-CTFN Only', 'csp-ctfn-only'),
        ('PSC-Head Only', 'psc-head-only'),
        ('SIoU Loss Only', 'siou-only'),
        ('LW-YOLOv8 Full', 'lw-yolov8-full')
    ]
    
    model_weights = {}
    for name, pattern in weight_patterns:
        weight_path = find_latest_weight(pattern)
        if weight_path:
            model_weights[name] = str(weight_path)
            print(f"✅ 找到模型: {name} -> {weight_path}")
        else:
            print(f"⚠️ 未找到模型: {name} (pattern: {pattern}*)")
    
    if not model_weights:
        print("❌ 没有找到任何模型权重文件")
        return
    
    print(f"\n📊 开始评估 {len(model_weights)} 个模型...")
    print("-" * 60)
    
    # 评估所有模型
    results = []
    for name, path in model_weights.items():
        info = get_model_info(path, name)
        if info:
            results.append(info)
            print(f"✅ {name} 评估完成")
        else:
            print(f"❌ {name} 评估失败")
    
    if not results:
        print("❌ 没有成功评估的模型")
        return
    
    # 创建对比表格
    df = pd.DataFrame(results)
    
    # 保存详细结果
    output_dir = Path('runs/compare')
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'model_comparison.csv', index=False)
    
    print(f"\n📊 模型对比结果:")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    
    # mAP50-95 性能排名
    if 'mAP50-95' in df.columns:
        df_sorted = df.sort_values('mAP50-95', ascending=False)
        print(f"\n🏆 mAP50-95 性能排名:")
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            print(f"  {i}. {row['Model']}: {row['mAP50-95']:.4f}")
    
    # mAP50 性能排名
    if 'mAP50' in df.columns:
        df_sorted = df.sort_values('mAP50', ascending=False)
        print(f"\n🎯 mAP50 性能排名:")
        for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
            print(f"  {i}. {row['Model']}: {row['mAP50']:.4f}")
    
    # 参数量对比
    if 'Parameters' in df.columns:
        df_params = df.sort_values('Parameters')
        print(f"\n📏 参数量排名 (从少到多):")
        for i, (_, row) in enumerate(df_params.iterrows(), 1):
            params_m = row['Parameters'] / 1_000_000
            print(f"  {i}. {row['Model']}: {params_m:.2f}M ({row['Parameters']:,})")
    
    # 模型大小对比
    if 'Model_Size_MB' in df.columns:
        df_size = df.sort_values('Model_Size_MB')
        print(f"\n💾 模型大小排名 (从小到大):")
        for i, (_, row) in enumerate(df_size.iterrows(), 1):
            print(f"  {i}. {row['Model']}: {row['Model_Size_MB']:.2f} MB")
    
    # 效果分析 - 与baseline对比
    baseline_data = df[df['Model'] == 'Baseline YOLOv8s']
    if not baseline_data.empty:
        baseline_map50_95 = baseline_data['mAP50-95'].iloc[0]
        baseline_map50 = baseline_data['mAP50'].iloc[0]
        baseline_params = baseline_data['Parameters'].iloc[0]
        
        print(f"\n📈 与Baseline YOLOv8s的对比分析:")
        print(f"   Baseline: mAP50-95={baseline_map50_95:.4f}, mAP50={baseline_map50:.4f}, 参数={baseline_params/1_000_000:.2f}M")
        print(f"   " + "-" * 80)
        
        for _, row in df.iterrows():
            if row['Model'] != 'Baseline YOLOv8s':
                # mAP50-95 改进
                map_improvement = row['mAP50-95'] - baseline_map50_95
                map_percent = (map_improvement / baseline_map50_95) * 100 if baseline_map50_95 > 0 else 0
                
                # 参数量变化
                param_reduction = (baseline_params - row['Parameters']) / baseline_params * 100
                
                status = "✅" if map_improvement >= 0 else "❌"
                print(f"   {status} {row['Model']}:")
                print(f"      mAP50-95: {row['mAP50-95']:.4f} ({map_improvement:+.4f}, {map_percent:+.2f}%)")
                print(f"      参数量: {row['Parameters']/1_000_000:.2f}M ({param_reduction:+.1f}% vs baseline)")
    
    # 推荐最佳模型
    if 'mAP50-95' in df.columns and 'Parameters' in df.columns:
        # 综合评分: 性能权重70% + 效率权重30%
        df['efficiency_score'] = (1 - df['Parameters'] / df['Parameters'].max()) * 0.3
        df['performance_score'] = (df['mAP50-95'] / df['mAP50-95'].max()) * 0.7
        df['combined_score'] = df['efficiency_score'] + df['performance_score']
        
        best_model = df.loc[df['combined_score'].idxmax()]
        best_performance = df.loc[df['mAP50-95'].idxmax()]
        best_efficiency = df.loc[df['Parameters'].idxmin()]
        
        print(f"\n🎖️ 模型推荐:")
        print(f"   🏆 综合最佳: {best_model['Model']} (综合评分: {best_model['combined_score']:.3f})")
        print(f"   🎯 性能最佳: {best_performance['Model']} (mAP50-95: {best_performance['mAP50-95']:.4f})")
        print(f"   ⚡ 效率最佳: {best_efficiency['Model']} (参数: {best_efficiency['Parameters']/1_000_000:.2f}M)")
    
    print(f"\n📁 详细结果已保存到: {output_dir / 'model_comparison.csv'}")
    print(f"🎉 模型对比分析完成！")

if __name__ == '__main__':
    main() 