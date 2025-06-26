#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LW-YOLOv8 一键运行脚本

这个脚本让您可以一键完成所有操作：
1. 训练LW-YOLOv8模型
2. 训练基线YOLOv8模型 
3. 进行模型对比
4. 测试推理

使用方法:
    # 完整流程（训练+对比）
    python run_all.py
    
    # 仅训练LW-YOLOv8
    python run_all.py --lw-only
    
    # 快速测试（少量epochs）
    python run_all.py --quick --epochs 10
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n🚀 {description}")
    print(f"执行命令: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {description} - 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - 失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='LW-YOLOv8 一键运行脚本')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--lw-only', action='store_true', help='仅训练LW-YOLOv8（不训练基线模型）')
    parser.add_argument('--quick', action='store_true', help='快速测试模式（使用更少epochs）')
    parser.add_argument('--no-compare', action='store_true', help='不进行模型对比')
    parser.add_argument('--no-inference', action='store_true', help='不进行推理测试')
    
    args = parser.parse_args()
    
    # 快速模式调整参数
    if args.quick:
        args.epochs = min(args.epochs, 50)
        print(f"🚀 快速测试模式：使用 {args.epochs} epochs")
    
    print("🎯 LW-YOLOv8 一键运行脚本")
    print("=" * 50)
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch}")
    print(f"仅LW-YOLOv8: {args.lw_only}")
    print(f"模型对比: {not args.no_compare and not args.lw_only}")
    print(f"推理测试: {not args.no_inference}")
    print("=" * 50)
    
    success_count = 0
    total_steps = 2 + (0 if args.lw_only else 1) + (0 if args.no_compare or args.lw_only else 1) + (0 if args.no_inference else 1)
    
    # 1. 训练LW-YOLOv8
    cmd_lw = [
        sys.executable, 'train_lw_yolov8.py',
        '--epochs', str(args.epochs),
        '--batch', str(args.batch)
    ]
    
    if run_command(cmd_lw, "步骤 1: 训练LW-YOLOv8模型"):
        success_count += 1
    else:
        print("❌ LW-YOLOv8训练失败，终止流程")
        return
    
    # 2. 训练基线YOLOv8（除非指定仅LW）
    if not args.lw_only:
        cmd_baseline = [
            sys.executable, '-c',
            f"""
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.train(
    data='datasets/dataset.yaml',
    epochs={args.epochs},
    batch={args.batch},
    project='runs/train',
    name='yolov8-baseline',
    amp=True
)
print('✅ 基线YOLOv8训练完成')
"""
        ]
        
        if run_command(cmd_baseline, "步骤 2: 训练基线YOLOv8模型"):
            success_count += 1
        else:
            print("⚠️ 基线YOLOv8训练失败，跳过模型对比")
            args.no_compare = True
    else:
        print("ℹ️ 跳过基线YOLOv8训练")
        success_count += 1
    
    # 3. 模型对比（如果有基线模型）
    if not args.no_compare and not args.lw_only:
        cmd_compare = [
            sys.executable, 'inference_lw_yolov8.py',
            '--compare'
        ]
        
        if run_command(cmd_compare, "步骤 3: 模型对比分析"):
            success_count += 1
        else:
            print("⚠️ 模型对比失败")
    elif args.lw_only:
        print("ℹ️ 跳过模型对比（仅训练LW-YOLOv8模式）")
        success_count += 1
    
    # 4. 推理测试
    if not args.no_inference:
        cmd_inference = [
            sys.executable, 'inference_lw_yolov8.py'
        ]
        
        if run_command(cmd_inference, "步骤 4: 推理测试"):
            success_count += 1
        else:
            print("⚠️ 推理测试失败")
    else:
        print("ℹ️ 跳过推理测试")
        success_count += 1
    
    # 5. 模型评估
    cmd_eval = [
        sys.executable, 'inference_lw_yolov8.py',
        '--evaluate'
    ]
    
    if run_command(cmd_eval, "步骤 5: 模型评估"):
        success_count += 1
    else:
        print("⚠️ 模型评估失败")
    
    # 总结
    print("\n" + "=" * 60)
    print("🎉 一键运行完成！")
    print("=" * 60)
    print(f"成功步骤: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("🎊 所有步骤都成功完成！")
    elif success_count >= total_steps - 1:
        print("✅ 主要步骤成功完成！")
    else:
        print("⚠️ 部分步骤失败，请检查输出日志")
    
    print("\n📁 主要输出文件:")
    
    # 检查生成的文件
    lw_weights = Path('runs/train/lw-yolov8/weights/best.pt')
    if lw_weights.exists():
        print(f"✅ LW-YOLOv8权重: {lw_weights}")
    
    baseline_weights = Path('runs/train/yolov8-baseline/weights/best.pt')
    if baseline_weights.exists():
        print(f"✅ 基线YOLOv8权重: {baseline_weights}")
    
    compare_dir = Path('runs/compare')
    if compare_dir.exists():
        print(f"✅ 对比报告: {compare_dir}")
    
    detect_dir = Path('runs/detect')
    if detect_dir.exists():
        print(f"✅ 推理结果: {detect_dir}")
    
    print("\n🚀 下一步建议:")
    if lw_weights.exists():
        print("1. 查看训练曲线和日志文件")
        print("2. 使用训练好的模型进行更多测试")
        print("3. 考虑模型优化和部署")
    
    if compare_dir.exists():
        print("4. 分析模型对比报告")
        print("5. 根据对比结果调整模型参数")
    
    print("\n📖 更多信息请参考: README_LW_YOLOv8.md")

if __name__ == '__main__':
    main() 