#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LW-YOLOv8 训练结果查看工具
"""

import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def find_training_results():
    """查找所有训练结果"""
    runs_dir = Path("runs/train")
    if not runs_dir.exists():
        print("❌ 没有找到训练结果目录")
        return []
    
    results = []
    for folder in runs_dir.iterdir():
        if folder.is_dir():
            # 查找results.csv文件
            csv_file = folder / "results.csv"
            if csv_file.exists():
                results.append({
                    "name": folder.name,
                    "path": folder,
                    "csv": csv_file
                })
    
    return results

def load_training_metrics(csv_file):
    """加载训练指标"""
    try:
        df = pd.read_csv(csv_file)
        # 获取最后一行的指标
        last_row = df.iloc[-1]
        
        metrics = {
            "epochs": len(df),
            "final_train_loss": last_row.get("train/box_loss", 0) + last_row.get("train/cls_loss", 0) + last_row.get("train/dfl_loss", 0),
            "best_map50": df["val/map50"].max() if "val/map50" in df.columns else 0,
            "best_map50_95": df["val/map50-95"].max() if "val/map50-95" in df.columns else 0,
            "final_lr": last_row.get("lr/pg0", 0)
        }
        return metrics, df
    except Exception as e:
        print(f"❌ 读取 {csv_file} 失败: {e}")
        return None, None

def display_summary(results):
    """显示训练结果总结"""
    print("\n🎯 LW-YOLOv8 训练结果总结")
    print("=" * 80)
    
    if not results:
        print("❌ 没有找到任何训练结果")
        return
    
    # 收集所有指标
    summary_data = []
    
    for result in results:
        metrics, df = load_training_metrics(result["csv"])
        if metrics:
            summary_data.append({
                "模型": result["name"],
                "训练轮数": metrics["epochs"],
                "最佳mAP@0.5": f"{metrics['best_map50']:.4f}",
                "最佳mAP@0.5:0.95": f"{metrics['best_map50_95']:.4f}",
                "最终训练损失": f"{metrics['final_train_loss']:.4f}"
            })
    
    if summary_data:
        # 创建DataFrame并显示
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        
        # 找出最佳模型
        best_map50_idx = df_summary["最佳mAP@0.5"].astype(float).idxmax()
        best_map50_95_idx = df_summary["最佳mAP@0.5:0.95"].astype(float).idxmax()
        
        print(f"\n🏆 最佳结果:")
        print(f"   mAP@0.5 最高: {df_summary.iloc[best_map50_idx]['模型']} ({df_summary.iloc[best_map50_idx]['最佳mAP@0.5']})")
        print(f"   mAP@0.5:0.95 最高: {df_summary.iloc[best_map50_95_idx]['模型']} ({df_summary.iloc[best_map50_95_idx]['最佳mAP@0.5:0.95']})")
        
    print(f"\n📁 详细结果位置: runs/train/")
    
    # 检查权重文件
    print(f"\n🎯 可用的模型权重:")
    for result in results:
        weights_dir = result["path"] / "weights"
        if weights_dir.exists():
            best_pt = weights_dir / "best.pt"
            last_pt = weights_dir / "last.pt"
            if best_pt.exists():
                print(f"   ✅ {result['name']}/weights/best.pt")
            if last_pt.exists():
                print(f"   ✅ {result['name']}/weights/last.pt")

def plot_training_curves(results, max_models=5):
    """绘制训练曲线对比"""
    if not results:
        return
        
    plt.figure(figsize=(15, 10))
    
    # 限制显示的模型数量
    display_results = results[:max_models]
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # 绘制mAP@0.5曲线
    plt.subplot(2, 2, 1)
    for i, result in enumerate(display_results):
        metrics, df = load_training_metrics(result["csv"])
        if df is not None and "val/map50" in df.columns:
            plt.plot(df.index, df["val/map50"], color=colors[i % len(colors)], label=result["name"])
    plt.title("mAP@0.5 训练曲线")
    plt.xlabel("Epoch")
    plt.ylabel("mAP@0.5")
    plt.legend()
    plt.grid(True)
    
    # 绘制mAP@0.5:0.95曲线
    plt.subplot(2, 2, 2)
    for i, result in enumerate(display_results):
        metrics, df = load_training_metrics(result["csv"])
        if df is not None and "val/map50-95" in df.columns:
            plt.plot(df.index, df["val/map50-95"], color=colors[i % len(colors)], label=result["name"])
    plt.title("mAP@0.5:0.95 训练曲线")
    plt.xlabel("Epoch")
    plt.ylabel("mAP@0.5:0.95")
    plt.legend()
    plt.grid(True)
    
    # 绘制训练损失
    plt.subplot(2, 2, 3)
    for i, result in enumerate(display_results):
        metrics, df = load_training_metrics(result["csv"])
        if df is not None:
            # 计算总损失
            if all(col in df.columns for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]):
                total_loss = df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"]
                plt.plot(df.index, total_loss, color=colors[i % len(colors)], label=result["name"])
    plt.title("训练总损失")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # 绘制验证损失
    plt.subplot(2, 2, 4)
    for i, result in enumerate(display_results):
        metrics, df = load_training_metrics(result["csv"])
        if df is not None:
            # 计算验证总损失
            if all(col in df.columns for col in ["val/box_loss", "val/cls_loss", "val/dfl_loss"]):
                val_loss = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]
                plt.plot(df.index, val_loss, color=colors[i % len(colors)], label=result["name"])
    plt.title("验证总损失")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig("training_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n📊 训练曲线对比图已保存: training_comparison.png")
    plt.show()

def main():
    """主函数"""
    print("🔍 正在查找训练结果...")
    
    results = find_training_results()
    
    if not results:
        print("❌ 没有找到任何训练结果")
        print("💡 请先运行: python run_all.py")
        return
    
    print(f"✅ 找到 {len(results)} 个训练结果")
    
    # 显示结果总结
    display_summary(results)
    
    # 绘制训练曲线（如果有matplotlib）
    try:
        plot_training_curves(results)
    except ImportError:
        print("\n📈 提示: 安装matplotlib可查看训练曲线")
        print("   pip install matplotlib")
    except Exception as e:
        print(f"\n❌ 绘制曲线失败: {e}")

if __name__ == "__main__":
    main() 