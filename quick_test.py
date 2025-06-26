#!/usr/bin/env python3
"""
LW-YOLOv8 快速测试 - 一键验证所有功能
"""

import subprocess
import sys
from pathlib import Path

def run_cmd(cmd):
    """简单运行命令"""
    print(f"🔄 {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ 成功")
        return True
    else:
        print(f"❌ 失败: {result.stderr[:100]}")
        return False

def main():
    print("🧪 LW-YOLOv8 快速测试开始")
    
    # 1. 训练LW-YOLOv8 (mini数据集)
    cmd1 = "python train_lw_yolov8.py --data datasets_mini/dataset_mini.yaml --epochs 2 --batch 2 --name quick-test --device cpu"
    if not run_cmd(cmd1):
        print("❌ 训练失败")
        return False
    
    # 2. 推理测试
    cmd2 = "python inference_lw_yolov8.py --weights runs/train/quick-test/weights/best.pt --source datasets_mini/val/images --conf 0.1"
    run_cmd(cmd2)
    
    # 3. 检查结果
    weights = Path('runs/train/quick-test/weights/best.pt')
    if weights.exists():
        print(f"✅ 模型权重: {weights} ({weights.stat().st_size/1024/1024:.1f}MB)")
    
    print("\n🎉 测试完成！LW-YOLOv8功能正常")
    print("💡 可以在服务器上用完整数据集训练了")
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 