#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器端YOLOv8 vs LW-YOLOv8对比训练脚本
适用于4090 GPU，完整数据集训练
"""

import subprocess
import time
import os
from pathlib import Path
import json

def log_print(msg):
    """带时间戳的日志打印"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

def run_training_job(name, cmd, gpu_id=0):
    """运行训练任务"""
    log_print(f"开始训练: {name}")
    log_print(f"命令: {cmd}")
    
    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    start_time = time.time()
    try:
        # 使用nohup在后台运行，输出重定向到日志文件
        log_file = f"logs/{name}.log"
        os.makedirs("logs", exist_ok=True)
        
        full_cmd = f"nohup {cmd} > {log_file} 2>&1 &"
        process = subprocess.Popen(full_cmd, shell=True, env=env)
        
        log_print(f"任务 {name} 已启动，PID: {process.pid}")
        log_print(f"日志文件: {log_file}")
        
        return process, log_file
        
    except Exception as e:
        log_print(f"启动训练失败: {e}")
        return None, None

def monitor_training(processes, log_files):
    """监控训练进度"""
    log_print("开始监控训练进度...")
    
    while True:
        all_finished = True
        
        for i, (process, log_file) in enumerate(zip(processes, log_files)):
            if process and process.poll() is None:
                all_finished = False
                
        if all_finished:
            log_print("所有训练任务已完成！")
            break
            
        # 每10分钟检查一次
        time.sleep(600)
        log_print("训练进行中...")

def main():
    log_print("🚀 服务器端YOLOv8对比训练开始")
    log_print("GPU: RTX 4090")
    log_print("数据集: 完整安全帽检测数据集")
    
    # 检查GPU状态
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        log_print("GPU状态检查:")
        print(result.stdout)
    except:
        log_print("警告: 无法检查GPU状态")
    
    # 训练配置
    epochs = 300
    batch_size = 16
    device = 0
    
    # 1. 训练基线YOLOv8
    baseline_cmd = f"""python -c "
from ultralytics import YOLO
import torch
print(f'使用设备: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}}')
model = YOLO('yolov8s.pt')
results = model.train(
    data='datasets/dataset.yaml',
    epochs={epochs},
    batch={batch_size},
    device={device},
    project='runs/train',
    name='yolov8-baseline-full',
    amp=True,
    save_period=50,
    plots=True,
    verbose=True
)
print('基线YOLOv8训练完成!')
"
"""
    
    # 2. 训练LW-YOLOv8
    lw_cmd = f"""python train_lw_yolov8.py \
--data datasets/dataset.yaml \
--epochs {epochs} \
--batch {batch_size} \
--device {device} \
--name lw-yolov8-full \
--workers 8 \
--save_period 50
"""
    
    # 启动训练任务
    processes = []
    log_files = []
    
    # 启动基线训练
    p1, l1 = run_training_job("baseline-yolov8", baseline_cmd, device)
    if p1:
        processes.append(p1)
        log_files.append(l1)
    
    # 等待5分钟后启动LW-YOLOv8训练（避免初始化冲突）
    log_print("等待5分钟后启动LW-YOLOv8训练...")
    time.sleep(300)
    
    p2, l2 = run_training_job("lw-yolov8", lw_cmd, device)
    if p2:
        processes.append(p2)
        log_files.append(l2)
    
    if not processes:
        log_print("❌ 没有训练任务启动成功")
        return
    
    # 监控训练
    monitor_training(processes, log_files)
    
    # 训练完成后的对比分析
    log_print("开始模型对比分析...")
    
    compare_cmd = f"""python inference_lw_yolov8.py \
--compare \
--lw-weights runs/train/lw-yolov8-full/weights/best.pt \
--yolo-weights runs/train/yolov8-baseline-full/weights/best.pt \
--data datasets/dataset.yaml \
--output runs/compare/full-dataset-comparison
"""
    
    try:
        subprocess.run(compare_cmd, shell=True, check=True)
        log_print("✅ 模型对比分析完成")
    except Exception as e:
        log_print(f"⚠️ 对比分析失败: {e}")
    
    # 结果总结
    log_print("="*60)
    log_print("🎯 训练完成总结")
    log_print("="*60)
    
    # 检查生成的文件
    baseline_weights = Path('runs/train/yolov8-baseline-full/weights/best.pt')
    lw_weights = Path('runs/train/lw-yolov8-full/weights/best.pt')
    
    if baseline_weights.exists():
        size = baseline_weights.stat().st_size / (1024*1024)
        log_print(f"✅ 基线YOLOv8权重: {baseline_weights} ({size:.1f}MB)")
    
    if lw_weights.exists():
        size = lw_weights.stat().st_size / (1024*1024)
        log_print(f"✅ LW-YOLOv8权重: {lw_weights} ({size:.1f}MB)")
    
    compare_dir = Path('runs/compare/full-dataset-comparison')
    if compare_dir.exists():
        log_print(f"✅ 对比报告: {compare_dir}")
    
    log_print("🎉 完整训练和对比流程结束！")
    
    # 生成tmux会话启动脚本
    tmux_script = """#!/bin/bash
# 服务器端训练启动脚本
tmux new-session -d -s yolo_training
tmux send-keys -t yolo_training "cd /root/ultralytics" Enter
tmux send-keys -t yolo_training "python server_train_compare.py" Enter
echo "训练已在tmux会话'yolo_training'中启动"
echo "使用 'tmux attach -t yolo_training' 查看进度"
echo "使用 'tmux list-sessions' 查看所有会话"
"""
    
    with open('start_training.sh', 'w') as f:
        f.write(tmux_script)
    
    os.chmod('start_training.sh', 0o755)
    log_print("已生成tmux启动脚本: start_training.sh")

if __name__ == '__main__':
    main() 