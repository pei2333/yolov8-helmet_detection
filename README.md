# LW-YOLOv8 轻量化目标检测模型

基于YOLOv8的轻量化改进版本，专门用于安全帽检测任务。

## 🎯 项目简介

LW-YOLOv8（Lightweight YOLOv8）通过以下三个核心模块对原始YOLOv8进行改进：

1. **CSP-CTFN模块**：结合CNN和Transformer的特征提取
2. **PSC检测头**：参数共享的卷积检测头  
3. **SIoU损失函数**：形状感知的IoU损失

## 📁 项目结构

```
ultralytics/
├── run_all.py                    # 🚀 主训练脚本
├── view_results.py               # 📊 结果查看工具
├── model_comparison.py           # 📈 模型对比脚本
├── dataset_OnHands/              # 📂 安全帽检测数据集
│   ├── data.yaml
│   ├── images/
│   └── labels/
├── ultralytics/cfg/models/v8/    # ⚙️ 模型配置文件
│   ├── csp-ctfn-only.yaml       # 仅CSP-CTFN模块
│   ├── psc-head-only.yaml       # 仅PSC检测头
│   ├── siou-only.yaml           # 仅SIoU损失
│   └── lw-yolov8-full.yaml      # 完整LW-YOLOv8
└── runs/train/                   # 📊 训练结果保存目录
```

## 🚀 快速开始

### 1. 数据集准备

确保 `dataset_OnHands` 目录包含：
- `data.yaml`: 数据集配置文件
- `images/train/`: 训练图像
- `images/valid/`: 验证图像  
- `labels/train/`: 训练标签
- `labels/valid/`: 验证标签

### 2. 运行训练

```bash
# 默认训练（10轮，batch=16）
python run_all.py

# 自定义参数
python run_all.py --epochs 50 --batch 32 --imgsz 640 --device cuda

# 查看帮助
python run_all.py --help
```

### 3. 查看结果

```bash
# 查看训练结果总结
python view_results.py

# 模型详细对比
python model_comparison.py
```

## 📊 训练配置

脚本会依次训练以下5个模型：

| 模型名称 | 描述 | 配置文件 |
|---------|------|----------|
| baseline-yolov8s | 基线YOLOv8s | yolov8s.pt |
| csp-ctfn-only | 仅CSP-CTFN模块 | csp-ctfn-only.yaml |
| psc-head-only | 仅PSC检测头 | psc-head-only.yaml |
| siou-only | 仅SIoU损失 | siou-only.yaml |
| lw-yolov8-full | 完整LW-YOLOv8 | lw-yolov8-full.yaml |

## 🔧 参数说明

- `--epochs`: 训练轮数（默认10）
- `--batch`: 批次大小（默认16）
- `--imgsz`: 图像尺寸（默认640）
- `--device`: 训练设备（cuda/cpu，默认cuda）

## 📈 结果分析

训练完成后，查看以下位置：

- **权重文件**: `runs/train/{model_name}/weights/best.pt`
- **训练日志**: `runs/train/{model_name}/results.csv`
- **可视化图**: `runs/train/{model_name}/`

使用 `view_results.py` 可以：
- 对比所有模型的mAP指标
- 生成训练曲线对比图
- 查看最佳性能模型

## 🎯 数据集信息

**OnHands 安全帽检测数据集**:
- 训练集：15,887张图像
- 验证集：4,842张图像
- 测试集：2,261张图像
- 类别：2类（head: 无安全帽，helmet: 戴安全帽）
