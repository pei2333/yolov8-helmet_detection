# LW-YOLOv8 轻量级安全帽检测算法

基于YOLOv8改进的轻量级安全帽佩戴检测算法，专为无人机视角下的安全帽检测任务优化设计。

## 🚀 项目特点

### 主要改进模块

1. **CSP-CTFN (Cross Stage Partial Convolutional Neural Network Transformer Fusion Net)**
   - 融合CNN局部特征提取和Transformer全局特征捕捉
   - 根据特征图层级自适应调整CNN和Transformer比例
   - 引入卷积门控线性单元(CGLU)增强非线性表达

2. **PSC-Head (Parameter Shared Convolution Head)**
   - 参数共享的检测头结构，减少参数冗余
   - 独立的BatchNorm层避免不同尺度特征规范化偏差
   - 显著降低模型复杂度同时保持检测性能

3. **SIoU Loss (Shape-aware IoU Loss)**
   - 考虑边界框形状和角度信息的IoU损失
   - 更适合处理无人机视角下的非正交目标
   - 提升对长条形或倾斜目标的定位精度

### 性能优势

- ✅ **轻量化设计**：参数量和模型大小显著减少
- ✅ **精度提升**：针对安全帽检测任务优化
- ✅ **边缘友好**：适合部署在资源受限的设备上
- ✅ **无人机视角优化**：专为俯视角度和复杂场景设计

## 📦 环境要求

```bash
Python >= 3.8
PyTorch >= 1.8.0
ultralytics >= 8.0.0
opencv-python >= 4.0.0
numpy >= 1.19.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
thop >= 0.1.1  # 可选，用于FLOPs计算
```

## 🛠️ 安装

1. **克隆项目**
```bash
git clone <your-repo-url>
cd ultralytics
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **安装thop（可选，用于模型复杂度分析）**
```bash
pip install thop
```

## 📊 数据集准备

### 数据集结构
```
datasets/
└── safety_helmet/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

### 数据集配置文件 (`datasets/dataset.yaml`)
```yaml
# 安全帽检测数据集配置
train: datasets/safety_helmet/train/images
val: datasets/safety_helmet/val/images
test: datasets/safety_helmet/test/images

# 类别数量
nc: 3

# 类别名称
names:
  0: person      # 人员
  1: helmet      # 佩戴安全帽
  2: no_helmet   # 未佩戴安全帽
```

## 🏋️ 模型训练（微调）

### 基本训练
```bash
# 使用默认参数训练LW-YOLOv8（所有路径已配置）
python train_lw_yolov8.py

# 自定义参数训练
python train_lw_yolov8.py \
    --epochs 300 \
    --batch 16 \
    --imgsz 640
```

### 高级训练配置
```bash
# 自定义训练参数
python train_lw_yolov8.py \
    --epochs 500 \
    --batch 32 \
    --imgsz 640 \
    --device 0 \
    --workers 8 \
    --optimizer AdamW \
    --lr0 0.001 \
    --weight-decay 0.0005 \
    --name lw-yolov8-experiment \
    --amp \
    --multi-scale
```

### 训练参数说明
- `--epochs`: 训练轮数（默认300）
- `--batch`: 批次大小（默认16）
- `--imgsz`: 输入图像尺寸（默认640）
- `--device`: 训练设备（cpu, 0, 1, 2, 3, auto）
- `--workers`: 数据加载器工作进程数（默认8）
- `--optimizer`: 优化器类型（默认AdamW）
- `--lr0`: 初始学习率（默认0.001）
- `--weight-decay`: 权重衰减（默认0.0005）
- `--amp`: 启用自动混合精度训练
- `--multi-scale`: 启用多尺度训练

> 注意：数据集路径、预训练权重路径等已配置为默认值，可直接运行

## 🔍 模型推理

### 1. 单张图像推理
```bash
# 基本推理（使用默认路径）
python inference_lw_yolov8.py

# 指定特定图像
python inference_lw_yolov8.py --source /path/to/your/image.jpg

# 显示结果
python inference_lw_yolov8.py --source /path/to/your/image.jpg --show
```

### 2. 批量图像推理
```bash
python inference_lw_yolov8.py \
    --weights runs/train/lw-yolov8/weights/best.pt \
    --source images/ \
    --output runs/detect/batch \
    --conf 0.25 \
    --iou 0.45
```

### 3. 视频推理
```bash
python inference_lw_yolov8.py \
    --weights runs/train/lw-yolov8/weights/best.pt \
    --source video.mp4 \
    --output runs/detect/video_output.mp4 \
    --video
```

## 📈 模型评估

### 单模型评估
```bash
python inference_lw_yolov8.py \
    --weights runs/train/lw-yolov8/weights/best.pt \
    --data datasets/dataset.yaml \
    --evaluate
```

### 模型对比评估
```bash
# 训练原始YOLOv8作为对比基线
python -m ultralytics.models.yolo.detect.train \
    data=datasets/dataset.yaml \
    model=yolov8s.pt \
    epochs=300 \
    project=runs/train \
    name=yolov8-baseline

# 对比LW-YOLOv8和原始YOLOv8
python inference_lw_yolov8.py \
    --compare \
    --lw-weights runs/train/lw-yolov8/weights/best.pt \
    --yolo-weights runs/train/yolov8-baseline/weights/best.pt \
    --data datasets/dataset.yaml
```

## 📊 性能分析

### 评估指标
- **检测精度**: mAP50, mAP50-95, Precision, Recall, F1
- **模型复杂度**: 参数量, 模型大小, FLOPs
- **推理速度**: FPS, 平均推理时间

### 对比结果示例
```
=== LW-YOLOv8 vs YOLOv8 对比报告 ===

性能指标对比:
指标             LW-YOLOv8    YOLOv8       改进幅度
--------------------------------------------------
mAP50           0.8520       0.8456       +0.76%
mAP50-95        0.6234       0.6198       +0.58%
precision       0.8745       0.8712       +0.38%
recall          0.8456       0.8423       +0.39%
f1              0.8598       0.8565       +0.39%

模型复杂度对比:
指标                  LW-YOLOv8       YOLOv8          减少幅度
------------------------------------------------------------
参数量                9,458,724       11,166,560      +15.29%
模型大小(MB)          18.43           21.75           +15.27%
FLOPs(G)             24.56           28.80           +14.72%
```

## 🎯 使用场景

### 1. 建筑工地安全监控
- 无人机巡检建筑工地
- 实时检测工人安全帽佩戴情况
- 自动生成安全报告

### 2. 工厂安全管理
- 生产车间安全监督
- 入场人员安全检查
- 违规行为自动预警

### 3. 边缘设备部署
- 移动设备实时检测
- 嵌入式系统集成
- 离线环境应用

## 🛠️ 模型部署

### 模型转换
```bash
# 转换为ONNX格式
python -c "
from ultralytics import YOLO
model = YOLO('runs/train/lw-yolov8/weights/best.pt')
model.export(format='onnx')
"

# 转换为TensorRT格式
python -c "
from ultralytics import YOLO
model = YOLO('runs/train/lw-yolov8/weights/best.pt')
model.export(format='engine')
"
```

### 推理优化
```python
import torch
from ultralytics import YOLO

# 加载模型并优化
model = YOLO('runs/train/lw-yolov8/weights/best.pt')

# 使用半精度推理加速
model.model.half()

# 设置为评估模式
model.model.eval()

# 推理
with torch.no_grad():
    results = model('image.jpg', device='cuda:0', half=True)
```

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch size
   - 降低图像分辨率
   - 启用梯度累积

2. **训练速度慢**
   - 增加workers数量
   - 启用AMP混合精度
   - 使用更快的数据存储

3. **模型精度低**
   - 增加训练epochs
   - 调整学习率
   - 检查数据质量

### 调试命令
```bash
# 检查数据集
python -c "
from ultralytics.data import build_dataset
dataset = build_dataset('datasets/dataset.yaml', mode='train')
print(f'Dataset size: {len(dataset)}')
"

# 验证模型配置
python -c "
from ultralytics import YOLO
model = YOLO('ultralytics/cfg/models/v8/lw-yolov8.yaml')
print(model.model)
"
```
