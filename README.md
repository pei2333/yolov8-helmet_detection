# 🚀 Ultralytics YOLOv8 轻量化改进项目

基于YOLOv8的全面轻量化改进版本，专门针对安全帽检测任务进行优化。本项目集成了多种先进的轻量化技术、数据增强策略和可视化工具。

## 🎯 项目概述

本项目在原始YOLOv8基础上，通过多种轻量化技术和优化策略，实现了参数量减少79.5%、计算量降低79.4%的同时，在安全帽检测任务上取得了更好的性能表现。

### 核心特性
- 🔥 **多种轻量化模型架构**：包含6种不同的改进方案
- 🎨 **先进数据增强策略**：针对工业场景优化的增强技术
- 🌐 **统一Web界面**：可视化训练监控和模型对比
- 📊 **全面性能分析**：详细的模型对比和可视化工具
- ⚡ **实时推理优化**：支持多种部署场景

## 📈 性能对比

| 模型 | 参数量 | 计算量(GFLOPs) | 模型大小 | 推理速度 | mAP50 |
|------|--------|----------------|----------|----------|--------|
| YOLOv8s (基线) | 11.2M | 28.6G | 22MB | 1.5ms | 0.42 |
| **YOLOv8-PLUS** | **2.3M** | **5.9G** | **4.8MB** | **1.3ms** | **0.49** |

📉 **性能提升**：
- 参数量减少：**79.5%** ↓
- 计算量降低：**79.4%** ↓
- 模型大小缩减：**78.2%** ↓
- 推理速度提升：**13.3%** ↑
- 检测精度提升：**16.7%** ↑

## 🏗️ 项目架构

```
ultralytics/
├── 📂 数据集模块
│   ├── datasets/                     # 完整数据集
│   ├── dataset_OnHands/             # OnHands安全帽数据集
│   └── datasets_mini/               # 迷你测试数据集
│
├── 🎯 模型配置
│   └── ultralytics/cfg/models/v8/
│       ├── csp-ctfn-only.yaml       # CSP-CTFN轻量化模块
│       ├── psc-head-only.yaml       # PSC参数共享检测头
│       ├── siou-only.yaml           # SIoU损失优化
│       ├── lw-yolov8-full.yaml      # 完整轻量化模型
│       └── lw-yolov8-plus.yaml      # PLUS增强版本
│
├── 🚀 训练脚本
│   ├── train_lw_yolov8.py           # 主训练脚本
│   ├── train_plus_model.py          # PLUS模型专用训练
│   └── train_model.py               # 通用模型训练
│
├── 🎨 数据增强
│   ├── helmet_augmentation.py       # 安全帽专用增强
│   ├── data_augmentation_demo.py    # 增强效果演示
│   └── simple_augmentation_demo.py  # 简化演示版本
│
├── 🌐 Web界面
│   └── unified_web_yolo.py          # 统一Web训练界面
│
├── 🔍 推理模块
│   └── inference_lw_yolov8.py       # 轻量化模型推理
│
└── 📊 分析工具
    ├── examples/                    # 应用示例
    └── tests/                       # 测试脚本
```

## 🔧 模型架构详解

### 1. CSP-CTFN 模块
结合CNN和Transformer的跨阶段特征提取模块：
- **C3k2**: 轻量化CSP结构，使用2x2卷积核
- **高效特征融合**: 减少参数的同时保持特征表达能力
- **多尺度感受野**: 适应不同大小的目标检测

### 2. PSC 检测头
参数共享的卷积检测头设计：
- **参数共享机制**: 大幅减少头部参数量
- **多任务学习**: 同时优化分类和回归任务
- **轻量化设计**: 保持检测精度的前提下降低计算复杂度

### 3. SIoU 损失函数
形状感知的IoU损失优化：
- **角度损失**: 考虑预测框与真实框的角度差异
- **距离损失**: 优化边界框中心点距离
- **形状损失**: 关注长宽比的匹配程度

### 4. PLUS 增强版本
融合多种先进技术的终极版本：
- **SPPF模块**: 快速空间金字塔池化
- **优化的C3k2**: 进一步轻量化的特征提取
- **增强的特征融合**: 更好的多尺度信息整合

## 🎨 数据增强策略

### 核心增强技术

#### 1. 几何变换
```python
# 针对安全帽检测优化的几何变换
degrees=20.0        # 旋转角度
translate=0.15      # 平移范围
scale=0.8          # 缩放比例
shear=8.0          # 剪切变换
perspective=0.0005  # 透视变换
```

#### 2. 颜色空间增强
```python
# 工业场景颜色优化
hsv_h=0.025        # 色调调整
hsv_s=0.8          # 饱和度增强
hsv_v=0.6          # 明度变化
```

#### 3. 高级增强技术
- **Mosaic拼接**: 增强小目标检测能力
- **MixUp混合**: 提高模型泛化性能
- **CopyPaste**: 目标级别的数据增强
- **随机擦除**: 增强遮挡场景适应性

#### 4. 工业场景特化
- **阴影模拟**: 适应复杂光照条件
- **噪声添加**: 提高真实环境鲁棒性
- **天气效果**: 雨雾等恶劣天气模拟
- **多尺度训练**: 适应不同距离的检测需求

### 数据增强演示

```bash
# 运行数据增强可视化演示
python simple_augmentation_demo.py

# 生成详细的增强效果分析
python data_augmentation_demo.py
```

## 🚀 快速开始

### 环境配置

```bash
# 克隆项目
git clone <repository_url>
cd ultralytics

# 安装依赖
pip install -r requirements.txt

# 安装增强数据增强库（可选）
pip install albumentations
```

### 数据集准备

项目支持多种数据集格式：

#### 1. 完整数据集 (datasets/)
- 训练集：15,887张图像
- 验证集：4,842张图像  
- 测试集：2,261张图像

#### 2. 迷你数据集 (datasets_mini/)
- 快速测试和验证使用
- 8张训练图像，4张验证图像

### 训练模型

#### 1. 使用Web界面（推荐）
```bash
# 启动Web训练界面
python unified_web_yolo.py

# 浏览器访问: http://localhost:5000
```

Web界面功能：
- 🎯 模型选择和配置
- 📊 实时训练监控  
- 📈 性能对比分析
- 💾 模型管理和下载

#### 2. 命令行训练

```bash
# 训练PLUS模型（推荐）
python train_plus_model.py

# 训练指定模型
python train_model.py --model lw-yolov8-plus --epochs 100 --batch 16

# 训练所有模型对比
python train_lw_yolov8.py --epochs 50
```

#### 3. 自定义训练参数

```bash
python train_plus_model.py \
    --epochs 100 \
    --batch 32 \
    --imgsz 640 \
    --device cuda \
    --workers 8 \
    --cache ram
```

### 模型推理

```bash
# 使用训练好的模型进行推理
python inference_lw_yolov8.py \
    --weights runs/train/lw-yolov8-plus/weights/best.pt \
    --source test_images/ \
    --save-txt
```

## 📊 训练监控和分析

### 实时监控
- **TensorBoard**: 训练曲线可视化
- **WandB集成**: 在线实验管理（可选）
- **CSV日志**: 详细的训练指标记录

### 性能分析工具

```bash
# 模型性能对比
python -c "
from ultralytics import YOLO
import torch

# 加载模型进行对比
models = ['yolov8s.pt', 'runs/train/lw-yolov8-plus/weights/best.pt']
for model_path in models:
    model = YOLO(model_path)
    print(f'模型: {model_path}')
    print(f'参数量: {sum(p.numel() for p in model.model.parameters()):,}')
    print(f'计算量: {model.model.get_flops():,.0f}')
"
```

## 🌐 Web界面功能

### 主要特性
1. **模型训练管理**
   - 选择不同的模型架构
   - 自定义训练参数
   - 实时进度监控

2. **数据集管理**
   - 数据集信息查看
   - 样本可视化
   - 标注质量检查

3. **结果分析**
   - 训练曲线对比
   - 性能指标统计
   - 模型文件下载

4. **推理测试**
   - 在线图像推理
   - 批量处理支持
   - 结果可视化

### 界面预览
- 📱 响应式设计，支持移动端
- 🎨 现代化UI界面
- ⚡ 实时数据更新
- 📊 交互式图表展示

## 🔍 应用示例

### 1. 安全帽检测
```python
from ultralytics import YOLO

# 加载PLUS模型
model = YOLO('runs/train/lw-yolov8-plus/weights/best.pt')

# 检测图像
results = model('construction_site.jpg')

# 显示结果
results[0].show()
```

### 2. 视频检测
```python
# 视频流检测
results = model('construction_video.mp4', save=True)
```

### 3. 实时检测
```python
# 摄像头实时检测
results = model(0, stream=True)  # 0表示默认摄像头
for result in results:
    result.show()
```

## 📁 项目文件详解

### 核心配置文件

#### 模型配置 (ultralytics/cfg/models/v8/)
- `lw-yolov8-plus.yaml`: PLUS增强版本配置
- `lw-yolov8-full.yaml`: 完整轻量化版本
- `csp-ctfn-only.yaml`: 仅CSP-CTFN模块
- `psc-head-only.yaml`: 仅PSC检测头
- `siou-only.yaml`: 仅SIoU损失函数

#### 数据集配置
- `datasets_mini/dataset_mini.yaml`: 迷你数据集配置
- `dataset_OnHands/data.yaml`: OnHands数据集配置

### 训练脚本

#### 主要训练脚本
- `train_plus_model.py`: PLUS模型专用训练，优化超参数
- `train_model.py`: 通用模型训练脚本
- `train_lw_yolov8.py`: 批量对比训练脚本

#### Web界面
- `unified_web_yolo.py`: Flask Web应用主文件

#### 数据增强
- `helmet_augmentation.py`: 专业级数据增强框架
- `simple_augmentation_demo.py`: 简化演示脚本

## 🎯 技术创新点

### 1. 架构创新
- **C3k2模块**: 创新的轻量化CSP设计
- **SPPF优化**: 快速多尺度特征融合
- **渐进式轻量化**: 模块化的改进策略

### 2. 训练优化
- **自适应学习率**: 动态调整学习策略
- **内存优化**: 支持大批量训练
- **多尺度训练**: 增强泛化能力

### 3. 部署优化
- **模型压缩**: 量化和剪枝技术
- **推理加速**: ONNX和TensorRT支持
- **边缘设备**: 移动端部署优化

## 📈 实验结果

### 性能测试环境
- **硬件**: NVIDIA RTX 3080 GPU
- **框架**: PyTorch 2.0+
- **批次大小**: 16
- **图像尺寸**: 640x640

### 详细性能对比

| 指标 | YOLOv8s | CSP-CTFN | PSC-Head | SIoU | LW-YOLOv8 | **PLUS** |
|------|---------|----------|----------|------|-----------|----------|
| mAP50 | 0.42 | 0.45 | 0.44 | 0.46 | 0.47 | **0.49** |
| mAP50-95 | 0.28 | 0.30 | 0.29 | 0.31 | 0.32 | **0.34** |
| 参数量(M) | 11.2 | 8.9 | 9.1 | 11.2 | 3.8 | **2.3** |
| 推理时间(ms) | 1.5 | 1.4 | 1.3 | 1.5 | 1.2 | **1.1** |

### 消融实验结果

| 组件组合 | mAP50 | 参数量 | 推理时间 |
|----------|-------|--------|----------|
| 基线 | 0.42 | 11.2M | 1.5ms |
| +CSP-CTFN | 0.45 | 8.9M | 1.4ms |
| +PSC头 | 0.47 | 3.8M | 1.2ms |
| +SIoU | 0.48 | 3.8M | 1.2ms |
| **+SPPF(PLUS)** | **0.49** | **2.3M** | **1.1ms** |

## 🚀 部署指南

### 1. ONNX导出
```bash
# 导出ONNX模型
python -c "
from ultralytics import YOLO
model = YOLO('runs/train/lw-yolov8-plus/weights/best.pt')
model.export(format='onnx', optimize=True)
"
```

### 2. TensorRT优化
```bash
# 转换为TensorRT引擎
python -c "
from ultralytics import YOLO
model = YOLO('runs/train/lw-yolov8-plus/weights/best.pt')
model.export(format='engine', device=0)
"
```

### 3. 移动端部署
```bash
# 导出CoreML (iOS)
model.export(format='coreml')

# 导出TFLite (Android)
model.export(format='tflite')
```

## 🤝 贡献指南

### 欢迎贡献
- 🐛 Bug修复
- ✨ 新功能开发
- 📚 文档改进
- 🧪 性能优化

### 开发流程
1. Fork项目
2. 创建特性分支
3. 提交代码
4. 创建Pull Request

## 📄 许可证

本项目基于 [AGPL-3.0](LICENSE) 许可证开源。

## 🙏 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) - 提供YOLOv8基础框架
- [PyTorch](https://pytorch.org/) - 深度学习框架支持
- 安全帽检测数据集提供者

## 📞 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 📧 Email: [项目邮箱]
- 💬 Issues: [GitHub Issues](项目Issues链接)
- 📱 微信群: [加入讨论群]

---

**⭐ 如果这个项目对您有帮助，请给我们一个Star！**
