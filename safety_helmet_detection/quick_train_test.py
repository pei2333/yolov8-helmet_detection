#!/usr/bin/env python3
import os
import torch
from pathlib import Path
from ultralytics import YOLO
import tempfile

def create_dummy_dataset():
    """创建一个虚拟的小数据集用于快速测试"""
    import yaml
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    
    # 创建目录结构
    train_images = temp_dir / "train" / "images"
    train_labels = temp_dir / "train" / "labels"
    val_images = temp_dir / "val" / "images"
    val_labels = temp_dir / "val" / "labels"
    
    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)
    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)
    
    # 创建虚拟图像和标签
    import numpy as np
    from PIL import Image
    
    for i in range(5):  # 5张训练图像
        # 创建虚拟图像 (640x640)
        img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        img.save(train_images / f"img_{i}.jpg")
        
        # 创建虚拟标签 (YOLO格式)
        with open(train_labels / f"img_{i}.txt", 'w') as f:
            # class_id center_x center_y width height
            f.write("0 0.5 0.5 0.2 0.2\n")  # person
            f.write("1 0.3 0.3 0.1 0.1\n")  # helmet
    
    for i in range(2):  # 2张验证图像
        img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        img.save(val_images / f"val_{i}.jpg")
        
        with open(val_labels / f"val_{i}.txt", 'w') as f:
            f.write("0 0.4 0.4 0.2 0.2\n")
            f.write("2 0.6 0.6 0.1 0.1\n")  # no_helmet
    
    # 创建数据集配置文件
    config_data = {
        'train': str(train_images),
        'val': str(val_images),
        'nc': 3,
        'names': ['person', 'helmet', 'no_helmet']
    }
    
    config_path = temp_dir / "dataset.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    print(f"✅ 创建虚拟数据集: {temp_dir}")
    return str(config_path)

def test_baseline_training():
    """测试基线YOLOv8模型快速训练"""
    print("🔄 开始基线模型微调测试...")
    
    try:
        # 创建虚拟数据集
        dataset_config = create_dummy_dataset()
        
        # 初始化YOLOv8n模型
        model = YOLO('yolov8n.pt')
        print("✅ 模型加载成功")
        
        # 极简训练配置
        train_args = {
            'data': dataset_config,
            'epochs': 2,  # 只训练2个epoch
            'batch': 2,   # 小批次
            'imgsz': 320, # 小图像尺寸
            'patience': 1,
            'save': False,
            'plots': False,
            'val': True,
            'verbose': True,
            'device': 'cpu'  # 强制使用CPU
        }
        
        print("🚀 开始微调训练...")
        results = model.train(**train_args)
        
        print("✅ 训练完成!")
        print(f"最终mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        
        # 测试推理
        print("\n🧪 测试推理...")
        test_img = "https://ultralytics.com/images/bus.jpg"
        results = model(test_img)
        print("✅ 推理测试成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_lightweight_modules():
    """测试轻量化模块的集成"""
    print("\n🔧 测试轻量化模块集成...")
    
    try:
        from modules.fasternet import FasterNetBlock, C2f_Fast
        from modules.attention import A2_Attention
        from modules.losses import FocalerCIOULoss
        
        # 创建一个简单的轻量化网络
        class LightweightNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fasternet = FasterNetBlock(256, 256)
                self.attention = A2_Attention(256)
                self.c2f_fast = C2f_Fast(256, 256, n=2)
                
            def forward(self, x):
                x = self.fasternet(x)
                x = self.attention(x)
                x = self.c2f_fast(x)
                return x
        
        # 测试前向传播
        net = LightweightNet()
        test_input = torch.randn(1, 256, 32, 32)
        output = net(test_input)
        
        print(f"✅ 轻量化网络测试: {test_input.shape} -> {output.shape}")
        
        # 测试损失函数
        loss_fn = FocalerCIOULoss()
        print("✅ 损失函数初始化成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 轻量化模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("🧪 轻量化安全帽检测系统 - 快速微调测试")
    print("="*60)
    
    # 测试1: 轻量化模块
    success1 = test_lightweight_modules()
    
    # 测试2: 基线训练
    success2 = test_baseline_training()
    
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    print(f"轻量化模块测试: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"基线训练测试: {'✅ 通过' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        print("\n🎉 所有测试通过！系统运行正常。")
        print("💡 可以开始正式的模型训练和优化工作。")
    else:
        print("\n⚠️  部分测试失败，请检查环境配置。")
    
    print("="*60)

if __name__ == "__main__":
    main() 