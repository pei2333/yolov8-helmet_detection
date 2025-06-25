import os
import sys
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class BaselineTrainer:
    def __init__(self, dataset_type="medium", dataset_size=1500):
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size
        self.project_root = Path(__file__).parent.parent
        
        self.results_dir = self.project_root / "results" / "baseline"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = {
            "model_size": "yolov8n",
            "epochs": 100,
            "batch_size": 16,
            "imgsz": 640,
            "lr0": 0.01,
            "weight_decay": 0.0005,
            "momentum": 0.937,
            "warmup_epochs": 3,
            "patience": 50,
            "save_period": 10,
            "device": "0" if torch.cuda.is_available() else "cpu"
        }
        
        self._set_seed(42)
        
    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def prepare_dataset(self):
        print(f"📊 准备{self.dataset_type}数据集 ({self.dataset_size}张图像)...")
        
        # 检查数据集是否存在
        dataset_dir = self.project_root / "datasets" / "safety_helmet"
        if not dataset_dir.exists():
            print("❌ 数据集不存在，请先运行数据集转换脚本")
            return False
        
        # 根据数据集类型创建子集
        if self.dataset_type in ["subset", "medium"]:
            return self._create_dataset_subset()
        
        return True
    
    def _create_dataset_subset(self):
        """创建数据集子集"""
        from utils.dataset_utils import create_dataset_subset
        
        try:
            subset_dir = self.project_root / "datasets" / f"safety_helmet_{self.dataset_type}"
            
            if not subset_dir.exists():
                print(f"创建{self.dataset_type}数据集子集...")
                create_dataset_subset(
                    source_dir=self.project_root / "datasets" / "safety_helmet",
                    target_dir=subset_dir,
                    train_size=int(self.dataset_size * 0.7),
                    val_size=int(self.dataset_size * 0.2),
                    test_size=int(self.dataset_size * 0.1)
                )
            
            # 更新配置文件路径
            self._update_config_file(subset_dir)
            return True
            
        except Exception as e:
            print(f"❌ 创建数据集子集失败: {e}")
            return False
    
    def _update_config_file(self, dataset_dir):
        """更新数据集配置文件"""
        config_path = self.project_root / "configs" / f"safety_helmet_{self.dataset_type}.yaml"
        
        config_data = {
            'train': str(dataset_dir / "train" / "images"),
            'val': str(dataset_dir / "val" / "images"),
            'test': str(dataset_dir / "test" / "images"),
            'nc': 3,
            'names': ['person', 'helmet', 'no_helmet']
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        
        self.data_config = str(config_path)
    
    def select_model_size(self):
        print("\n🔧 选择YOLOv8模型大小:")
        print("1. YOLOv8n (Nano) - 最快，参数最少")
        print("2. YOLOv8s (Small) - 平衡速度和精度")
        print("3. YOLOv8m (Medium) - 较高精度")
        print("4. YOLOv8l (Large) - 高精度")
        print("5. YOLOv8x (XLarge) - 最高精度")
        
        while True:
            choice = input("请选择模型大小 (1-5，默认1): ").strip() or "1"
            
            if choice == "1":
                self.config["model_size"] = "yolov8n"
                break
            elif choice == "2":
                self.config["model_size"] = "yolov8s"
                break
            elif choice == "3":
                self.config["model_size"] = "yolov8m"
                break
            elif choice == "4":
                self.config["model_size"] = "yolov8l"
                break
            elif choice == "5":
                self.config["model_size"] = "yolov8x"
                break
            else:
                print("❌ 无效选择，请重新输入")
    
    def configure_training(self):
        """配置训练参数"""
        print(f"\n⚙️ 配置训练参数 (当前: {self.config['model_size']})...")
        
        # 根据模型大小调整默认参数
        if self.config["model_size"] in ["yolov8l", "yolov8x"]:
            self.config["batch_size"] = 8  # 大模型减少batch size
            self.config["epochs"] = 150   # 大模型增加训练轮数
        elif self.config["model_size"] == "yolov8n":
            self.config["batch_size"] = 32  # 小模型增加batch size
            self.config["epochs"] = 200    # 小模型需要更多轮数
        
        # 根据数据集大小调整参数
        if self.dataset_type == "subset":
            self.config["epochs"] = 50
            self.config["patience"] = 20
        elif self.dataset_type == "full":
            self.config["epochs"] = 300
            self.config["patience"] = 100
        
        # 显示当前配置
        print(f"📋 训练配置:")
        print(f"   模型: {self.config['model_size']}")
        print(f"   轮数: {self.config['epochs']}")
        print(f"   批次大小: {self.config['batch_size']}")
        print(f"   学习率: {self.config['lr0']}")
        print(f"   图像大小: {self.config['imgsz']}")
        print(f"   设备: {self.config['device']}")
    
    def train(self):
        """执行训练"""
        print("\n🚀 开始基线YOLOv8模型训练...")
        
        # 1. 准备数据集
        if not self.prepare_dataset():
            return False
        
        # 2. 选择模型
        self.select_model_size()
        
        # 3. 配置训练参数
        self.configure_training()
        
        # 4. 初始化模型
        try:
            model = YOLO(f"{self.config['model_size']}.pt")
            print(f"✅ 成功加载 {self.config['model_size']} 预训练模型")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
        
        # 5. 开始训练
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = f"baseline_{self.config['model_size']}_{self.dataset_type}"
            
            results = model.train(
                data=self.data_config,
                epochs=self.config["epochs"],
                batch=self.config["batch_size"],
                imgsz=self.config["imgsz"],
                lr0=self.config["lr0"],
                weight_decay=self.config["weight_decay"],
                momentum=self.config["momentum"],
                warmup_epochs=self.config["warmup_epochs"],
                patience=self.config["patience"],
                save_period=self.config["save_period"],
                project=str(self.results_dir),
                name=f"{project_name}_{timestamp}",
                device=self.config["device"],
                verbose=True,
                plots=True
            )
            
            print(f"✅ 训练完成！结果保存在: {self.results_dir}")
            
            # 6. 生成训练报告
            self._generate_training_report(results, project_name, timestamp)
            
            return True
            
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_training_report(self, results, project_name, timestamp):
        """生成训练报告"""
        print("\n📊 生成训练报告...")
        
        try:
            # 创建报告目录
            report_dir = self.results_dir / f"{project_name}_{timestamp}"
            
            # 获取训练指标
            metrics = results.results_dict if hasattr(results, 'results_dict') else {}
            
            # 生成Markdown报告
            report_content = f"""# 基线YOLOv8训练报告
            
## 训练配置
- **模型**: {self.config['model_size']}
- **数据集**: {self.dataset_type} ({self.dataset_size}张图像)
- **训练轮数**: {self.config['epochs']}
- **批次大小**: {self.config['batch_size']}
- **学习率**: {self.config['lr0']}
- **设备**: {self.config['device']}
- **训练时间**: {timestamp}

## 训练结果
- **最终mAP50**: {metrics.get('metrics/mAP50(B)', 'N/A')}
- **最终mAP50-95**: {metrics.get('metrics/mAP50-95(B)', 'N/A')}
- **最佳Precision**: {metrics.get('metrics/precision(B)', 'N/A')}
- **最佳Recall**: {metrics.get('metrics/recall(B)', 'N/A')}

## 文件结构
```
{report_dir}/
├── weights/
│   ├── best.pt          # 最佳模型权重
│   └── last.pt          # 最后一轮权重
├── results.png          # 训练曲线图
├── confusion_matrix.png # 混淆矩阵
├── val_batch0_pred.jpg  # 验证结果示例
└── train_batch0.jpg     # 训练批次示例
```

## 模型使用
```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('{report_dir}/weights/best.pt')

# 进行预测
results = model('image.jpg')
```
"""
            
            # 保存报告
            report_path = report_dir / "training_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"📋 训练报告已保存: {report_path}")
            
        except Exception as e:
            print(f"⚠️ 生成报告时出错: {e}")
    
    def evaluate_model(self, model_path=None):
        """评估训练好的模型"""
        if model_path is None:
            # 查找最新的最佳模型
            model_path = self._find_latest_model()
        
        if not model_path or not os.path.exists(model_path):
            print("❌ 找不到模型文件")
            return None
        
        try:
            model = YOLO(model_path)
            
            # 在验证集上评估
            results = model.val(data=self.data_config)
            
            print(f"📊 模型评估结果:")
            print(f"   mAP50: {results.box.map50:.4f}")
            print(f"   mAP50-95: {results.box.map:.4f}")
            print(f"   Precision: {results.box.mp:.4f}")
            print(f"   Recall: {results.box.mr:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ 模型评估失败: {e}")
            return None
    
    def _find_latest_model(self):
        """查找最新训练的模型"""
        try:
            # 查找最新的训练结果目录
            result_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
            if not result_dirs:
                return None
            
            latest_dir = max(result_dirs, key=lambda x: x.stat().st_mtime)
            model_path = latest_dir / "weights" / "best.pt"
            
            return str(model_path) if model_path.exists() else None
            
        except Exception:
            return None

# 数据集工具函数
class DatasetUtils:
    """数据集处理工具类"""
    
    @staticmethod
    def create_dataset_subset(source_dir, target_dir, train_size, val_size, test_size):
        """创建数据集子集"""
        import shutil
        
        # 创建目标目录
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                (target_dir / split / subdir).mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        source_images = list((source_dir / "train" / "images").glob("*.jpg")) + \
                        list((source_dir / "train" / "images").glob("*.png"))
        
        # 随机采样
        random.shuffle(source_images)
        
        # 分配样本
        train_samples = source_images[:train_size]
        val_samples = source_images[train_size:train_size + val_size]
        test_samples = source_images[train_size + val_size:train_size + val_size + test_size]
        
        # 复制文件
        for samples, split in [(train_samples, 'train'), (val_samples, 'val'), (test_samples, 'test')]:
            for img_path in samples:
                # 复制图像
                shutil.copy2(img_path, target_dir / split / "images" / img_path.name)
                
                # 复制对应的标签
                label_name = img_path.stem + ".txt"
                label_path = source_dir / "train" / "labels" / label_name
                if label_path.exists():
                    shutil.copy2(label_path, target_dir / split / "labels" / label_name)
        
        print(f"✅ 数据集子集创建完成: {target_dir}")
        print(f"   训练集: {len(train_samples)} 张")
        print(f"   验证集: {len(val_samples)} 张")
        print(f"   测试集: {len(test_samples)} 张")

# 测试函数
if __name__ == "__main__":
    print("测试基线训练器...")
    
    # 创建训练器实例
    trainer = BaselineTrainer(dataset_type="subset", dataset_size=100)
    
    # 测试配置
    trainer.select_model_size()
    trainer.configure_training()
    
    print("基线训练器测试完成! ✅") 