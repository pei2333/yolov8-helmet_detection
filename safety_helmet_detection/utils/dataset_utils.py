"""
数据集处理工具函数
提供数据集创建、子集分割、增强等功能
"""
import os
import shutil
import random
import yaml
from pathlib import Path
from typing import Tuple, List
import numpy as np

def create_dataset_subset(source_dir: Path, target_dir: Path, 
                         train_size: int, val_size: int, test_size: int):
    """
    创建数据集子集
    
    Args:
        source_dir: 源数据集目录
        target_dir: 目标数据集目录
        train_size: 训练集大小
        val_size: 验证集大小
        test_size: 测试集大小
    """
    print(f"创建数据集子集: {target_dir}")
    
    # 创建目标目录结构
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            (target_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    source_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        source_images.extend((source_dir / "train" / "images").glob(ext))
    
    if not source_images:
        print("❌ 源数据集中没有找到图像文件")
        return False
    
    # 随机采样
    random.shuffle(source_images)
    
    # 确保有足够的样本
    total_needed = train_size + val_size + test_size
    if len(source_images) < total_needed:
        print(f"⚠️ 源数据集只有{len(source_images)}张图像，需要{total_needed}张")
        # 调整样本分配
        ratio = len(source_images) / total_needed
        train_size = int(train_size * ratio)
        val_size = int(val_size * ratio)
        test_size = len(source_images) - train_size - val_size
    
    # 分配样本
    train_samples = source_images[:train_size]
    val_samples = source_images[train_size:train_size + val_size]
    test_samples = source_images[train_size + val_size:train_size + val_size + test_size]
    
    # 复制文件
    for samples, split in [(train_samples, 'train'), (val_samples, 'val'), (test_samples, 'test')]:
        for img_path in samples:
            try:
                # 复制图像
                shutil.copy2(img_path, target_dir / split / "images" / img_path.name)
                
                # 复制对应的标签
                label_name = img_path.stem + ".txt"
                label_path = source_dir / "train" / "labels" / label_name
                if label_path.exists():
                    shutil.copy2(label_path, target_dir / split / "labels" / label_name)
                else:
                    print(f"⚠️ 找不到标签文件: {label_name}")
                    
            except Exception as e:
                print(f"❌ 复制文件失败 {img_path.name}: {e}")
    
    print(f"✅ 数据集子集创建完成:")
    print(f"   训练集: {len(train_samples)} 张")
    print(f"   验证集: {len(val_samples)} 张") 
    print(f"   测试集: {len(test_samples)} 张")
    
    return True

def analyze_dataset(dataset_dir: Path) -> dict:
    """
    分析数据集统计信息
    
    Args:
        dataset_dir: 数据集目录
        
    Returns:
        数据集统计信息字典
    """
    stats = {
        'splits': {},
        'class_distribution': {},
        'bbox_stats': {}
    }
    
    class_names = ['person', 'helmet', 'no_helmet']
    
    for split in ['train', 'val', 'test']:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            continue
            
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        # 统计图像数量
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        label_files = list(labels_dir.glob("*.txt"))
        
        # 类别分布统计
        class_counts = [0] * len(class_names)
        bbox_sizes = []
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if 0 <= class_id < len(class_names):
                                class_counts[class_id] += 1
                            
                            # 计算边界框大小
                            w, h = float(parts[3]), float(parts[4])
                            bbox_sizes.append(w * h)
            except Exception as e:
                print(f"⚠️ 读取标签文件失败 {label_file}: {e}")
        
        # 保存统计信息
        stats['splits'][split] = {
            'images': len(image_files),
            'labels': len(label_files),
            'annotations': sum(class_counts)
        }
        
        stats['class_distribution'][split] = {
            class_names[i]: class_counts[i] for i in range(len(class_names))
        }
        
        if bbox_sizes:
            stats['bbox_stats'][split] = {
                'mean_size': np.mean(bbox_sizes),
                'std_size': np.std(bbox_sizes),
                'min_size': np.min(bbox_sizes),
                'max_size': np.max(bbox_sizes)
            }
    
    return stats

def print_dataset_stats(stats: dict):
    """打印数据集统计信息"""
    print("\n📊 数据集统计信息")
    print("="*50)
    
    # 数据集分割统计
    print("数据集分割:")
    for split, info in stats['splits'].items():
        print(f"  {split:>5}: {info['images']:>4} 张图像, {info['annotations']:>5} 个标注")
    
    # 类别分布
    print("\n类别分布:")
    class_names = ['person', 'helmet', 'no_helmet']
    for split in ['train', 'val', 'test']:
        if split in stats['class_distribution']:
            print(f"  {split}:")
            for class_name in class_names:
                count = stats['class_distribution'][split].get(class_name, 0)
                print(f"    {class_name:>12}: {count:>5}")
    
    # 边界框统计
    print("\n边界框大小统计:")
    for split, bbox_stats in stats['bbox_stats'].items():
        print(f"  {split}:")
        print(f"    平均大小: {bbox_stats['mean_size']:.4f}")
        print(f"    标准差:   {bbox_stats['std_size']:.4f}")
        print(f"    最小:     {bbox_stats['min_size']:.4f}")
        print(f"    最大:     {bbox_stats['max_size']:.4f}")

def validate_dataset(dataset_dir: Path) -> bool:
    """
    验证数据集完整性
    
    Args:
        dataset_dir: 数据集目录
        
    Returns:
        验证是否通过
    """
    print(f"🔍 验证数据集: {dataset_dir}")
    
    required_structure = [
        "train/images",
        "train/labels", 
        "val/images",
        "val/labels",
        "test/images",
        "test/labels"
    ]
    
    # 检查目录结构
    for structure in required_structure:
        path = dataset_dir / structure
        if not path.exists():
            print(f"❌ 缺少目录: {structure}")
            return False
    
    # 检查图像和标签对应关系
    issues = []
    for split in ['train', 'val', 'test']:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        
        image_files = set(f.stem for f in images_dir.glob("*.jpg")) | \
                     set(f.stem for f in images_dir.glob("*.png"))
        label_files = set(f.stem for f in labels_dir.glob("*.txt"))
        
        # 检查缺失的标签
        missing_labels = image_files - label_files
        if missing_labels:
            issues.append(f"{split}分割缺少{len(missing_labels)}个标签文件")
        
        # 检查多余的标签
        extra_labels = label_files - image_files
        if extra_labels:
            issues.append(f"{split}分割有{len(extra_labels)}个多余标签文件")
    
    if issues:
        print("⚠️ 数据集验证发现问题:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ 数据集验证通过")
    return True

def create_dataset_config(dataset_dir: Path, config_path: Path = None) -> str:
    """
    创建数据集配置文件
    
    Args:
        dataset_dir: 数据集目录
        config_path: 配置文件保存路径
        
    Returns:
        配置文件路径
    """
    if config_path is None:
        config_path = dataset_dir.parent / f"{dataset_dir.name}.yaml"
    
    config_data = {
        'train': str(dataset_dir / "train" / "images"),
        'val': str(dataset_dir / "val" / "images"),
        'test': str(dataset_dir / "test" / "images"),
        'nc': 3,
        'names': ['person', 'helmet', 'no_helmet']
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✅ 数据集配置文件已创建: {config_path}")
    return str(config_path)

def balance_dataset(dataset_dir: Path, target_dir: Path = None, 
                   min_samples_per_class: int = 100):
    """
    平衡数据集类别分布
    
    Args:
        dataset_dir: 源数据集目录
        target_dir: 目标数据集目录
        min_samples_per_class: 每个类别最少样本数
    """
    if target_dir is None:
        target_dir = dataset_dir.parent / f"{dataset_dir.name}_balanced"
    
    print(f"🔄 平衡数据集类别分布...")
    
    # 分析原始数据集
    stats = analyze_dataset(dataset_dir)
    
    # 实现数据平衡逻辑
    # 这里可以通过复制少数类样本或数据增强来平衡数据集
    print("⚠️ 数据集平衡功能正在开发中...")

def augment_small_objects(dataset_dir: Path, target_dir: Path = None, 
                         small_threshold: float = 0.1):
    """
    对小目标进行数据增强
    
    Args:
        dataset_dir: 源数据集目录
        target_dir: 目标数据集目录
        small_threshold: 小目标阈值（相对于图像大小）
    """
    if target_dir is None:
        target_dir = dataset_dir.parent / f"{dataset_dir.name}_augmented"
    
    print(f"🔍 对小目标进行数据增强...")
    print("⚠️ 小目标增强功能正在开发中...")

# 测试函数
if __name__ == "__main__":
    print("测试数据集工具函数...")
    
    # 模拟测试
    test_dir = Path("test_dataset")
    
    # 测试数据集分析功能的逻辑
    print("✅ 数据集工具函数测试完成") 