#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化数据增强演示脚本
专注展示针对安全帽检测的图像增强效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import albumentations as A

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class SimpleAugmentationDemo:
    def __init__(self):
        """初始化增强演示器"""
        self.setup_augmentations()
    
    def setup_augmentations(self):
        """设置各种数据增强变换"""
        # 1. 几何变换
        self.geometric_transform = A.Compose([
            A.Rotate(limit=25, p=1.0),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=0,
                p=1.0
            )
        ])
        
        # 2. 颜色空间增强
        self.color_transform = A.Compose([
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            )
        ])
        
        # 3. 噪声和模糊
        self.noise_transform = A.Compose([
            A.GaussNoise(var_limit=(20, 80), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=0.8)
        ])
        
        # 4. 工业环境模拟
        self.industrial_transform = A.Compose([
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1), 
                num_shadows_lower=1, 
                num_shadows_upper=3, 
                p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.7
            )
        ])
        
        # 5. 复合增强
        self.complex_transform = A.Compose([
            A.Rotate(limit=15, p=0.8),
            A.HueSaturationValue(
                hue_shift_limit=15, 
                sat_shift_limit=25, 
                val_shift_limit=15, 
                p=0.9
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.8
            ),
            A.GaussNoise(var_limit=(10, 40), p=0.4),
            A.RandomShadow(p=0.4)
        ])
    
    def load_image(self, image_path):
        """加载图片"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法加载图片: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def load_and_draw_labels(self, image, label_path):
        """加载并绘制标签"""
        img_copy = image.copy()
        h, w = image.shape[:2]
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        class_names = ['helmet', 'head', 'person']
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # 转换为像素坐标
                        x1 = int((x_center - width/2) * w)
                        y1 = int((y_center - height/2) * h)
                        x2 = int((x_center + width/2) * w)
                        y2 = int((y_center + height/2) * h)
                        
                        # 绘制边界框
                        color = colors[class_id % len(colors)]
                        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
                        
                        # 绘制类别标签
                        label = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(img_copy, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
                        cv2.putText(img_copy, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return img_copy
    
    def apply_augmentation(self, image, transform):
        """应用数据增强"""
        try:
            augmented = transform(image=image)
            return augmented['image']
        except Exception as e:
            print(f"增强失败: {e}")
            return image
    
    def create_demo_visualization(self, image_path, label_path, output_path='simple_augmentation_demo.png'):
        """创建数据增强演示可视化"""
        # 加载原始图片
        original_image = self.load_image(image_path)
        
        # 绘制带标签的原始图片
        original_with_labels = self.load_and_draw_labels(original_image, label_path)
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('安全帽检测数据增强演示 - YOLOv8 PLUS', fontsize=16, fontweight='bold')
        
        # 原始图片
        axes[0, 0].imshow(original_with_labels)
        axes[0, 0].set_title('原始图片', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 应用各种增强
        transforms = [
            (self.geometric_transform, '几何变换\n(旋转+缩放+平移)'),
            (self.color_transform, '颜色空间增强\n(HSV+亮度对比度)'),
            (self.noise_transform, '噪声和模糊\n(高斯噪声+模糊)'),
            (self.industrial_transform, '工业环境模拟\n(随机阴影)'),
            (self.complex_transform, '复合增强\n(多种技术组合)')
        ]
        
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for i, ((transform, title), pos) in enumerate(zip(transforms, positions)):
            try:
                aug_image = self.apply_augmentation(original_image, transform)
                
                axes[pos].imshow(aug_image)
                axes[pos].set_title(title, fontsize=12, fontweight='bold')
                axes[pos].axis('off')
                
            except Exception as e:
                print(f"增强 {title} 失败: {e}")
                axes[pos].imshow(original_image)
                axes[pos].set_title(f'{title}\n(失败)', fontsize=12, fontweight='bold')
                axes[pos].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"数据增强演示已保存到: {output_path}")
        
        # 输出统计信息
        print("\n=== 数据增强统计信息 ===")
        print(f"原始图片尺寸: {original_image.shape}")
        print(f"图片路径: {image_path}")
        print(f"标签路径: {label_path}")
        
        # 统计标签信息
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                print(f"检测目标数量: {len(lines)}")
                
                class_counts = [0, 0, 0]
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        if class_id < 3:
                            class_counts[class_id] += 1
                
                print(f"类别分布: helmet={class_counts[0]}, head={class_counts[1]}, person={class_counts[2]}")
        else:
            print("标签文件不存在")

def main():
    """主函数"""
    print("=== 安全帽检测数据增强演示 - YOLOv8 PLUS ===")
    
    # 设置路径
    image_path = "datasets/train/images/hard_hat_workers972.png"
    label_path = "datasets/train/labels/hard_hat_workers972.txt"
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在 {image_path}")
        return
    
    if not os.path.exists(label_path):
        print(f"警告: 标签文件不存在 {label_path}")
    
    # 创建演示
    demo = SimpleAugmentationDemo()
    demo.create_demo_visualization(image_path, label_path)
    
    print("\n=== 增强技术说明 ===")
    print("1. 几何变换: 通过旋转、缩放、平移增加数据多样性")
    print("2. 颜色空间增强: 调整HSV和亮度对比度，适应不同光照条件")
    print("3. 噪声和模糊: 模拟真实环境中的图像质量问题")
    print("4. 工业环境模拟: 添加阴影等工业场景特有的视觉干扰")
    print("5. 复合增强: 组合多种技术，提供最丰富的数据变化")
    print("\n这些增强技术专门针对安全帽检测任务优化，")
    print("能够有效提升YOLOv8 PLUS模型在复杂工业环境下的检测精度！")
    print("\n演示完成！")

if __name__ == "__main__":
    main() 