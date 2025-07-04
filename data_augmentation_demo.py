#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强演示脚本
展示针对安全帽检测的各种数据增强效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import random
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class HelmetAugmentationDemo:
    def __init__(self):
        """初始化增强演示器"""
        self.setup_augmentations()
    
    def setup_augmentations(self):
        """设置各种数据增强变换"""
        # 1. 几何变换
        self.geometric_transform = A.Compose([
            A.Rotate(limit=20, p=1.0),
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.2,
                rotate_limit=0,
                p=1.0
            )
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # 2. 颜色空间增强
        self.color_transform = A.Compose([
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=15,
                p=1.0
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0
            )
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # 3. 噪声和模糊
        self.noise_transform = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.5),
                A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=0.5),
            ], p=1.0),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.7)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # 4. 工业环境模拟
        self.industrial_transform = A.Compose([
            A.OneOf([
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.3),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, p=0.2),
            ], p=0.5),
            A.RandomRain(
                slant_lower=-10, slant_upper=10,
                drop_length=10, drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=1,
                brightness_coefficient=0.7,
                rain_type='drizzle',
                p=0.3
            )
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # 5. 复合增强
        self.complex_transform = A.Compose([
            A.Rotate(limit=15, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.8),
            A.GaussNoise(var_limit=(5, 25), p=0.3),
            A.RandomShadow(p=0.3)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def load_image_and_labels(self, image_path, label_path):
        """加载图片和标签"""
        # 读取图片
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取标签
        bboxes = []
        class_labels = []
        
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
                        
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        
        return image, bboxes, class_labels
    
    def apply_augmentation(self, image, bboxes, class_labels, transform):
        """应用数据增强"""
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            return augmented['image'], augmented['bboxes'], augmented['class_labels']
        except Exception as e:
            print(f"增强失败: {e}")
            return image, bboxes, class_labels
    
    def draw_bboxes(self, image, bboxes, class_labels):
        """在图片上绘制边界框"""
        img_copy = image.copy()
        h, w = image.shape[:2]
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        class_names = ['helmet', 'head', 'person']
        
        for bbox, class_id in zip(bboxes, class_labels):
            if len(bbox) == 4:
                x_center, y_center, width, height = bbox
                
                # 转换为像素坐标
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                # 绘制边界框
                color = colors[class_id % len(colors)]
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                
                # 绘制类别标签
                label = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
                cv2.putText(img_copy, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img_copy
    
    def create_demo_visualization(self, image_path, label_path, output_path='augmentation_demo.png'):
        """创建数据增强演示可视化"""
        # 加载原始图片和标签
        original_image, bboxes, class_labels = self.load_image_and_labels(image_path, label_path)
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('安全帽检测数据增强演示', fontsize=16, fontweight='bold')
        
        # 原始图片
        original_with_boxes = self.draw_bboxes(original_image, bboxes, class_labels)
        axes[0, 0].imshow(original_with_boxes)
        axes[0, 0].set_title('原始图片', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 应用各种增强
        transforms = [
            (self.geometric_transform, '几何变换（旋转+位移）'),
            (self.color_transform, '颜色空间增强'),
            (self.noise_transform, '噪声和模糊'),
            (self.industrial_transform, '工业环境模拟'),
            (self.complex_transform, '复合增强')
        ]
        
        positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for i, ((transform, title), pos) in enumerate(zip(transforms, positions)):
            try:
                aug_image, aug_bboxes, aug_class_labels = self.apply_augmentation(
                    original_image, bboxes, class_labels, transform
                )
                aug_image_with_boxes = self.draw_bboxes(aug_image, aug_bboxes, aug_class_labels)
                
                axes[pos].imshow(aug_image_with_boxes)
                axes[pos].set_title(title, fontsize=12, fontweight='bold')
                axes[pos].axis('off')
                
            except Exception as e:
                print(f"增强 {title} 失败: {e}")
                axes[pos].imshow(original_with_boxes)
                axes[pos].set_title(f'{title} (失败)', fontsize=12, fontweight='bold')
                axes[pos].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"数据增强演示已保存到: {output_path}")
        
        # 输出统计信息
        print("\n=== 数据增强统计信息 ===")
        print(f"原始图片尺寸: {original_image.shape[:2]}")
        print(f"检测目标数量: {len(bboxes)}")
        print(f"类别分布: {dict(zip(['helmet', 'head', 'person'], [class_labels.count(i) for i in range(3)]))}")

def main():
    """主函数"""
    print("=== 安全帽检测数据增强演示 ===")
    
    # 设置路径
    image_path = "datasets/train/images/hard_hat_workers972.png"
    label_path = "datasets/train/labels/hard_hat_workers972.txt"
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在 {image_path}")
        return
    
    if not os.path.exists(label_path):
        print(f"警告: 标签文件不存在 {label_path}，将使用空标签")
        # 创建空标签文件
        with open(label_path, 'w') as f:
            pass
    
    # 创建演示
    demo = HelmetAugmentationDemo()
    demo.create_demo_visualization(image_path, label_path)
    
    print("\n演示完成！")

if __name__ == "__main__":
    main() 