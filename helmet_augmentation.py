#!/usr/bin/env python3
"""
安全帽检测专用数据增强策略
针对工业场景的特殊需求设计
"""

import cv2
import numpy as np
import random
from typing import Tuple, List, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2

class HelmetDetectionAugmentation:
    """安全帽检测专用数据增强类"""
    
    def __init__(self, mode='train', img_size=640):
        self.mode = mode
        self.img_size = img_size
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
    
    def _get_train_transform(self):
        """训练时的数据增强"""
        return A.Compose([
            # 1. 几何变换（适应不同拍摄角度）
            A.ShiftScaleRotate(
                shift_limit=0.15,      # 平移范围增大
                scale_limit=0.3,       # 缩放范围增大，适应远近距离
                rotate_limit=20,       # 旋转角度适中
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.7
            ),
            
            # 2. 透视变换（模拟真实拍摄角度）
            A.Perspective(
                scale=(0.05, 0.1),     # 适度透视变换
                p=0.3
            ),
            
            # 3. 光照和颜色增强（适应不同环境条件）
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.8),
            
            # 4. 色调饱和度调整（适应不同天气）
            A.HueSaturationValue(
                hue_shift_limit=15,    # 色调变化
                sat_shift_limit=30,    # 饱和度变化
                val_shift_limit=25,    # 亮度变化
                p=0.7
            ),
            
            # 5. 噪声和模糊（模拟真实环境干扰）
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.4),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            # 6. 天气模拟（雨雾等恶劣天气）
            A.OneOf([
                A.RandomRain(
                    slant_lower=-10, slant_upper=10,
                    drop_length=10, drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=1, brightness_coefficient=0.7,
                    rain_type=None, p=1.0
                ),
                A.RandomFog(
                    fog_coef_lower=0.1, fog_coef_upper=0.3,
                    alpha_coef=0.08, p=1.0
                ),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    angle_lower=0, angle_upper=1,
                    num_flare_circles_lower=6, num_flare_circles_upper=10,
                    p=1.0
                ),
            ], p=0.2),
            
            # 7. 遮挡模拟（模拟部分遮挡情况）
            A.OneOf([
                A.CoarseDropout(
                    max_holes=8, max_height=32, max_width=32,
                    min_holes=1, min_height=8, min_width=8,
                    fill_value=0, p=1.0
                ),
                A.GridDropout(
                    ratio=0.3, unit_size_min=10, unit_size_max=20,
                    holes_number_x=3, holes_number_y=3,
                    shift_x=0, shift_y=0, random_offset=False, p=1.0
                ),
            ], p=0.3),
            
            # 8. 工业环境特有的增强
            A.OneOf([
                # 模拟金属反射
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                # 模拟灰尘环境
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
                # 模拟机械振动
                A.ElasticTransform(
                    alpha=1, sigma=50, alpha_affine=50,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REFLECT_101, p=1.0
                ),
            ], p=0.2),
            
            # 9. 最终调整
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=100,        # 最小目标面积
            min_visibility=0.3,  # 最小可见度
        ))
    
    def _get_val_transform(self):
        """验证时的数据变换"""
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
        ))
    
    def __call__(self, image, bboxes=None, class_labels=None):
        """执行数据增强"""
        if self.mode == 'train':
            transform = self.train_transform
        else:
            transform = self.val_transform
        
        if bboxes is not None and class_labels is not None:
            transformed = transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            return transformed['image'], transformed['bboxes'], transformed['class_labels']
        else:
            transformed = transform(image=image)
            return transformed['image']

def get_helmet_detection_config():
    """获取安全帽检测的增强配置"""
    return {
        # 基础几何变换
        'degrees': 20.0,        # 旋转角度
        'translate': 0.15,      # 平移范围
        'scale': 0.8,           # 缩放范围
        'shear': 8.0,           # 剪切角度
        'perspective': 0.0005,  # 透视变换
        
        # 颜色空间变换
        'hsv_h': 0.025,         # 色调变化
        'hsv_s': 0.8,           # 饱和度变化
        'hsv_v': 0.6,           # 亮度变化
        
        # 高级增强
        'mosaic': 1.0,          # 马赛克增强（重要！）
        'mixup': 0.15,          # 混合增强
        'copy_paste': 0.3,      # 复制粘贴增强
        'erasing': 0.4,         # 随机擦除
        
        # 翻转设置
        'fliplr': 0.5,          # 水平翻转
        'flipud': 0.0,          # 不使用垂直翻转
        
        # 自动增强
        'auto_augment': 'randaugment',
        'augment': True,
        
        # 训练设置
        'rect': False,          # 不使用矩形训练
        'multi_scale': True,    # 多尺度训练
    }

class WorkSiteAugmentation:
    """工地场景特殊增强"""
    
    @staticmethod
    def add_construction_noise(image, intensity=0.3):
        """添加建筑工地噪声"""
        noise = np.random.normal(0, intensity * 255, image.shape).astype(np.uint8)
        return cv2.add(image, noise)
    
    @staticmethod
    def simulate_dust_effect(image, density=0.2):
        """模拟灰尘效果"""
        dust_mask = np.random.random(image.shape[:2]) < density
        dust_color = np.random.randint(200, 255, 3)
        image[dust_mask] = image[dust_mask] * 0.7 + dust_color * 0.3
        return image.astype(np.uint8)
    
    @staticmethod
    def simulate_helmet_reflection(image, bbox_list, intensity=0.3):
        """模拟安全帽反射光线"""
        for bbox in bbox_list:
            if len(bbox) >= 4:  # 确保有足够的坐标
                x1, y1, x2, y2 = map(int, bbox[:4])
                # 在安全帽区域添加高光
                highlight = np.random.randint(200, 255, 3)
                cv2.circle(image, 
                          ((x1 + x2) // 2, (y1 + y2) // 2), 
                          max(5, (x2 - x1) // 8),
                          highlight.tolist(), -1)
        return image

def create_helmet_training_config(epochs=150, batch_size=16):
    """创建完整的安全帽检测训练配置"""
    helmet_config = get_helmet_detection_config()
    
    training_config = {
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': [640, 640],    # 多尺度训练
        'workers': 8,
        'cache': 'ram',
        'fraction': 1.0,        # 使用全量数据
        'seed': 42,
        
        # 优化器设置
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # 学习率调度
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'cos_lr': True,         # 余弦学习率调度
        
        # 损失函数权重（针对安全帽检测优化）
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # 安全设置
        'patience': 50,
        'save_period': 10,      # 每10轮保存一次
        'val': True,
        'plots': True,
        'verbose': True,
    }
    
    # 合并增强配置
    training_config.update(helmet_config)
    
    return training_config

if __name__ == "__main__":
    # 测试增强效果
    print("安全帽检测数据增强配置:")
    config = get_helmet_detection_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n完整训练配置:")
    full_config = create_helmet_training_config()
    print(f"  总参数数量: {len(full_config)}")
    print(f"  使用数据比例: {full_config['fraction']*100}%")
    print(f"  数据增强强度: 高强度工业场景适配") 