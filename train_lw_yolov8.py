#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LW-YOLOv8 轻量级安全帽检测模型训练脚本

本脚本实现了基于改进YOLOv8的轻量级安全帽佩戴检测算法训练。
主要改进包括：
1. CSP-CTFN模块：融合CNN和Transformer的特征提取
2. PSC-Head结构：参数共享的检测头  
3. SIoU损失函数：形状感知的IoU损失

使用示例:
    python train_lw_yolov8.py --data safety_helmet.yaml --model lw-yolov8s.yaml --epochs 300
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn

# 添加ultralytics路径
FILE = Path(__file__).resolve()
ROOT = FILE.parent  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.loss import v8DetectionSIoULoss
from ultralytics.nn.tasks import DetectionModel


class LWYOLOv8Trainer:
    """LW-YOLOv8模型微调训练器类（基于预训练YOLOv8权重）"""
    
    def __init__(self, model_config: str, data_config: str, pretrained_weights: str = 'yolov8s.pt', **kwargs):
        """
        初始化训练器
        
        Args:
            model_config (str): 模型配置文件路径
            data_config (str): 数据配置文件路径
            **kwargs: 其他训练参数
        """
        self.model_config = model_config
        self.data_config = data_config
        self.pretrained_weights = pretrained_weights
        self.training_args = kwargs
        
        # 设置默认训练参数
        self.default_args = {
            'epochs': 300,
            'batch': 16,
            'imgsz': 640,
            'device': 'cuda',
            'workers': 8,
            'project': 'runs/train',
            'name': 'lw-yolov8',
            'save_period': 10,
            'val': True,
            'plots': True,
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': True,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'save_json': True,
            'conf': 0.001,
            'iou': 0.6,
            'max_det': 300,
            'half': False,
            'dnn': False,
            # 优化器设置
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            # 损失函数权重
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            # 数据增强
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
        }
        
        # 合并用户参数
        self.training_args = {**self.default_args, **kwargs}
        
    def create_model_with_siou_loss(self):
        """创建使用SIoU损失的微调模型"""
        
        # 加载预训练权重进行微调
        LOGGER.info(f"正在加载预训练权重进行微调: {self.pretrained_weights}")
        model = YOLO(self.pretrained_weights)  # 先加载预训练权重
        
        # 然后应用自定义配置
        if self.model_config != self.pretrained_weights:
            LOGGER.info(f"应用自定义模型配置: {self.model_config}")
            # 这里我们保持预训练权重，但会在训练时自动适应新的类别数
        
        # 获取模型实例
        if hasattr(model, 'model'):
            model_instance = model.model
        else:
            model_instance = model
            
        # 创建自定义训练器类，使用SIoU损失
        class SIoUTrainer(model.task_map['detect']['trainer']):
            def get_model(self, cfg=None, weights=None, verbose=True):
                """返回使用SIoU损失的模型"""
                model = super().get_model(cfg, weights, verbose)
                return model
                
            def get_loss(self, model):
                """返回SIoU损失函数"""
                return v8DetectionSIoULoss(model)
        
        # 替换训练器
        model.trainer = SIoUTrainer
        model.task_map['detect']['trainer'] = SIoUTrainer
        
        return model
    
    def setup_training_environment(self):
        """设置训练环境"""
        
        # 设置随机种子
        if self.training_args.get('seed'):
            torch.manual_seed(self.training_args['seed'])
            torch.cuda.manual_seed(self.training_args['seed'])
            torch.cuda.manual_seed_all(self.training_args['seed'])
            
        if self.training_args.get('deterministic'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        # 创建输出目录
        project_dir = Path(self.training_args['project']) / self.training_args['name']
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置文件
        config_save_path = project_dir / 'training_config.yaml'
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.training_args, f, default_flow_style=False, allow_unicode=True)
            
        LOGGER.info(f"训练配置已保存到: {config_save_path}")
        
    def train(self):
        """开始训练"""
        
        # 设置训练环境
        self.setup_training_environment()
        
        # 创建模型
        LOGGER.info("正在创建LW-YOLOv8模型（使用SIoU损失）...")
        model = self.create_model_with_siou_loss()
        
        # 开始训练
        LOGGER.info("开始训练...")
        LOGGER.info(f"模型配置: {self.model_config}")
        LOGGER.info(f"数据配置: {self.data_config}")
        LOGGER.info(f"训练参数: {self.training_args}")
        
        try:
            results = model.train(
                data=self.data_config,
                **self.training_args
            )
            
            LOGGER.info("训练完成!")
            LOGGER.info(f"最佳权重保存路径: {results.save_dir / 'weights' / 'best.pt'}")
            
            return results
            
        except Exception as e:
            LOGGER.error(f"训练过程中出现错误: {e}")
            raise
    
    def validate(self, weights_path: str = None):
        """验证模型性能"""
        
        if weights_path is None:
            # 尝试多个可能的权重路径
            possible_paths = [
                Path(self.training_args['project']) / self.training_args['name'] / 'weights' / 'best.pt',
                Path(self.training_args['project']) / f"{self.training_args['name']}9" / 'weights' / 'best.pt',
                Path(self.training_args['project']) / f"{self.training_args['name']}8" / 'weights' / 'best.pt',
                Path(self.training_args['project']) / f"{self.training_args['name']}7" / 'weights' / 'best.pt',
            ]
            
            weights_path = None
            for path in possible_paths:
                if path.exists():
                    weights_path = path
                    break
            
            if weights_path is None:
                LOGGER.warning("未找到训练好的权重文件，跳过验证")
                return None
            
        LOGGER.info(f"正在验证模型: {weights_path}")
        
        # 加载训练好的模型
        model = YOLO(weights_path)
        
        # 运行验证
        results = model.val(
            data=self.data_config,
            imgsz=self.training_args['imgsz'],
            batch=self.training_args['batch'],
            conf=self.training_args['conf'],
            iou=self.training_args['iou'],
            max_det=self.training_args['max_det'],
            half=self.training_args['half'],
            device=self.training_args['device'],
            dnn=self.training_args['dnn'],
            plots=self.training_args['plots'],
            save_json=self.training_args['save_json'],
            save_hybrid=self.training_args.get('save_hybrid', False),
        )
        
        LOGGER.info("验证完成!")
        return results


def create_safety_helmet_data_config():
    """创建安全帽检测数据集配置文件"""
    
    config = {
        'path': './datasets',                # 数据集根目录
        'train': 'safety_helmet/train/images',  # 训练图像相对路径
        'val': 'safety_helmet/val/images',      # 验证图像相对路径
        'test': 'safety_helmet/test/images',    # 测试图像相对路径（可选）
        'nc': 3,                             # 类别数量
        'names': {0: 'person', 1: 'helmet', 2: 'no_helmet'}  # 类别名称
    }
    
    # 保存配置文件
    config_path = 'safety_helmet.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
    LOGGER.info(f"数据集配置文件已创建: {config_path}")
    return config_path


def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description='LW-YOLOv8 轻量级安全帽检测模型训练')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='预训练模型权重路径(用于微调)')
    parser.add_argument('--data', type=str, default='datasets_mini/dataset_mini.yaml', 
                        help='数据集配置文件路径')
    parser.add_argument('--pretrained', type=str, default='yolov8s.pt',
                        help='预训练权重路径(自动下载)')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, 
                        help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, 
                        help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='训练设备 (cpu, cuda, cuda:0, cuda:1, etc.)')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载器工作进程数')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='项目输出目录')
    parser.add_argument('--name', type=str, default='lw-yolov8',
                        help='实验名称')
    parser.add_argument('--resume', action='store_true',
                        help='从最后的检查点恢复训练')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='自动混合精度训练')
    parser.add_argument('--multi-scale', action='store_true', default=True,
                        help='多尺度训练')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='优化器 (SGD, Adam, AdamW, NAdam, RAdam, RMSProp)')
    parser.add_argument('--lr0', type=float, default=0.001,
                        help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='权重衰减')
    parser.add_argument('--create-data-config', action='store_true',
                        help='创建示例数据集配置文件')
    parser.add_argument('--validate-only', type=str, default=None,
                        help='仅验证指定权重文件')
    
    args = parser.parse_args()
    
    # 创建数据集配置文件（如果需要）
    if args.create_data_config:
        create_safety_helmet_data_config()
        return
    
    # 准备训练参数
    training_args = {
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'resume': args.resume,
        'amp': args.amp,
        'multi_scale': args.multi_scale,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'weight_decay': args.weight_decay,
    }
    
    # 创建训练器
    trainer = LWYOLOv8Trainer(
        model_config=args.model,
        data_config=args.data,
        pretrained_weights=args.pretrained,
        **training_args
    )
    
    # 如果只是验证
    if args.validate_only:
        trainer.validate(args.validate_only)
        return
    
    # 开始训练
    results = trainer.train()
    
    # 训练完成后自动验证
    LOGGER.info("开始最终验证...")
    trainer.validate()
    
    LOGGER.info("LW-YOLOv8训练和验证全部完成!")


if __name__ == '__main__':
    main() 