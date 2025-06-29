#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LW-YOLOv8 轻量级安全帽检测模型推理和评估脚本

本脚本实现了基于改进YOLOv8的轻量级安全帽佩戴检测算法的推理和评估功能。
主要功能包括：
1. 单张图像推理
2. 批量图像推理
3. 视频推理
4. 模型性能评估
5. 与原始YOLOv8性能对比
6. 模型复杂度分析

使用示例:
    # 单张图像推理
    python inference_lw_yolov8.py --weights runs/train/lw-yolov8/weights/best.pt --source image.jpg
    
    # 批量推理
    python inference_lw_yolov8.py --weights runs/train/lw-yolov8/weights/best.pt --source images/
    
    # 性能评估
    python inference_lw_yolov8.py --weights runs/train/lw-yolov8/weights/best.pt --evaluate --data datasets/dataset.yaml
    
    # 模型对比
    python inference_lw_yolov8.py --compare --lw-weights runs/train/lw-yolov8/weights/best.pt --yolo-weights runs/train/yolov8/weights/best.pt --data datasets/dataset.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# 添加ultralytics路径
FILE = Path(__file__).resolve()
ROOT = FILE.parent  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import LOGGER

try:
    import thop  # 用于计算FLOPs
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    LOGGER.warning("thop未安装，无法计算FLOPs")


class SafetyHelmetInferencer:
    """安全帽检测推理器"""
    
    def __init__(self, weights: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45, device: str = 'cpu'):
        """
        初始化推理器
        
        Args:
            weights (str): 模型权重文件路径
            conf_threshold (float): 置信度阈值
            iou_threshold (float): NMS IoU阈值
            device (str): 推理设备选择 (cpu 或 cuda)
        """
        self.weights = weights
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # 加载模型
        LOGGER.info(f"正在加载模型: {weights}")
        self.model = YOLO(weights)
        
        # 设置设备
        if device != 'cpu':
            self.model.to(device)
        
        # 类别名称（根据您的数据集）
        self.class_names = {
            0: "head",     # 无安全帽的头部
            1: "helmet"    # 戴安全帽的头部
        }
        
        # 类别颜色
        self.class_colors = {
            0: (0, 0, 255),      # 红色 - head (无安全帽，危险)
            1: (0, 255, 0),      # 绿色 - helmet (戴安全帽，安全)
        }
        
    def predict_image(self, image_path: str, save_path: str = None, show: bool = False) -> Dict:
        """
        单张图像推理
        
        Args:
            image_path (str): 图像路径
            save_path (str): 保存路径
            show (bool): 是否显示结果
            
        Returns:
            Dict: 推理结果
        """
        # 推理
        results = self.model(image_path, conf=self.conf_threshold, iou=self.iou_threshold, device=self.device)
        
        # 解析结果
        result = results[0]
        boxes = result.boxes
        
        if boxes is not None:
            # 提取检测信息
            detections = {
                'boxes': boxes.xyxy.cpu().numpy(),
                'confidences': boxes.conf.cpu().numpy(),
                'class_ids': boxes.cls.cpu().numpy().astype(int),
                'count': len(boxes)
            }
            
            # 统计各类别数量
            class_counts = defaultdict(int)
            for class_id in detections['class_ids']:
                class_counts[self.class_names[class_id]] += 1
                
            detections['class_counts'] = dict(class_counts)
            
        else:
            detections = {
                'boxes': np.array([]),
                'confidences': np.array([]),
                'class_ids': np.array([]),
                'count': 0,
                'class_counts': {}
            }
        
        # 绘制结果
        if save_path or show:
            annotated_image = self._draw_detections(image_path, detections)
            
            if save_path:
                cv2.imwrite(save_path, annotated_image)
                LOGGER.info(f"结果已保存到: {save_path}")
                
            if show:
                cv2.imshow('Safety Helmet Detection', annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        return detections
    
    def predict_batch(self, source_dir: str, output_dir: str = "runs/detect") -> List[Dict]:
        """
        批量图像推理
        
        Args:
            source_dir (str): 输入图像目录
            output_dir (str): 输出目录
            
        Returns:
            List[Dict]: 所有图像的推理结果
        """
        source_path = Path(source_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 获取所有图像文件
        if source_path.is_file():
            image_files = [source_path]
        else:
            image_files = [f for f in source_path.rglob('*') 
                          if f.suffix.lower() in image_extensions]
        
        LOGGER.info(f"找到 {len(image_files)} 张图像")
        
        all_results = []
        start_time = time.time()
        
        for i, image_file in enumerate(image_files):
            LOGGER.info(f"处理第 {i+1}/{len(image_files)} 张图像: {image_file.name}")
            
            # 生成输出路径
            save_path = output_path / f"annotated_{image_file.name}"
            
            # 推理
            result = self.predict_image(str(image_file), str(save_path))
            result['image_path'] = str(image_file)
            result['image_name'] = image_file.name
            all_results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / len(image_files)
        
        LOGGER.info(f"批量推理完成!")
        LOGGER.info(f"总时间: {total_time:.2f}s")
        LOGGER.info(f"平均每张: {avg_time:.3f}s")
        LOGGER.info(f"FPS: {1/avg_time:.1f}")
        
        # 保存统计结果
        self._save_batch_statistics(all_results, output_path / "statistics.txt")
        
        return all_results
    
    def predict_video(self, video_path: str, output_path: str = "runs/detect/video_output.mp4") -> Dict:
        """
        视频推理
        
        Args:
            video_path (str): 输入视频路径
            output_path (str): 输出视频路径
            
        Returns:
            Dict: 视频推理统计信息
        """
        cap = cv2.VideoCapture(video_path)
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 创建输出视频
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        LOGGER.info(f"处理视频: {video_path}")
        LOGGER.info(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
        
        frame_results = []
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 30 == 0:  # 每30帧打印一次进度
                LOGGER.info(f"处理进度: {frame_count}/{total_frames}")
            
            # 推理当前帧
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, device=self.device)
            result = results[0]
            
            # 解析检测结果
            frame_detection = self._parse_detection_result(result)
            frame_results.append(frame_detection)
            
            # 绘制检测结果
            annotated_frame = self._draw_detections_on_frame(frame, frame_detection)
            
            # 写入输出视频
            out.write(annotated_frame)
        
        cap.release()
        out.release()
        
        end_time = time.time()
        processing_time = end_time - start_time
        processing_fps = frame_count / processing_time
        
        # 统计信息
        stats = self._calculate_video_statistics(frame_results, processing_time, processing_fps)
        
        LOGGER.info(f"视频处理完成!")
        LOGGER.info(f"输出保存到: {output_path}")
        LOGGER.info(f"处理时间: {processing_time:.2f}s")
        LOGGER.info(f"处理FPS: {processing_fps:.1f}")
        
        return stats
    
    def _draw_detections(self, image_path: str, detections: Dict) -> np.ndarray:
        """绘制检测结果"""
        image = cv2.imread(image_path)
        
        if detections['count'] > 0:
            for i in range(detections['count']):
                box = detections['boxes'][i]
                conf = detections['confidences'][i]
                class_id = detections['class_ids'][i]
                
                # 绘制边界框
                x1, y1, x2, y2 = box.astype(int)
                color = self.class_colors[class_id]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"{self.class_names[class_id]}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def _draw_detections_on_frame(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """在视频帧上绘制检测结果"""
        if detections['count'] > 0:
            for i in range(detections['count']):
                box = detections['boxes'][i]
                conf = detections['confidences'][i]
                class_id = detections['class_ids'][i]
                
                # 绘制边界框
                x1, y1, x2, y2 = box.astype(int)
                color = self.class_colors[class_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"{self.class_names[class_id]}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _parse_detection_result(self, result) -> Dict:
        """解析检测结果"""
        boxes = result.boxes
        
        if boxes is not None:
            detections = {
                'boxes': boxes.xyxy.cpu().numpy(),
                'confidences': boxes.conf.cpu().numpy(),
                'class_ids': boxes.cls.cpu().numpy().astype(int),
                'count': len(boxes)
            }
        else:
            detections = {
                'boxes': np.array([]),
                'confidences': np.array([]),
                'class_ids': np.array([]),
                'count': 0
            }
        
        return detections
    
    def _save_batch_statistics(self, results: List[Dict], save_path: str):
        """保存批量推理统计信息"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=== 批量推理统计报告 ===\n\n")
            
            total_images = len(results)
            total_detections = sum(r['count'] for r in results)
            
            f.write(f"总图像数: {total_images}\n")
            f.write(f"总检测数: {total_detections}\n")
            f.write(f"平均每张图像检测数: {total_detections/total_images:.2f}\n\n")
            
            # 统计各类别
            class_stats = defaultdict(int)
            for result in results:
                for class_name, count in result.get('class_counts', {}).items():
                    class_stats[class_name] += count
            
            f.write("类别统计:\n")
            for class_name, count in class_stats.items():
                f.write(f"  {class_name}: {count}\n")
            
            f.write("\n详细结果:\n")
            for result in results:
                f.write(f"图像: {result['image_name']}, 检测数: {result['count']}\n")
                for class_name, count in result.get('class_counts', {}).items():
                    f.write(f"  {class_name}: {count}\n")
    
    def _calculate_video_statistics(self, frame_results: List[Dict], processing_time: float, processing_fps: float) -> Dict:
        """计算视频统计信息"""
        total_frames = len(frame_results)
        total_detections = sum(r['count'] for r in frame_results)
        
        # 各类别统计
        class_stats = defaultdict(int)
        for result in frame_results:
            for class_id in result['class_ids']:
                class_stats[self.class_names[class_id]] += 1
        
        stats = {
            'total_frames': total_frames,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / total_frames,
            'processing_time': processing_time,
            'processing_fps': processing_fps,
            'class_statistics': dict(class_stats)
        }
        
        return stats


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path: str, data_config: str, device: str = 'cpu'):
        """
        初始化评估器
        
        Args:
            model_path (str): 模型权重路径
            data_config (str): 数据配置文件路径
            device (str): 评估设备选择 (cpu 或 cuda)
        """
        self.model_path = model_path
        self.data_config = data_config
        self.device = device
        self.model = YOLO(model_path)
        
        # 设置设备
        if device != 'cpu':
            self.model.to(device)
        
        # 加载数据配置
        with open(data_config, 'r', encoding='utf-8') as f:
            self.data_info = yaml.safe_load(f)
        
        self.class_names = self.data_info['names']
        self.nc = self.data_info['nc']
        
    def evaluate(self, save_dir: str = "runs/evaluate") -> Dict:
        """
        评估模型性能
        
        Args:
            save_dir (str): 保存目录
            
        Returns:
            Dict: 评估结果
        """
        LOGGER.info("开始模型评估...")
        
        # 运行验证
        results = self.model.val(
            data=self.data_config,
            save_json=True,
            save_dir=save_dir,
            plots=True,
            verbose=True,
            device=self.device
        )
        
        # 提取评估指标
        metrics = {
            'mAP50': results.box.map50 if hasattr(results.box, 'map50') else 0.0,
            'mAP50-95': results.box.map if hasattr(results.box, 'map') else 0.0,
            'precision': results.box.mp if hasattr(results.box, 'mp') else 0.0,
            'recall': results.box.mr if hasattr(results.box, 'mr') else 0.0,
        }
        
        # 计算F1分数
        if metrics['precision'] > 0 and metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0
        
        # 各类别AP
        class_ap = {}
        if hasattr(results.box, 'ap_class_index') and results.box.ap_class_index is not None:
            for i, class_idx in enumerate(results.box.ap_class_index):
                class_name = self.class_names[class_idx] if isinstance(self.class_names, list) else self.class_names[class_idx]
                class_ap[class_name] = {
                    'AP50': results.box.ap50[i] if hasattr(results.box, 'ap50') else 0.0,
                    'AP50-95': results.box.ap[i] if hasattr(results.box, 'ap') else 0.0
                }
        
        metrics['class_ap'] = class_ap
        
        LOGGER.info("模型评估完成!")
        return metrics
    
    def analyze_model_complexity(self) -> Dict:
        """分析模型复杂度"""
        LOGGER.info("分析模型复杂度...")
        
        # 获取模型信息
        model = self.model.model
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 计算模型大小（MB）
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        # 计算FLOPs
        flops_g = None
        if THOP_AVAILABLE:
            try:
                input_tensor = torch.randn(1, 3, 640, 640)
                flops, _ = thop.profile(model, inputs=(input_tensor,), verbose=False)
                flops_g = flops / 1e9  # 转换为GFLOPs
            except Exception as e:
                LOGGER.warning(f"无法计算FLOPs: {e}")
        
        complexity = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'flops_g': flops_g
        }
        
        LOGGER.info(f"模型复杂度分析完成:")
        LOGGER.info(f"  总参数量: {total_params:,}")
        LOGGER.info(f"  可训练参数: {trainable_params:,}")
        LOGGER.info(f"  模型大小: {model_size_mb:.2f} MB")
        if flops_g:
            LOGGER.info(f"  FLOPs: {flops_g:.2f} G")
        
        return complexity


class ModelComparator:
    """模型对比器"""
    
    def __init__(self, lw_yolo_weights: str, yolo_weights: str, data_config: str, device: str = 'cpu'):
        """
        初始化对比器
        
        Args:
            lw_yolo_weights (str): LW-YOLOv8权重路径
            yolo_weights (str): 原始YOLOv8权重路径  
            data_config (str): 数据配置文件路径
            device (str): 对比设备选择 (cpu 或 cuda)
        """
        self.lw_yolo_weights = lw_yolo_weights
        self.yolo_weights = yolo_weights
        self.data_config = data_config
        
        self.device = device
        
        self.lw_evaluator = ModelEvaluator(lw_yolo_weights, data_config, device)
        self.yolo_evaluator = ModelEvaluator(yolo_weights, data_config, device)
        
    def compare_models(self, save_dir: str = "runs/compare") -> Dict:
        """
        对比两个模型
        
        Args:
            save_dir (str): 保存目录
            
        Returns:
            Dict: 对比结果
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        LOGGER.info("开始模型对比...")
        
        # 评估LW-YOLOv8
        LOGGER.info("评估LW-YOLOv8...")
        lw_metrics = self.lw_evaluator.evaluate(save_dir=str(save_path / "lw_yolo"))
        lw_complexity = self.lw_evaluator.analyze_model_complexity()
        
        # 评估原始YOLOv8
        LOGGER.info("评估原始YOLOv8...")
        yolo_metrics = self.yolo_evaluator.evaluate(save_dir=str(save_path / "yolo"))
        yolo_complexity = self.yolo_evaluator.analyze_model_complexity()
        
        # 对比结果
        comparison = {
            'lw_yolo': {
                'metrics': lw_metrics,
                'complexity': lw_complexity
            },
            'yolo': {
                'metrics': yolo_metrics,
                'complexity': yolo_complexity
            }
        }
        
        # 计算改进幅度
        improvements = {}
        for metric in ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']:
            if metric in lw_metrics and metric in yolo_metrics and yolo_metrics[metric] > 0:
                improvement = ((lw_metrics[metric] - yolo_metrics[metric]) / yolo_metrics[metric]) * 100
                improvements[metric] = improvement
        
        # 模型复杂度对比
        if yolo_complexity['total_parameters'] > 0:
            param_reduction = ((yolo_complexity['total_parameters'] - lw_complexity['total_parameters']) / 
                              yolo_complexity['total_parameters']) * 100
            improvements['parameter_reduction'] = param_reduction
            
        if yolo_complexity['model_size_mb'] > 0:
            size_reduction = ((yolo_complexity['model_size_mb'] - lw_complexity['model_size_mb']) / 
                             yolo_complexity['model_size_mb']) * 100
            improvements['size_reduction'] = size_reduction
        
        if lw_complexity['flops_g'] and yolo_complexity['flops_g'] and yolo_complexity['flops_g'] > 0:
            flops_reduction = ((yolo_complexity['flops_g'] - lw_complexity['flops_g']) / 
                              yolo_complexity['flops_g']) * 100
            improvements['flops_reduction'] = flops_reduction
        
        comparison['improvements'] = improvements
        
        # 保存对比报告
        self._save_comparison_report(comparison, save_path / "comparison_report.txt")
        
        # 生成对比图表
        self._plot_comparison(comparison, save_path)
        
        LOGGER.info("模型对比完成!")
        return comparison
    
    def _save_comparison_report(self, comparison: Dict, save_path: Path):
        """保存对比报告"""
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=== LW-YOLOv8 vs YOLOv8 对比报告 ===\n\n")
            
            # 性能指标对比
            f.write("性能指标对比:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'指标':<15} {'LW-YOLOv8':<12} {'YOLOv8':<12} {'改进幅度':<12}\n")
            f.write("-" * 50 + "\n")
            
            lw_metrics = comparison['lw_yolo']['metrics']
            yolo_metrics = comparison['yolo']['metrics']
            improvements = comparison['improvements']
            
            for metric in ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']:
                if metric in lw_metrics and metric in yolo_metrics:
                    improvement = improvements.get(metric, 0)
                    f.write(f"{metric:<15} {lw_metrics[metric]:<12.4f} {yolo_metrics[metric]:<12.4f} {improvement:+7.2f}%\n")
            
            f.write("\n\n模型复杂度对比:\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'指标':<20} {'LW-YOLOv8':<15} {'YOLOv8':<15} {'减少幅度':<12}\n")
            f.write("-" * 60 + "\n")
            
            lw_complex = comparison['lw_yolo']['complexity']
            yolo_complex = comparison['yolo']['complexity']
            
            f.write(f"{'参数量':<20} {lw_complex['total_parameters']:<15,} {yolo_complex['total_parameters']:<15,} {improvements.get('parameter_reduction', 0):+7.2f}%\n")
            f.write(f"{'模型大小(MB)':<20} {lw_complex['model_size_mb']:<15.2f} {yolo_complex['model_size_mb']:<15.2f} {improvements.get('size_reduction', 0):+7.2f}%\n")
            
            if 'flops_reduction' in improvements:
                f.write(f"{'FLOPs(G)':<20} {lw_complex['flops_g']:<15.2f} {yolo_complex['flops_g']:<15.2f} {improvements['flops_reduction']:+7.2f}%\n")
            
            # 各类别AP对比
            f.write("\n\n各类别AP对比:\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'类别':<15} {'LW-YOLOv8 AP50':<15} {'YOLOv8 AP50':<15} {'LW-YOLOv8 AP50-95':<18} {'YOLOv8 AP50-95':<15}\n")
            f.write("-" * 70 + "\n")
            
            lw_class_ap = lw_metrics.get('class_ap', {})
            yolo_class_ap = yolo_metrics.get('class_ap', {})
            
            for class_name in lw_class_ap.keys():
                if class_name in yolo_class_ap:
                    f.write(f"{class_name:<15} {lw_class_ap[class_name]['AP50']:<15.4f} {yolo_class_ap[class_name]['AP50']:<15.4f} ")
                    f.write(f"{lw_class_ap[class_name]['AP50-95']:<18.4f} {yolo_class_ap[class_name]['AP50-95']:<15.4f}\n")
    
    def _plot_comparison(self, comparison: Dict, save_dir: Path):
        """生成对比图表"""
        try:
            # 设置图形样式
            plt.style.use('default')
            
            # 性能指标对比图
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 主要指标对比
            metrics = ['mAP50', 'mAP50-95', 'precision', 'recall', 'f1']
            lw_values = [comparison['lw_yolo']['metrics'].get(m, 0) for m in metrics]
            yolo_values = [comparison['yolo']['metrics'].get(m, 0) for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax1.bar(x - width/2, lw_values, width, label='LW-YOLOv8', color='skyblue')
            ax1.bar(x + width/2, yolo_values, width, label='YOLOv8', color='lightcoral')
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Value')
            ax1.set_title('Performance Metrics Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 模型复杂度对比
            complexity_metrics = ['Parameters(M)', 'Model Size(MB)']
            lw_complex_values = [
                comparison['lw_yolo']['complexity']['total_parameters'] / 1e6,
                comparison['lw_yolo']['complexity']['model_size_mb']
            ]
            yolo_complex_values = [
                comparison['yolo']['complexity']['total_parameters'] / 1e6,
                comparison['yolo']['complexity']['model_size_mb']
            ]
            
            if comparison['lw_yolo']['complexity']['flops_g']:
                complexity_metrics.append('FLOPs(G)')
                lw_complex_values.append(comparison['lw_yolo']['complexity']['flops_g'])
                yolo_complex_values.append(comparison['yolo']['complexity']['flops_g'])
            
            x2 = np.arange(len(complexity_metrics))
            ax2.bar(x2 - width/2, lw_complex_values, width, label='LW-YOLOv8', color='lightgreen')
            ax2.bar(x2 + width/2, yolo_complex_values, width, label='YOLOv8', color='orange')
            ax2.set_xlabel('Metrics')
            ax2.set_ylabel('Value')
            ax2.set_title('Model Complexity Comparison')
            ax2.set_xticks(x2)
            ax2.set_xticklabels(complexity_metrics)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. 改进幅度图
            improvements = comparison['improvements']
            improvement_metrics = [k for k in improvements.keys() if 'reduction' not in k]
            improvement_values = [improvements[k] for k in improvement_metrics]
            
            colors_improve = ['green' if v > 0 else 'red' for v in improvement_values]
            ax3.bar(improvement_metrics, improvement_values, color=colors_improve, alpha=0.7)
            ax3.set_xlabel('Metrics')
            ax3.set_ylabel('Improvement (%)')
            ax3.set_title('Performance Improvement')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.get_xticklabels(), rotation=45)
            
            # 4. 模型减少幅度图
            reduction_metrics = [k.replace('_reduction', '').replace('_', ' ') for k in improvements.keys() if 'reduction' in k]
            reduction_values = [improvements[k] for k in improvements.keys() if 'reduction' in k]
            
            ax4.bar(reduction_metrics, reduction_values, color='blue', alpha=0.7)
            ax4.set_xlabel('Metrics')
            ax4.set_ylabel('Reduction (%)')
            ax4.set_title('Model Complexity Reduction')
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(save_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            LOGGER.info(f"对比图表已保存到: {save_dir / 'model_comparison.png'}")
            
        except Exception as e:
            LOGGER.warning(f"生成对比图表失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LW-YOLOv8 轻量级安全帽检测模型推理和评估')
    
    # 推理参数
    parser.add_argument('--weights', type=str, default='runs/train/lw-yolov8/weights/best.pt', help='模型权重文件路径')
    parser.add_argument('--source', type=str, default='datasets/val/images', help='输入源（图像/目录/视频）')
    parser.add_argument('--output', type=str, default='runs/detect', help='输出目录')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU阈值')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='推理设备选择 (cpu 或 cuda)')
    parser.add_argument('--show', action='store_true', help='显示结果')
    
    # 评估参数
    parser.add_argument('--evaluate', action='store_true', help='评估模型')
    parser.add_argument('--data', type=str, default='datasets/dataset.yaml', help='数据配置文件路径')
    
    # 对比参数
    parser.add_argument('--compare', action='store_true', help='对比模型')
    parser.add_argument('--lw-weights', type=str, default='runs/train/lw-yolov8/weights/best.pt', help='LW-YOLOv8权重路径')
    parser.add_argument('--yolo-weights', type=str, default='runs/train/yolov8-baseline/weights/best.pt', help='原始YOLOv8权重路径')
    
    # 视频推理
    parser.add_argument('--video', action='store_true', help='视频推理模式')
    
    args = parser.parse_args()
    
    # 模型对比
    if args.compare:
        if not args.lw_weights or not args.yolo_weights or not args.data:
            parser.error("对比模式需要 --lw-weights, --yolo-weights 和 --data 参数")
        
        comparator = ModelComparator(args.lw_weights, args.yolo_weights, args.data, args.device)
        comparison_results = comparator.compare_models()
        
        # 打印简要结果
        improvements = comparison_results['improvements']
        LOGGER.info("=== 对比结果摘要 ===")
        LOGGER.info(f"mAP50 改进: {improvements.get('mAP50', 0):+.2f}%")
        LOGGER.info(f"mAP50-95 改进: {improvements.get('mAP50-95', 0):+.2f}%")
        LOGGER.info(f"参数量减少: {improvements.get('parameter_reduction', 0):+.2f}%")
        LOGGER.info(f"模型大小减少: {improvements.get('size_reduction', 0):+.2f}%")
        if 'flops_reduction' in improvements:
            LOGGER.info(f"FLOPs减少: {improvements.get('flops_reduction', 0):+.2f}%")
        
        return
    
    # 模型评估
    if args.evaluate:
        if not args.weights or not args.data:
            parser.error("评估模式需要 --weights 和 --data 参数")
        
        evaluator = ModelEvaluator(args.weights, args.data, args.device)
        metrics = evaluator.evaluate()
        complexity = evaluator.analyze_model_complexity()
        
        LOGGER.info("=== 评估结果 ===")
        LOGGER.info(f"mAP50: {metrics['mAP50']:.4f}")
        LOGGER.info(f"mAP50-95: {metrics['mAP50-95']:.4f}")
        LOGGER.info(f"Precision: {metrics['precision']:.4f}")
        LOGGER.info(f"Recall: {metrics['recall']:.4f}")
        LOGGER.info(f"F1: {metrics['f1']:.4f}")
        
        return
    
    # 推理模式
    if not args.weights or not args.source:
        parser.error("推理模式需要 --weights 和 --source 参数")
    
    # 创建推理器
    inferencer = SafetyHelmetInferencer(args.weights, args.conf, args.iou, args.device)
    
    # 视频推理
    if args.video:
        stats = inferencer.predict_video(args.source, args.output)
        LOGGER.info("=== 视频推理统计 ===")
        LOGGER.info(f"总帧数: {stats['total_frames']}")
        LOGGER.info(f"总检测数: {stats['total_detections']}")
        LOGGER.info(f"平均每帧检测数: {stats['avg_detections_per_frame']:.2f}")
        LOGGER.info(f"处理FPS: {stats['processing_fps']:.1f}")
        for class_name, count in stats['class_statistics'].items():
            LOGGER.info(f"{class_name}: {count}")
    
    # 图像推理
    else:
        source_path = Path(args.source)
        
        # 单张图像
        if source_path.is_file():
            output_path = Path(args.output) / f"annotated_{source_path.name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result = inferencer.predict_image(args.source, str(output_path), args.show)
            
            LOGGER.info("=== 推理结果 ===")
            LOGGER.info(f"检测数量: {result['count']}")
            for class_name, count in result.get('class_counts', {}).items():
                LOGGER.info(f"{class_name}: {count}")
        
        # 批量推理
        else:
            results = inferencer.predict_batch(args.source, args.output)
            
            LOGGER.info("=== 批量推理统计 ===")
            total_images = len(results)
            total_detections = sum(r['count'] for r in results)
            LOGGER.info(f"总图像数: {total_images}")
            LOGGER.info(f"总检测数: {total_detections}")
            LOGGER.info(f"平均每张检测数: {total_detections/total_images:.2f}")


if __name__ == '__main__':
    main() 