import torch
import torch.nn as nn
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


class SimplifiedSIoULoss(BboxLoss):
    """
    简化的SIoU损失函数
    
    只保留最关键的组件：
    1. 基础IoU
    2. 简化的角度损失（用于处理倾斜目标）
    3. 简化的形状损失（用于处理长条形目标）
    """
    
    def __init__(self, reg_max=16):
        """初始化简化的SIoU损失"""
        super().__init__(reg_max)
        
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, 
                target_scores, target_scores_sum, fg_mask):
        """计算简化的SIoU损失"""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        
        # 计算简化的SIoU
        siou = self.simplified_siou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - siou) * weight).sum() / target_scores_sum
        
        # DFL损失保持不变
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
            
        return loss_iou, loss_dfl
    
    def simplified_siou(self, pred_boxes, target_boxes, eps=1e-7):
        """
        计算简化的SIoU
        
        Args:
            pred_boxes: 预测框 [N, 4] (x1, y1, x2, y2)
            target_boxes: 真实框 [N, 4] (x1, y1, x2, y2)
            
        Returns:
            siou: 简化的SIoU值 [N]
        """
        # 计算基础IoU
        iou = bbox_iou(pred_boxes, target_boxes, xywh=False, CIoU=False).squeeze(-1)
        
        # 计算中心点
        pred_cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_cx = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_cy = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        
        # 计算宽高
        pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
        pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
        target_w = target_boxes[:, 2] - target_boxes[:, 0]
        target_h = target_boxes[:, 3] - target_boxes[:, 1]
        
        # 1. 简化的角度损失
        # 计算中心点偏移的角度
        dx = target_cx - pred_cx
        dy = target_cy - pred_cy
        
        # 角度相关的权重（当中心点偏移方向与宽高比不匹配时惩罚更大）
        angle_weight = torch.abs(dx * dy) / (torch.sqrt(dx**2 + eps) * torch.sqrt(dy**2 + eps) + eps)
        angle_loss = angle_weight * 0.1  # 缩小权重
        
        # 2. 简化的形状损失
        # 宽高比差异
        wh_ratio_pred = pred_w / (pred_h + eps)
        wh_ratio_target = target_w / (target_h + eps)
        wh_ratio_loss = torch.abs(wh_ratio_pred - wh_ratio_target) / (wh_ratio_target + eps)
        
        # 形状损失（对于安全帽这种相对规则的目标，权重可以小一些）
        shape_loss = wh_ratio_loss * 0.05
        
        # 综合SIoU
        siou = iou - angle_loss - shape_loss
        
        return torch.clamp(siou, min=0.0, max=1.0)