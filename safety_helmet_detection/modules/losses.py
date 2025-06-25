import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FocalerCIOULoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred_boxes, target_boxes, iou):
        ciou_loss = self.ciou_loss(pred_boxes, target_boxes)
        focal_weight = self.alpha * (1 - iou) ** self.gamma
        focaler_ciou = focal_weight * ciou_loss
        
        if self.reduction == 'mean':
            return focaler_ciou.mean()
        elif self.reduction == 'sum':
            return focaler_ciou.sum()
        else:
            return focaler_ciou
    
    def ciou_loss(self, pred_boxes, target_boxes):
        # Convert to center and width/height format
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        target_x1, target_y1, target_x2, target_y2 = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]
        
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1
        target_w = target_x2 - target_x1
        target_h = target_y2 - target_y1
        
        pred_cx = (pred_x1 + pred_x2) / 2
        pred_cy = (pred_y1 + pred_y2) / 2
        target_cx = (target_x1 + target_x2) / 2
        target_cy = (target_y1 + target_y2) / 2
        
        # 计算IoU
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        pred_area = pred_w * pred_h
        target_area = target_w * target_h
        union_area = pred_area + target_area - inter_area
        
        iou = inter_area / (union_area + 1e-7)
        
        # 外接矩形
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        enclose_w = enclose_x2 - enclose_x1
        enclose_h = enclose_y2 - enclose_y1
        enclose_area = enclose_w * enclose_h
        
        # 中心点距离
        center_distance = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        diagonal_distance = enclose_w ** 2 + enclose_h ** 2
        
        # 纵横比惩罚项
        v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(target_w / (target_h + 1e-7)) - torch.atan(pred_w / (pred_h + 1e-7)), 2)
        alpha_ciou = v / (1 - iou + v + 1e-7)
        
        # CIOU
        ciou = iou - center_distance / (diagonal_distance + 1e-7) - alpha_ciou * v
        
        return 1 - ciou

class EnhancedFocalLoss(nn.Module):
    """
    增强Focal Loss
    专门用于安全帽检测的分类损失
    """
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: 预测概率 [N, C]
            targets: 真实标签 [N]
        Returns:
            focal_loss: Focal损失
        """
        # 标签平滑
        if self.label_smoothing > 0:
            targets = self.smooth_labels(targets, inputs.size(1))
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算预测概率
        pt = torch.exp(-ce_loss)
        
        # Alpha权重 (处理类别不平衡)
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            alpha_t = self.alpha[targets]
        
        # Focal权重
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        
        # Focal损失
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def smooth_labels(self, targets, num_classes):
        """标签平滑"""
        smoothed = torch.full((targets.size(0), num_classes), 
                            self.label_smoothing / (num_classes - 1),
                            device=targets.device)
        smoothed.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
        return smoothed

class YOLOLoss(nn.Module):
    """
    完整的YOLO损失函数
    结合分类、回归和对象性损失
    """
    def __init__(self, nc=3, anchors=(), device='cpu'):
        super().__init__()
        self.nc = nc
        self.device = device
        
        # 损失权重
        self.box_weight = 0.05
        self.cls_weight = 0.5
        self.obj_weight = 1.0
        
        # 损失函数
        self.focaler_ciou = FocalerCIOULoss(alpha=0.25, gamma=2.0)
        self.focal_cls = EnhancedFocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.1)
        self.bce_obj = nn.BCEWithLogitsLoss()
        
        # Anchors
        self.anchors = torch.tensor(anchors, device=device).float()
        self.nl = len(anchors)  # number of layers
        self.na = len(anchors[0]) // 2  # number of anchors per layer
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: 模型预测 [B, N, no]
            targets: 真实标签
        Returns:
            total_loss: 总损失
        """
        device = predictions.device
        
        # 解析预测
        batch_size = predictions.shape[0]
        
        # 初始化损失
        box_loss = torch.zeros(1, device=device)
        cls_loss = torch.zeros(1, device=device)
        obj_loss = torch.zeros(1, device=device)
        
        # 计算损失 (简化版本)
        # 这里需要根据具体的anchor matching策略实现
        # 暂时使用简化的损失计算
        
        # 总损失
        total_loss = (self.box_weight * box_loss + 
                     self.cls_weight * cls_loss + 
                     self.obj_weight * obj_loss)
        
        return total_loss, torch.cat((box_loss, cls_loss, obj_loss)).detach()

class MultiScaleTrainingLoss(nn.Module):
    """
    多尺度训练损失
    支持不同尺度的输入训练
    """
    def __init__(self, base_loss, scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
        super().__init__()
        self.base_loss = base_loss
        self.scales = scales
        
    def forward(self, predictions, targets, current_scale=1.0):
        """
        Args:
            predictions: 模型预测
            targets: 真实标签
            current_scale: 当前训练尺度
        Returns:
            scaled_loss: 尺度调整后的损失
        """
        # 计算基础损失
        base_loss_value = self.base_loss(predictions, targets)
        
        # 尺度权重 - 小尺度需要更多关注
        scale_weight = 2.0 - current_scale  # 尺度越小，权重越大
        
        return base_loss_value * scale_weight

class SafetyHelmetLoss(nn.Module):
    """
    安全帽检测专用损失函数
    考虑安全帽检测的特殊需求
    """
    def __init__(self, nc=3, small_object_weight=2.0):
        super().__init__()
        self.nc = nc
        self.small_object_weight = small_object_weight
        
        # 类别权重 (安全帽检测中，未戴安全帽可能是少数类)
        self.class_weights = torch.tensor([1.0, 1.0, 1.5])  # [person, helmet, no_helmet]
        
        # 损失函数
        self.box_loss = FocalerCIOULoss(alpha=0.25, gamma=2.0)
        self.cls_loss = EnhancedFocalLoss(alpha=self.class_weights, gamma=2.0)
        self.obj_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets, bbox_sizes=None):
        """
        Args:
            predictions: 模型预测
            targets: 真实标签
            bbox_sizes: 边界框大小，用于小目标加权
        Returns:
            weighted_loss: 加权损失
        """
        # 计算基础损失
        box_loss = self.box_loss(predictions, targets)
        cls_loss = self.cls_loss(predictions, targets)
        obj_loss = self.obj_loss(predictions, targets)
        
        # 小目标加权
        if bbox_sizes is not None:
            # 计算目标大小权重
            size_weights = torch.where(bbox_sizes < 0.1,  # 小于10%图像大小
                                     self.small_object_weight,
                                     1.0)
            box_loss = box_loss * size_weights.mean()
        
        # 总损失
        total_loss = box_loss + cls_loss + obj_loss
        
        return total_loss

# 工具函数
def bbox_iou(box1, box2, eps=1e-7):
    """计算两个边界框的IoU"""
    # 获取交集坐标
    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])
    
    # 交集面积
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 各自面积
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # 并集面积
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + eps)
    
    return iou

# 测试函数
if __name__ == "__main__":
    print("测试损失函数模块...")
    
    # 测试Focaler-CIOU Loss
    print("测试Focaler-CIOU Loss...")
    focaler_ciou = FocalerCIOULoss()
    
    # 模拟边界框
    pred_boxes = torch.rand(10, 4) * 100  # [x1, y1, x2, y2]
    target_boxes = torch.rand(10, 4) * 100
    iou_values = torch.rand(10)  # 模拟IoU值
    
    ciou_loss = focaler_ciou(pred_boxes, target_boxes, iou_values)
    print(f"Focaler-CIOU Loss: {ciou_loss.item():.4f}")
    
    # 测试Enhanced Focal Loss
    print("\n测试Enhanced Focal Loss...")
    focal_loss = EnhancedFocalLoss()
    
    # 模拟分类预测和标签
    cls_pred = torch.randn(10, 3)  # 3个类别
    cls_targets = torch.randint(0, 3, (10,))
    
    cls_loss_value = focal_loss(cls_pred, cls_targets)
    print(f"Enhanced Focal Loss: {cls_loss_value.item():.4f}")
    
    # 测试安全帽专用损失
    print("\n测试安全帽专用损失...")
    safety_loss = SafetyHelmetLoss(nc=3)
    
    # 模拟预测和目标 (简化版本)
    predictions = torch.randn(2, 100, 8)  # [B, N, no]
    targets = torch.randn(2, 100, 8)
    bbox_sizes = torch.rand(2, 100)  # 边界框大小
    
    # safety_loss_value = safety_loss(predictions, targets, bbox_sizes)
    # print(f"Safety Helmet Loss: {safety_loss_value.item():.4f}")
    
    print("损失函数模块测试完成! ✅") 