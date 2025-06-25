"""
LSCD模块 - 轻量化共享卷积检测头 (Lightweight Shared Convolution Detection)
保留分类和定位特征，同时利用共享卷积减少参数和计算量
目标：减少19%参数和10%计算量，同时保持检测精度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedConvBlock(nn.Module):
    """
    共享卷积块
    在分类和回归任务间共享部分卷积层
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super().__init__()
        
        # 共享的特征提取层
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                     padding=kernel_size//2, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 任务特定的轻量化调节层
        self.cls_adapter = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.reg_adapter = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        
    def forward(self, x, task='both'):
        """
        Args:
            x: 输入特征 [B, C, H, W]
            task: 'cls', 'reg', or 'both'
        Returns:
            features for classification and/or regression
        """
        # 共享特征提取
        shared_feat = self.shared_conv(x)
        
        if task == 'cls':
            return self.cls_adapter(shared_feat)
        elif task == 'reg':
            return self.reg_adapter(shared_feat)
        else:  # both
            cls_feat = self.cls_adapter(shared_feat)
            reg_feat = self.reg_adapter(shared_feat)
            return cls_feat, reg_feat

class LightweightDetectionHead(nn.Module):
    """
    轻量化检测头
    使用共享卷积减少参数，同时保持分类和回归性能
    """
    def __init__(self, nc=3, anchors=None, ch=(256, 512, 1024), inplace=True):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        
        # 处理anchors
        if anchors is None or len(anchors) == 0:
            # 默认YOLOv8 anchors
            anchors = [
                [10, 13, 16, 30, 33, 23],      # P3
                [30, 61, 62, 45, 59, 119],     # P4  
                [116, 90, 156, 198, 373, 326]  # P5
            ]
        
        self.nl = len(anchors) if isinstance(anchors, list) else len(ch)  # number of detection layers
        
        # 处理不同格式的anchors
        if isinstance(anchors, (list, tuple)) and len(anchors) > 0:
            if isinstance(anchors[0], (list, tuple)):
                # 格式: [[a1, a2, ...], [b1, b2, ...], ...]
                self.na = len(anchors[0]) // 2  # 每层anchor数量
                anchor_tensor = torch.tensor(anchors).float()
                if anchor_tensor.dim() == 2:
                    anchor_tensor = anchor_tensor.view(self.nl, -1, 2)
            else:
                # 格式: [a1, a2, a3, ...]
                self.na = 3  # 默认3个anchor
                anchor_tensor = torch.tensor(anchors).float().view(-1, 2)
                if anchor_tensor.shape[0] >= self.nl * self.na:
                    anchor_tensor = anchor_tensor[:self.nl * self.na].view(self.nl, self.na, 2)
                else:
                    anchor_tensor = anchor_tensor.repeat(self.nl * self.na // anchor_tensor.shape[0] + 1, 1)
                    anchor_tensor = anchor_tensor[:self.nl * self.na].view(self.nl, self.na, 2)
        else:
            self.na = 3
            anchor_tensor = torch.ones(self.nl, self.na, 2)
        
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        
        # 注册anchors
        self.register_buffer('anchors', anchor_tensor)
        
        # 确保通道数匹配层数
        if len(ch) != self.nl:
            ch = ch[:self.nl] if len(ch) > self.nl else list(ch) + [ch[-1]] * (self.nl - len(ch))
        
        # 共享特征提取骨干
        self.shared_backbone = nn.ModuleList()
        for i in range(self.nl):
            # 输入通道调节
            input_conv = nn.Sequential(
                nn.Conv2d(ch[i], 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            
            # 共享卷积块
            shared_blocks = nn.Sequential(
                SharedConvBlock(256, 128, kernel_size=3),
                SharedConvBlock(128, 128, kernel_size=3),
            )
            
            self.shared_backbone.append(nn.Sequential(input_conv, shared_blocks))
        
        # 任务特定的输出头
        self.cls_heads = nn.ModuleList()  # 分类头
        self.reg_heads = nn.ModuleList()  # 回归头
        
        for i in range(self.nl):
            # 分类头 - 预测类别概率和对象性
            cls_head = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, self.na * (self.nc + 1), 1)  # +1 for objectness
            )
            
            # 回归头 - 预测边界框坐标
            reg_head = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, self.na * 4, 1)  # 4 coordinates (x, y, w, h)
            )
            
            self.cls_heads.append(cls_head)
            self.reg_heads.append(reg_head)
        
        # 权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化检测头权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: 来自FPN的特征图列表 [P3, P4, P5]
        Returns:
            检测结果
        """
        outputs = []
        
        for i, xi in enumerate(x):
            # 共享特征提取
            shared_feat = self.shared_backbone[i](xi)
            
            # 任务特定预测
            cls_feat, reg_feat = shared_feat[1](shared_feat[0], task='both')
            
            # 分类预测
            cls_pred = self.cls_heads[i](cls_feat)
            
            # 回归预测
            reg_pred = self.reg_heads[i](reg_feat)
            
            # 重塑输出格式
            bs, _, ny, nx = cls_pred.shape
            
            cls_pred = cls_pred.view(bs, self.na, self.nc + 1, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            reg_pred = reg_pred.view(bs, self.na, 4, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            # 合并分类和回归预测
            pred = torch.cat([reg_pred, cls_pred[..., :1], cls_pred[..., 1:]], dim=-1)
            
            outputs.append(pred.view(bs, -1, self.no))
        
        return torch.cat(outputs, 1) if self.training else (torch.cat(outputs, 1), x)

class AdaptiveFusion(nn.Module):
    """
    自适应特征融合模块
    动态调节分类和回归特征的融合权重
    """
    def __init__(self, channels):
        super().__init__()
        
        # 特征重要性评估
        self.importance_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, 2, 1),  # 2个任务的权重
            nn.Softmax(dim=1)
        )
        
        # 特征调节
        self.feature_modulator = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, shared_feat):
        """
        Args:
            shared_feat: 共享特征 [B, C, H, W]
        Returns:
            cls_feat, reg_feat: 任务特定特征
        """
        # 计算任务重要性权重
        importance = self.importance_net(shared_feat)  # [B, 2, 1, 1]
        cls_weight, reg_weight = importance[:, 0:1], importance[:, 1:2]
        
        # 调节共享特征
        modulated_feat = self.feature_modulator(shared_feat)
        
        # 生成任务特定特征
        cls_feat = modulated_feat * cls_weight
        reg_feat = modulated_feat * reg_weight
        
        return cls_feat, reg_feat

class LSCD_Head(nn.Module):
    """
    完整的LSCD检测头
    集成所有轻量化设计
    """
    def __init__(self, nc=3, anchors=None, ch=(256, 512, 1024)):
        super().__init__()
        self.nc = nc
        self.no = nc + 5  # 输出数量: classes + objectness + 4 bbox
        
        # 处理anchors
        if anchors is None:
            # 默认YOLOv8 anchors
            anchors = [
                [10, 13, 16, 30, 33, 23],      # P3
                [30, 61, 62, 45, 59, 119],     # P4  
                [116, 90, 156, 198, 373, 326]  # P5
            ]
        
        self.nl = len(anchors) if isinstance(anchors, list) else len(ch)  # 检测层数
        
        # 处理不同格式的anchors
        if isinstance(anchors, (list, tuple)) and len(anchors) > 0:
            if isinstance(anchors[0], (list, tuple)):
                # 格式: [[a1, a2, ...], [b1, b2, ...], ...]
                self.na = len(anchors[0]) // 2  # 每层anchor数量
                anchor_tensor = torch.tensor(anchors).float()
                if anchor_tensor.dim() == 2:
                    anchor_tensor = anchor_tensor.view(self.nl, -1, 2)
            else:
                # 格式: [a1, a2, a3, ...]
                self.na = 3  # 默认3个anchor
                anchor_tensor = torch.tensor(anchors).float().view(-1, 2)
                if anchor_tensor.shape[0] >= self.nl * self.na:
                    anchor_tensor = anchor_tensor[:self.nl * self.na].view(self.nl, self.na, 2)
                else:
                    # 扩展到所需大小
                    anchor_tensor = anchor_tensor.repeat(self.nl * self.na // anchor_tensor.shape[0] + 1, 1)
                    anchor_tensor = anchor_tensor[:self.nl * self.na].view(self.nl, self.na, 2)
        else:
            # 无anchor或空anchor，使用默认值
            self.na = 3
            anchor_tensor = torch.ones(self.nl, self.na, 2)
        
        # 注册anchors
        self.register_buffer('anchors', anchor_tensor)
        
        # 确保通道数匹配层数
        if len(ch) != self.nl:
            ch = ch[:self.nl] if len(ch) > self.nl else list(ch) + [ch[-1]] * (self.nl - len(ch))
        
        # 每层的共享特征提取器
        self.feature_extractors = nn.ModuleList()
        for i in range(self.nl):
            extractor = nn.Sequential(
                nn.Conv2d(ch[i], 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                SharedConvBlock(128, 128),
                AdaptiveFusion(128)
            )
            self.feature_extractors.append(extractor)
        
        # 轻量化输出层
        self.output_layers = nn.ModuleList()
        for i in range(self.nl):
            # 使用深度可分离卷积进一步减少参数
            output_layer = nn.Sequential(
                # 深度卷积
                nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                # 点卷积
                nn.Conv2d(128, self.na * self.no, 1)
            )
            self.output_layers.append(output_layer)
    
    def forward(self, x):
        """前向传播"""
        outputs = []
        
        # 确保输入特征数量与层数匹配
        if len(x) != self.nl:
            x = x[:self.nl] if len(x) > self.nl else list(x) + [x[-1]] * (self.nl - len(x))
        
        for i, xi in enumerate(x):
            # 提取共享特征
            feat_layers = self.feature_extractors[i]
            shared_feat = feat_layers[:3](xi)  # 前3层
            
            # 自适应特征融合
            cls_feat, reg_feat = feat_layers[3](shared_feat)
            fused_feat = (cls_feat + reg_feat) / 2
            pred = self.output_layers[i](fused_feat)
            bs, _, ny, nx = pred.shape
            pred = pred.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            outputs.append(pred.view(bs, -1, self.no))
        
        return torch.cat(outputs, 1)

# 测试函数
if __name__ == "__main__":
    print("测试LSCD轻量化检测头...")
    
    # 模拟anchors (YOLOv8格式)
    anchors = [
        [10, 13, 16, 30, 33, 23],      # P3
        [30, 61, 62, 45, 59, 119],     # P4  
        [116, 90, 156, 198, 373, 326]  # P5
    ]
    
    # 创建测试输入
    batch_size = 2
    num_classes = 3
    features = [
        torch.randn(batch_size, 256, 80, 80),   # P3
        torch.randn(batch_size, 512, 40, 40),   # P4
        torch.randn(batch_size, 1024, 20, 20),  # P5
    ]
    
    # 测试共享卷积块
    print("测试共享卷积块...")
    shared_block = SharedConvBlock(256, 128)
    cls_feat, reg_feat = shared_block(features[0])
    print(f"共享卷积块输出 - 分类: {cls_feat.shape}, 回归: {reg_feat.shape}")
    
    # 测试自适应融合
    print("\n测试自适应融合...")
    adaptive_fusion = AdaptiveFusion(128)
    cls_adaptive, reg_adaptive = adaptive_fusion(cls_feat)
    print(f"自适应融合输出 - 分类: {cls_adaptive.shape}, 回归: {reg_adaptive.shape}")
    
    # 测试完整LSCD头
    print("\n测试LSCD检测头...")
    lscd_head = LSCD_Head(nc=num_classes, anchors=anchors, ch=[256, 512, 1024])
    lscd_output = lscd_head(features)
    print(f"LSCD检测头输出: {lscd_output.shape}")
    
    # 参数量对比
    shared_params = sum(p.numel() for p in shared_block.parameters())
    fusion_params = sum(p.numel() for p in adaptive_fusion.parameters())
    lscd_params = sum(p.numel() for p in lscd_head.parameters())
    
    print(f"\n参数量统计:")
    print(f"共享卷积块: {shared_params:,} 参数")
    print(f"自适应融合: {fusion_params:,} 参数")
    print(f"LSCD检测头: {lscd_params:,} 参数")
    
    # 模拟标准检测头进行对比
    class StandardHead(nn.Module):
        def __init__(self, nc, ch):
            super().__init__()
            self.heads = nn.ModuleList()
            for c in ch:
                head = nn.Sequential(
                    nn.Conv2d(c, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 3 * (nc + 5), 1)
                )
                self.heads.append(head)
    
    standard_head = StandardHead(num_classes, [256, 512, 1024])
    standard_params = sum(p.numel() for p in standard_head.parameters())
    
    reduction_ratio = (standard_params - lscd_params) / standard_params * 100
    print(f"标准检测头: {standard_params:,} 参数")
    print(f"参数减少: {reduction_ratio:.1f}%")
    print("LSCD轻量化检测头测试完成! ✅") 