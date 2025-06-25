import torch
import torch.nn as nn
import torch.nn.functional as F

class FSDI(nn.Module):
    def __init__(self, channels_list, fusion_channels=256):
        super().__init__()
        self.channels_list = channels_list
        self.fusion_channels = fusion_channels
        
        self.semantic_branch = nn.ModuleList()
        for i, channels in enumerate(channels_list):
            self.semantic_branch.append(
                nn.Sequential(
                    nn.Conv2d(channels, fusion_channels, 1, bias=False),
                    nn.BatchNorm2d(fusion_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.detail_branch = nn.ModuleList()
        for i, channels in enumerate(channels_list):
            self.detail_branch.append(
                nn.Sequential(
                    nn.Conv2d(channels, fusion_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(fusion_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.cross_scale_fusion = nn.ModuleList()
        for i in range(len(channels_list)):
            self.cross_scale_fusion.append(
                nn.Sequential(
                    nn.Conv2d(fusion_channels * 2, fusion_channels, 1, bias=False),
                    nn.BatchNorm2d(fusion_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        self.attention_weights = nn.ModuleList()
        for i in range(len(channels_list)):
            self.attention_weights.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(fusion_channels, fusion_channels // 16, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(fusion_channels // 16, fusion_channels, 1),
                    nn.Sigmoid()
                )
            )
    
    def forward(self, features):
        """
        Args:
            features: 来自不同层的特征图列表 [P3, P4, P5]
        Returns:
            融合后的特征图列表
        """
        # 确保输入特征数量正确
        assert len(features) == len(self.channels_list), \
            f"Expected {len(self.channels_list)} features, got {len(features)}"
        
        # 1. 提取语义和细节特征
        semantic_features = []
        detail_features = []
        
        for i, feat in enumerate(features):
            # 语义特征提取
            semantic = self.semantic_branch[i](feat)
            semantic_features.append(semantic)
            
            # 细节特征提取
            detail = self.detail_branch[i](feat)
            detail_features.append(detail)
        
        # 2. 跨尺度特征传播
        enhanced_features = []
        
        for i in range(len(features)):
            # 当前层的语义和细节特征
            current_semantic = semantic_features[i]
            current_detail = detail_features[i]
            
            # 从其他尺度获取上下文信息
            if i > 0:  # 从更高层获取语义信息
                high_semantic = F.interpolate(
                    semantic_features[i-1], 
                    size=current_semantic.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                current_semantic = current_semantic + high_semantic
            
            if i < len(features) - 1:  # 从更低层获取细节信息
                low_detail = F.adaptive_avg_pool2d(
                    detail_features[i+1], 
                    current_detail.shape[2:]
                )
                current_detail = current_detail + low_detail
            
            # 3. 语义和细节融合
            fused = torch.cat([current_semantic, current_detail], dim=1)
            fused = self.cross_scale_fusion[i](fused)
            
            # 4. 自适应权重调节
            attention = self.attention_weights[i](fused)
            fused = fused * attention
            
            enhanced_features.append(fused)
        
        return enhanced_features

class FSDI_Neck(nn.Module):
    """
    基于FSDI的颈部网络
    替换标准PANet结构
    """
    def __init__(self, in_channels=[256, 512, 1024], out_channels=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # FSDI融合模块
        self.fsdi = FSDI(in_channels, out_channels)
        
        # 输出调整层
        self.output_convs = nn.ModuleList()
        for _ in range(len(in_channels)):
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
    
    def forward(self, features):
        """
        Args:
            features: 来自骨干网络的特征图 [P3, P4, P5]
        Returns:
            增强后的特征图列表
        """
        # FSDI特征融合
        enhanced_features = self.fsdi(features)
        
        # 输出调整
        outputs = []
        for i, feat in enumerate(enhanced_features):
            output = self.output_convs[i](feat)
            outputs.append(output)
        
        return outputs

# 测试函数
if __name__ == "__main__":
    print("测试FSDI模块...")
    
    # 创建测试输入 (模拟不同尺度的特征图)
    batch_size = 2
    features = [
        torch.randn(batch_size, 256, 80, 80),   # P3
        torch.randn(batch_size, 512, 40, 40),   # P4  
        torch.randn(batch_size, 1024, 20, 20),  # P5
    ]
    
    # 测试FSDI模块
    fsdi = FSDI([256, 512, 1024], fusion_channels=256)
    enhanced_features = fsdi(features)
    
    print("FSDI增强特征:")
    for i, feat in enumerate(enhanced_features):
        print(f"P{i+3}: {feat.shape}")
    
    # 测试FSDI Neck
    fsdi_neck = FSDI_Neck([256, 512, 1024], 256)
    neck_outputs = fsdi_neck(features)
    
    print("\nFSDI Neck输出:")
    for i, output in enumerate(neck_outputs):
        print(f"Output P{i+3}: {output.shape}")
    
    # 参数量统计
    fsdi_params = sum(p.numel() for p in fsdi.parameters())
    neck_params = sum(p.numel() for p in fsdi_neck.parameters())
    
    print(f"\n参数量统计:")
    print(f"FSDI模块: {fsdi_params:,} 参数")
    print(f"FSDI Neck: {neck_params:,} 参数")
    print("FSDI模块测试完成! ✅") 