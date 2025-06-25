"""
MB-FPN模块 - 多分支特征金字塔网络 (Multi-Branch Feature Pyramid Network)
基于MS-YOLO的设计，有效结合不同分辨率的特征信息
目标：解决上采样过程中小目标容易产生错误信息的问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiBranchConv(nn.Module):
    """
    多分支卷积模块
    使用不同内核大小的卷积来捕获多尺度特征
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5]):
        super().__init__()
        self.branches = nn.ModuleList()
        
        # 每个分支使用不同的卷积核
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2
            branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // len(kernel_sizes), 
                         kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels // len(kernel_sizes)),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 并行处理各分支
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # 拼接所有分支输出
        concat_output = torch.cat(branch_outputs, dim=1)
        
        # 特征融合
        output = self.fusion(concat_output)
        return output

class AdaptiveUpsampling(nn.Module):
    """
    自适应上采样模块
    根据特征内容自动调整上采样策略
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        
        # 上采样路径选择
        self.path_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 2, 1),
            nn.Softmax(dim=1)
        )
        
        # 双线性插值分支
        self.bilinear_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 反卷积分支
        self.deconv_branch = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 
                              stride=scale_factor, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 获取路径权重
        weights = self.path_selector(x)  # [B, 2, 1, 1]
        w1, w2 = weights[:, 0:1], weights[:, 1:2]
        
        # 双线性插值分支
        bilinear_out = F.interpolate(x, scale_factor=self.scale_factor, 
                                    mode='bilinear', align_corners=False)
        bilinear_out = self.bilinear_branch(bilinear_out)
        
        # 反卷积分支
        deconv_out = self.deconv_branch(x)
        
        # 加权融合
        output = w1 * bilinear_out + w2 * deconv_out
        return output

class MB_FPN(nn.Module):
    """
    多分支特征金字塔网络
    集成多分支卷积和自适应上采样，提高小目标检测性能
    """
    def __init__(self, in_channels=[256, 512, 1024], out_channels=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 侧边连接 - 降维
        self.lateral_convs = nn.ModuleList()
        for in_ch in in_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 多分支卷积层
        self.mb_convs = nn.ModuleList()
        for _ in range(len(in_channels)):
            self.mb_convs.append(
                MultiBranchConv(out_channels, out_channels)
            )
        
        # 自适应上采样层
        self.adaptive_upsample = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.adaptive_upsample.append(
                AdaptiveUpsampling(out_channels, out_channels)
            )
        
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
        
        # 下采样层用于底向上路径
        self.downsample_convs = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.downsample_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
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
        # 1. 侧边连接 - 通道统一
        laterals = []
        for i, (feat, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            laterals.append(lateral_conv(feat))
        
        # 2. 自顶向下路径 (Top-down pathway)
        top_down_features = []
        
        # 从最高层开始
        current = laterals[-1]  # P5
        top_down_features.append(current)
        
        # 逐层向下融合
        for i in range(len(laterals) - 2, -1, -1):  # P4, P3
            # 自适应上采样
            upsampled = self.adaptive_upsample[i](current)
            
            # 与当前层特征融合
            current = laterals[i] + upsampled
            
            # 多分支卷积增强
            current = self.mb_convs[i](current)
            
            top_down_features.append(current)
        
        # 反转列表，使其顺序为 [P3, P4, P5]
        top_down_features = top_down_features[::-1]
        
        # 3. 自底向上路径 (Bottom-up pathway)
        bottom_up_features = []
        
        # 从最低层开始
        current = top_down_features[0]  # P3
        bottom_up_features.append(current)
        
        # 逐层向上融合
        for i in range(1, len(top_down_features)):
            # 下采样当前特征
            downsampled = self.downsample_convs[i-1](current)
            
            # 与对应层特征融合
            current = top_down_features[i] + downsampled
            
            # 多分支卷积增强
            current = self.mb_convs[i](current)
            
            bottom_up_features.append(current)
        
        # 4. 输出调整
        outputs = []
        for i, feat in enumerate(bottom_up_features):
            output = self.output_convs[i](feat)
            outputs.append(output)
        
        return outputs

class SmallObjectEnhancer(nn.Module):
    """
    小目标增强模块
    专门用于增强小目标的特征表示
    """
    def __init__(self, channels, enhancement_factor=2):
        super().__init__()
        self.enhancement_factor = enhancement_factor
        
        # 小目标检测分支
        self.small_object_branch = nn.Sequential(
            nn.Conv2d(channels, channels * enhancement_factor, 1, bias=False),
            nn.BatchNorm2d(channels * enhancement_factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * enhancement_factor, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意力权重
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 小目标特征增强
        enhanced = self.small_object_branch(x)
        
        # 注意力调节
        attention_weights = self.attention(enhanced)
        enhanced = enhanced * attention_weights
        
        # 残差连接
        output = x + enhanced
        return output

# 测试函数
if __name__ == "__main__":
    print("测试MB-FPN模块...")
    
    # 创建测试输入
    batch_size = 2
    features = [
        torch.randn(batch_size, 256, 80, 80),   # P3
        torch.randn(batch_size, 512, 40, 40),   # P4
        torch.randn(batch_size, 1024, 20, 20),  # P5
    ]
    
    # 测试多分支卷积
    mb_conv = MultiBranchConv(256, 256)
    mb_output = mb_conv(features[0])
    print(f"多分支卷积输出: {mb_output.shape}")
    
    # 测试自适应上采样
    adaptive_up = AdaptiveUpsampling(512, 256)
    up_output = adaptive_up(features[1])
    print(f"自适应上采样输出: {up_output.shape}")
    
    # 测试MB-FPN
    mb_fpn = MB_FPN([256, 512, 1024], 256)
    fpn_outputs = mb_fpn(features)
    
    print("\nMB-FPN输出:")
    for i, output in enumerate(fpn_outputs):
        print(f"P{i+3}: {output.shape}")
    
    # 测试小目标增强器
    enhancer = SmallObjectEnhancer(256)
    enhanced_output = enhancer(fpn_outputs[0])
    print(f"\n小目标增强输出: {enhanced_output.shape}")
    
    # 参数量统计
    mb_fpn_params = sum(p.numel() for p in mb_fpn.parameters())
    enhancer_params = sum(p.numel() for p in enhancer.parameters())
    
    print(f"\n参数量统计:")
    print(f"MB-FPN: {mb_fpn_params:,} 参数")
    print(f"小目标增强器: {enhancer_params:,} 参数")
    print("MB-FPN模块测试完成! ✅") 