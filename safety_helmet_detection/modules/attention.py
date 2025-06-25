import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class A2_Attention(nn.Module):
    def __init__(self, channels, num_heads=8, area_size=7):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.area_size = area_size
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        
        self.pos_embedding = nn.Parameter(
            torch.randn(area_size * area_size, channels)
        )
        
        self.area_pool = nn.AdaptiveAvgPool2d(area_size)
        self.area_unpool = nn.Upsample(scale_factor=area_size, mode='nearest')
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        pooled = self.area_pool(x)
        pooled_flat = pooled.flatten(2).transpose(1, 2)
        pooled_flat = pooled_flat + self.pos_embedding.unsqueeze(0)
        
        Q = self.q_proj(pooled_flat)
        K = self.k_proj(pooled_flat)
        V = self.v_proj(pooled_flat)
        
        Q = Q.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).reshape(B, -1, C)
        attended = self.out_proj(attended)
        attended = self.norm(attended + pooled_flat)
        attended = attended.transpose(1, 2).reshape(B, C, self.area_size, self.area_size)
        attended = F.interpolate(attended, size=(H, W), mode='bilinear', align_corners=False)
        
        output = identity + attended
        return output

class PAM_Attention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        channel_att = self.channel_attention(x)
        channel_enhanced = x * channel_att
        
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        
        spatial_att = self.spatial_attention(spatial_input)
        spatial_enhanced = x * spatial_att
        
        x_flat = x.flatten(2).transpose(1, 2)
        self_att_out, _ = self.self_attention(x_flat, x_flat, x_flat)
        self_enhanced = self_att_out.transpose(1, 2).reshape(B, C, H, W)
        
        fused_features = torch.cat([
            channel_enhanced, 
            spatial_enhanced, 
            self_enhanced
        ], dim=1)
        
        fused = self.fusion_conv(fused_features)
        output = self.output_conv(fused)
        output = output + identity
        
        return output

class A2C2f(nn.Module):
    """
    集成A2注意力的C2f模块
    结合A2注意力机制和标准C2f结构
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        
        # 输入卷积
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, bias=False)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1, bias=False)
        
        # A2注意力增强的Bottleneck
        self.blocks = nn.ModuleList()
        for _ in range(n):
            self.blocks.append(
                nn.Sequential(
                    Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0),
                    A2_Attention(self.c, num_heads=8, area_size=7)
                )
            )
    
    def forward(self, x):
        """Forward pass through A2C2f module."""
        y = list(self.cv1(x).chunk(2, 1))
        
        for block in self.blocks:
            y.append(block(y[-1]))
        
        return self.cv2(torch.cat(y, 1))

class Bottleneck(nn.Module):
    """标准瓶颈块，用于A2C2f"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, k[0], 1, autopad(k[0]), bias=False)
        self.cv2 = nn.Conv2d(c_, c2, k[1], 1, autopad(k[1]), groups=g, bias=False)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

def autopad(k, p=None, d=1):
    """自动计算填充"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class HybridAttention(nn.Module):
    """
    混合注意力模块
    结合A2和PAM注意力的优势
    """
    def __init__(self, channels, area_size=7, reduction=16):
        super().__init__()
        
        # A2区域注意力
        self.a2_attention = A2_Attention(channels, num_heads=8, area_size=area_size)
        
        # PAM并行注意力
        self.pam_attention = PAM_Attention(channels, reduction=reduction)
        
        # 注意力融合权重
        self.fusion_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        
        # 输出调节
        self.output_norm = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            enhanced_x: [B, C, H, W]
        """
        # 计算两种注意力
        a2_out = self.a2_attention(x)
        pam_out = self.pam_attention(x)
        
        # 加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = weights[0] * a2_out + weights[1] * pam_out
        
        # 归一化
        output = self.output_norm(fused)
        
        return output

# 测试函数
if __name__ == "__main__":
    print("测试注意力机制模块...")
    
    # 创建测试输入
    batch_size = 2
    channels = 256
    height, width = 80, 80
    
    x = torch.randn(batch_size, channels, height, width)
    
    # 测试A2注意力
    print("测试A2注意力...")
    a2_attention = A2_Attention(channels, num_heads=8, area_size=7)
    a2_output = a2_attention(x)
    print(f"A2注意力输出: {a2_output.shape}")
    
    # 测试PAM注意力
    print("\n测试PAM注意力...")
    pam_attention = PAM_Attention(channels, reduction=16)
    pam_output = pam_attention(x)
    print(f"PAM注意力输出: {pam_output.shape}")
    
    # 测试混合注意力
    print("\n测试混合注意力...")
    hybrid_attention = HybridAttention(channels)
    hybrid_output = hybrid_attention(x)
    print(f"混合注意力输出: {hybrid_output.shape}")
    
    # 参数量统计
    a2_params = sum(p.numel() for p in a2_attention.parameters())
    pam_params = sum(p.numel() for p in pam_attention.parameters())
    hybrid_params = sum(p.numel() for p in hybrid_attention.parameters())
    
    print(f"\n参数量统计:")
    print(f"A2注意力: {a2_params:,} 参数")
    print(f"PAM注意力: {pam_params:,} 参数")
    print(f"混合注意力: {hybrid_params:,} 参数")
    print("注意力机制模块测试完成! ✅") 