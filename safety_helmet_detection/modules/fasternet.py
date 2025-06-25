import torch
import torch.nn as nn
import torch.nn.functional as F

class PConv(nn.Module):
    def __init__(self, channels, ratio=0.25, kernel_size=3):
        super().__init__()
        self.partial_channels = max(1, int(channels * ratio))
        self.conv = nn.Conv2d(
            self.partial_channels, 
            self.partial_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            groups=self.partial_channels,
            bias=False
        )
        self.bn = nn.BatchNorm2d(self.partial_channels)
        
    def forward(self, x):
        x1, x2 = torch.split(x, [self.partial_channels, x.size(1) - self.partial_channels], dim=1)
        x1 = self.conv(x1)
        x1 = self.bn(x1)
        return torch.cat([x1, x2], dim=1)

class FasterNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2, drop_path=0.):
        super().__init__()
        hidden_channels = int(in_channels * expand_ratio)
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.act1 = nn.ReLU(inplace=True)
        
        self.pconv = PConv(hidden_channels, ratio=0.25)
        
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        shortcut = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.pconv(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)
        
        x = self.drop_path(x) + shortcut
        return x

class C2f_Fast(nn.Module):
    """
    轻量化C2f模块
    使用FasterNet Block替换标准的Bottleneck
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        
        # 输入卷积
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, bias=False)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1, bias=False)
        
        # FasterNet Blocks
        self.blocks = nn.ModuleList([
            FasterNetBlock(self.c, self.c, expand_ratio=2)
            for _ in range(n)
        ])
        
    def forward(self, x):
        """Forward pass through C2f_Fast module."""
        y = list(self.cv1(x).chunk(2, 1))
        
        for block in self.blocks:
            y.append(block(y[-1]))
        
        return self.cv2(torch.cat(y, 1))

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

def count_flops_params(model, input_size=(1, 3, 640, 640)):
    """
    计算模型的FLOPs和参数量
    """
    from thop import profile
    input_tensor = torch.randn(input_size)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    return flops, params

# 测试函数
if __name__ == "__main__":
    # 测试FasterNet Block
    print("测试FasterNet模块...")
    
    # 创建标准C2f和轻量化C2f_Fast进行对比
    input_tensor = torch.randn(1, 256, 80, 80)
    
    # FasterNet Block测试
    faster_block = FasterNetBlock(256, 256)
    output = faster_block(input_tensor)
    print(f"FasterNet Block - Input: {input_tensor.shape}, Output: {output.shape}")
    
    # C2f_Fast测试
    c2f_fast = C2f_Fast(256, 256, n=3)
    output_fast = c2f_fast(input_tensor)
    print(f"C2f_Fast - Input: {input_tensor.shape}, Output: {output_fast.shape}")
    
    # 参数量对比
    faster_params = sum(p.numel() for p in faster_block.parameters())
    c2f_params = sum(p.numel() for p in c2f_fast.parameters())
    
    print(f"\n参数量对比:")
    print(f"FasterNet Block: {faster_params:,} 参数")
    print(f"C2f_Fast: {c2f_params:,} 参数")
    print(f"轻量化改进完成! ✅") 