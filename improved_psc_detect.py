import torch
import torch.nn as nn
import math
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.modules.conv import Conv, DWConv


class ImprovedPSCDetect(Detect):
    """
    改进的PSC-Head：部分参数共享检测头
    
    在共享参数和尺度特异性之间取得平衡：
    1. 前两层卷积共享参数（提取通用特征）
    2. 最后一层卷积独立（尺度特异性）
    3. 所有BatchNorm层独立
    """
    
    def __init__(self, nc=80, ch=()):
        """初始化改进的PSC检测头"""
        super().__init__(nc, ch)
        
        # 计算统一的中间通道数
        c_unified = int(sum(ch) / len(ch))  # 使用平均值
        c2 = max((16, c_unified // 4, self.reg_max * 4))
        c3 = max(c_unified, min(self.nc, 100))
        
        # 输入适配层（独立）
        self.input_adapters_reg = nn.ModuleList([
            Conv(ch_i, c_unified, 1, act=False) for ch_i in ch
        ])
        self.input_adapters_cls = nn.ModuleList([
            Conv(ch_i, c_unified, 1, act=False) for ch_i in ch
        ])
        
        # 共享的前两层卷积（不包含BN和激活）
        self.shared_conv_reg = nn.ModuleList([
            nn.Conv2d(c_unified, c2, 3, 1, 1, bias=False),  # 第一层
            nn.Conv2d(c2, c2, 3, 1, 1, bias=False),         # 第二层
        ])
        
        self.shared_conv_cls = nn.ModuleList([
            nn.Conv2d(c_unified, c3, 3, 1, 1, bias=False),  # 第一层
            nn.Conv2d(c3, c3, 3, 1, 1, bias=False),         # 第二层
        ])
        
        # 独立的最后一层（保持尺度特异性）
        self.final_conv_reg = nn.ModuleList([
            nn.Conv2d(c2, 4 * self.reg_max, 1, bias=True) for _ in ch
        ])
        self.final_conv_cls = nn.ModuleList([
            nn.Conv2d(c3, self.nc, 1, bias=True) for _ in ch
        ])
        
        # 独立的BatchNorm层
        self.bn_reg = nn.ModuleList([
            nn.ModuleList([
                nn.BatchNorm2d(c2),  # 第一层BN
                nn.BatchNorm2d(c2),  # 第二层BN
            ]) for _ in range(self.nl)
        ])
        
        self.bn_cls = nn.ModuleList([
            nn.ModuleList([
                nn.BatchNorm2d(c3),  # 第一层BN
                nn.BatchNorm2d(c3),  # 第二层BN
            ]) for _ in range(self.nl)
        ])
        
        # 激活函数
        self.act = nn.SiLU()
        
        # 清空原始的cv2和cv3
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()
        
    def forward(self, x):
        """前向传播"""
        for i in range(self.nl):
            # 输入适配
            x_reg = self.input_adapters_reg[i](x[i])
            x_cls = self.input_adapters_cls[i](x[i])
            
            # 回归分支
            x_reg = self.act(x_reg)
            x_reg = self.act(self.bn_reg[i][0](self.shared_conv_reg[0](x_reg)))
            x_reg = self.act(self.bn_reg[i][1](self.shared_conv_reg[1](x_reg)))
            x_reg = self.final_conv_reg[i](x_reg)
            
            # 分类分支
            x_cls = self.act(x_cls)
            x_cls = self.act(self.bn_cls[i][0](self.shared_conv_cls[0](x_cls)))
            x_cls = self.act(self.bn_cls[i][1](self.shared_conv_cls[1](x_cls)))
            x_cls = self.final_conv_cls[i](x_cls)
            
            # 拼接
            x[i] = torch.cat([x_reg, x_cls], 1)
            
        if self.training:
            return x
        
        # 推理时的后处理
        return self._inference(x)
        
    def bias_init(self):
        """初始化偏置"""
        # 初始化回归分支偏置
        for conv in self.final_conv_reg:
            conv.bias.data[:] = 1.0
            
        # 初始化分类分支偏置
        for i, (conv, s) in enumerate(zip(self.final_conv_cls, self.stride)):
            conv.bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)