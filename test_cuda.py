#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA诊断脚本
用于检查PyTorch的CUDA支持情况
"""

import torch
import sys
import os

def test_cuda():
    """测试CUDA支持情况"""
    print("=" * 60)
    print("🔍 CUDA诊断报告")
    print("=" * 60)
    
    # 1. 基本信息
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"PyTorch构建信息: {torch.version.cuda}")
    
    # 2. CUDA可用性
    print(f"\n📊 CUDA支持情况:")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.current_device()}")
        
        # 3. GPU信息
        print(f"\n🖥️ GPU详细信息:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            print(f"  计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
        
        # 4. 测试CUDA张量
        print(f"\n🧪 CUDA张量测试:")
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = torch.mm(x, y)
            print("✅ CUDA张量运算测试通过")
            print(f"   结果形状: {z.shape}")
            print(f"   设备: {z.device}")
        except Exception as e:
            print(f"❌ CUDA张量运算测试失败: {e}")
        
        # 5. 内存信息
        print(f"\n💾 GPU内存信息:")
        print(f"已分配: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"已缓存: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        print(f"总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f} MB")
        
    else:
        print("❌ CUDA不可用")
        
        # 检查可能的原因
        print(f"\n🔍 可能的原因:")
        print("1. PyTorch安装的是CPU版本")
        print("2. CUDA驱动程序版本不匹配")
        print("3. 环境变量设置问题")
        
        # 检查环境变量
        print(f"\n🌍 环境变量检查:")
        cuda_home = os.environ.get('CUDA_HOME')
        cuda_path = os.environ.get('CUDA_PATH')
        print(f"CUDA_HOME: {cuda_home}")
        print(f"CUDA_PATH: {cuda_path}")
    
    # 6. 设备选择测试
    print(f"\n🎯 设备选择测试:")
    try:
        # 测试auto设备选择
        device = torch.device('auto' if torch.cuda.is_available() else 'cpu')
        print(f"自动设备选择: {device}")
        
        # 测试CUDA设备选择
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print(f"CUDA设备选择: {device}")
            
            # 测试张量到CUDA
            x = torch.randn(2, 2)
            x_cuda = x.to(device)
            print(f"张量设备转换: {x.device} -> {x_cuda.device}")
            
    except Exception as e:
        print(f"❌ 设备选择测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)

if __name__ == '__main__':
    test_cuda() 