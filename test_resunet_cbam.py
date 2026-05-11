#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 ResUNet-CBAM 模型
"""
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_resunet_cbam():
    print("=" * 60)
    print("测试 ResUNet-CBAM 模型")
    print("=" * 60)
    
    import torch
    from model.ResUNet.model_ResUNet_CBAM import ResUNet_CBAM
    
    print("\n1️⃣  创建模型...")
    model = ResUNet_CBAM(num_classes=1, input_channels=1)
    print("   ✓ ResUNet-CBAM 创建成功")
    
    print("\n2️⃣  统计模型参数...")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   ✓ 模型参数总数: {params:.2f}M")
    
    print("\n3️⃣  测试前向传播...")
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(f"   ✓ 输入形状: {x.shape}")
    print(f"   ✓ 输出形状: {y.shape}")
    assert y.shape == (1, 1, 256, 256), "输出形状不正确"
    
    print("\n4️⃣  测试批量处理...")
    x_batch = torch.randn(4, 1, 256, 256)
    y_batch = model(x_batch)
    print(f"   ✓ 批量输入: {x_batch.shape}")
    print(f"   ✓ 批量输出: {y_batch.shape}")
    assert y_batch.shape == (4, 1, 256, 256), "批量处理输出形状不正确"
    
    print("\n5️⃣  测试模型在 Net 中的注册...")
    from net import Net
    net = Net('ResUNet-CBAM', 'train')
    print("   ✓ ResUNet-CBAM 在 Net 中注册成功")
    
    print("\n6️⃣  测试 Net 的前向传播...")
    y_net = net(x)
    print(f"   ✓ Net 输出形状: {y_net.shape}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！ResUNet-CBAM 可以使用")
    print("=" * 60)
    print("\n快速开始训练:")
    print("  python train.py --model_names ResUNet-CBAM --dataset_names NUDT-SIRST --batchSize 8 --nEpochs 10")
    print()

if __name__ == '__main__':
    try:
        test_resunet_cbam()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
