#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ResUNet vs ResUNet-CBAM 对比实验
用于验证注意力机制的有效性
"""

import os
import sys
import torch
import argparse
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def analyze_models():
    """分析和对比两个模型的结构与参数"""
    from model.ResUNet.model_ResUNet import ResUNet
    from model.ResUNet.model_ResUNet_CBAM import ResUNet_CBAM
    
    print_header("📊 模型结构对比分析")
    
    print("创建模型...")
    baseline = ResUNet(num_classes=1, input_channels=1)
    improved = ResUNet_CBAM(num_classes=1, input_channels=1)
    
    baseline_params = sum(p.numel() for p in baseline.parameters())
    improved_params = sum(p.numel() for p in improved.parameters())
    
    print(f"\n模型        | 参数数量        | 参数量(M)  | 相对增长")
    print("-" * 60)
    print(f"ResUNet     | {baseline_params:14,} | {baseline_params/1e6:9.2f} | 基线")
    print(f"ResUNet-CBAM| {improved_params:14,} | {improved_params/1e6:9.2f} | +{(improved_params/baseline_params-1)*100:.1f}%")
    
    print(f"\nCBAM 模块增加的参数: {improved_params - baseline_params:,} ({(improved_params/baseline_params-1)*100:.1f}%)")
    
    # 测试前向传播
    print("\n测试前向传播...")
    x = torch.randn(1, 1, 256, 256)
    
    with torch.no_grad():
        y_baseline = baseline(x)
        y_improved = improved(x)
    
    print(f"✓ ResUNet 输出: {y_baseline.shape}")
    print(f"✓ ResUNet-CBAM 输出: {y_improved.shape}")
    
    # 计算计算量 (理论)
    print("\n关键点:")
    print("• CBAM 模块在解码器的 3 个层级添加")
    print("• 通道注意力: 使用 FC 层学习权重")
    print("• 空间注意力: 使用 7×7 卷积学习空间权重")
    print("• 两部分串联: 先通道后空间")
    
    return baseline, improved

def test_model_inference_speed():
    """简单的推理速度测试"""
    from model.ResUNet.model_ResUNet import ResUNet
    from model.ResUNet.model_ResUNet_CBAM import ResUNet_CBAM
    import time
    
    print_header("⚡ 推理速度测试 (CPU)")
    
    models = {
        'ResUNet': ResUNet(num_classes=1, input_channels=1),
        'ResUNet-CBAM': ResUNet_CBAM(num_classes=1, input_channels=1)
    }
    
    x = torch.randn(4, 1, 256, 256)
    
    for name, model in models.items():
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)
        
        # 计时
        with torch.no_grad():
            start = time.time()
            for _ in range(20):
                _ = model(x)
            elapsed = time.time() - start
        
        avg_time = elapsed / 20 * 1000  # 毫秒
        throughput = 4 / (elapsed / 20)  # 图片/秒
        
        print(f"{name:20} | 平均推理时间: {avg_time:6.2f}ms | 吞吐量: {throughput:6.1f} img/s")

def create_comparison_report():
    """生成对比实验报告模板"""
    print_header("📋 对比实验操作指南")
    
    report = """
【实验目标】
验证 CBAM 注意力机制在红外小目标检测中的有效性

【实验设计】
- 数据集: NUDT-SIRST (1,327 张 256×256 红外图像)
- 基线: ResUNet (无注意力)
- 改进: ResUNet-CBAM (通道+空间注意力)
- 其他配置: 保持一致 (优化器、学习率、数据增强、epoch等)

【关键指标】
- mIoU (平均交集并集): 越高越好，目标 +1-3%
- PD (检测率/召回): 越高越好，目标 +2-5%  
- FA (误报率): 越低越好，目标 -1-2%

【训练命令】

# 1️⃣ 验证模型可用
python test_resunet_cbam.py

# 2️⃣ 基线训练 (ResUNet)
python train.py \\
  --model_names ResUNet \\
  --dataset_names NUDT-SIRST \\
  --batchSize 8 \\
  --nEpochs 20

# 3️⃣ 改进训练 (ResUNet-CBAM)
python train.py \\
  --model_names ResUNet-CBAM \\
  --dataset_names NUDT-SIRST \\
  --batchSize 8 \\
  --nEpochs 20

# 4️⃣ 查看结果
# 基线日志: log/NUDT-SIRST_ResUNet_*.txt
# 改进日志: log/NUDT-SIRST_ResUNet-CBAM_*.txt

【性能对比示例】
┌────────────────┬──────────┬──────────┬──────────┐
│    模型        │  mIoU   │   PD     │   FA     │
├────────────────┼──────────┼──────────┼──────────┤
│ ResUNet        │  0.0226  │ 0.8148   │ 0.0201   │
│ ResUNet-CBAM   │  0.0235  │ 0.8320   │ 0.0185   │
│ 改进幅度       │ +3.8%    │ +2.1%    │ -8.0%    │
└────────────────┴──────────┴──────────┴──────────┘

【成功标准】
✓ 至少一个指标有显著提升 (> 1%)
✓ 不出现性能下降
✓ 运行时间增长 < 10%

【如果没有提升怎么办?】
1. 检查模型是否正确加载 (test_resunet_cbam.py)
2. 增加训练 epoch (如 50 或 100)
3. 调整学习率或优化器
4. 查看训练曲线 (log 文件) 是否正常收敛

【下一步方向】
✓ 实验 1 成功 → 进入第二阶段: 添加多尺度 MAEB 模块
✓ 需要更多改进 → 调整 CBAM 参数 (reduction_ratio, kernel_size)
✓ 想要特别优化 → 在某些层添加额外的注意力机制
"""
    
    print(report)
    
    # 保存为文件
    report_file = 'EXPERIMENT_GUIDE.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✓ 实验指南已保存到: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='ResUNet-CBAM 模型分析与对比')
    parser.add_argument('--analyze', action='store_true', help='执行模型分析')
    parser.add_argument('--speed', action='store_true', help='执行速度测试')
    parser.add_argument('--guide', action='store_true', help='生成实验指南')
    parser.add_argument('--all', action='store_true', help='执行所有测试')
    
    args = parser.parse_args()
    
    if args.all:
        args.analyze = True
        args.speed = True
        args.guide = True
    
    if not any([args.analyze, args.speed, args.guide]):
        args.analyze = True
        args.guide = True
    
    try:
        if args.analyze:
            analyze_models()
        
        if args.speed:
            test_model_inference_speed()
        
        if args.guide:
            create_comparison_report()
        
        print_header("✅ 所有分析完成")
        print("下一步: 根据上面的指南，运行训练命令进行对比实验\n")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
