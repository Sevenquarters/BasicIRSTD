#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ResUNet vs ResUNet-GCA 对比实验
用于验证 GCA 模块在红外小目标检测中的效果
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def analyze_models():
    from model.ResUNet.model_ResUNet import ResUNet
    from model.ResUNet.model_ResUNet_GCA import ResUNet_GCA

    print_header("📊 ResUNet / ResUNet-GCA 结构对比分析")

    baseline = ResUNet(num_classes=1, input_channels=1)
    improved = ResUNet_GCA(num_classes=1, input_channels=1)

    baseline_params = sum(p.numel() for p in baseline.parameters())
    improved_params = sum(p.numel() for p in improved.parameters())

    print("模型              | 参数数量        | 参数量(M)  | 相对增长")
    print("-" * 66)
    print(f"ResUNet           | {baseline_params:14,} | {baseline_params / 1e6:9.2f} | 基线")
    print(f"ResUNet-GCA       | {improved_params:14,} | {improved_params / 1e6:9.2f} | +{(improved_params / baseline_params - 1) * 100:.1f}%")
    print(f"\nGCA 模块参数增量: {improved_params - baseline_params:,}")

    x = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        y_baseline = baseline(x)
        y_improved = improved(x)

    print("\n前向传播检查:")
    print(f"✓ ResUNet 输出: {tuple(y_baseline.shape)}")
    print(f"✓ ResUNet-GCA 输出: {tuple(y_improved.shape)}")

    print("\nGCA 连接位置:")
    print("• gca_2_0: 作用于 x2_0 (H/4, W/4) 后再与 up(x3_0) 融合")
    print("• gca_1_0: 作用于 x1_0 (H/2, W/2) 后再与 up(x2_1) 融合")
    print("• gca_0_0: 作用于 x0_0 (H, W) 后再与 up(x1_2) 融合")


def test_inference_speed():
    from model.ResUNet.model_ResUNet import ResUNet
    from model.ResUNet.model_ResUNet_GCA import ResUNet_GCA

    print_header("⚡ 推理速度测试 (CPU)")

    models = {
        "ResUNet": ResUNet(num_classes=1, input_channels=1).eval(),
        "ResUNet-GCA": ResUNet_GCA(num_classes=1, input_channels=1).eval(),
    }
    x = torch.randn(4, 1, 256, 256)

    for name, model in models.items():
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)

            start = time.time()
            for _ in range(20):
                _ = model(x)
            elapsed = time.time() - start

        avg_ms = elapsed / 20 * 1000
        throughput = 4 / (elapsed / 20)
        print(f"{name:16} | 平均推理时间: {avg_ms:7.2f}ms | 吞吐量: {throughput:6.1f} img/s")


def create_guide():
    print_header("📋 GCA 对比实验命令")
    print(
        """# 1) 快速验证模型
python test_resunet_gca.py

# 2) 基线训练
python train.py --model_names ResUNet --dataset_names NUDT-SIRST --batchSize 8 --nEpochs 20

# 3) GCA 训练
python train.py --model_names ResUNet-GCA --dataset_names NUDT-SIRST --batchSize 8 --nEpochs 20

# 4) 对比日志
# log/NUDT-SIRST_ResUNet_*.txt
# log/NUDT-SIRST_ResUNet-GCA_*.txt
"""
    )


def main():
    parser = argparse.ArgumentParser(description="ResUNet-GCA 对比分析")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--speed", action="store_true")
    parser.add_argument("--guide", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        args.analyze = True
        args.speed = True
        args.guide = True
    if not any([args.analyze, args.speed, args.guide]):
        args.analyze = True
        args.guide = True

    if args.analyze:
        analyze_models()
    if args.speed:
        test_inference_speed()
    if args.guide:
        create_guide()

    print_header("✅ 分析完成")


if __name__ == "__main__":
    main()

