#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
针对 NUDT 数据集全自动筛选并生成 ResUNet 与 ResUNet-GCA 的 2x3 特征/预测硬对比热力图。
若两模型检测结果无差异，则自动跳过；若有显著差异，则保存差异图。

运行示例：
python visualize_diff_nudt.py \
  --img_dir ./datasets/NUDT-SIRST/images \
  --mask_dir ./datasets/NUDT-SIRST/masks \
  --dataset NUDT-SIRST \
  --resunet_ckpt ./log/NUDT-SIRST/ResUNet_best.pth.tar \
  --gca_ckpt ./log/NUDT-SIRST/ResUNet-GCA_best.pth.tar \
  --out_dir ./plots/NUDT_Diff_Results
"""

import argparse
import os
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches

from utils import Normalized, get_img_norm_cfg, PadImg
from model.ResUNet.model_ResUNet import ResUNet
from model.ResUNet.model_ResUNet_GCA import ResUNet_GCA


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cleaned = {}
    for key, value in state_dict.items():
        k = key
        while True:
            changed = False
            for prefix in ("module.", "model."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    changed = True
            if not changed:
                break
        cleaned[k] = value
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model


def load_image(img_path: str, dataset_name: str, dataset_dir: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
    img = Image.open(img_path).convert("I")
    img_np = np.array(img, dtype=np.float32)
    img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
    img_np = Normalized(img_np, img_norm_cfg)
    h, w = img_np.shape
    img_np = PadImg(img_np)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_np)).unsqueeze(0).unsqueeze(0)
    return img_tensor, (h, w)


def load_mask(mask_path: str, target_shape: Tuple[int, int]) -> np.ndarray:
    if not mask_path or not os.path.exists(mask_path):
        return np.zeros(target_shape, dtype=np.float32)
    mask = Image.open(mask_path)
    mask_np = np.array(mask, dtype=np.float32) / 255.0
    if mask_np.ndim > 2:
        mask_np = mask_np[:, :, 0]
    mask_np = PadImg(mask_np)
    return mask_np[: target_shape[0], : target_shape[1]]


def crop_tensor(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return x[:, :, :h, :w]


def normalize_map(x: np.ndarray) -> np.ndarray:
    x_min = float(x.min())
    x_max = float(x.max())
    return (x - x_min) / (x_max - x_min + 1e-6)


def channel_reduce(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "max":
        return torch.max(x, dim=1, keepdim=True).values
    return torch.mean(x, dim=1, keepdim=True)


def main():
    parser = argparse.ArgumentParser(description="Auto-find differences and plot Heatmaps for NUDT dataset")
    parser.add_argument("--img_dir", required=True, help="Directory of NUDT input images")
    parser.add_argument("--mask_dir", required=True, help="Directory of NUDT Ground-Truth masks")
    parser.add_argument("--dataset", default="NUDT-SIRST", help="Dataset name for normalization profile")
    parser.add_argument("--dataset_dir", default="./datasets", help="Dataset root path")
    parser.add_argument("--resunet_ckpt", required=True, help="ResUNet baseline checkpoint")
    parser.add_argument("--gca_ckpt", required=True, help="ResUNet-GCA checkpoint")
    parser.add_argument("--out_dir", default="./plots/NUDT_Diff_Results", help="Output directory for diff cases")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--feature_mode", choices=["mean", "max"], default="mean")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for二值化 mask difference")
    parser.add_argument("--min_diff_pixels", type=int, default=15, help="Threshold of different pixels to trigger plot")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. 静态载入两部核心模型
    resunet = load_checkpoint(ResUNet(num_classes=1, input_channels=1), args.resunet_ckpt, device)
    gca = load_checkpoint(ResUNet_GCA(num_classes=1, input_channels=1), args.gca_ckpt, device)

    # 2. 获取测试文件夹内的全部红外图像列表
    img_names = [f for f in os.listdir(args.img_dir) if f.endswith(('.png', '.jpg', '.bmp'))]
    print(f"开始遍历数据集，共计发现 {len(img_names)} 张样本。开始挖掘硬解耦差异样本...")

    for idx, img_name in enumerate(img_names):
        img_path = os.path.join(args.img_dir, img_name)
        mask_name = img_name  # 保持文件名对齐
        mask_path = os.path.join(args.mask_dir, mask_name)

        if not os.path.exists(mask_path):
            continue

        # 载入单张图像与GT
        img_tensor, (h, w) = load_image(img_path, args.dataset, args.dataset_dir)
        gt_mask = load_mask(mask_path, (h, w))
        img_tensor = img_tensor.to(device)

        # 3. 挂载动态中间特征拦截钩子 (Hook)
        features = {}
        hooks = []
        hooks.append(resunet.conv2_0.register_forward_hook(lambda m, i, o: features.update({"res_conv2_0": o})))
        hooks.append(gca.gca_2_0.register_forward_hook(lambda m, i, o: features.update({"gca_2_0": o})))

        # 前向传播推理
        with torch.no_grad():
            pred_res = resunet(img_tensor)
            pred_gca = gca(img_tensor)

        # 释放钩子防止内存溢出
        for hdl in hooks:
            hdl.remove()

        # 4. 二值化预测掩膜判定
        pred_res_np = crop_tensor(pred_res, h, w).squeeze().detach().cpu().numpy()
        pred_gca_np = crop_tensor(pred_gca, h, w).squeeze().detach().cpu().numpy()

        bin_res = (pred_res_np >= args.threshold).astype(np.uint8)
        bin_gca = (pred_gca_np >= args.threshold).astype(np.uint8)

        # 核心算法：对点像素绝对差值计算，自动揪出差异样本
        diff_mask = np.abs(bin_res - bin_gca)
        diff_pixels = int(np.sum(diff_mask > 0))

        # 若两模型结果高度一致，说明属于普通无争议工况，为了高效科研，直接跳过不画图
        if diff_pixels < args.min_diff_pixels:
            continue

        print(f"发现极具分析价值的黄金对比工况! 图像名: {img_name}, 差异像素点个数: {diff_pixels}")

        # 5. 提取真正具备学术烈焰响应特征的热力图
        res_feat = features["res_conv2_0"]
        gca_feat = features["gca_2_0"][0] if isinstance(features["gca_2_0"], (tuple, list)) else features["gca_2_0"]

        res_map = channel_reduce(res_feat, args.feature_mode)
        gca_map = channel_reduce(gca_feat, args.feature_mode)

        res_map = F.interpolate(res_map, size=(h, w), mode="bilinear", align_corners=False)
        gca_map = F.interpolate(gca_map, size=(h, w), mode="bilinear", align_corners=False)

        res_map_np = normalize_map(res_map.squeeze().detach().cpu().numpy())
        gca_map_np = normalize_map(gca_map.squeeze().detach().cpu().numpy())

        # 6. 优雅绘制精简版 2x3 六宫格学术看板
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax in axes.flatten():
            ax.axis("off")

        # 第一行：Input 与 内部特征图对比 (展现 GCA 如何在特征层净化树枝/线条噪声)
        input_img = normalize_map(crop_tensor(img_tensor, h, w).squeeze().detach().cpu().numpy())
        axes[0, 0].imshow(input_img, cmap="gray")
        axes[0, 0].set_title(f"Input ({img_name})", fontsize=11, weight="bold")
        
        axes[0, 1].imshow(res_map_np, cmap="jet")
        axes[0, 1].set_title("Baseline (ResUNet) Feature Map", fontsize=11, weight="bold")
        
        axes[0, 2].imshow(gca_map_np, cmap="jet")
        axes[0, 2].set_title("ResUNet-GCA Feature Map", fontsize=11, weight="bold")

        # 第二行：Ground Truth 与 最终 Mask 预测二值化对比 (展现 GCA 在最终掩膜阶段如何消灭虚警/漏检)
        axes[1, 0].imshow(gt_mask, cmap="gray")
        axes[1, 0].set_title("Ground Truth Mask", fontsize=11, weight="bold")
        
        axes[1, 1].imshow(pred_res_np, cmap="jet")
        axes[1, 1].set_title("Baseline (ResUNet) Prediction", fontsize=11, weight="bold")
        
        axes[1, 2].imshow(pred_gca_np, cmap="jet")
        axes[1, 2].set_title("ResUNet-GCA Prediction", fontsize=11, weight="bold")

        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f"vis_compare_{os.path.splitext(img_name)[0]}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"成功导出硬核差异分析图: {out_path}")

    print("🎉 全集差异挖掘和热力图绘制大满贯结束！请前往 plots/NUDT_Diff_Results 查看精选黄金样本。")


if __name__ == "__main__":
    main()