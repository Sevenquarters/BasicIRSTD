#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize internal feature maps and predictions for ResUNet / ResUNet-GCA / ResUNet-GCA-DSPG.

Example:
python visualize_features.py ^
  --image ./datasets/IRSTD-1K/images/XDU109.png ^
  --gt ./datasets/IRSTD-1K/masks/XDU109.png ^
  --dataset IRSTD-1K ^
  --resunet_ckpt ./log/IRSTD-1K/ResUNet_best.pth.tar ^
  --gca_ckpt ./log/IRSTD-1K/ResUNet-GCA_best.pth.tar ^
  --dspg_ckpt ./log/IRSTD-1K/ResUNet-GCA-DSPG_best.pth.tar ^
  --out ./plots/vis_IRSTD1K.png
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
from model.ResUNet.model_ResUNet_GCA import ResUNet_GCA, ResUNet_GCA_DSPG


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


def parse_boxes(box_str: str) -> List[Tuple[int, int, int, int]]:
    if not box_str:
        return []
    boxes = []
    for item in box_str.split(";"):
        coords = [int(v) for v in item.split(",") if v.strip()]
        if len(coords) == 4:
            x1, y1, x2, y2 = coords
            boxes.append((x1, y1, x2, y2))
    return boxes


def find_component_boxes(mask: np.ndarray, min_area: int = 20, max_boxes: int = 1) -> List[Tuple[int, int, int, int]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            min_y = max_y = y
            min_x = max_x = x
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if area >= min_area:
                boxes.append((min_x, min_y, max_x, max_y, area))
    boxes.sort(key=lambda b: b[4], reverse=True)
    return [(b[0], b[1], b[2], b[3]) for b in boxes[:max_boxes]]


def add_boxes(ax, boxes: List[Tuple[int, int, int, int]], label: str = None):
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        rect = patches.Rectangle(
            (x1, y1),
            max(1, x2 - x1),
            max(1, y2 - y1),
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        if label and i == 0:
            ax.text(x1, max(0, y1 - 5), label, color="red", fontsize=9, weight="bold")


def main():
    parser = argparse.ArgumentParser(description="Visualize ResUNet/GCA/DSPG feature maps and predictions")
    parser.add_argument("--image", required=True, help="Input IR image path")
    parser.add_argument("--gt", default=None, help="Ground-truth mask path (optional)")
    parser.add_argument("--dataset", default="IRSTD-1K", help="Dataset name for normalization")
    parser.add_argument("--dataset_dir", default="./datasets", help="Dataset root")
    parser.add_argument("--resunet_ckpt", required=True, help="ResUNet checkpoint")
    parser.add_argument("--gca_ckpt", required=True, help="ResUNet-GCA checkpoint")
    parser.add_argument("--dspg_ckpt", required=True, help="ResUNet-GCA-DSPG checkpoint")
    parser.add_argument("--out", default="visualization.png", help="Output image path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--feature_mode", choices=["mean", "max"], default="mean")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for auto box generation")
    parser.add_argument("--fp_boxes", default="", help="Manual boxes for baseline FP (x1,y1,x2,y2;...)")
    parser.add_argument("--fn_boxes", default="", help="Manual boxes for baseline FN (x1,y1,x2,y2;...)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    img_tensor, (h, w) = load_image(args.image, args.dataset, args.dataset_dir)
    gt_mask = load_mask(args.gt, (h, w))

    img_tensor = img_tensor.to(device)

    # Load models
    resunet = load_checkpoint(ResUNet(num_classes=1, input_channels=1), args.resunet_ckpt, device)
    gca = load_checkpoint(ResUNet_GCA(num_classes=1, input_channels=1), args.gca_ckpt, device)
    dspg = load_checkpoint(ResUNet_GCA_DSPG(num_classes=1, input_channels=1), args.dspg_ckpt, device)

    features = {}
    hooks = []

    # ResUNet: conv2_0 output
    hooks.append(resunet.conv2_0.register_forward_hook(lambda m, i, o: features.update({"res_conv2_0": o})))

    # GCA: gca_2_0 output (tuple)
    hooks.append(gca.gca_2_0.register_forward_hook(lambda m, i, o: features.update({"gca_2_0": o})))

    # DSPG: gca_2_0 output and conv2_1 input (pre-hook)
    hooks.append(dspg.gca_2_0.register_forward_hook(lambda m, i, o: features.update({"dspg_gca_2_0": o})))

    def conv2_1_pre_hook(module, inputs):
        features["dspg_conv2_1_in"] = inputs[0]

    hooks.append(dspg.conv2_1.register_forward_pre_hook(conv2_1_pre_hook))

    with torch.no_grad():
        pred_res = resunet(img_tensor)
        pred_gca = gca(img_tensor)
        pred_dspg = dspg(img_tensor)

    for hdl in hooks:
        hdl.remove()

    # Feature maps
    res_feat = features["res_conv2_0"]
    gca_feat = features["gca_2_0"][0] if isinstance(features["gca_2_0"], (tuple, list)) else features["gca_2_0"]

    dspg_conv2_1_in = features["dspg_conv2_1_in"]
    dspg_gca_feat = features["dspg_gca_2_0"][0] if isinstance(features["dspg_gca_2_0"], (tuple, list)) else features["dspg_gca_2_0"]
    split_c = dspg_gca_feat.shape[1]
    dspg_prior_guided = dspg_conv2_1_in[:, split_c:, :, :]

    # Reduce channels and resize to input size
    res_map = channel_reduce(res_feat, args.feature_mode)
    gca_map = channel_reduce(gca_feat, args.feature_mode)
    dspg_map = channel_reduce(dspg_prior_guided, args.feature_mode)

    res_map = F.interpolate(res_map, size=(h, w), mode="bilinear", align_corners=False)
    gca_map = F.interpolate(gca_map, size=(h, w), mode="bilinear", align_corners=False)
    dspg_map = F.interpolate(dspg_map, size=(h, w), mode="bilinear", align_corners=False)

    res_map_np = normalize_map(res_map.squeeze().detach().cpu().numpy())
    gca_map_np = normalize_map(gca_map.squeeze().detach().cpu().numpy())
    dspg_map_np = normalize_map(dspg_map.squeeze().detach().cpu().numpy())

    # Predictions
    pred_res_np = crop_tensor(pred_res, h, w).squeeze().detach().cpu().numpy()
    pred_gca_np = crop_tensor(pred_gca, h, w).squeeze().detach().cpu().numpy()
    pred_dspg_np = crop_tensor(pred_dspg, h, w).squeeze().detach().cpu().numpy()

    # Auto boxes if not provided
    fp_boxes = parse_boxes(args.fp_boxes)
    fn_boxes = parse_boxes(args.fn_boxes)
    if not fp_boxes and not fn_boxes:
        fp_mask = (pred_res_np >= args.threshold) & (pred_dspg_np < args.threshold)
        fn_mask = (pred_res_np < args.threshold) & (pred_dspg_np >= args.threshold)
        fp_boxes = find_component_boxes(fp_mask, min_area=20, max_boxes=1)
        fn_boxes = find_component_boxes(fn_mask, min_area=20, max_boxes=1)

    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax in axes.flatten():
        ax.axis("off")

    # Row 1: features
    input_img = normalize_map(crop_tensor(img_tensor, h, w).squeeze().detach().cpu().numpy())
    axes[0, 0].imshow(input_img, cmap="gray")
    axes[0, 0].set_title("Input")
    axes[0, 1].imshow(res_map_np, cmap="jet")
    axes[0, 1].set_title("ResUNet Feature")
    axes[0, 2].imshow(gca_map_np, cmap="jet")
    axes[0, 2].set_title("ResUNet-GCA Feature")
    axes[0, 3].imshow(dspg_map_np, cmap="jet")
    axes[0, 3].set_title("DSPG Prior-Guided Feature")

    # Row 2: predictions
    axes[1, 0].imshow(gt_mask, cmap="gray")
    axes[1, 0].set_title("Ground Truth")
    axes[1, 1].imshow(pred_res_np, cmap="jet")
    axes[1, 1].set_title("ResUNet Pred")
    axes[1, 2].imshow(pred_gca_np, cmap="jet")
    axes[1, 2].set_title("ResUNet-GCA Pred")
    axes[1, 3].imshow(pred_dspg_np, cmap="jet")
    axes[1, 3].set_title("DSPG Pred")

    # Highlight boxes on prediction row
    for ax in [axes[1, 1], axes[1, 2], axes[1, 3]]:
        add_boxes(ax, fp_boxes, label="FP suppressed")
        add_boxes(ax, fn_boxes, label="DSPG detects")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
