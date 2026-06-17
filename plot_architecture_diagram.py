#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a 1x3 architecture comparison diagram for:
1) Baseline Skip Connection
2) Single-Stream GCA
3) Dual-Stream DSPG

Output: PDF (vector) with dpi=300.
"""

import argparse
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import patheffects


def setup_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11
    plt.rcParams["figure.facecolor"] = "white"


def add_block(ax, xy, w, h, label, facecolor, edgecolor="#333333", shadow=True):
    x, y = xy
    if shadow:
        shadow_rect = patches.FancyBboxPatch(
            (x + 0.01, y - 0.01),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=0,
            facecolor="black",
            alpha=0.15,
            zorder=1,
        )
        ax.add_patch(shadow_rect)

    rect = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    rect.set_path_effects([patheffects.Normal()])
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center")


def add_arrow(ax, start, end, text=None, text_offset=(0, 0), color="#333333"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.6, color=color),
        zorder=3,
    )
    if text:
        tx = (start[0] + end[0]) / 2 + text_offset[0]
        ty = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(tx, ty, text, ha="center", va="center", color=color)


def add_concat_label(ax, pos):
    ax.text(pos[0], pos[1], "Concat", ha="center", va="center", weight="bold")


def add_multiply_node(ax, center, radius=0.035):
    circle = patches.Circle(center, radius=radius, facecolor="white", edgecolor="#333333", lw=1.2, zorder=4)
    ax.add_patch(circle)
    ax.text(center[0], center[1], "x", ha="center", va="center", weight="bold")


def draw_baseline(ax):
    ax.set_title("Baseline Skip")
    add_block(ax, (0.08, 0.55), 0.30, 0.18, "Encoder Feature", "#d9d9d9")
    add_block(ax, (0.62, 0.55), 0.30, 0.18, "Decoder Up-sampled", "#d9d9d9")
    add_arrow(ax, (0.38, 0.64), (0.62, 0.64), text="Skip Connection", text_offset=(0, 0.06))
    add_concat_label(ax, (0.50, 0.58))


def draw_gca(ax):
    ax.set_title("Single-Stream GCA")
    add_block(ax, (0.08, 0.55), 0.30, 0.18, "Encoder Feature", "#d9d9d9")
    add_block(ax, (0.62, 0.55), 0.30, 0.18, "Decoder Up-sampled", "#d9d9d9")
    add_block(ax, (0.40, 0.78), 0.20, 0.12, "GCA", "#7fd4d4", edgecolor="#2c7a7b")

    add_arrow(ax, (0.38, 0.64), (0.40, 0.84), text="Feature", text_offset=(-0.02, 0.04))
    add_arrow(ax, (0.50, 0.78), (0.56, 0.64), text="GCA Enhanced", text_offset=(0.02, 0.05))
    add_arrow(ax, (0.62, 0.64), (0.56, 0.64), text="Decoder", text_offset=(0.02, -0.06))
    add_concat_label(ax, (0.56, 0.58))


def draw_dspg(ax):
    ax.set_title("Dual-Stream DSPG (Ours)")
    add_block(ax, (0.08, 0.55), 0.30, 0.18, "Encoder Feature", "#d9d9d9")
    add_block(ax, (0.62, 0.55), 0.30, 0.18, "Decoder Up-sampled", "#d9d9d9")
    add_block(ax, (0.36, 0.78), 0.22, 0.12, "GCA", "#7fd4d4", edgecolor="#2c7a7b")

    # Feature stream
    add_arrow(ax, (0.38, 0.64), (0.38, 0.84), text="Feature Stream", text_offset=(-0.08, 0.04))
    add_arrow(ax, (0.47, 0.78), (0.60, 0.64), text="Feature Flow", text_offset=(0.02, 0.06))

    # Prior stream to multiply node
    add_arrow(ax, (0.47, 0.78), (0.52, 0.64), text="Prior Stream [B, 1, H, W]", text_offset=(0.04, 0.08))
    add_multiply_node(ax, (0.56, 0.64))
    add_arrow(ax, (0.62, 0.64), (0.59, 0.64), text="Decoder", text_offset=(0.02, -0.06))
    add_arrow(ax, (0.56, 0.64), (0.60, 0.64), text="Element-wise Product", text_offset=(0.00, -0.08))

    add_concat_label(ax, (0.62, 0.58))


def main():
    parser = argparse.ArgumentParser(description="Plot architecture comparison diagram")
    parser.add_argument("--out", default="architecture_compare.pdf", help="Output PDF file")
    args = parser.parse_args()

    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    draw_baseline(axes[0])
    draw_gca(axes[1])
    draw_dspg(axes[2])

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
