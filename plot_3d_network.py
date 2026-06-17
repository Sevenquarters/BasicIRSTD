#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot a 3D feature-map box diagram (academic-style) and export to PDF.
"""

import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def box_faces(x, y, z, dx, dy, dz):
    # 8 vertices of a cuboid
    v = [
        (x, y, z),
        (x + dx, y, z),
        (x + dx, y + dy, z),
        (x, y + dy, z),
        (x, y, z + dz),
        (x + dx, y, z + dz),
        (x + dx, y + dy, z + dz),
        (x, y + dy, z + dz),
    ]
    # 6 faces
    return [
        [v[0], v[1], v[2], v[3]],  # bottom
        [v[4], v[5], v[6], v[7]],  # top
        [v[0], v[1], v[5], v[4]],  # front
        [v[2], v[3], v[7], v[6]],  # back
        [v[1], v[2], v[6], v[5]],  # right
        [v[0], v[3], v[7], v[4]],  # left
    ]


def draw_box(ax, x, y, z, dx, dy, dz, color, label):
    faces = box_faces(x, y, z, dx, dy, dz)
    poly = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors="#333333",
        linewidths=0.8,
        alpha=0.6,
    )
    ax.add_collection3d(poly)
    ax.text(x + dx / 2, y + dy + 0.2, z + dz / 2, label, ha="center", va="bottom")


def draw_arrow(ax, start, end, color="#333333"):
    sx, sy, sz = start
    ex, ey, ez = end
    ax.quiver(
        sx, sy, sz,
        ex - sx, ey - sy, ez - sz,
        color=color, arrow_length_ratio=0.12, linewidth=1.2
    )


def main():
    parser = argparse.ArgumentParser(description="3D feature-map box diagram")
    parser.add_argument("--out", default="feature_map_3d.pdf", help="Output PDF path")
    args = parser.parse_args()

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # Boxes
    draw_box(ax, x=0.0, y=0.0, z=0.0, dx=1.0, dy=10.0, dz=10.0, color="#b7d8ff", label="C0 x H x W")
    draw_box(ax, x=2.0, y=1.5, z=1.5, dx=2.0, dy=5.0, dz=5.0, color="#7fc8ff", label="C1 x H/2 x W/2")
    draw_box(ax, x=4.8, y=3.0, z=3.0, dx=4.0, dy=2.5, dz=2.5, color="#65e0e0", label="C2 x H/4 x W/4")

    # Forward arrows
    draw_arrow(ax, start=(1.1, 5.0, 5.0), end=(2.0, 4.0, 4.0))
    draw_arrow(ax, start=(4.1, 4.0, 4.0), end=(4.8, 4.0, 4.0))

    # Skip connection (long-range)
    draw_arrow(ax, start=(1.0, 9.5, 9.5), end=(4.8, 5.0, 4.2), color="#aa3333")

    # Clean axes
    ax.set_axis_off()
    ax.view_init(elev=15, azim=-55)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
