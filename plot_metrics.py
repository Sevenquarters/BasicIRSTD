#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot training metrics from CSV logs.

Supports:
- CSV logs with columns: epoch, loss, pixAcc, mIoU, PD, FA
- TXT logs from train.py output
"""

import argparse
import csv
import re
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def read_metrics(csv_path: str) -> List[Dict[str, float]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"File not found: {csv_path}. "
            "For full curves, make sure train.py was run with --eval_intervals 1 "
            "so each epoch metrics were recorded."
        )

    if csv_path.lower().endswith(".txt"):
        return read_metrics_from_txt(csv_path)

    rows: List[Dict[str, float]] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "loss": float(row["loss"]) if row["loss"] else None,
                    "pixAcc": float(row["pixAcc"]) if row["pixAcc"] else None,
                    "mIoU": float(row["mIoU"]) if row["mIoU"] else None,
                    "PD": float(row["PD"]) if row["PD"] else None,
                    "FA": float(row["FA"]) if row["FA"] else None,
                }
            )
    return rows


def read_metrics_from_txt(txt_path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    epoch_re = re.compile(r"Epoch---(\d+), total_loss---([0-9eE\.\+\-]+)")
    miou_re = re.compile(r"pixAcc, mIoU:\s*\(\s*([0-9eE\.\+\-]+)\s*,\s*(?:np\.float64\()?\s*([0-9eE\.\+\-]+)\s*\)?\s*\)")
    pd_fa_re = re.compile(r"PD, FA:\s*\(\s*([0-9eE\.\+\-]+)\s*,\s*([0-9eE\.\+\-]+)\s*\)")

    current = {}
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = epoch_re.search(line)
            if m:
                current = {"epoch": int(m.group(1)), "loss": float(m.group(2)), "pixAcc": None, "mIoU": None, "PD": None, "FA": None}
                continue

            m = miou_re.search(line)
            if m and current:
                current["pixAcc"] = float(m.group(1))
                current["mIoU"] = float(m.group(2))
                continue

            m = pd_fa_re.search(line)
            if m and current:
                current["PD"] = float(m.group(1))
                current["FA"] = float(m.group(2))
                rows.append(current)
                current = {}

    if len(rows) == 0:
        raise RuntimeError(
            f"No per-epoch records found in {txt_path}. "
            "Please re-run training with --eval_intervals 1."
        )
    return rows


def plot_single(metrics: List[Dict[str, float]], title: str, out_path: str):
    epochs = [r["epoch"] for r in metrics]
    loss = [r["loss"] for r in metrics]
    miou = [r["mIoU"] for r in metrics]
    pd = [r["PD"] for r in metrics]
    fa = [r["FA"] for r in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)

    axes[0, 0].plot(epochs, loss, marker="o", linewidth=1.5)
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, miou, marker="o", linewidth=1.5, color="tab:green")
    axes[0, 1].set_title("mIoU")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("mIoU")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, pd, marker="o", linewidth=1.5, color="tab:blue")
    axes[1, 0].set_title("PD")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("PD")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, fa, marker="o", linewidth=1.5, color="tab:red")
    axes[1, 1].set_title("FA")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("FA")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_compare(metric_sets: List[List[Dict[str, float]]], labels: List[str], out_path: str, metric_name: str):
    plt.figure(figsize=(8, 5))
    for metrics, label in zip(metric_sets, labels):
        epochs = [r["epoch"] for r in metrics]
        values = [r[metric_name] for r in metrics]
        plt.plot(epochs, values, marker="o", linewidth=1.5, label=label)

    plt.title(metric_name)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def default_title(csv_path: str) -> str:
    base = os.path.basename(csv_path)
    return os.path.splitext(base)[0]


def main():
    parser = argparse.ArgumentParser(description="Plot metrics from CSV logs.")
    parser.add_argument("csvs", nargs="+", help="One or more metrics CSV files.")
    parser.add_argument("--outdir", default="./plots", help="Output directory for figures.")
    parser.add_argument("--show", action="store_true", help="Show figures interactively.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    all_metrics = []
    labels = []
    for csv_path in args.csvs:
        metrics = read_metrics(csv_path)
        if len(metrics) == 0:
            raise RuntimeError(f"No rows found in {csv_path}")
        all_metrics.append(metrics)
        labels.append(os.path.splitext(os.path.basename(csv_path))[0])

        single_out = os.path.join(args.outdir, f"{labels[-1]}_curves.png")
        plot_single(metrics, labels[-1], single_out)
        print(f"Saved: {single_out}")

    if len(all_metrics) > 1:
        for metric_name in ["loss", "mIoU", "PD", "FA"]:
            compare_out = os.path.join(args.outdir, f"compare_{metric_name}.png")
            plot_compare(all_metrics, labels, compare_out, metric_name)
            print(f"Saved: {compare_out}")

    if args.show:
        # Re-open the last figure for interactive review if desired.
        # Figures are already saved to disk.
        plt.show()


if __name__ == "__main__":
    main()

