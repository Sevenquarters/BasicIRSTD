#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os

import matplotlib.pyplot as plt
import numpy as np


ROOT = r"D:\Program Files (x86)\IRSTD\BasicIRSTD"

SPECS = {
    "NUDT-SIRST": {
        "DNANet": os.path.join(ROOT, "log", "exp_baseline_dnanet_nudt_40e", "NUDT-SIRST_DNANet_metrics.csv"),
        "DNANet-LDEM": os.path.join(ROOT, "log", "exp_dnanet_ldem_nudt_40e", "NUDT-SIRST_DNANet-LDEM_metrics.csv"),
        "DNANet-LDEM-Gate": os.path.join(ROOT, "log", "exp_dnanet_ldem_gate_nudt_40e", "NUDT-SIRST_DNANet-LDEM-Gate_metrics.csv"),
    },
    "IRSTD-1K": {
        "DNANet": os.path.join(ROOT, "log", "exp_baseline_dnanet_irstd1k_40e", "IRSTD-1K_DNANet_metrics.csv"),
        "DNANet-LDEM": os.path.join(ROOT, "log", "exp_dnanet_ldem_irstd1k_40e", "IRSTD-1K_DNANet-LDEM_metrics.csv"),
        "DNANet-LDEM-Gate": os.path.join(ROOT, "log", "exp_dnanet_ldem_gate_irstd1k_40e", "IRSTD-1K_DNANet-LDEM-Gate_metrics.csv"),
    },
}

OUT_DIRS = {
    "NUDT-SIRST": os.path.join(ROOT, "docs", "experiments", "figures", "dnanet_family_nudt"),
    "IRSTD-1K": os.path.join(ROOT, "docs", "experiments", "figures", "dnanet_family_irstd1k"),
    "summary": os.path.join(ROOT, "docs", "experiments", "figures", "dnanet_family_summary"),
}

COLORS = {
    "DNANet": "#1f77b4",
    "DNANet-LDEM": "#ff7f0e",
    "DNANet-LDEM-Gate": "#2ca02c",
}


def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "loss": float(row["loss"]),
                    "pixAcc": float(row["pixAcc"]),
                    "mIoU": float(row["mIoU"]),
                    "PD": float(row["PD"]),
                    "FA": float(row["FA"]),
                }
            )
    return rows


def best_row(rows):
    return max(rows, key=lambda x: x["mIoU"])


def final_row(rows):
    return rows[-1]


def save_curve_grid(dataset_name, model_rows, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"{dataset_name} - DNANet Family Training Curves", fontsize=14)
    metric_info = [
        ("loss", "Loss"),
        ("mIoU", "mIoU"),
        ("PD", "PD"),
        ("FA", "FA"),
    ]

    for ax, (metric, title) in zip(axes.flatten(), metric_info):
        for label, rows in model_rows.items():
            epochs = [r["epoch"] for r in rows]
            values = [r[metric] for r in rows]
            ax.plot(
                epochs,
                values,
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=label,
                color=COLORS[label],
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metric_bar(dataset_name, summary_rows, out_path):
    labels = list(summary_rows.keys())
    best_miou = [summary_rows[k]["best"]["mIoU"] for k in labels]
    final_miou = [summary_rows[k]["final"]["mIoU"] for k in labels]
    best_pd = [summary_rows[k]["best"]["PD"] for k in labels]

    x = np.arange(len(labels))
    width = 0.23

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, best_miou, width, label="Best mIoU", color="#4c78a8")
    ax.bar(x, final_miou, width, label="Final mIoU", color="#f58518")
    ax.bar(x + width, best_pd, width, label="Best PD", color="#54a24b")

    ax.set_title(f"{dataset_name} - Key Metric Summary")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_cross_dataset_best(summary, out_path):
    models = ["DNANet", "DNANet-LDEM", "DNANet-LDEM-Gate"]
    nudt = [summary["NUDT-SIRST"][m]["best"]["mIoU"] for m in models]
    irstd = [summary["IRSTD-1K"][m]["best"]["mIoU"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, nudt, width, label="NUDT-SIRST", color="#1f77b4")
    ax.bar(x + width / 2, irstd, width, label="IRSTD-1K", color="#ff7f0e")
    ax.set_title("Best mIoU Across Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    for outdir in OUT_DIRS.values():
        os.makedirs(outdir, exist_ok=True)

    summary = {}
    for dataset_name, model_paths in SPECS.items():
        model_rows = {label: read_csv(path) for label, path in model_paths.items()}
        summary[dataset_name] = {
            label: {"best": best_row(rows), "final": final_row(rows)}
            for label, rows in model_rows.items()
        }

        save_curve_grid(
            dataset_name,
            model_rows,
            os.path.join(OUT_DIRS[dataset_name], f"{dataset_name}_dnanet_family_curves.png"),
        )
        save_metric_bar(
            dataset_name,
            summary[dataset_name],
            os.path.join(OUT_DIRS[dataset_name], f"{dataset_name}_dnanet_family_summary.png"),
        )

    save_cross_dataset_best(
        summary,
        os.path.join(OUT_DIRS["summary"], "dnanet_family_best_miou_cross_dataset.png"),
    )

    print("Saved comparison figures.")


if __name__ == "__main__":
    main()
