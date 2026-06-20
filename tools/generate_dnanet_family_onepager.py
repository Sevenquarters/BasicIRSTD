#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os

import matplotlib.pyplot as plt
import numpy as np


ROOT = r"D:\Program Files (x86)\IRSTD\BasicIRSTD"
OUT_DIR = os.path.join(ROOT, "docs", "experiments", "figures", "dnanet_family_summary")

CSV_PATHS = {
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

COLORS = {
    "DNANet": "#4c78a8",
    "DNANet-LDEM": "#f58518",
    "DNANet-LDEM-Gate": "#54a24b",
}


def read_metrics(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "loss": float(row["loss"]),
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


def format_fa(value):
    return f"{value:.2e}"


def build_summary():
    summary = {}
    for dataset, spec in CSV_PATHS.items():
        summary[dataset] = {}
        for model, path in spec.items():
            rows = read_metrics(path)
            summary[dataset][model] = {
                "best": best_row(rows),
                "final": final_row(rows),
            }
    return summary


def add_table(ax, dataset_name, dataset_summary):
    ax.axis("off")
    col_labels = ["Model", "Best mIoU", "Best PD", "Best FA", "Final mIoU", "Gap"]
    table_rows = []
    for model in ["DNANet", "DNANet-LDEM", "DNANet-LDEM-Gate"]:
        best = dataset_summary[model]["best"]
        final = dataset_summary[model]["final"]
        gap = best["mIoU"] - final["mIoU"]
        table_rows.append(
            [
                model,
                f"{best['mIoU']:.4f}",
                f"{best['PD']:.4f}",
                format_fa(best["FA"]),
                f"{final['mIoU']:.4f}",
                f"{gap:.4f}",
            ]
        )

    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.05, 1.5)
    ax.set_title(dataset_name, fontsize=13, pad=8, fontweight="bold")


def add_best_final_bar(ax, summary):
    models = ["DNANet", "DNANet-LDEM", "DNANet-LDEM-Gate"]
    x = np.arange(len(models))
    width = 0.18

    nudt_best = [summary["NUDT-SIRST"][m]["best"]["mIoU"] for m in models]
    nudt_final = [summary["NUDT-SIRST"][m]["final"]["mIoU"] for m in models]
    irstd_best = [summary["IRSTD-1K"][m]["best"]["mIoU"] for m in models]
    irstd_final = [summary["IRSTD-1K"][m]["final"]["mIoU"] for m in models]

    ax.bar(x - 1.5 * width, nudt_best, width, label="NUDT Best", color="#4c78a8")
    ax.bar(x - 0.5 * width, nudt_final, width, label="NUDT Final", color="#9ecae9")
    ax.bar(x + 0.5 * width, irstd_best, width, label="IRSTD Best", color="#f58518")
    ax.bar(x + 1.5 * width, irstd_final, width, label="IRSTD Final", color="#ffbf79")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.45, 0.88)
    ax.set_title("Best vs Final mIoU")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9)


def add_stability_bar(ax, summary):
    models = ["DNANet", "DNANet-LDEM", "DNANet-LDEM-Gate"]
    x = np.arange(len(models))
    width = 0.35

    nudt_gap = [
        summary["NUDT-SIRST"][m]["best"]["mIoU"] - summary["NUDT-SIRST"][m]["final"]["mIoU"]
        for m in models
    ]
    irstd_gap = [
        summary["IRSTD-1K"][m]["best"]["mIoU"] - summary["IRSTD-1K"][m]["final"]["mIoU"]
        for m in models
    ]

    ax.bar(x - width / 2, nudt_gap, width, label="NUDT Gap", color="#54a24b")
    ax.bar(x + width / 2, irstd_gap, width, label="IRSTD Gap", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title("Stability Gap (Best - Final mIoU)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9)


def add_takeaways(ax, summary):
    ax.axis("off")

    nudt_best = summary["NUDT-SIRST"]["DNANet-LDEM"]["best"]["mIoU"]
    gate_nudt_final = summary["NUDT-SIRST"]["DNANet-LDEM-Gate"]["final"]["mIoU"]
    irstd_best = summary["IRSTD-1K"]["DNANet-LDEM-Gate"]["best"]["mIoU"]
    irstd_final = summary["IRSTD-1K"]["DNANet-LDEM-Gate"]["final"]["mIoU"]

    lines = [
        "Key Takeaways",
        "",
        f"1. NUDT-SIRST: LDEM and LDEM-Gate both outperform DNANet.",
        f"2. NUDT-SIRST best performer: DNANet-LDEM ({nudt_best:.4f} best mIoU).",
        f"3. NUDT-SIRST most stable final result: DNANet-LDEM-Gate ({gate_nudt_final:.4f}).",
        "",
        f"4. IRSTD-1K best peak: DNANet-LDEM-Gate ({irstd_best:.4f} best mIoU).",
        f"5. IRSTD-1K final drop: DNANet-LDEM-Gate falls to {irstd_final:.4f}.",
        "6. V2 solves peak performance, but not late-stage stability.",
        "",
        "V3 Direction",
        "Gate stabilization + semantic counter-correction",
        "to reduce late-stage oscillation on IRSTD-1K.",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = build_summary()

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], width_ratios=[1.2, 1.2, 1.0])
    fig.suptitle("DNANet Family One-Page Experiment Overview", fontsize=18, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[:, 2])

    add_table(ax1, "NUDT-SIRST", summary["NUDT-SIRST"])
    add_table(ax2, "IRSTD-1K", summary["IRSTD-1K"])
    add_best_final_bar(ax3, summary)
    add_stability_bar(ax4, summary)
    add_takeaways(ax5, summary)

    png_path = os.path.join(OUT_DIR, "dnanet_family_onepage_overview.png")
    pdf_path = os.path.join(OUT_DIR, "dnanet_family_onepage_overview.pdf")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
