"""Copy summary metrics and aggregate plots from logs/ (git-ignored) into
results/ (tracked) and draw a cross-model comparison chart.

No patient images (prediction overlays, IG / uncertainty / counterfactual
examples) and no NIfTI files are copied.

Usage: python collect_results.py
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
LOG_ROOT = BASE_DIR / "logs"
OUT_ROOT = BASE_DIR / "results"

# run folder -> display name (order = chart order)
RUNS = {
    "baseline_unet_v2": "U-Net",
    "distance_map_unet_v2": "Distance U-Net",
    "unetpp_v2": "U-Net++",
    "distance_map_unetpp_v2": "Distance U-Net++",
    "baseline_swin_unetr_v2": "SwinUNETR",
}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
TARGETS = ["ventricle", "normal_wmh", "pv_abnormal_wmh", "deep_abnormal_wmh", "abnormal_wmh", "all_wmh"]
METRICS = ["dice", "sensitivity", "precision", "hd95"]

RUN_FILES = [
    "cv_summary/cv_mean_std.csv",
    "cv_summary/postprocessing_choice.json",
    "cv_summary/plots/cv_dice_bar.png",
    "cv_summary/plots/cv_selection_curves.png",
    "cv_summary/plots/cv_all_val_dice_boxplot.png",
    "final_results/final_plan.json",
    "final_results/test30_summary.csv",
    "final_results/train_cv_test_report.csv",
    "final_results/plots/final70_training_curves.png",
    "final_results/plots/test30_dice_boxplot.png",
    "final_results/plots/test30_hd95_boxplot.png",
    "final_results/plots/test30_confusion_matrix.png",
    "final_results/plots/test30_volume_agreement.png",
]

EXPLAIN_FILES = [
    "explainability_config.json",
    "ablation/ablation_summary.csv",
    "ablation/plots/ablation_dice.png",
    "ablation/plots/distance_counterfactual.png",
    "ablation/plots/counterfactual_per_patient.png",
    "attribution/ig_summary.csv",
    "attribution/plots/ig_channel_shares.png",
    "distance_bands/distance_band_metrics.csv",
    "distance_bands/pvdeep_label_accuracy_by_band.csv",
    "distance_bands/abnormal_wmh_by_distance.png",
    "distance_bands/normal_wmh_by_distance.png",
    "distance_bands/pvdeep_label_accuracy_by_distance.png",
    "rule_based/rule_based_summary.csv",
    "rule_based/rule_based_comparison.csv",
    "uncertainty/uncertainty_summary.csv",
]


def copy_files(src_root, dst_root, rel_paths):
    for rel in rel_paths:
        src = src_root / rel
        if not src.exists():
            print(f"  missing, skipped: {src}")
            continue
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def read_summary(run):
    path = LOG_ROOT / run / "final_results" / "test30_summary.csv"
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["metric"]] = float(row["value"])
            except ValueError:
                pass
    return out


def main():
    for run in RUNS:
        print(f"Copying {run}")
        copy_files(LOG_ROOT / run, OUT_ROOT / run, RUN_FILES)
    print("Copying explainability")
    copy_files(LOG_ROOT / "explainability", OUT_ROOT / "explainability", EXPLAIN_FILES)

    # comparison table: test-set mean ± SD per model / target / metric, plus CV Dice
    rows = []
    summaries = {run: read_summary(run) for run in RUNS}
    for run, name in RUNS.items():
        s = summaries[run]
        cv = json.loads((LOG_ROOT / run / "cv_summary" / "cv_mean_std.json").read_text(encoding="utf-8"))
        plan = json.loads((LOG_ROOT / run / "final_results" / "final_plan.json").read_text(encoding="utf-8"))
        for t in TARGETS:
            row = {"model": name, "run": run, "in_channels": plan.get("in_channels"),
                   "final_epochs": plan.get("final_epochs"), "target": t}
            for m in METRICS:
                row[f"test_{m}_mean"] = s.get(f"{t}_{m}_mean", math.nan)
                row[f"test_{m}_std"] = s.get(f"{t}_{m}_std", math.nan)
            row["cv_dice_mean"] = cv.get(f"{t}_dice_mean", {}).get("mean", math.nan)
            row["cv_dice_std"] = cv.get(f"{t}_dice_mean", {}).get("std", math.nan)
            rows.append(row)
    comp_dir = OUT_ROOT / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)
    with (comp_dir / "model_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # grouped bar chart of test Dice (mean ± SD over 30 patients)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    n = len(RUNS)
    width = 0.8 / n
    x = np.arange(len(TARGETS))
    for i, (run, name) in enumerate(RUNS.items()):
        s = summaries[run]
        means = [s.get(f"{t}_dice_mean", math.nan) for t in TARGETS]
        stds = [s.get(f"{t}_dice_std", math.nan) for t in TARGETS]
        ax.bar(x + (i - (n - 1) / 2) * width, means, width * 0.92, yerr=stds, label=name,
               color=COLORS[i], edgecolor="white", linewidth=1,
               error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "#555555"})
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", " ") for t in TARGETS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice (mean ± SD, 30 test patients)")
    ax.set_title("Held-out test set: per-patient 3D Dice by class and model")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(ncol=n, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    fig.savefig(comp_dir / "test_dice_comparison.png", dpi=130)
    plt.close(fig)
    print(f"Results written to {OUT_ROOT}")


if __name__ == "__main__":
    main()
