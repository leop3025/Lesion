"""
22_zone_artifact_check_and_gap_significance.py

Step 1: Checks how many patients have Dice==1.0 in the out-of-zone
(deep) normal-WMH metric for each model, since this usually indicates
"no ground truth AND no prediction in this region" rather than a
genuine correct segmentation, and can destabilize the reported mean/std.

Step 2: Computes the in-zone-minus-out-zone Dice GAP per patient for
each model, then runs paired Wilcoxon signed-rank tests between models
on that gap (not on the raw scores) — this directly tests whether
PRAL's effect is more/less spatially localized to the periventricular
zone than baseline/distance-only, rather than just comparing overall
performance.

Reads:
    logs/in_zone_out_zone_eval/in_zone_out_zone_per_patient.csv

Usage:
    python 22_zone_artifact_check_and_gap_significance.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"C:\Users\Preethi\Documents\LESION")

PER_PATIENT_CSV = (
    PROJECT_DIR / "logs" / "in_zone_out_zone_eval" / "in_zone_out_zone_per_patient.csv"
)

OUTPUT_DIR = PROJECT_DIR / "logs" / "in_zone_out_zone_eval"
ARTIFACT_REPORT_CSV = OUTPUT_DIR / "step1_dice1_artifact_report.csv"
GAP_PER_PATIENT_CSV = OUTPUT_DIR / "step2_zone_gap_per_patient.csv"
GAP_SIGNIFICANCE_CSV = OUTPUT_DIR / "step2_zone_gap_significance.csv"

MODELS = ["baseline", "distance_only", "distance_pral_lambda4"]


# ============================================================
# STEP 1 — DICE==1.0 ARTIFACT CHECK
# ============================================================

def step1_artifact_check(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 90)
    print("STEP 1 — Dice==1.0 ARTIFACT CHECK (out-of-zone normal WMH)")
    print("=" * 90)

    rows = []

    metrics_to_check = [
        ("out_zone_deep_dice_normal_wmh", "out-zone (deep) normal WMH"),
        ("out_zone_deep_dice_abnormal_wmh", "out-zone (deep) abnormal WMH"),
        ("in_zone_pv_dice_normal_wmh", "in-zone (PV) normal WMH"),
        ("in_zone_pv_dice_abnormal_wmh", "in-zone (PV) abnormal WMH"),
    ]

    for model in MODELS:
        sub = df[df["model"] == model]
        n_total = len(sub)

        print(f"\nModel: {model}  (n={n_total} patients)")

        for col, label in metrics_to_check:
            if col not in sub.columns:
                print(f"  WARNING: column '{col}' not found, skipping.")
                continue

            values = sub[col].astype(float)
            n_exact_one = int((values == 1.0).sum())
            pct = 100.0 * n_exact_one / n_total if n_total else float("nan")

            # Recompute mean/std WITH and WITHOUT the exact-1.0 patients
            mean_with = float(values.mean())
            std_with = float(values.std(ddof=1)) if len(values) > 1 else 0.0

            without = values[values != 1.0]
            mean_without = float(without.mean()) if len(without) else float("nan")
            std_without = float(without.std(ddof=1)) if len(without) > 1 else 0.0

            print(
                f"  {label:<32}: {n_exact_one}/{n_total} patients ({pct:.1f}%) "
                f"have Dice==1.0"
            )
            print(
                f"      mean incl. Dice=1.0 patients : {mean_with:.4f} ± {std_with:.4f}"
            )
            print(
                f"      mean EXCL. Dice=1.0 patients  : {mean_without:.4f} ± {std_without:.4f} "
                f"(n={len(without)})"
            )

            rows.append({
                "model": model,
                "metric": col,
                "n_total_patients": n_total,
                "n_dice_exactly_1": n_exact_one,
                "pct_dice_exactly_1": pct,
                "mean_including_1s": mean_with,
                "std_including_1s": std_with,
                "mean_excluding_1s": mean_without,
                "std_excluding_1s": std_without,
                "n_excluding_1s": len(without),
            })

    report_df = pd.DataFrame(rows)
    report_df.to_csv(ARTIFACT_REPORT_CSV, index=False)
    print(f"\nFull artifact report written to:\n{ARTIFACT_REPORT_CSV}")

    return report_df


# ============================================================
# STEP 2 — IN-ZONE MINUS OUT-ZONE GAP + SIGNIFICANCE
# ============================================================

def step2_gap_and_significance(df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("STEP 2 — IN-ZONE vs OUT-ZONE DICE GAP (per patient) + SIGNIFICANCE")
    print("=" * 90)

    gap_metrics = [
        ("dice_normal_wmh", "in_zone_pv_dice_normal_wmh", "out_zone_deep_dice_normal_wmh"),
        ("dice_abnormal_wmh", "in_zone_pv_dice_abnormal_wmh", "out_zone_deep_dice_abnormal_wmh"),
    ]

    # ------------------------------------------------------------------
    # Build a wide table: one row per patient, one gap column per model
    # ------------------------------------------------------------------
    gap_frames = {}

    for base_name, in_col, out_col in gap_metrics:
        for model in MODELS:
            sub = df[df["model"] == model].set_index("patient_id")
            gap = sub[in_col].astype(float) - sub[out_col].astype(float)
            gap_frames[f"{model}__gap_{base_name}"] = gap

    gap_wide = pd.DataFrame(gap_frames)
    gap_wide.index.name = "patient_id"
    gap_wide.to_csv(GAP_PER_PATIENT_CSV)
    print(f"\nPer-patient in-zone-minus-out-zone gaps written to:\n{GAP_PER_PATIENT_CSV}")

    # ------------------------------------------------------------------
    # Print mean gap per model per metric
    # ------------------------------------------------------------------
    print("\nMean (in-zone minus out-zone) Dice gap per model:")
    for base_name, _, _ in gap_metrics:
        print(f"\n  {base_name}:")
        for model in MODELS:
            col = f"{model}__gap_{base_name}"
            vals = gap_wide[col].dropna()
            print(f"    {model:<24}: {vals.mean():+.4f} ± {vals.std(ddof=1):.4f}")

    # ------------------------------------------------------------------
    # Paired Wilcoxon tests on the GAP itself, model vs model
    # Tests whether zone-localization differs between models, not just
    # whether raw performance differs.
    # ------------------------------------------------------------------
    print("\n" + "-" * 90)
    print("Paired Wilcoxon signed-rank tests on the GAP (baseline vs distance vs PRAL)")
    print("-" * 90)

    comparisons = [
        ("baseline", "distance_only"),
        ("distance_only", "distance_pral_lambda4"),
        ("baseline", "distance_pral_lambda4"),
    ]

    sig_rows = []

    for base_name, _, _ in gap_metrics:
        print(f"\n{base_name}:")
        for model_a, model_b in comparisons:
            col_a = f"{model_a}__gap_{base_name}"
            col_b = f"{model_b}__gap_{base_name}"

            paired = gap_wide[[col_a, col_b]].dropna()
            a_vals = paired[col_a].values
            b_vals = paired[col_b].values

            if len(paired) < 2:
                print(f"  {model_a} vs {model_b}: not enough paired patients, skipped.")
                continue

            diff = b_vals - a_vals

            try:
                stat, pval = wilcoxon(a_vals, b_vals)
            except ValueError:
                pval = float("nan")

            n_b_more_localized = int((diff < 0).sum())  # smaller gap = more localized effect? sign depends on metric direction
            n_pairs = len(paired)

            print(
                f"  {model_a:<22} vs {model_b:<22}: "
                f"mean gap {a_vals.mean():+.4f} -> {b_vals.mean():+.4f}   "
                f"p={pval:.4f}   n={n_pairs}"
            )

            sig_rows.append({
                "metric": base_name,
                "model_a": model_a,
                "model_b": model_b,
                "mean_gap_a": float(a_vals.mean()),
                "mean_gap_b": float(b_vals.mean()),
                "wilcoxon_p": pval,
                "n_pairs": n_pairs,
            })

    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(GAP_SIGNIFICANCE_CSV, index=False)
    print(f"\nSignificance results written to:\n{GAP_SIGNIFICANCE_CSV}")


# ============================================================
# MAIN
# ============================================================

def main():
    if not PER_PATIENT_CSV.exists():
        raise FileNotFoundError(
            f"Missing per-patient CSV from the eval script:\n{PER_PATIENT_CSV}\n"
            "Run 21_in_zone_out_zone_eval.py first."
        )

    df = pd.read_csv(PER_PATIENT_CSV)

    missing_models = set(MODELS) - set(df["model"].unique())
    if missing_models:
        raise RuntimeError(f"Missing expected models in CSV: {missing_models}")

    step1_artifact_check(df)
    step2_gap_and_significance(df)

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)


if __name__ == "__main__":
    main()