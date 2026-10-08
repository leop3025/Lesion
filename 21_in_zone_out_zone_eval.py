"""
21_in_zone_out_zone_eval.py

Splits the sealed 30-patient test-set predictions into periventricular
(PV, <=10mm from ventricle) vs deep (>10mm) zones, using the Stage-15/18
PV-deep zone maps, and recomputes Dice, Sensitivity, and region-restricted
HD95 separately for each zone — for baseline, distance-only, and
distance+PRAL models.

Reuses the EXACT trained checkpoints, Dataset classes, and inference logic
already used to produce each model's test30_per_patient_metrics.csv, so
results stay consistent with numbers already reported.

Usage:
    python 21_in_zone_out_zone_eval.py
"""

from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(r"C:\Users\Preethi\Documents\LESION")

DISTANCE_ROOT = PROJECT_DIR / "distance_labels"
SPLIT_ROOT = PROJECT_DIR / "splits" / "baseline_2d"

STEP13_FILE = PROJECT_DIR / "13_train_baseline_unet_with_graphs.py"
STEP17_FILE = PROJECT_DIR / "17_train_final_distance_model_and_test.py"
STEP19_FILE = PROJECT_DIR / "19_train_pral_distance_unet.py"

BASELINE_FINAL_CHECKPOINT = (
    PROJECT_DIR / "checkpoints" / "baseline_final70" / "final70_model.pth"
)

DISTANCE_FINAL_CHECKPOINT = (
    PROJECT_DIR / "checkpoints" / "distance_conditioned_final70"
    / "final70_distance_model.pth"
)

PRAL_FINAL_CHECKPOINT = (
    PROJECT_DIR / "checkpoints" / "pral_final70" / "final70_pral_model.pth"
)

OUTPUT_DIR = PROJECT_DIR / "logs" / "in_zone_out_zone_eval"
OUTPUT_CSV = OUTPUT_DIR / "in_zone_out_zone_per_patient.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "in_zone_out_zone_summary.csv"

NUM_CLASSES = 3
CLASS_NAMES = ["background", "normal_wmh", "abnormal_wmh"]
BASE_CHANNELS = 32
BATCH_SIZE = 8

VENTRICLE_ZONE_LABEL = 0
PV_ZONE_LABEL = 1
DEEP_ZONE_LABEL = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


# ============================================================
# DYNAMICALLY IMPORT EXISTING MODULES
# (reuse locked, already-validated Dataset/model code)
# ============================================================

def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


print("Importing existing pipeline modules (this re-runs their module-level code)...")
step13 = import_module(STEP13_FILE, "step13_module")
step17 = import_module(STEP17_FILE, "step17_module")
step19 = import_module(STEP19_FILE, "step19_module")

UNet2D = step17.UNet2D  # same class used everywhere

BaselineSliceDataset = step13.BaselineSliceDataset
find_flair_for_patient = step13.find_flair_for_patient  # identical across scripts
read_patient_ids = step13.read_patient_ids

DistanceDataset = step17.DistanceDataset
PRALDistanceDataset = step19.PRALDistanceDataset


# ============================================================
# ZONE MAP LOADING (same convention as 18_pral_loss.py)
# ============================================================

def pv_zone_path_for_patient(patient_id: str) -> Path:
    patient_id = str(patient_id).zfill(3)
    return DISTANCE_ROOT / patient_id / f"{patient_id}_pv_deep_zone_map.nii.gz"


def load_zone_map(patient_id: str, expected_shape: tuple[int, int, int]) -> np.ndarray:
    path = pv_zone_path_for_patient(patient_id)
    if not path.exists():
        raise FileNotFoundError(f"Missing PV/deep zone map:\n{path}")

    zone_img = nib.load(str(path))
    zone_map = np.asarray(zone_img.dataobj).astype(np.int16, copy=False)

    if zone_map.shape != expected_shape:
        raise RuntimeError(
            f"{patient_id}: zone map shape {zone_map.shape} != "
            f"prediction shape {expected_shape}"
        )

    return zone_map


# ============================================================
# ZONE-RESTRICTED METRICS
# ============================================================

def dice_and_sensitivity_in_region(
    prediction: np.ndarray,
    target: np.ndarray,
    class_index: int,
    region_mask: np.ndarray,
) -> tuple[float, float]:
    pred_mask = (prediction == class_index) & region_mask
    target_mask = (target == class_index) & region_mask

    tp = np.count_nonzero(pred_mask & target_mask)
    fp = np.count_nonzero(pred_mask & ~target_mask)
    fn = np.count_nonzero(~pred_mask & target_mask)

    denom = 2 * tp + fp + fn
    dice = 1.0 if denom == 0 else (2 * tp / denom)

    positives = tp + fn
    sensitivity = math.nan if positives == 0 else (tp / positives)

    return float(dice), float(sensitivity)


def hd95_in_region(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    region_mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """
    Surfaces are computed on the FULL volume (so distances stay physically
    correct), but only surface points that fall inside region_mask
    contribute to the percentile. Avoids artificial edge effects from
    cropping to the region first.
    """
    pred_mask = pred_mask.astype(bool)
    target_mask = target_mask.astype(bool)

    pred_exists = bool((pred_mask & region_mask).any())
    target_exists = bool((target_mask & region_mask).any())

    if not pred_exists and not target_exists:
        return 0.0
    if pred_exists != target_exists:
        return math.nan

    structure = np.ones((3, 3, 3), dtype=bool)

    pred_surface_full = pred_mask ^ binary_erosion(
        pred_mask, structure=structure, border_value=0
    )
    target_surface_full = target_mask ^ binary_erosion(
        target_mask, structure=structure, border_value=0
    )

    dist_to_target = distance_transform_edt(~target_surface_full, sampling=spacing)
    dist_to_pred = distance_transform_edt(~pred_surface_full, sampling=spacing)

    pred_surface_in_region = pred_surface_full & region_mask
    target_surface_in_region = target_surface_full & region_mask

    if not pred_surface_in_region.any() or not target_surface_in_region.any():
        return math.nan

    distances = np.concatenate([
        dist_to_target[pred_surface_in_region],
        dist_to_pred[target_surface_in_region],
    ])

    return float(np.percentile(distances, 95))


# ============================================================
# RUN INFERENCE FOR ONE MODEL, RETURN FULL VOLUMES
# ============================================================

@torch.no_grad()
def run_inference_get_volumes(model, loader, patient_ids):
    model.eval()

    pred_volumes = {}
    target_volumes = {}
    spacing_map = {}

    for patient_id in sorted(patient_ids):
        flair_img = nib.load(str(find_flair_for_patient(patient_id)))
        shape = flair_img.shape[:3]
        spacing = tuple(float(x) for x in flair_img.header.get_zooms()[:3])

        pred_volumes[patient_id] = np.zeros(shape, dtype=np.uint8)
        target_volumes[patient_id] = np.zeros(shape, dtype=np.uint8)
        spacing_map[patient_id] = spacing

    for batch in loader:
        image = batch["image"].to(DEVICE, non_blocking=True)

        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = model(image)

        prediction = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
        target = batch["label"].numpy().astype(np.uint8)
        patient_batch = list(batch["patient_id"])

        slice_indices = batch["slice_index"]
        if torch.is_tensor(slice_indices):
            slice_indices = slice_indices.cpu().numpy().tolist()

        for i, patient_id in enumerate(patient_batch):
            z = int(slice_indices[i])
            pred_volumes[patient_id][:, :, z] = prediction[i]
            target_volumes[patient_id][:, :, z] = target[i]

    return pred_volumes, target_volumes, spacing_map


# ============================================================
# EVALUATE ONE MODEL: IN-ZONE VS OUT-OF-ZONE
# ============================================================

def evaluate_model_by_zone(model_name, model, loader, patient_ids):
    print(f"\nRunning inference for: {model_name}")

    pred_volumes, target_volumes, spacing_map = run_inference_get_volumes(
        model, loader, patient_ids
    )

    rows = []

    for patient_id in sorted(patient_ids):
        prediction = pred_volumes[patient_id]
        target = target_volumes[patient_id]
        spacing = spacing_map[patient_id]

        zone_map = load_zone_map(patient_id, prediction.shape)

        pv_mask = zone_map == PV_ZONE_LABEL
        deep_mask = zone_map == DEEP_ZONE_LABEL

        if not pv_mask.any():
            raise RuntimeError(f"{patient_id}: no PV-zone voxels found.")
        if not deep_mask.any():
            raise RuntimeError(f"{patient_id}: no deep-zone voxels found.")

        row = {"patient_id": patient_id, "model": model_name}

        for zone_name, region_mask in [("in_zone_pv", pv_mask), ("out_zone_deep", deep_mask)]:
            for class_index, class_name in enumerate(CLASS_NAMES):
                if class_name == "background":
                    continue

                dice, sensitivity = dice_and_sensitivity_in_region(
                    prediction, target, class_index, region_mask
                )

                hd = hd95_in_region(
                    prediction == class_index,
                    target == class_index,
                    region_mask,
                    spacing,
                )

                row[f"{zone_name}_dice_{class_name}"] = dice
                row[f"{zone_name}_sensitivity_{class_name}"] = sensitivity
                row[f"{zone_name}_hd95_{class_name}"] = hd

        rows.append(row)

    return rows


# ============================================================
# SUMMARY STATS
# ============================================================

def summarize(rows, model_name):
    filtered = [r for r in rows if r["model"] == model_name]
    keys = [k for k in filtered[0].keys() if k not in ("patient_id", "model")]

    summary = {"model": model_name, "n_patients": len(filtered)}

    for key in keys:
        values = np.asarray([r[key] for r in filtered], dtype=float)
        finite = values[np.isfinite(values)]
        summary[f"{key}_mean"] = float(finite.mean()) if finite.size else math.nan
        summary[f"{key}_std"] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    test_ids = set(read_patient_ids(SPLIT_ROOT / "test_30_patients.csv"))
    print(f"Test patients: {len(test_ids)}")

    all_rows = []

    # ----- Baseline model (3 channels, no distance) -----
    baseline_test_dataset = BaselineSliceDataset(
        partition_root=SPLIT_ROOT / "test",
        expected_patient_ids=test_ids,
        augment=False,
    )
    baseline_test_loader = DataLoader(
        baseline_test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"), drop_last=False,
    )

    baseline_model = UNet2D(in_channels=3, num_classes=NUM_CLASSES, base_channels=BASE_CHANNELS).to(DEVICE)
    baseline_ckpt = torch.load(BASELINE_FINAL_CHECKPOINT, map_location=DEVICE)
    baseline_model.load_state_dict(baseline_ckpt["model_state"])

    all_rows += evaluate_model_by_zone(
        "baseline", baseline_model, baseline_test_loader, test_ids
    )

    # ----- Distance-conditioned model -----
    distance_test_dataset = DistanceDataset(
        partition_roots=[SPLIT_ROOT / "test"],
        expected_patient_ids=test_ids,
        augment=False,
    )
    distance_test_loader = DataLoader(
        distance_test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"), drop_last=False,
    )

    distance_model = UNet2D(in_channels=4, num_classes=NUM_CLASSES, base_channels=BASE_CHANNELS).to(DEVICE)
    distance_ckpt = torch.load(DISTANCE_FINAL_CHECKPOINT, map_location=DEVICE)
    distance_model.load_state_dict(distance_ckpt["model_state"])

    all_rows += evaluate_model_by_zone(
        "distance_only", distance_model, distance_test_loader, test_ids
    )

    # ----- PRAL model -----
    pral_test_dataset = PRALDistanceDataset(
        partition_root=SPLIT_ROOT / "test",
        expected_patient_ids=test_ids,
        augment=False,
    )
    pral_test_loader = DataLoader(
        pral_test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"), drop_last=False,
    )

    pral_model = UNet2D(in_channels=4, num_classes=NUM_CLASSES, base_channels=BASE_CHANNELS).to(DEVICE)
    pral_ckpt = torch.load(PRAL_FINAL_CHECKPOINT, map_location=DEVICE)
    pral_model.load_state_dict(pral_ckpt["model_state"])

    all_rows += evaluate_model_by_zone(
        "distance_pral_lambda4", pral_model, pral_test_loader, test_ids
    )

    # ----- Save per-patient results -----
    fieldnames = ["patient_id", "model"] + [
        k for k in all_rows[0].keys() if k not in ("patient_id", "model")
    ]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nPer-patient in-zone/out-zone metrics written to:\n{OUTPUT_CSV}")

    # ----- Save summary -----
    summaries = [
        summarize(all_rows, "baseline"),
        summarize(all_rows, "distance_only"),
        summarize(all_rows, "distance_pral_lambda4"),
    ]

    summary_fieldnames = list(summaries[0].keys())
    with OUTPUT_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Summary written to:\n{OUTPUT_SUMMARY_CSV}")

    # ----- Print headline comparison -----
    print("\n" + "=" * 96)
    print("IN-ZONE (PV, <=10mm) vs OUT-OF-ZONE (deep, >10mm) — ABNORMAL WMH DICE")
    print("=" * 96)
    for s in summaries:
        print(
            f"{s['model']:<24} "
            f"in-zone Dice: {s['in_zone_pv_dice_abnormal_wmh_mean']:.4f} ± "
            f"{s['in_zone_pv_dice_abnormal_wmh_std']:.4f}   "
            f"out-zone Dice: {s['out_zone_deep_dice_abnormal_wmh_mean']:.4f} ± "
            f"{s['out_zone_deep_dice_abnormal_wmh_std']:.4f}"
        )

    print("\n" + "=" * 96)
    print("IN-ZONE vs OUT-OF-ZONE — NORMAL WMH DICE")
    print("=" * 96)
    for s in summaries:
        print(
            f"{s['model']:<24} "
            f"in-zone Dice: {s['in_zone_pv_dice_normal_wmh_mean']:.4f} ± "
            f"{s['in_zone_pv_dice_normal_wmh_std']:.4f}   "
            f"out-zone Dice: {s['out_zone_deep_dice_normal_wmh_mean']:.4f} ± "
            f"{s['out_zone_deep_dice_normal_wmh_std']:.4f}"
        )


if __name__ == "__main__":
    main()