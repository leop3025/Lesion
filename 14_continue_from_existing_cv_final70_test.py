from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import Dataset, DataLoader

THIS_FILE = Path(__file__).resolve()
BASE_DIR = THIS_FILE.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
from unet2d import UNet2D

PROJECT_DIR = Path(r"C:\Users\Preethi\Documents\LESION")
SPLIT_ROOT = PROJECT_DIR / "splits" / "baseline_2d"
PREPROCESSED_ROOT = PROJECT_DIR / "preprocessed"
CV_LOG_ROOT = PROJECT_DIR / "logs" / "baseline_unet"
FINAL_LOG_DIR = PROJECT_DIR / "logs" / "baseline_unet" / "final results"
FINAL_CKPT_DIR = PROJECT_DIR / "checkpoints" / "baseline_final70"
OVERALL_CV_JSON = FINAL_LOG_DIR / "overall_cv_metrics.json"
OVERALL_CV_CSV = FINAL_LOG_DIR / "overall_cv_metrics.csv"
FINAL_TRAIN_LOG = FINAL_LOG_DIR / "final70_training_history.csv"
FINAL_MODEL_PATH = FINAL_CKPT_DIR / "final70_model.pth"
TRAIN_PATIENT_METRICS_CSV = FINAL_LOG_DIR / "train70_per_patient_metrics.csv"
TRAIN_METRICS_JSON = FINAL_LOG_DIR / "train70_final_metrics.json"
TRAIN_METRICS_CSV = FINAL_LOG_DIR / "train70_final_metrics.csv"
TEST_PATIENT_METRICS_CSV = FINAL_LOG_DIR / "test30_per_patient_metrics.csv"
TEST_METRICS_JSON = FINAL_LOG_DIR / "test30_final_metrics.json"
TEST_METRICS_CSV = FINAL_LOG_DIR / "test30_final_metrics.csv"
FINAL_REPORT_CSV = FINAL_LOG_DIR / "baseline_global_train_cv_test_summary.csv"
EVALUATION_PLAN_JSON = FINAL_LOG_DIR / "evaluation_plan.json"

NUM_CLASSES = 3
CLASS_NAMES = ["background", "normal_wmh", "abnormal_wmh"]
CHANNEL_ORDER = ["T1", "T2", "FLAIR"]
BASE_CHANNELS = 32
BATCH_SIZE = 8
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 20260911
CE_LOSS_WEIGHT = 1.0
DICE_LOSS_WEIGHT = 1.0
CLASS_WEIGHT_POWER = 0.5
INCLUDE_BACKGROUND_IN_DICE_LOSS = False
USE_AUGMENTATION = True
P_HORIZONTAL_FLIP = 0.5
P_VERTICAL_FLIP = 0.5
P_ROT90 = 0.5
P_INTENSITY_SCALE = 0.3
INTENSITY_SCALE_RANGE = (0.90, 1.10)
P_INTENSITY_SHIFT = 0.3
INTENSITY_SHIFT_RANGE = (-0.10, 0.10)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

METRIC_COLUMNS = [
    f"{metric}_{class_name}"
    for metric in ["dice", "sensitivity", "hd95"]
    for class_name in CLASS_NAMES
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_torch_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def read_patient_ids(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty split file: {path}")
    ids = [str(row["patient_id"]).strip().zfill(3) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate patient IDs in {path}")
    return ids


def patient_id_from_filename(path):
    match = re.match(r"(\d{3})_slice_\d+\.npz$", path.name)
    if not match:
        raise RuntimeError(f"Unexpected slice filename: {path.name}")
    return match.group(1)


def finite_mean_std(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return math.nan, math.nan
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    return mean, std


def load_completed_cv_metrics():
    rows = []
    for fold in range(1, 6):
        path = CV_LOG_ROOT / f"fold_{fold}" / "final_metrics.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Fold {fold} final metrics not found:\n{path}\nRun all 5 CV folds before Stage 14."
            )
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if int(metrics.get("fold", -1)) != fold:
            raise RuntimeError(f"Fold identity mismatch in {path}")
        if "best_epoch" not in metrics:
            raise RuntimeError(f"best_epoch missing in {path}")
        for metric in METRIC_COLUMNS:
            if metric not in metrics:
                raise RuntimeError(f"{metric} missing in {path}")
        rows.append(metrics)
    return rows


def save_overall_cv_metrics(cv_rows):
    result = {
        "n_folds": 5,
        "fold_best_epochs": [int(row["best_epoch"]) for row in cv_rows],
    }
    for metric in METRIC_COLUMNS:
        mean, std = finite_mean_std([float(row[metric]) for row in cv_rows])
        result[f"{metric}_mean"] = mean
        result[f"{metric}_std"] = std

    atomic_write_text(OVERALL_CV_JSON, json.dumps(result, indent=2))
    OVERALL_CV_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OVERALL_CV_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std"])
        writer.writeheader()
        for metric in METRIC_COLUMNS:
            writer.writerow({
                "metric": metric,
                "mean": result[f"{metric}_mean"],
                "std": result[f"{metric}_std"],
            })
    return result


def choose_final_training_epochs(cv_rows):
    best_epochs = np.asarray([int(row["best_epoch"]) for row in cv_rows], dtype=int)
    final_epochs = int(np.rint(np.median(best_epochs)))
    if final_epochs < 1:
        raise RuntimeError(f"Invalid final epoch count: {final_epochs}")
    return final_epochs


class MultiPartitionSliceDataset(Dataset):
    def __init__(self, partition_roots, expected_patient_ids, augment=False):
        self.partition_roots = [Path(p) for p in partition_roots]
        self.expected_patient_ids = set(expected_patient_ids)
        self.augment = augment
        self.samples = []
        seen_names = set()

        for partition_root in self.partition_roots:
            image_dir = partition_root / "images"
            label_dir = partition_root / "labels"
            if not image_dir.exists():
                raise FileNotFoundError(f"Missing image directory: {image_dir}")
            if not label_dir.exists():
                raise FileNotFoundError(f"Missing label directory: {label_dir}")
            image_paths = sorted(image_dir.glob("*.npz"))
            if not image_paths:
                raise RuntimeError(f"No images found in {image_dir}")

            for image_path in image_paths:
                pid = patient_id_from_filename(image_path)
                if pid not in self.expected_patient_ids:
                    raise RuntimeError(
                        f"{image_path.name} belongs to patient {pid}, who is not allowed in this dataset."
                    )
                if image_path.name in seen_names:
                    raise RuntimeError(f"Duplicate slice detected across partitions: {image_path.name}")
                seen_names.add(image_path.name)
                label_path = label_dir / image_path.name
                if not label_path.exists():
                    raise FileNotFoundError(f"Missing matching label: {label_path}")
                self.samples.append((image_path, label_path))

        actual_patients = {patient_id_from_filename(p) for p, _ in self.samples}
        if actual_patients != self.expected_patient_ids:
            missing = sorted(self.expected_patient_ids - actual_patients)
            extra = sorted(actual_patients - self.expected_patient_ids)
            raise RuntimeError(f"Dataset patient membership mismatch. missing={missing}, extra={extra}")

    def __len__(self):
        return len(self.samples)

    def _augment(self, image, label):
        if random.random() < P_HORIZONTAL_FLIP:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=1).copy()
        if random.random() < P_VERTICAL_FLIP:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=0).copy()
        if random.random() < P_ROT90:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            label = np.rot90(label, k=k, axes=(0, 1)).copy()
        if random.random() < P_INTENSITY_SCALE:
            image = (image * random.uniform(*INTENSITY_SCALE_RANGE)).astype(np.float32, copy=False)
        if random.random() < P_INTENSITY_SHIFT:
            image = (image + random.uniform(*INTENSITY_SHIFT_RANGE)).astype(np.float32, copy=False)
        return image, label

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]
        with np.load(image_path, allow_pickle=False) as image_npz:
            image = image_npz["image"].astype(np.float32, copy=True)
            patient_id = str(image_npz["patient_id"]).zfill(3)
            slice_index = int(image_npz["slice_index"])
            channel_order = [str(x) for x in image_npz["channel_order"].tolist()]
        if channel_order != CHANNEL_ORDER:
            raise RuntimeError(f"Channel order mismatch in {image_path}: {channel_order}")
        with np.load(label_path, allow_pickle=False) as label_npz:
            label = label_npz["label"].astype(np.int64, copy=True)
        if image.ndim != 3 or image.shape[0] != 3:
            raise RuntimeError(f"Expected image [3,H,W], got {image.shape}")
        if label.ndim != 2:
            raise RuntimeError(f"Expected label [H,W], got {label.shape}")
        unique = set(np.unique(label).tolist())
        if not unique.issubset({0, 1, 2}):
            raise RuntimeError(f"Unexpected labels {unique} in {label_path}")
        if self.augment and USE_AUGMENTATION:
            image, label = self._augment(image, label)
        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "patient_id": patient_id,
            "slice_index": slice_index,
        }


def compute_class_voxel_counts(dataset):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    print("\nComputing class counts from ALL 70 DEVELOPMENT PATIENTS ONLY...")
    for _, label_path in dataset.samples:
        with np.load(label_path, allow_pickle=False) as data:
            label = data["label"]
        counts += np.bincount(label.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    if np.any(counts == 0):
        raise RuntimeError(f"At least one class has zero development voxels: {counts.tolist()}")
    return counts


def class_weights_from_counts(counts):
    counts = counts.astype(np.float64)
    frequencies = counts / counts.sum()
    weights = 1.0 / np.power(frequencies, CLASS_WEIGHT_POWER)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


class MulticlassDiceLoss(nn.Module):
    def __init__(self, include_background=False, smooth=1e-6):
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(target, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
        first_class = 0 if self.include_background else 1
        losses = []
        for cls in range(first_class, NUM_CLASSES):
            p = probs[:, cls]
            t = one_hot[:, cls]
            intersection = torch.sum(p * t, dim=(1, 2))
            denominator = torch.sum(p, dim=(1, 2)) + torch.sum(t, dim=(1, 2))
            dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
            losses.append(1.0 - dice)
        return torch.stack(losses, dim=0).mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = MulticlassDiceLoss(include_background=INCLUDE_BACKGROUND_IN_DICE_LOSS)

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        dice_loss = self.dice(logits, target)
        total_loss = CE_LOSS_WEIGHT * ce_loss + DICE_LOSS_WEIGHT * dice_loss
        return total_loss, ce_loss, dice_loss


def dice_from_logits(logits, target):
    pred = torch.argmax(logits, dim=1)
    scores = []
    for cls in range(NUM_CLASSES):
        p = pred == cls
        t = target == cls
        intersection = torch.sum(p & t).float()
        denominator = torch.sum(p).float() + torch.sum(t).float()
        if float(denominator) == 0.0:
            score = 1.0
        else:
            score = float((2.0 * intersection / denominator).detach().cpu())
        scores.append(score)
    return scores


def run_train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_losses, ce_losses, dice_losses = [], [], []
    dice_values = [[], [], []]
    for batch in loader:
        image = batch["image"].to(DEVICE, non_blocking=True)
        label = batch["label"].to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = model(image)
            loss, ce_loss, dice_loss = criterion(logits, label)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_losses.append(float(loss.detach().cpu()))
        ce_losses.append(float(ce_loss.detach().cpu()))
        dice_losses.append(float(dice_loss.detach().cpu()))
        scores = dice_from_logits(logits.detach(), label)
        for cls in range(NUM_CLASSES):
            dice_values[cls].append(scores[cls])
    return {
        "loss": float(np.mean(total_losses)),
        "ce_loss": float(np.mean(ce_losses)),
        "dice_loss": float(np.mean(dice_losses)),
        "dice_background": float(np.mean(dice_values[0])),
        "dice_normal_wmh": float(np.mean(dice_values[1])),
        "dice_abnormal_wmh": float(np.mean(dice_values[2])),
    }


def write_final_training_log_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch", "train_loss", "train_ce_loss", "train_dice_loss",
        "train_dice_background", "train_dice_normal_wmh", "train_dice_abnormal_wmh",
        "learning_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()


def append_final_training_log(path, epoch, metrics, optimizer):
    fields = [
        "epoch", "train_loss", "train_ce_loss", "train_dice_loss",
        "train_dice_background", "train_dice_normal_wmh", "train_dice_abnormal_wmh",
        "learning_rate",
    ]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerow({
            "epoch": epoch,
            "train_loss": metrics["loss"],
            "train_ce_loss": metrics["ce_loss"],
            "train_dice_loss": metrics["dice_loss"],
            "train_dice_background": metrics["dice_background"],
            "train_dice_normal_wmh": metrics["dice_normal_wmh"],
            "train_dice_abnormal_wmh": metrics["dice_abnormal_wmh"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })


def dice_and_sensitivity(pred, target, cls):
    p = pred == cls
    t = target == cls
    tp = np.count_nonzero(p & t)
    fp = np.count_nonzero(p & ~t)
    fn = np.count_nonzero(~p & t)
    denom = 2 * tp + fp + fn
    dice = 1.0 if denom == 0 else 2.0 * tp / denom
    gt_positive = tp + fn
    sensitivity = math.nan if gt_positive == 0 else tp / gt_positive
    return float(dice), float(sensitivity)


def hd95(pred_mask, target_mask, spacing):
    pred_mask = pred_mask.astype(bool)
    target_mask = target_mask.astype(bool)
    pred_exists = bool(pred_mask.any())
    target_exists = bool(target_mask.any())
    if not pred_exists and not target_exists:
        return 0.0
    if pred_exists != target_exists:
        return math.nan
    structure = np.ones((3, 3, 3), dtype=bool)
    pred_surface = pred_mask ^ binary_erosion(pred_mask, structure=structure, border_value=0)
    target_surface = target_mask ^ binary_erosion(target_mask, structure=structure, border_value=0)
    if not pred_surface.any() or not target_surface.any():
        return math.nan
    distance_to_target = distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([
        distance_to_target[pred_surface],
        distance_to_pred[target_surface],
    ])
    return float(np.percentile(distances, 95))


def find_flair_for_patient(patient_id):
    patient_dir = PREPROCESSED_ROOT / patient_id
    candidates = [
        p for p in patient_dir.rglob("*.nii*")
        if "flair" in p.name.lower() and "mask" not in p.name.lower()
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"{patient_id}: expected exactly one FLAIR in {patient_dir}; got {len(candidates)}"
        )
    return candidates[0]


@torch.no_grad()
def evaluate_test_3d(model, test_loader, test_patient_ids):
    model.eval()
    pred_volumes, target_volumes, spacing_map = {}, {}, {}
    for patient_id in sorted(test_patient_ids):
        flair_img = nib.load(str(find_flair_for_patient(patient_id)))
        shape = flair_img.shape[:3]
        spacing = tuple(float(x) for x in flair_img.header.get_zooms()[:3])
        pred_volumes[patient_id] = np.zeros(shape, dtype=np.uint8)
        target_volumes[patient_id] = np.zeros(shape, dtype=np.uint8)
        spacing_map[patient_id] = spacing

    for batch in test_loader:
        image = batch["image"].to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = model(image)
        prediction = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
        target = batch["label"].numpy().astype(np.uint8)
        patient_ids = list(batch["patient_id"])
        slice_indices = batch["slice_index"]
        if torch.is_tensor(slice_indices):
            slice_indices = slice_indices.cpu().numpy().tolist()
        for i, patient_id in enumerate(patient_ids):
            z = int(slice_indices[i])
            pred_volumes[patient_id][:, :, z] = prediction[i]
            target_volumes[patient_id][:, :, z] = target[i]

    per_patient_rows = []
    for patient_id in sorted(test_patient_ids):
        pred = pred_volumes[patient_id]
        target = target_volumes[patient_id]
        spacing = spacing_map[patient_id]
        row = {"patient_id": patient_id}
        for cls, class_name in enumerate(CLASS_NAMES):
            dice, sensitivity = dice_and_sensitivity(pred, target, cls)
            hausdorff = hd95(pred == cls, target == cls, spacing)
            row[f"dice_{class_name}"] = dice
            row[f"sensitivity_{class_name}"] = sensitivity
            row[f"hd95_{class_name}"] = hausdorff
        per_patient_rows.append(row)

    aggregate = {"n_test_patients": len(per_patient_rows)}
    for metric in METRIC_COLUMNS:
        values = [float(row[metric]) for row in per_patient_rows]
        mean, std = finite_mean_std(values)
        finite_count = int(np.count_nonzero(np.isfinite(np.asarray(values, dtype=float))))
        aggregate[f"{metric}_mean"] = mean
        aggregate[f"{metric}_std"] = std
        aggregate[f"{metric}_n_finite"] = finite_count
    return per_patient_rows, aggregate


def save_partition_results(
    per_patient_rows,
    aggregate,
    per_patient_csv,
    aggregate_json,
    aggregate_csv,
):
    per_patient_csv.parent.mkdir(parents=True, exist_ok=True)

    with per_patient_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id"] + METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(per_patient_rows)

    atomic_write_text(
        aggregate_json,
        json.dumps(aggregate, indent=2),
    )

    with aggregate_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "mean", "std", "n_finite"],
        )
        writer.writeheader()

        for metric in METRIC_COLUMNS:
            writer.writerow({
                "metric": metric,
                "mean": aggregate[f"{metric}_mean"],
                "std": aggregate[f"{metric}_std"],
                "n_finite": aggregate[f"{metric}_n_finite"],
            })


def save_combined_final_report(train_summary, cv_summary, test_summary):
    FINAL_REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with FINAL_REPORT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "train70_mean",
                "train70_std",
                "cv_mean",
                "cv_std",
                "test30_mean",
                "test30_std",
            ],
        )
        writer.writeheader()

        for metric in METRIC_COLUMNS:
            writer.writerow({
                "metric": metric,
                "train70_mean": train_summary[f"{metric}_mean"],
                "train70_std": train_summary[f"{metric}_std"],
                "cv_mean": cv_summary[f"{metric}_mean"],
                "cv_std": cv_summary[f"{metric}_std"],
                "test30_mean": test_summary[f"{metric}_mean"],
                "test30_std": test_summary[f"{metric}_std"],
            })


def save_evaluation_plan():
    plan = {
        "current_baseline_outputs": {
            "global_train70": [
                "Dice per class",
                "Sensitivity per class",
                "HD95 mm per class",
                "Per-patient metrics"
            ],
            "global_5fold_cv": [
                "Mean and SD Dice per class",
                "Mean and SD Sensitivity per class",
                "Mean and SD HD95 mm per class"
            ],
            "global_test30": [
                "Dice per class",
                "Sensitivity per class",
                "HD95 mm per class",
                "Per-patient metrics"
            ],
            "training_history": [
                "Total loss",
                "Cross-entropy loss",
                "Dice loss",
                "Background Dice",
                "Normal-WMH Dice",
                "Abnormal-WMH Dice"
            ]
        },
        "full_study_evaluation_target": {
            "global": (
                "Global Dice, Hausdorff95, and sensitivity per class across "
                "all 4 backbones, comparing with versus without distance conditioning."
            ),
            "regional": (
                "Regional Dice and false-positive rate inside the PV zone "
                "versus the deep zone."
            ),
            "fixed_10mm_rule_comparison": (
                "Compare against the fixed 10 mm-rule baseline using the "
                "predefined false-positive-reduction metric."
            )
        },
        "important": (
            "This Stage 14 file computes the current plain U-Net baseline global "
            "train/CV/test metrics only. The 4-backbone, distance-conditioned, "
            "regional PV/deep, and fixed-10-mm comparison metrics must be computed "
            "after those additional model outputs and the exact FP-reduction "
            "definition are available; they are not fabricated here."
        )
    }

    atomic_write_text(
        EVALUATION_PLAN_JSON,
        json.dumps(plan, indent=2),
    )


def main():
    seed_everything(RANDOM_SEED)

    print("=" * 88)
    print("STAGE 14 — CONTINUE FROM EXISTING 5-FOLD CV RESULTS")
    print("=" * 88)
    print("\nDevice:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # IMPORTANT:
    # This script DOES NOT rerun Fold 1-5 training.
    # It only reads the already-generated fold_X/final_metrics.json files.
    cv_rows = load_completed_cv_metrics()
    best_epochs = [int(row["best_epoch"]) for row in cv_rows]
    final_epochs = choose_final_training_epochs(cv_rows)

    print("\nAll 5 existing CV folds found.")
    print("Fold 1-5 training will NOT be rerun.")
    print("Continuing from saved CV results/checkpoints only.")
    print("Fold best epochs:", best_epochs)
    print("Final training epochs (median CV best epoch):", final_epochs)

    fold1_dir = SPLIT_ROOT / "fold_1"
    dev_train_ids = set(read_patient_ids(fold1_dir / "train_patients.csv"))
    dev_val_ids = set(read_patient_ids(fold1_dir / "val_patients.csv"))
    test_ids = set(read_patient_ids(SPLIT_ROOT / "test_30_patients.csv"))

    if dev_train_ids & dev_val_ids:
        raise RuntimeError("Fold 1 train/val overlap detected.")
    development_ids = dev_train_ids | dev_val_ids
    if len(development_ids) != 70:
        raise RuntimeError(f"Expected exactly 70 development patients, got {len(development_ids)}")
    if len(test_ids) != 30:
        raise RuntimeError(f"Expected exactly 30 test patients, got {len(test_ids)}")
    if development_ids & test_ids:
        raise RuntimeError("LEAKAGE: development and final test patients overlap.")

    print("\nFinal membership safeguard: PASS")
    print("  development patients:", len(development_ids))
    print("  final test patients :", len(test_ids))
    print("  overlap             : 0")

    development_dataset = MultiPartitionSliceDataset(
        partition_roots=[fold1_dir / "train", fold1_dir / "val"],
        expected_patient_ids=development_ids,
        augment=True,
    )
    development_eval_dataset = MultiPartitionSliceDataset(
        partition_roots=[fold1_dir / "train", fold1_dir / "val"],
        expected_patient_ids=development_ids,
        augment=False,
    )
    test_dataset = MultiPartitionSliceDataset(
        partition_roots=[SPLIT_ROOT / "test"],
        expected_patient_ids=test_ids,
        augment=False,
    )

    print("\nDevelopment slices:", len(development_dataset))
    print("Test slices       :", len(test_dataset))

    if FINAL_LOG_DIR.exists():
        shutil.rmtree(FINAL_LOG_DIR)
    if FINAL_CKPT_DIR.exists():
        shutil.rmtree(FINAL_CKPT_DIR)
    FINAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    cv_summary = save_overall_cv_metrics(cv_rows)

    class_counts = compute_class_voxel_counts(development_dataset)
    class_weights = class_weights_from_counts(class_counts)

    print("\nFinal 70-patient class distribution:")
    for cls, name in enumerate(CLASS_NAMES):
        print(
            f"  {name:<14} voxels={int(class_counts[cls]):>14,d} "
            f"weight={float(class_weights[cls]):.6f}"
        )

    atomic_write_text(
        FINAL_LOG_DIR / "final70_config.json",
        json.dumps(
            {
                "class_counts": {CLASS_NAMES[i]: int(class_counts[i]) for i in range(NUM_CLASSES)},
                "class_weights": {CLASS_NAMES[i]: float(class_weights[i]) for i in range(NUM_CLASSES)},
                "class_weight_power": CLASS_WEIGHT_POWER,
                "training_patients": 70,
                "test_patients": 30,
                "final_epochs": final_epochs,
                "cv_best_epochs": best_epochs,
                "epoch_selection_rule": "median of 5 CV best epochs; fixed before test evaluation",
            },
            indent=2,
        ),
    )

    development_loader = DataLoader(
        development_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )
    development_eval_loader = DataLoader(
        development_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
    )

    model = UNet2D(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    criterion = CombinedLoss(class_weight_tensor)
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    write_final_training_log_header(FINAL_TRAIN_LOG)

    print("\n" + "=" * 88)
    print(f"TRAINING FINAL MODEL ON ALL 70 DEVELOPMENT PATIENTS FOR {final_epochs} EPOCHS")
    print("=" * 88)

    for epoch in range(1, final_epochs + 1):
        metrics = run_train_epoch(model, development_loader, optimizer, criterion, scaler)
        print(f"\nFinal70 | Epoch {epoch:03d}/{final_epochs}")
        print(
            "  train total/CE/Dice loss: "
            f"{metrics['loss']:.5f} / {metrics['ce_loss']:.5f} / {metrics['dice_loss']:.5f}"
        )
        print(
            "  train Dice BG/nWMH/abWMH: "
            f"{metrics['dice_background']:.4f}/"
            f"{metrics['dice_normal_wmh']:.4f}/"
            f"{metrics['dice_abnormal_wmh']:.4f}"
        )
        append_final_training_log(FINAL_TRAIN_LOG, epoch, metrics, optimizer)

    atomic_torch_save(
        FINAL_MODEL_PATH,
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epochs": final_epochs,
            "epoch_selection_rule": "median of 5 CV best epochs",
            "cv_best_epochs": best_epochs,
            "class_counts": class_counts.tolist(),
            "class_weights": class_weights.tolist(),
            "channel_order": CHANNEL_ORDER,
            "num_classes": NUM_CLASSES,
            "development_patients": 70,
        },
    )

    print("\nFinal 70-patient model saved:")
    print(FINAL_MODEL_PATH)

    print("\n" + "=" * 88)
    print("FINAL 70-PATIENT TRAINING METRICS — 3D PATIENT-WISE EVALUATION")
    print("=" * 88)

    train_per_patient_rows, train_summary = evaluate_test_3d(
        model,
        development_eval_loader,
        development_ids,
    )
    train_summary["partition"] = "train70"
    train_summary["final_training_epochs"] = final_epochs

    save_partition_results(
        train_per_patient_rows,
        train_summary,
        TRAIN_PATIENT_METRICS_CSV,
        TRAIN_METRICS_JSON,
        TRAIN_METRICS_CSV,
    )

    print("\n" + "=" * 88)
    print("RUNNING ONE-TIME FINAL EVALUATION ON THE 30-PATIENT TEST SET")
    print("=" * 88)

    per_patient_rows, test_summary = evaluate_test_3d(model, test_loader, test_ids)
    test_summary["final_training_epochs"] = final_epochs
    test_summary["cv_best_epochs"] = best_epochs
    test_summary["test_evaluation_policy"] = (
        "single evaluation after all model/epoch choices were fixed from CV"
    )

    save_partition_results(
        per_patient_rows,
        test_summary,
        TEST_PATIENT_METRICS_CSV,
        TEST_METRICS_JSON,
        TEST_METRICS_CSV,
    )
    save_combined_final_report(
        train_summary,
        cv_summary,
        test_summary,
    )
    save_evaluation_plan()

    print("\n" + "=" * 88)
    print("FINAL 70-PATIENT TRAINING METRICS — PATIENT MEAN ± SD")
    print("=" * 88)
    for class_name in ["normal_wmh", "abnormal_wmh"]:
        print("\n" + class_name.upper())
        for metric in ["dice", "sensitivity", "hd95"]:
            key = f"{metric}_{class_name}"
            mean = train_summary[f"{key}_mean"]
            std = train_summary[f"{key}_std"]
            n_finite = train_summary[f"{key}_n_finite"]
            unit = " mm" if metric == "hd95" else ""
            print(
                f"  {metric:<12}: {mean:.6f} ± {std:.6f}{unit} "
                f"(finite n={n_finite}/70)"
            )

    print("\n" + "=" * 88)
    print("OVERALL 5-FOLD CROSS-VALIDATION METRICS — MEAN ± SD")
    print("=" * 88)
    for class_name in ["normal_wmh", "abnormal_wmh"]:
        print("\n" + class_name.upper())
        for metric in ["dice", "sensitivity", "hd95"]:
            key = f"{metric}_{class_name}"
            mean = cv_summary[f"{key}_mean"]
            std = cv_summary[f"{key}_std"]
            unit = " mm" if metric == "hd95" else ""
            print(f"  {metric:<12}: {mean:.6f} ± {std:.6f}{unit}")

    print("\n" + "=" * 88)
    print("FINAL 30-PATIENT TEST METRICS — PATIENT MEAN ± SD")
    print("=" * 88)
    for class_name in ["normal_wmh", "abnormal_wmh"]:
        print("\n" + class_name.upper())
        for metric in ["dice", "sensitivity", "hd95"]:
            key = f"{metric}_{class_name}"
            mean = test_summary[f"{key}_mean"]
            std = test_summary[f"{key}_std"]
            n_finite = test_summary[f"{key}_n_finite"]
            unit = " mm" if metric == "hd95" else ""
            print(
                f"  {metric:<12}: {mean:.6f} ± {std:.6f}{unit} "
                f"(finite n={n_finite}/30)"
            )

    print("\n" + "=" * 88)
    print("STAGE 14 COMPLETE")
    print("=" * 88)
    print("\nFinal results folder:")
    print(FINAL_LOG_DIR)
    print("\nTraining history:")
    print(FINAL_TRAIN_LOG)
    print("\nFinal 70-patient train metrics:")
    print(TRAIN_METRICS_CSV)
    print("\nPer-patient train metrics:")
    print(TRAIN_PATIENT_METRICS_CSV)
    print("\nOverall 5-fold CV metrics:")
    print(OVERALL_CV_CSV)
    print("\nPer-patient test metrics:")
    print(TEST_PATIENT_METRICS_CSV)
    print("\nFinal 30-patient test metrics:")
    print(TEST_METRICS_CSV)
    print("\nCombined train + CV + test report:")
    print(FINAL_REPORT_CSV)
    print("\nFull-study evaluation plan:")
    print(EVALUATION_PLAN_JSON)


if __name__ == "__main__":
    main()
