from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
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


# ============================================================
# STAGE 16 — DISTANCE-MAP U-NET (4 input channels, 5 classes)
# ============================================================
#
# Input  : splits/baseline_2d slices (T1, T2, FLAIR) + distance channel
#          taken on the fly from distance_map/<id>/<id>_VentDist_norm
#          at the same slice index  ->  [4, H, W]
# Target : six_class_masks/<id>/<id>_SixClass_Mask (5 labels, see below)
#          taken at the same slice index
#
#   0 = Background
#   1 = Ventricle
#   2 = Normal WMH
#   3 = PV Abnormal WMH    (d <= 10 mm)
#   4 = Deep Abnormal WMH  (d >  10 mm)
#
# Same patients / folds / slices as the baseline (splits/baseline_2d).
#
# Pipeline:
#   1. 5-fold CV on the 70 development patients
#   2. retrain on all 70 for median(best epochs) epochs
#   3. evaluate the 30 test patients ONCE (refuses to re-run)
#
# Fixes vs. scripts 13/14:
#   - seed reset per fold (RANDOM_SEED + fold)
#   - Dice loss computed over the whole batch (no empty-slice penalty)
#   - best epoch = patient-level 3D Dice of abnormal WMH (labels 3+4)
#   - Dice / sensitivity only over patients that have the class;
#     excluded and NaN-HD95 counts are reported
#   - intensity augmentation on brain voxels of T1/T2/FLAIR only;
#     distance channel never intensity-augmented
#   - evaluation against the FULL 3D ground-truth volume
#
# Usage:
#   python 16_distance_map_unet.py              # CV (all folds) + final + test
#   python 16_distance_map_unet.py --fold 3     # one CV fold only
#   python 16_distance_map_unet.py --cv-only    # all CV folds, no final/test
#   python 16_distance_map_unet.py --final-only # final-70 + test (CV must exist)
# ============================================================


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = BASE_DIR

SPLIT_ROOT = PROJECT_DIR / "splits" / "baseline_2d"
PREPROCESSED_ROOT = PROJECT_DIR / "preprocessed"
DISTANCE_ROOT = PROJECT_DIR / "distance_map"
LABEL_ROOT = PROJECT_DIR / "six_class_masks"

LOG_ROOT = PROJECT_DIR / "logs" / "distance_map_unet"
CHECKPOINT_ROOT = PROJECT_DIR / "checkpoints" / "distance_map_unet"
FINAL_LOG_DIR = LOG_ROOT / "final_results"
FINAL_CKPT_DIR = CHECKPOINT_ROOT / "final70"
TEST_METRICS_JSON = FINAL_LOG_DIR / "test30_summary.json"

NUM_CLASSES = 5
CLASS_NAMES = ["background", "ventricle", "normal_wmh", "pv_abnormal_wmh", "deep_abnormal_wmh"]
CLASS_COLORS = ["black", "cyan", "yellow", "red", "magenta"]

# evaluation targets: name -> label values merged into one binary mask
EVAL_TARGETS = {
    "ventricle": (1,),
    "normal_wmh": (2,),
    "pv_abnormal_wmh": (3,),
    "deep_abnormal_wmh": (4,),
    "abnormal_wmh": (3, 4),      # comparable to baseline abWMH
    "all_wmh": (2, 3, 4),
}
SELECTION_TARGET = "abnormal_wmh"

IMAGE_CHANNEL_ORDER = ["T1", "T2", "FLAIR"]
CHANNEL_ORDER = ["T1", "T2", "FLAIR", "DIST"]
IN_CHANNELS = len(CHANNEL_ORDER)
BASE_CHANNELS = 32

EPOCHS = 50
BATCH_SIZE = 8
NUM_WORKERS = 0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 20260911

CE_LOSS_WEIGHT = 1.0
DICE_LOSS_WEIGHT = 1.0
DICE_SMOOTH = 1.0
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

N_VAL_PNG_PATIENTS = 4          # prediction PNGs per CV fold
N_PNG_SLICES = 3                # slices per patient PNG
SAVE_VAL_PNG_EVERY = 10         # also save val prediction PNGs every N epochs (0 = off)
CHECKPOINT_EVERY = 0            # also keep epoch_XXX.pth every N epochs (0 = off)

# LR scheduler: cosine annealing LEARNING_RATE -> MIN_LEARNING_RATE over
# EPOCHS, stepped once per epoch. Fixed schedule (no validation needed),
# so the final model follows the same curve and stops at the median best
# epoch. The baseline must use the same setting for a fair comparison.
# No gradient clipping (same as baseline scripts 13/14).
USE_LR_SCHEDULER = True
MIN_LEARNING_RATE = 1e-6

# --smoke: fold 1 only, a few dev patients, 2 epochs, separate output
# folders, never touches final training or the test set.
SMOKE_TEST = "--smoke" in sys.argv
SMOKE_TRAIN_PATIENTS = 3
SMOKE_VAL_PATIENTS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

ONLY_FOLD = None
CV_ONLY = "--cv-only" in sys.argv
FINAL_ONLY = "--final-only" in sys.argv
if "--fold" in sys.argv:
    idx = sys.argv.index("--fold")
    if idx + 1 >= len(sys.argv):
        raise RuntimeError("--fold requires a value from 1 to 5")
    ONLY_FOLD = int(sys.argv[idx + 1])
    if ONLY_FOLD not in {1, 2, 3, 4, 5}:
        raise RuntimeError("--fold must be between 1 and 5")

if SMOKE_TEST:
    LOG_ROOT = PROJECT_DIR / "logs" / "distance_map_unet_smoke"
    CHECKPOINT_ROOT = PROJECT_DIR / "checkpoints" / "distance_map_unet_smoke"
    EPOCHS = 2
    SAVE_VAL_PNG_EVERY = 1
    CHECKPOINT_EVERY = 1
    ONLY_FOLD = ONLY_FOLD or 1
    if "--epochs" in sys.argv:
        EPOCHS = int(sys.argv[sys.argv.index("--epochs") + 1])


# ============================================================
# UTILITIES
# ============================================================

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


def write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, indent=2, default=float))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_patient_ids(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    rows = read_csv(path)
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


def finite_stats(values):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": math.nan, "std": math.nan, "median": math.nan, "n": 0}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)),
        "n": int(finite.size),
    }


# ============================================================
# 3D VOLUME CACHE (distance channel, labels, FLAIR, geometry)
# ============================================================

_VOLUME_CACHE = {}


def get_patient_volumes(pid):
    if pid not in _VOLUME_CACHE:
        dist_img = nib.load(str(DISTANCE_ROOT / pid / f"{pid}_VentDist_norm.nii.gz"))
        label_img = nib.load(str(LABEL_ROOT / pid / f"{pid}_SixClass_Mask.nii.gz"))
        flair_img = nib.load(str(PREPROCESSED_ROOT / pid / f"{pid}_FLAIR.nii.gz"))

        dist = np.asanyarray(dist_img.dataobj).astype(np.float32)
        label = np.asanyarray(label_img.dataobj).astype(np.uint8)

        if not (dist.shape == label.shape == flair_img.shape[:3]):
            raise RuntimeError(f"{pid}: shape mismatch dist={dist.shape} label={label.shape} flair={flair_img.shape}")
        if not np.allclose(dist_img.affine, flair_img.affine, atol=1e-4):
            raise RuntimeError(f"{pid}: distance map affine differs from FLAIR")
        if not np.allclose(label_img.affine, flair_img.affine, atol=1e-4):
            raise RuntimeError(f"{pid}: label affine differs from FLAIR")
        if dist.min() < 0 or dist.max() > 1 or not np.isfinite(dist).all():
            raise RuntimeError(f"{pid}: distance channel outside [0, 1]")
        if label.max() >= NUM_CLASSES:
            raise RuntimeError(f"{pid}: unexpected label {label.max()}")

        _VOLUME_CACHE[pid] = {
            "dist": dist,
            "label": label,
            "affine": flair_img.affine,
            "header": flair_img.header,
            "spacing": tuple(float(x) for x in flair_img.header.get_zooms()[:3]),
        }
    return _VOLUME_CACHE[pid]


def load_flair(pid):
    return np.asanyarray(nib.load(str(PREPROCESSED_ROOT / pid / f"{pid}_FLAIR.nii.gz")).dataobj).astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class DistanceSliceDataset(Dataset):
    def __init__(self, partition_roots, expected_patient_ids, augment=False, allow_subset=False):
        # allow_subset (smoke test only): skip slices of patients not in expected_patient_ids
        self.expected_patient_ids = set(expected_patient_ids)
        self.augment = augment
        self.samples = []
        seen = set()

        for root in partition_roots:
            image_dir = Path(root) / "images"
            if not image_dir.exists():
                raise FileNotFoundError(f"Missing image directory: {image_dir}")
            image_paths = sorted(image_dir.glob("*.npz"))
            if not image_paths:
                raise RuntimeError(f"No images found in {image_dir}")
            for image_path in image_paths:
                pid = patient_id_from_filename(image_path)
                if pid not in self.expected_patient_ids:
                    if allow_subset:
                        continue
                    raise RuntimeError(f"{image_path.name}: patient {pid} not allowed in this dataset")
                if image_path.name in seen:
                    raise RuntimeError(f"Duplicate slice across partitions: {image_path.name}")
                seen.add(image_path.name)
                with np.load(image_path, allow_pickle=False) as z:
                    slice_index = int(z["slice_index"])
                    stored_pid = str(z["patient_id"]).zfill(3)
                if stored_pid != pid:
                    raise RuntimeError(f"{image_path.name}: stored patient_id {stored_pid} != filename {pid}")
                self.samples.append((image_path, pid, slice_index))

        actual = {pid for _, pid, _ in self.samples}
        if actual != self.expected_patient_ids:
            raise RuntimeError(
                f"Dataset patient mismatch. missing={sorted(self.expected_patient_ids - actual)}, "
                f"extra={sorted(actual - self.expected_patient_ids)}"
            )
        for pid in sorted(actual):
            get_patient_volumes(pid)

    def __len__(self):
        return len(self.samples)

    def _augment(self, image, label):
        # spatial: identical for all 4 channels and the label
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

        # intensity: MRI channels only, brain voxels only
        brain = np.any(image[:3] != 0, axis=0)
        if random.random() < P_INTENSITY_SCALE:
            factor = random.uniform(*INTENSITY_SCALE_RANGE)
            image[:3, brain] *= factor
        if random.random() < P_INTENSITY_SHIFT:
            offset = random.uniform(*INTENSITY_SHIFT_RANGE)
            image[:3, brain] += offset
        return image, label

    def __getitem__(self, index):
        image_path, pid, z = self.samples[index]
        with np.load(image_path, allow_pickle=False) as npz:
            image = npz["image"].astype(np.float32, copy=True)
            channel_order = [str(x) for x in npz["channel_order"].tolist()]
        if channel_order != IMAGE_CHANNEL_ORDER:
            raise RuntimeError(f"Channel order mismatch in {image_path}: {channel_order}")
        if image.ndim != 3 or image.shape[0] != 3:
            raise RuntimeError(f"Expected image [3,H,W], got {image.shape} in {image_path}")

        vols = get_patient_volumes(pid)
        dist = vols["dist"][:, :, z]
        label = vols["label"][:, :, z].astype(np.int64)
        if dist.shape != image.shape[1:]:
            raise RuntimeError(f"{image_path.name}: distance slice {dist.shape} vs image {image.shape}")

        image = np.concatenate([image, dist[None]], axis=0).astype(np.float32)

        if self.augment and USE_AUGMENTATION:
            image, label = self._augment(image, label)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "label": torch.from_numpy(np.ascontiguousarray(label)),
            "patient_id": pid,
            "slice_index": z,
        }


def make_loader(dataset, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        generator=generator,
        drop_last=False,
    )


def compute_class_voxel_counts(dataset):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for _, pid, z in dataset.samples:
        counts += np.bincount(get_patient_volumes(pid)["label"][:, :, z].ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    if np.any(counts == 0):
        raise RuntimeError(f"At least one class has zero training voxels: {counts.tolist()}")
    return counts


def class_weights_from_counts(counts):
    frequencies = counts.astype(np.float64) / counts.sum()
    weights = 1.0 / np.power(frequencies, CLASS_WEIGHT_POWER)
    return (weights / weights.mean()).astype(np.float32)


# ============================================================
# LOSS
# ============================================================

class BatchDiceLoss(nn.Module):
    """Soft Dice over the whole batch (dim 0,2,3), so empty slices do not dominate."""

    def __init__(self, include_background=False, smooth=DICE_SMOOTH):
        super().__init__()
        self.first_class = 0 if include_background else 1
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits.float(), dim=1)
        one_hot = nn.functional.one_hot(target, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dim=dims)
        denominator = torch.sum(probs, dim=dims) + torch.sum(one_hot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return (1.0 - dice[self.first_class:]).mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = BatchDiceLoss(include_background=INCLUDE_BACKGROUND_IN_DICE_LOSS)

    def forward(self, logits, target):
        ce_loss = self.ce(logits.float(), target)
        dice_loss = self.dice(logits, target)
        return CE_LOSS_WEIGHT * ce_loss + DICE_LOSS_WEIGHT * dice_loss, ce_loss, dice_loss


# ============================================================
# TRAIN / PREDICT
# ============================================================

def global_dice_from_counts(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else math.nan


def confusion_from_labels(target, pred):
    """[NUM_CLASSES, NUM_CLASSES] voxel counts, rows = ground truth, cols = prediction."""
    idx = target.reshape(-1).to(torch.int64) * NUM_CLASSES + pred.reshape(-1).to(torch.int64)
    return torch.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES).cpu().numpy()


def target_dice_from_confusion(confusion):
    """Global Dice per evaluation target from a confusion matrix.
    Used for BOTH train and val, so the two are calculated the same way."""
    out = {}
    for name, labels in EVAL_TARGETS.items():
        inside = np.isin(np.arange(NUM_CLASSES), labels)
        tp = int(confusion[np.ix_(inside, inside)].sum())
        fp = int(confusion[np.ix_(~inside, inside)].sum())
        fn = int(confusion[np.ix_(inside, ~inside)].sum())
        out[name] = global_dice_from_counts(tp, fp, fn)
    return out


def run_train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    totals = {"loss": 0.0, "ce_loss": 0.0, "dice_loss": 0.0}
    n_batches = 0
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

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

        totals["loss"] += float(loss.detach())
        totals["ce_loss"] += float(ce_loss.detach())
        totals["dice_loss"] += float(dice_loss.detach())
        n_batches += 1

        confusion += confusion_from_labels(label, torch.argmax(logits.detach(), dim=1))

    out = {k: v / max(n_batches, 1) for k, v in totals.items()}
    for name, dice in target_dice_from_confusion(confusion).items():
        out[f"dice_{name}"] = dice
    return out


@torch.no_grad()
def predict_volumes(model, loader, patient_ids, criterion=None):
    """Run the model on all slices and rebuild 3D prediction volumes.
    Slices not present in the split (empty images) stay background."""
    model.eval()
    preds = {pid: np.zeros(get_patient_volumes(pid)["label"].shape, dtype=np.uint8) for pid in patient_ids}
    totals = {"loss": 0.0, "ce_loss": 0.0, "dice_loss": 0.0}
    n_batches = 0

    for batch in loader:
        image = batch["image"].to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = model(image)
        if criterion is not None:
            label = batch["label"].to(DEVICE, non_blocking=True)
            loss, ce_loss, dice_loss = criterion(logits, label)
            totals["loss"] += float(loss)
            totals["ce_loss"] += float(ce_loss)
            totals["dice_loss"] += float(dice_loss)
            n_batches += 1
        prediction = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
        slice_indices = batch["slice_index"]
        if torch.is_tensor(slice_indices):
            slice_indices = slice_indices.cpu().numpy().tolist()
        for i, pid in enumerate(batch["patient_id"]):
            preds[pid][:, :, int(slice_indices[i])] = prediction[i]

    losses = {k: v / max(n_batches, 1) for k, v in totals.items()} if criterion is not None else None
    return preds, losses


# ============================================================
# METRICS
# ============================================================

def surface(mask):
    return mask ^ binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)


def hd95(pred_mask, target_mask, spacing):
    """NaN if either mask is empty (reported as a count, never silently dropped)."""
    if not pred_mask.any() or not target_mask.any():
        return math.nan
    ps, ts = surface(pred_mask), surface(target_mask)
    d_to_t = distance_transform_edt(~ts, sampling=spacing)
    d_to_p = distance_transform_edt(~ps, sampling=spacing)
    return float(np.percentile(np.concatenate([d_to_t[ps], d_to_p[ts]]), 95))


def binary_metrics(pred_mask, target_mask, spacing):
    voxel_ml = float(np.prod(spacing)) / 1000.0
    tp = int(np.count_nonzero(pred_mask & target_mask))
    fp = int(np.count_nonzero(pred_mask & ~target_mask))
    fn = int(np.count_nonzero(~pred_mask & target_mask))
    gt = tp + fn
    has_gt = gt > 0
    return {
        "has_gt": int(has_gt),
        "dice": (2.0 * tp / (2 * tp + fp + fn)) if has_gt else math.nan,   # = F1
        "iou": (tp / (tp + fp + fn)) if has_gt else math.nan,
        "sensitivity": (tp / gt) if has_gt else math.nan,                  # = recall
        "precision": (tp / (tp + fp)) if (tp + fp) > 0 else math.nan,
        "hd95": hd95(pred_mask, target_mask, spacing),
        "gt_volume_ml": gt * voxel_ml,
        "pred_volume_ml": (tp + fp) * voxel_ml,
        "fp_volume_ml": fp * voxel_ml,
        "fn_volume_ml": fn * voxel_ml,
        "tp": tp, "fp": fp, "fn": fn,
    }


def evaluate_patients(preds, patient_ids):
    rows = []
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for pid in sorted(patient_ids):
        vols = get_patient_volumes(pid)
        gt, pred, spacing = vols["label"], preds[pid], vols["spacing"]
        confusion += np.bincount(
            gt.astype(np.int64).ravel() * NUM_CLASSES + pred.astype(np.int64).ravel(),
            minlength=NUM_CLASSES * NUM_CLASSES,
        ).reshape(NUM_CLASSES, NUM_CLASSES)
        row = {"patient_id": pid}
        for name, labels in EVAL_TARGETS.items():
            m = binary_metrics(np.isin(pred, labels), np.isin(gt, labels), spacing)
            for k, v in m.items():
                row[f"{name}_{k}"] = v
        rows.append(row)
    return rows, confusion


def summarize_patient_rows(rows):
    summary = {"n_patients": len(rows)}
    for name in EVAL_TARGETS:
        n_with_gt = sum(int(r[f"{name}_has_gt"]) for r in rows)
        summary[f"{name}_n_patients_with_gt"] = n_with_gt
        summary[f"{name}_n_excluded_no_gt"] = len(rows) - n_with_gt
        hd = [r[f"{name}_hd95"] for r in rows]
        summary[f"{name}_n_hd95_nan"] = int(sum(1 for v in hd if not np.isfinite(v)))
        for metric in ("dice", "iou", "sensitivity", "precision", "hd95",
                       "gt_volume_ml", "pred_volume_ml", "fp_volume_ml", "fn_volume_ml"):
            stats = finite_stats([r[f"{name}_{metric}"] for r in rows])
            for k, v in stats.items():
                if k != "n":
                    summary[f"{name}_{metric}_{k}"] = v
        tp = sum(r[f"{name}_tp"] for r in rows)
        fp = sum(r[f"{name}_fp"] for r in rows)
        fn = sum(r[f"{name}_fn"] for r in rows)
        summary[f"{name}_global_dice"] = global_dice_from_counts(tp, fp, fn)
    return summary


def selection_score(rows):
    return finite_stats([r[f"{SELECTION_TARGET}_dice"] for r in rows])["mean"]


# ============================================================
# OUTPUT: NIfTI + PNG + PLOTS
# ============================================================

def save_prediction_nifti(pid, pred, out_dir):
    vols = get_patient_volumes(pid)
    hdr = vols["header"].copy()
    hdr.set_data_dtype(np.uint8)
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), vols["affine"], hdr), str(out_dir / f"{pid}_pred.nii.gz"))


LABEL_CMAP = ListedColormap(CLASS_COLORS)
LABEL_NORM = BoundaryNorm(np.arange(-0.5, NUM_CLASSES + 0.5, 1), NUM_CLASSES)


def pick_slices(gt, n):
    wmh = np.isin(gt, (2, 3, 4)).sum(axis=(0, 1))
    if wmh.max() == 0:
        wmh = (gt > 0).sum(axis=(0, 1))
    order = np.argsort(wmh)[::-1]
    return sorted(int(z) for z in order[:n] if wmh[z] > 0) or [int(gt.shape[2] // 2)]


def overlay(ax, flair2d, label2d, title):
    ax.imshow(np.rot90(flair2d), cmap="gray")
    masked = np.ma.masked_where(label2d == 0, label2d)
    ax.imshow(np.rot90(masked), cmap=LABEL_CMAP, norm=LABEL_NORM, alpha=0.55, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_prediction_png(pid, pred, out_dir, tag=""):
    vols = get_patient_volumes(pid)
    gt, dist = vols["label"], vols["dist"]
    flair = load_flair(pid)
    slices = pick_slices(gt, N_PNG_SLICES)

    fig, axes = plt.subplots(len(slices), 5, figsize=(17, 3.6 * len(slices)), squeeze=False)
    for r, z in enumerate(slices):
        a = axes[r]
        a[0].imshow(np.rot90(flair[:, :, z]), cmap="gray")
        a[0].set_title(f"{pid} z={z} FLAIR", fontsize=9)
        a[0].axis("off")
        a[1].imshow(np.rot90(dist[:, :, z]), cmap="jet", vmin=0, vmax=1)
        a[1].set_title("distance channel", fontsize=9)
        a[1].axis("off")
        overlay(a[2], flair[:, :, z], gt[:, :, z], "ground truth")
        overlay(a[3], flair[:, :, z], pred[:, :, z], "prediction")

        # abnormal WMH error map: TP green, FP red, FN blue
        g = np.isin(gt[:, :, z], (3, 4))
        p = np.isin(pred[:, :, z], (3, 4))
        err = np.zeros(g.shape + (3,), dtype=np.float32)
        err[g & p] = (0, 1, 0)
        err[~g & p] = (1, 0, 0)
        err[g & ~p] = (0, 0.4, 1)
        a[4].imshow(np.rot90(flair[:, :, z]), cmap="gray")
        a[4].imshow(np.rot90(err), alpha=np.rot90((g | p).astype(float)) * 0.8)
        a[4].set_title("abnormal WMH: TP green / FP red / FN blue", fontsize=9)
        a[4].axis("off")

    handles = [Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(1, NUM_CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=NUM_CLASSES - 1, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{pid}{tag}_prediction.png", dpi=110)
    plt.close(fig)


def plot_training_curves(epoch_rows, best_epoch, out_path, title):
    epochs = [int(r["epoch"]) for r in epoch_rows]
    f = lambda key: [float(r[key]) if r.get(key, "") not in ("", None) else math.nan for r in epoch_rows]
    has_val = "val_loss" in epoch_rows[0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(epochs, f("train_loss"), label="train total")
    ax.plot(epochs, f("train_ce_loss"), "--", label="train CE")
    ax.plot(epochs, f("train_dice_loss"), ":", label="train Dice")
    if has_val:
        ax.plot(epochs, f("val_loss"), label="val total")
        ax.plot(epochs, f("val_ce_loss"), "--", label="val CE")
        ax.plot(epochs, f("val_dice_loss"), ":", label="val Dice")
    ax.set_title("Loss")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for i, name in enumerate(EVAL_TARGETS):
        line, = ax.plot(epochs, f(f"train_dice_{name}"), label=f"train {name}")
        if has_val:
            ax.plot(epochs, f(f"val_global_dice_{name}"), "--", color=line.get_color())
    ax.set_title("Global Dice per class (solid = train, dashed = val)")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    if has_val:
        for name in EVAL_TARGETS:
            ax.plot(epochs, f(f"val_{name}_dice"), label=name)
        ax.set_title("Validation patient-level 3D Dice (mean over patients with GT)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    else:
        ax.axis("off")
    ax.set_xlabel("epoch")

    ax = axes[1, 1]
    ax.plot(epochs, f("learning_rate"))
    ax.set_title("Learning rate")
    ax.set_xlabel("epoch")

    if best_epoch is not None:
        for a in axes.ravel():
            if a.has_data():
                a.axvline(best_epoch, color="k", ls="--", lw=0.8)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_class_distribution(counts, weights, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(NUM_CLASSES)
    axes[0].bar(x, counts, color=CLASS_COLORS, edgecolor="k")
    axes[0].set_yscale("log")
    axes[0].set_title("voxel count (log scale)")
    axes[1].bar(x, weights, color=CLASS_COLORS, edgecolor="k")
    axes[1].set_title(f"CE class weight (1 / freq^{CLASS_WEIGHT_POWER}, mean 1)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, rotation=25, ha="right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_patient_boxplot(rows, metric, out_path, title):
    names = list(EVAL_TARGETS)
    data = []
    for name in names:
        vals = np.asarray([r[f"{name}_{metric}"] for r in rows], dtype=float)
        data.append(vals[np.isfinite(vals)])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.boxplot(data, showmeans=True)
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.random.normal(i, 0.04, len(vals)), vals, s=10, alpha=0.6)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([f"{n}\n(n={len(d)})" for n, d in zip(names, data)], fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion(confusion, out_path, title):
    row_sum = confusion.sum(axis=1, keepdims=True)
    norm = np.divide(confusion, row_sum, out=np.zeros_like(confusion, dtype=float), where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, f"{norm[i, j]:.2f}\n{confusion[i, j]}", ha="center", va="center",
                    fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_volume_agreement(rows, out_path, title):
    names = ["normal_wmh", "pv_abnormal_wmh", "deep_abnormal_wmh", "abnormal_wmh"]
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.6))
    for ax, name in zip(axes, names):
        gt = np.asarray([r[f"{name}_gt_volume_ml"] for r in rows], dtype=float)
        pr = np.asarray([r[f"{name}_pred_volume_ml"] for r in rows], dtype=float)
        ax.scatter(gt, pr, s=14)
        lim = max(gt.max(initial=0), pr.max(initial=0)) * 1.05 + 1e-6
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        r = np.corrcoef(gt, pr)[0, 1] if len(gt) > 1 and gt.std() > 0 and pr.std() > 0 else math.nan
        ax.set_title(f"{name}  r={r:.3f}", fontsize=9)
        ax.set_xlabel("GT volume (ml)")
        ax.set_ylabel("predicted volume (ml)")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_evaluation_outputs(preds, rows, confusion, summary, out_dir, prefix, png_patients, save_nifti=True):
    write_csv(out_dir / f"{prefix}_per_patient_metrics.csv", rows)
    write_json(out_dir / f"{prefix}_summary.json", summary)
    write_csv(out_dir / f"{prefix}_summary.csv",
              [{"metric": k, "value": v} for k, v in summary.items()], ["metric", "value"])
    np.savetxt(out_dir / f"{prefix}_confusion_matrix.csv", confusion, fmt="%d", delimiter=",",
               header=",".join(CLASS_NAMES), comments="")

    plot_dir = out_dir / "plots"
    plot_patient_boxplot(rows, "dice", plot_dir / f"{prefix}_dice_boxplot.png", f"{prefix}: per-patient 3D Dice")
    plot_patient_boxplot(rows, "sensitivity", plot_dir / f"{prefix}_sensitivity_boxplot.png",
                         f"{prefix}: per-patient sensitivity")
    plot_patient_boxplot(rows, "hd95", plot_dir / f"{prefix}_hd95_boxplot.png", f"{prefix}: per-patient HD95 (mm)")
    plot_confusion(confusion, plot_dir / f"{prefix}_confusion_matrix.png", f"{prefix}: voxel confusion (row-normalized)")
    plot_volume_agreement(rows, plot_dir / f"{prefix}_volume_agreement.png", f"{prefix}: volume agreement")

    if save_nifti:
        for pid, pred in preds.items():
            save_prediction_nifti(pid, pred, out_dir / "predictions_nifti")
    for pid in png_patients:
        save_prediction_png(pid, preds[pid], out_dir / "predictions_png")


def burden_ordered_png_patients(patient_ids, n):
    """Half highest abnormal burden, rest spread over the remaining patients."""
    burden = sorted(patient_ids, key=lambda p: -int(np.isin(get_patient_volumes(p)["label"], (3, 4)).sum()))
    if n >= len(burden):
        return burden
    top = burden[: max(1, n // 2)]
    rest = burden[len(top):]
    step = max(1, len(rest) // max(1, n - len(top)))
    return top + rest[::step][: n - len(top)]


# ============================================================
# CHECKPOINTS
# ============================================================

def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(map_location=cuda) moves these to GPU; RNG states must be CPU ByteTensors
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def build_model():
    return UNet2D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, base_channels=BASE_CHANNELS).to(DEVICE)


def build_scheduler(optimizer):
    if not USE_LR_SCHEDULER:
        return None
    # T_max = EPOCHS for CV and final model alike (final stops early on the same curve)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LEARNING_RATE)


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def run_config():
    return {
        "channel_order": CHANNEL_ORDER,
        "class_names": CLASS_NAMES,
        "eval_targets": {k: list(v) for k, v in EVAL_TARGETS.items()},
        "selection_target": SELECTION_TARGET,
        "selection_rule": "mean per-patient 3D Dice over val patients with GT",
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "lr_scheduler": (f"CosineAnnealingLR(T_max={EPOCHS}, eta_min={MIN_LEARNING_RATE}), per epoch"
                         if USE_LR_SCHEDULER else "none"),
        "seed": RANDOM_SEED, "seed_rule": "RANDOM_SEED + fold (final model: RANDOM_SEED, as script 14)",
        "dice_loss": f"batch-level soft Dice, smooth={DICE_SMOOTH}, background excluded",
        "class_weight_power": CLASS_WEIGHT_POWER,
        "augmentation": {
            "spatial": "h-flip, v-flip, rot90 on all 4 channels + label",
            "intensity": "scale/shift on T1/T2/FLAIR brain voxels only; DIST never",
        },
        "distance_config": json.loads((DISTANCE_ROOT / "distance_config.json").read_text(encoding="utf-8")),
        "split_root": str(SPLIT_ROOT),
        "label_root": str(LABEL_ROOT),
    }


# ============================================================
# CROSS-VALIDATION FOLD
# ============================================================

def train_fold(fold):
    seed = RANDOM_SEED + fold
    seed_everything(seed)

    fold_dir = SPLIT_ROOT / f"fold_{fold}"
    log_dir = LOG_ROOT / f"fold_{fold}"
    ckpt_dir = CHECKPOINT_ROOT / f"fold_{fold}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if (log_dir / "final_metrics.json").exists():
        print(f"Fold {fold}: final_metrics.json exists, skipping.")
        return json.loads((log_dir / "final_metrics.json").read_text(encoding="utf-8"))

    train_ids = read_patient_ids(fold_dir / "train_patients.csv")
    val_ids = read_patient_ids(fold_dir / "val_patients.csv")
    if set(train_ids) & set(val_ids):
        raise RuntimeError(f"Fold {fold}: train/val overlap")
    test_ids = set(read_patient_ids(SPLIT_ROOT / "test_30_patients.csv"))
    if (set(train_ids) | set(val_ids)) & test_ids:
        raise RuntimeError(f"Fold {fold}: LEAKAGE into test patients")
    if SMOKE_TEST:
        train_ids = train_ids[:SMOKE_TRAIN_PATIENTS]
        val_ids = val_ids[:SMOKE_VAL_PATIENTS]
        print(f"SMOKE TEST: train {train_ids}, val {val_ids}, {EPOCHS} epochs -> {LOG_ROOT}")

    train_ds = DistanceSliceDataset([fold_dir / "train"], train_ids, augment=True, allow_subset=SMOKE_TEST)
    val_ds = DistanceSliceDataset([fold_dir / "val"], val_ids, augment=False, allow_subset=SMOKE_TEST)
    train_loader = make_loader(train_ds, shuffle=True, seed=seed)
    val_loader = make_loader(val_ds, shuffle=False, seed=seed)

    counts = compute_class_voxel_counts(train_ds)
    weights = class_weights_from_counts(counts)
    print(f"\n=== Fold {fold}: {len(train_ids)} train ({len(train_ds)} slices), "
          f"{len(val_ids)} val ({len(val_ds)} slices) ===")
    print("class counts :", counts.tolist())
    print("class weights:", np.round(weights, 4).tolist())
    plot_class_distribution(counts, weights, log_dir / "plots" / "class_distribution.png",
                            f"Fold {fold} training class distribution")

    model = build_model()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    criterion = CombinedLoss(torch.tensor(weights, device=DEVICE))
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    epoch_csv = log_dir / "epoch_log.csv"
    last_path = ckpt_dir / "last.pth"
    best_path = ckpt_dir / "best.pth"
    start_epoch, best_score, best_epoch = 1, -math.inf, None
    epoch_rows = []

    if last_path.exists():
        ckpt = torch.load(last_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        set_rng_state(ckpt["rng"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("loader_generator") is not None:
            train_loader.generator.set_state(ckpt["loader_generator"].cpu())
        start_epoch = ckpt["epoch"] + 1
        best_score, best_epoch = ckpt["best_score"], ckpt["best_epoch"]
        epoch_rows = read_csv(epoch_csv)[: ckpt["epoch"]] if epoch_csv.exists() else []
        print(f"Resuming fold {fold} from epoch {start_epoch} (best epoch so far {best_epoch})")

    write_json(log_dir / "run_config.json", {**run_config(), "fold": fold,
                                             "class_voxel_counts": counts.tolist(),
                                             "class_weights": weights.tolist(),
                                             "train_patients": train_ids, "val_patients": val_ids})

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        train_m = run_train_epoch(model, train_loader, optimizer, criterion, scaler)
        preds, val_losses = predict_volumes(model, val_loader, val_ids, criterion)
        val_rows, val_confusion = evaluate_patients(preds, val_ids)
        score = selection_score(val_rows)

        is_best = np.isfinite(score) and score > best_score
        if is_best:
            best_score, best_epoch = score, epoch
            atomic_torch_save(best_path, {"model": model.state_dict(), "epoch": epoch, "score": score,
                                          "fold": fold, "config": run_config()})

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_m.items()})
        row.update({f"val_{k}": v for k, v in val_losses.items()})
        for name in EVAL_TARGETS:
            row[f"val_{name}_dice"] = finite_stats([r[f"{name}_dice"] for r in val_rows])["mean"]
        for name, dice in target_dice_from_confusion(val_confusion).items():
            row[f"val_global_dice_{name}"] = dice
        row["selection_score"] = score
        row["is_best"] = int(is_best)
        row["learning_rate"] = optimizer.param_groups[0]["lr"]
        row["seconds"] = round(time.time() - t0, 1)
        epoch_rows.append({k: str(v) for k, v in row.items()})
        write_csv(epoch_csv, epoch_rows)
        if scheduler is not None:
            scheduler.step()   # after logging, so learning_rate = LR used in this epoch

        last_payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch, "rng": rng_state(),
            "loader_generator": train_loader.generator.get_state(),
        }
        atomic_torch_save(last_path, last_payload)
        if CHECKPOINT_EVERY and epoch % CHECKPOINT_EVERY == 0:
            atomic_torch_save(ckpt_dir / f"epoch_{epoch:03d}.pth", last_payload)

        if SAVE_VAL_PNG_EVERY and (epoch % SAVE_VAL_PNG_EVERY == 0 or epoch == 1):
            for pid in burden_ordered_png_patients(val_ids, N_VAL_PNG_PATIENTS):
                save_prediction_png(pid, preds[pid], log_dir / "epoch_predictions" / f"epoch_{epoch:03d}")

        print(f"fold {fold} ep {epoch:3d}/{EPOCHS} | train {train_m['loss']:.4f} | val {val_losses['loss']:.4f} | "
              f"val {SELECTION_TARGET} Dice {score:.4f}{'  *best*' if is_best else ''} | {row['seconds']}s")

    if best_epoch is None:
        raise RuntimeError(f"Fold {fold}: no valid best epoch")

    plot_training_curves(epoch_rows, best_epoch, log_dir / "plots" / "training_curves.png",
                         f"Distance U-Net fold {fold} (best epoch {best_epoch})")

    # final evaluation of the best checkpoint
    best = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best["model"])
    preds, val_losses = predict_volumes(model, val_loader, val_ids, criterion)
    rows, confusion = evaluate_patients(preds, val_ids)
    summary = summarize_patient_rows(rows)
    summary.update({"fold": fold, "best_epoch": best_epoch, "selection_score": best_score,
                    "val_loss": val_losses["loss"]})
    save_evaluation_outputs(preds, rows, confusion, summary, log_dir, "val",
                            burden_ordered_png_patients(val_ids, N_VAL_PNG_PATIENTS))
    write_json(log_dir / "final_metrics.json", summary)
    print(f"Fold {fold} done: best epoch {best_epoch}, {SELECTION_TARGET} Dice {best_score:.4f}")
    return summary


# ============================================================
# CROSS-FOLD SUMMARY
# ============================================================

def summarize_cv():
    fold_rows = []
    all_patient_rows = []
    for fold in range(1, 6):
        path = LOG_ROOT / f"fold_{fold}" / "final_metrics.json"
        if not path.exists():
            return None
        fold_rows.append(json.loads(path.read_text(encoding="utf-8")))
        prows = read_csv(LOG_ROOT / f"fold_{fold}" / "val_per_patient_metrics.csv")
        for r in prows:
            r["fold"] = fold
        all_patient_rows.extend(prows)

    out = LOG_ROOT / "cv_summary"
    write_csv(out / "cv_fold_metrics.csv", fold_rows)
    write_csv(out / "cv_all_val_patients_metrics.csv", all_patient_rows)

    keys = [k for k in fold_rows[0] if isinstance(fold_rows[0][k], (int, float)) and k != "fold"]
    summary_rows = []
    for k in keys:
        s = finite_stats([float(r[k]) for r in fold_rows])
        summary_rows.append({"metric": k, "mean": s["mean"], "std": s["std"], "median": s["median"], "n_folds": s["n"]})
    write_csv(out / "cv_mean_std.csv", summary_rows)
    write_json(out / "cv_mean_std.json", {r["metric"]: {"mean": r["mean"], "std": r["std"]} for r in summary_rows})

    # bar plot: mean +- SD across folds (per-patient Dice mean)
    names = list(EVAL_TARGETS)
    means = [finite_stats([r[f"{n}_dice_mean"] for r in fold_rows])["mean"] for n in names]
    stds = [finite_stats([r[f"{n}_dice_mean"] for r in fold_rows])["std"] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(names)), means, yerr=stds, capsize=4, color="steelblue")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice (mean ± SD over 5 folds)")
    ax.set_title("Distance U-Net — 5-fold CV")
    fig.tight_layout()
    (out / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "plots" / "cv_dice_bar.png", dpi=120)
    plt.close(fig)

    # per-fold curves of the selection target
    fig, ax = plt.subplots(figsize=(9, 5))
    for fold in range(1, 6):
        er = read_csv(LOG_ROOT / f"fold_{fold}" / "epoch_log.csv")
        ax.plot([int(r["epoch"]) for r in er], [float(r["selection_score"]) for r in er], label=f"fold {fold}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"val {SELECTION_TARGET} patient-level Dice")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Validation selection metric per fold")
    fig.tight_layout()
    fig.savefig(out / "plots" / "cv_selection_curves.png", dpi=120)
    plt.close(fig)

    for r in all_patient_rows:
        for k, v in r.items():
            if k not in ("patient_id", "fold"):
                r[k] = float(v)
    plot_patient_boxplot(all_patient_rows, "dice", out / "plots" / "cv_all_val_dice_boxplot.png",
                         "5-fold CV: per-patient 3D Dice (all 70 val patients)")
    print(f"CV summary written to {out}")
    return fold_rows


# ============================================================
# FINAL 70 + TEST (once)
# ============================================================

def train_final_and_test():
    if SMOKE_TEST:
        raise RuntimeError("Smoke test must never train the final model or touch the test set.")
    if TEST_METRICS_JSON.exists():
        raise RuntimeError(
            f"Test results already exist:\n{TEST_METRICS_JSON}\n"
            "The test set must be evaluated only once. Refusing to re-run."
        )

    fold_rows = summarize_cv()
    if fold_rows is None:
        raise RuntimeError("All 5 CV folds must be completed before final training.")

    best_epochs = [int(r["best_epoch"]) for r in fold_rows]
    final_epochs = int(np.rint(np.median(best_epochs)))
    if final_epochs < 1:
        raise RuntimeError(f"Invalid final epoch count: {final_epochs}")

    fold1 = SPLIT_ROOT / "fold_1"
    dev_ids = read_patient_ids(SPLIT_ROOT / "train_70_patients.csv")
    test_ids = read_patient_ids(SPLIT_ROOT / "test_30_patients.csv")
    if len(dev_ids) != 70 or len(test_ids) != 30:
        raise RuntimeError(f"Expected 70/30 patients, got {len(dev_ids)}/{len(test_ids)}")
    if set(dev_ids) & set(test_ids):
        raise RuntimeError("LEAKAGE: development and test patients overlap")

    seed = RANDOM_SEED          # same as script 14
    seed_everything(seed)
    train_ds = DistanceSliceDataset([fold1 / "train", fold1 / "val"], dev_ids, augment=True)
    train_loader = make_loader(train_ds, shuffle=True, seed=seed)
    counts = compute_class_voxel_counts(train_ds)
    weights = class_weights_from_counts(counts)

    FINAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(counts, weights, FINAL_LOG_DIR / "plots" / "final70_class_distribution.png",
                            "Final model (70 dev patients) class distribution")
    write_json(FINAL_LOG_DIR / "final_plan.json", {
        **run_config(),
        "cv_best_epochs": best_epochs,
        "final_epochs": final_epochs,
        "epoch_rule": "median of 5 CV best epochs, fixed before test evaluation",
        "lr_schedule_rule": "same cosine curve as CV (T_max=EPOCHS), stopped at final_epochs",
        "class_voxel_counts": counts.tolist(),
        "class_weights": weights.tolist(),
        "n_dev_patients": len(dev_ids), "n_test_patients": len(test_ids),
    })

    print(f"\n=== Final model: {len(dev_ids)} patients ({len(train_ds)} slices), {final_epochs} epochs "
          f"(CV best epochs {best_epochs}) ===")

    model = build_model()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    criterion = CombinedLoss(torch.tensor(weights, device=DEVICE))
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None

    epoch_rows = []
    for epoch in range(1, final_epochs + 1):
        t0 = time.time()
        m = run_train_epoch(model, train_loader, optimizer, criterion, scaler)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in m.items()},
               "learning_rate": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
        epoch_rows.append({k: str(v) for k, v in row.items()})
        write_csv(FINAL_LOG_DIR / "final70_epoch_log.csv", epoch_rows)
        if scheduler is not None:
            scheduler.step()
        print(f"final ep {epoch:3d}/{final_epochs} | train {m['loss']:.4f} | "
              f"pv {m['dice_pv_abnormal_wmh']:.3f} deep {m['dice_deep_abnormal_wmh']:.3f} | {row['seconds']}s")

    atomic_torch_save(FINAL_CKPT_DIR / "final70_model.pth", {
        "model": model.state_dict(), "epochs": final_epochs, "config": run_config()})
    plot_training_curves(epoch_rows, None, FINAL_LOG_DIR / "plots" / "final70_training_curves.png",
                         f"Distance U-Net final model (70 patients, {final_epochs} epochs)")

    # ---- final model on its own 70 training patients (no augmentation), as script 14 ----
    dev_eval_ds = DistanceSliceDataset([fold1 / "train", fold1 / "val"], dev_ids, augment=False)
    dev_eval_loader = make_loader(dev_eval_ds, shuffle=False, seed=seed)
    dev_preds, _ = predict_volumes(model, dev_eval_loader, dev_ids)
    dev_rows, dev_confusion = evaluate_patients(dev_preds, dev_ids)
    train_summary = summarize_patient_rows(dev_rows)
    save_evaluation_outputs(dev_preds, dev_rows, dev_confusion, train_summary, FINAL_LOG_DIR, "train70",
                            [], save_nifti=False)
    del dev_preds

    # ---- single test evaluation ----
    test_ds = DistanceSliceDataset([SPLIT_ROOT / "test"], test_ids, augment=False)
    test_loader = make_loader(test_ds, shuffle=False, seed=seed)
    preds, _ = predict_volumes(model, test_loader, test_ids)
    rows, confusion = evaluate_patients(preds, test_ids)
    summary = summarize_patient_rows(rows)
    summary.update({"final_epochs": final_epochs, "cv_best_epochs": best_epochs})
    save_evaluation_outputs(preds, rows, confusion, summary, FINAL_LOG_DIR, "test30",
                            sorted(test_ids))   # PNG for every test patient

    # combined CV vs test report
    cv = json.loads((LOG_ROOT / "cv_summary" / "cv_mean_std.json").read_text(encoding="utf-8"))
    report = []
    for name in EVAL_TARGETS:
        for metric in ("dice", "sensitivity", "hd95", "fp_volume_ml"):
            key = f"{name}_{metric}_mean"
            report.append({
                "target": name, "metric": metric,
                "train70_mean": train_summary.get(key, math.nan),
                "cv_mean": cv.get(key, {}).get("mean", math.nan),
                "cv_std": cv.get(key, {}).get("std", math.nan),
                "test_mean": summary.get(key, math.nan),
                "test_std": summary.get(f"{name}_{metric}_std", math.nan),
                "test_median": summary.get(f"{name}_{metric}_median", math.nan),
                "test_n_with_gt": summary[f"{name}_n_patients_with_gt"],
                "test_n_hd95_nan": summary[f"{name}_n_hd95_nan"],
            })
    write_csv(FINAL_LOG_DIR / "train_cv_test_report.csv", report)

    print("\n=== TEST (30 patients) ===")
    for name in EVAL_TARGETS:
        print(f"{name:20s} Dice {summary[f'{name}_dice_mean']:.4f} ± {summary[f'{name}_dice_std']:.4f} "
              f"(n={summary[f'{name}_n_patients_with_gt']})  HD95 {summary[f'{name}_hd95_mean']:.2f} mm "
              f"(NaN={summary[f'{name}_n_hd95_nan']})")
    print(f"Outputs in {FINAL_LOG_DIR}")


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Device: {DEVICE}  AMP: {USE_AMP}")
    for p in (SPLIT_ROOT, DISTANCE_ROOT / "distance_config.json", LABEL_ROOT):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    if not FINAL_ONLY:
        folds = [ONLY_FOLD] if ONLY_FOLD else [1, 2, 3, 4, 5]
        for fold in folds:
            try:
                train_fold(fold)
            except Exception:
                with (LOG_ROOT / "fold_failures.txt").open("a", encoding="utf-8") as f:
                    f.write(f"\n=== fold {fold} {time.ctime()} ===\n{traceback.format_exc()}")
                raise
        if ONLY_FOLD is None:
            summarize_cv()

    if ONLY_FOLD is None and not CV_ONLY:
        train_final_and_test()


if __name__ == "__main__":
    main()
