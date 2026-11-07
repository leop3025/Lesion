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
from scipy.ndimage import binary_erosion, distance_transform_edt, label as connected_components
from torch.utils.data import Dataset, DataLoader

THIS_FILE = Path(__file__).resolve()
BASE_DIR = THIS_FILE.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from unet2d import DoubleConv


# ============================================================
# STAGE 21 — DISTANCE-MAP UNet++ v2 (2.5D input, 5 classes)
#
# Identical to script 17 except the network: UNet2D -> UNet++ 2D
# (nested skips, no deep supervision, same DoubleConv blocks/widths).
# ============================================================
#
# Improvements over script 16 (all chosen on CV, never on the test set):
#   - 2.5D input: T1/T2/FLAIR of slices z-1, z, z+1 (CONTEXT_SLICES = 1),
#     read from the full preprocessed volumes (edge slices replicated)
#   - cosine LR schedule: CV over EPOCHS = 80, final model over final_epochs
#   - flip test-time augmentation (TTA) for all final evaluations
#   - small-lesion post-processing: abnormal-WMH components smaller than
#     a minimum size are removed; the size is chosen on pooled 5-fold CV
#     predictions, then applied unchanged to the test set
#   - --no-distance: identical pipeline WITHOUT the distance channel
#     (the fair baseline, written to separate folders)
#
# Input  : patient/slice list from splits/baseline_2d; images built from
#          preprocessed/<id>/<id>_{T1,T2,FLAIR}.nii.gz + distance channel
#          distance_map/<id>/<id>_VentDist_norm (centre slice)
#          ->  [3 * (2*CONTEXT_SLICES + 1) + 1, H, W]  (10 channels)
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
#   python 21_distance_map_unetpp.py              # CV (all folds) + final + test
#   python 21_distance_map_unetpp.py --fold 3     # one CV fold only
#   python 21_distance_map_unetpp.py --cv-only    # all CV folds + post-proc choice
#   python 21_distance_map_unetpp.py --final-only # final-70 + test (CV must exist)
#   python 21_distance_map_unetpp.py --smoke      # quick check, dev data only
#   add --allow-cpu only to debug without a GPU (otherwise it refuses to run on CPU)
#   add --no-distance to any command for the matching baseline (9 channels)
# ============================================================


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = BASE_DIR

SPLIT_ROOT = PROJECT_DIR / "splits" / "baseline_2d"
PREPROCESSED_ROOT = PROJECT_DIR / "preprocessed"
DISTANCE_ROOT = PROJECT_DIR / "distance_map"
LABEL_ROOT = PROJECT_DIR / "six_class_masks"

USE_DISTANCE = "--no-distance" not in sys.argv
RUN_NAME = "distance_map_unetpp_v2" if USE_DISTANCE else "baseline_unetpp_v2"
MODEL_NAME = "Distance UNet++ v2" if USE_DISTANCE else "Baseline UNet++ v2"

LOG_ROOT = PROJECT_DIR / "logs" / RUN_NAME
CHECKPOINT_ROOT = PROJECT_DIR / "checkpoints" / RUN_NAME

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

MRI_MODALITIES = ["T1", "T2", "FLAIR"]
CONTEXT_SLICES = 1              # 1 = slices z-1, z, z+1 (2.5D); 0 = plain 2D
SLICE_OFFSETS = list(range(-CONTEXT_SLICES, CONTEXT_SLICES + 1))
CHANNEL_ORDER = [f"{m}_z{o:+d}" for o in SLICE_OFFSETS for m in MRI_MODALITIES]
N_MRI_CHANNELS = len(CHANNEL_ORDER)
if USE_DISTANCE:
    CHANNEL_ORDER = CHANNEL_ORDER + ["DIST"]
IN_CHANNELS = len(CHANNEL_ORDER)
BASE_CHANNELS = 32

EPOCHS = 120
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

# LR scheduler: cosine annealing LEARNING_RATE -> MIN_LEARNING_RATE, stepped
# once per epoch. CV folds: T_max = EPOCHS (all folds train the full EPOCHS).
# Final model: T_max = final_epochs (median of the 5 CV best epochs), so its
# LR anneals fully within its own, shorter run. The baseline must use the
# same setting for a fair comparison.
# No gradient clipping (same as baseline scripts 13/14).
USE_LR_SCHEDULER = True
MIN_LEARNING_RATE = 1e-6

# Flip TTA (none, left-right, up-down, both; softmax averaged) for the
# best-checkpoint CV evaluation, the train70 evaluation and the test set.
# Not used for per-epoch selection (4x slower).
USE_TTA = True

# Post-processing: remove 3D-connected (26-neighbour) abnormal-WMH
# components smaller than N voxels. N is chosen on the pooled 70 CV
# validation patients (TTA predictions of each fold's best model).
POSTPROC_TARGET_LABELS = (3, 4)
MIN_LESION_VOXEL_CANDIDATES = [0, 2, 3, 5, 8, 10, 15, 20, 30, 50]
POSTPROC_TOLERANCE = 0.001      # pick the smallest size within 0.001 Dice of the best

# --smoke: fold 1 only, a few dev patients, 2 epochs, separate output
# folders, never touches final training or the test set.
SMOKE_TEST = "--smoke" in sys.argv
SMOKE_TRAIN_PATIENTS = 3
SMOKE_VAL_PATIENTS = 2

# GPU is required: stop immediately instead of silently training on CPU.
# Pass --allow-cpu only for debugging without a GPU.
REQUIRE_GPU = "--allow-cpu" not in sys.argv
if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not available, refusing to train on CPU.\n"
        f"  torch {torch.__version__}, built with CUDA {torch.version.cuda}\n"
        "  Check: NVIDIA driver installed (nvidia-smi), CUDA-enabled PyTorch installed,\n"
        "  CUDA_VISIBLE_DEVICES not hiding the GPU. Use --allow-cpu only for debugging."
    )
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
if DEVICE.type == "cuda":
    torch.cuda.set_device(DEVICE)

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
    LOG_ROOT = PROJECT_DIR / "logs" / f"{RUN_NAME}_smoke"
    CHECKPOINT_ROOT = PROJECT_DIR / "checkpoints" / f"{RUN_NAME}_smoke"
    EPOCHS = 2
    SAVE_VAL_PNG_EVERY = 1
    CHECKPOINT_EVERY = 1
    ONLY_FOLD = ONLY_FOLD or 1
    if "--epochs" in sys.argv:
        EPOCHS = int(sys.argv[sys.argv.index("--epochs") + 1])

FINAL_LOG_DIR = LOG_ROOT / "final_results"
FINAL_CKPT_DIR = CHECKPOINT_ROOT / "final70"
TEST_METRICS_JSON = FINAL_LOG_DIR / "test30_summary.json"
POSTPROC_JSON = LOG_ROOT / "cv_summary" / "postprocessing_choice.json"


# ============================================================
# MODEL: UNet++ (2D)
# ============================================================

class UNetPlusPlus2D(nn.Module):
    """
    2D UNet++ (nested U-Net), no deep supervision.

    Node X[i][j]: i = depth (0..4), j = position along the skip pathway.
      X[i][0] = encoder:  DoubleConv(maxpool(X[i-1][0]))
      X[i][j] = DoubleConv(cat(X[i][0..j-1], up(X[i+1][j-1])))
    Output = 1x1 conv on X[0][4].
    Same DoubleConv block (conv-BN-ReLU x2) and widths as UNet2D.
    """

    DEPTH = 5

    def __init__(self, in_channels: int = 3, num_classes: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        nb = [base_channels * 2 ** i for i in range(self.DEPTH)]
        self.pool = nn.MaxPool2d(2)
        self.nodes = nn.ModuleDict()
        for i in range(self.DEPTH):
            for j in range(self.DEPTH - i):
                if j == 0:
                    in_ch = in_channels if i == 0 else nb[i - 1]
                else:
                    in_ch = nb[i] * j + nb[i + 1]
                self.nodes[f"x{i}_{j}"] = DoubleConv(in_ch, nb[i])
        self.out = nn.Conv2d(nb[0], num_classes, kernel_size=1)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        # upsample to the exact size of the skip (handles odd H/W)
        return nn.functional.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X = {}
        for i in range(self.DEPTH):
            X[i, 0] = self.nodes[f"x{i}_0"](x if i == 0 else self.pool(X[i - 1, 0]))
        for j in range(1, self.DEPTH):
            for i in range(self.DEPTH - j):
                feats = [X[i, k] for k in range(j)] + [self._up(X[i + 1, j - 1], X[i, 0])]
                X[i, j] = self.nodes[f"x{i}_{j}"](torch.cat(feats, dim=1))
        return self.out(X[0, self.DEPTH - 1])


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

        # MRI volumes for 2.5D input, stored float16 to save RAM: [3, X, Y, Z]
        mri = []
        for m in MRI_MODALITIES:
            img = nib.load(str(PREPROCESSED_ROOT / pid / f"{pid}_{m}.nii.gz"))
            if img.shape[:3] != flair_img.shape[:3] or not np.allclose(img.affine, flair_img.affine, atol=1e-4):
                raise RuntimeError(f"{pid}: {m} geometry differs from FLAIR")
            mri.append(np.asanyarray(img.dataobj).astype(np.float16))
        mri = np.stack(mri, axis=0)

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
            "mri": mri,
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
        # load volumes and check that the volume-built centre slice equals the
        # split's .npz image (same patient, same slice index, same values)
        checked = set()
        for image_path, pid, z in self.samples:
            mri = get_patient_volumes(pid)["mri"]
            if pid in checked:
                continue
            checked.add(pid)
            with np.load(image_path, allow_pickle=False) as npz:
                ref = npz["image"].astype(np.float32)
            if not np.allclose(mri[:, :, :, z].astype(np.float32), ref, atol=0.05, rtol=1e-3):
                raise RuntimeError(f"{image_path.name}: volume slice {z} does not match the split image")

    def __len__(self):
        return len(self.samples)

    def _augment(self, image, label):
        # spatial: identical for all channels and the label
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

        # intensity: MRI channels only (all context slices), brain voxels only;
        # the distance channel (last) is never changed
        n = N_MRI_CHANNELS
        brain = np.any(image[:n] != 0, axis=0)
        if random.random() < P_INTENSITY_SCALE:
            factor = random.uniform(*INTENSITY_SCALE_RANGE)
            image[:n, brain] *= factor
        if random.random() < P_INTENSITY_SHIFT:
            offset = random.uniform(*INTENSITY_SHIFT_RANGE)
            image[:n, brain] += offset
        return image, label

    def __getitem__(self, index):
        image_path, pid, z = self.samples[index]
        vols = get_patient_volumes(pid)
        mri = vols["mri"]
        n_slices = mri.shape[3]
        zs = [min(max(z + o, 0), n_slices - 1) for o in SLICE_OFFSETS]   # replicate edges
        # [3, X, Y, K] -> [K, 3, X, Y] -> [3K, X, Y]  (offset-major, matches CHANNEL_ORDER)
        image = np.transpose(mri[:, :, :, zs], (3, 0, 1, 2)).reshape(-1, mri.shape[1], mri.shape[2])
        image = image.astype(np.float32)
        if USE_DISTANCE:
            image = np.concatenate([image, vols["dist"][:, :, z][None]], axis=0)
        label = vols["label"][:, :, z].astype(np.int64)
        if image.shape[0] != IN_CHANNELS:
            raise RuntimeError(f"{image_path.name}: expected {IN_CHANNELS} channels, got {image.shape[0]}")

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
def predict_volumes(model, loader, patient_ids, criterion=None, tta=False):
    """Run the model on all slices and rebuild 3D prediction volumes.
    Slices not present in the split (empty images) stay background.
    tta=True: average softmax over identity, left-right, up-down and both flips."""
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
        if tta:
            probs = torch.softmax(logits.float(), dim=1)
            for dims in ((3,), (2,), (2, 3)):
                with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                    flipped = model(torch.flip(image, dims=dims))
                probs += torch.flip(torch.softmax(flipped.float(), dim=1), dims=dims)
            prediction = torch.argmax(probs, dim=1).cpu().numpy().astype(np.uint8)
        else:
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
# POST-PROCESSING (small abnormal-WMH component removal)
# ============================================================

def remove_small_components(pred, min_voxels):
    """Set abnormal-WMH components (labels 3+4 merged, 26-connected) with
    fewer than min_voxels voxels to background."""
    if min_voxels <= 1:
        return pred
    out = pred.copy()
    mask = np.isin(out, POSTPROC_TARGET_LABELS)
    comps, n = connected_components(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return out
    small = np.bincount(comps.ravel()) < min_voxels
    small[0] = False
    out[small[comps]] = 0
    return out


def choose_postprocessing(folds=(1, 2, 3, 4, 5)):
    """Pick the minimum component size on pooled CV validation predictions
    (raw TTA predictions of each fold's best model). Never uses test data."""
    preds = {}
    for fold in folds:
        val_ids = read_patient_ids(SPLIT_ROOT / f"fold_{fold}" / "val_patients.csv")
        if SMOKE_TEST:
            val_ids = val_ids[:SMOKE_VAL_PATIENTS]
        for pid in val_ids:
            path = LOG_ROOT / f"fold_{fold}" / "predictions_nifti" / f"{pid}_pred.nii.gz"
            preds[pid] = np.asanyarray(nib.load(str(path)).dataobj).astype(np.uint8)

    rows = []
    for n in MIN_LESION_VOXEL_CANDIDATES:
        dice, sens, prec, fp_ml = [], [], [], []
        for pid, pred in preds.items():
            vols = get_patient_volumes(pid)
            m = binary_metrics(np.isin(remove_small_components(pred, n), POSTPROC_TARGET_LABELS),
                               np.isin(vols["label"], POSTPROC_TARGET_LABELS), vols["spacing"])
            dice.append(m["dice"])
            sens.append(m["sensitivity"])
            prec.append(m["precision"])
            fp_ml.append(m["fp_volume_ml"])
        rows.append({
            "min_voxels": n,
            "mean_dice": finite_stats(dice)["mean"],
            "median_dice": finite_stats(dice)["median"],
            "mean_sensitivity": finite_stats(sens)["mean"],
            "mean_precision": finite_stats(prec)["mean"],
            "mean_fp_volume_ml": finite_stats(fp_ml)["mean"],
        })
        print(f"  post-proc min {n:3d} voxels: CV abnormal Dice {rows[-1]['mean_dice']:.4f}")

    # smallest (least aggressive) size whose mean Dice is within POSTPROC_TOLERANCE
    # of the best one, so tiny CV differences do not pick an aggressive filter
    top = max(r["mean_dice"] for r in rows)
    best = min((r for r in rows if r["mean_dice"] >= top - POSTPROC_TOLERANCE), key=lambda r: r["min_voxels"])
    choice = {
        "min_lesion_voxels": int(best["min_voxels"]),
        "criterion": f"smallest size within {POSTPROC_TOLERANCE} of the best mean per-patient "
                     "abnormal-WMH (labels 3+4) 3D Dice",
        "selected_on": f"{len(preds)} pooled CV validation patients, folds {list(folds)}, TTA={USE_TTA}",
        "dice_without_postprocessing": rows[0]["mean_dice"] if rows[0]["min_voxels"] == 0 else None,
        "dice_with_choice": best["mean_dice"],
        "candidates": rows,
    }

    # pooled CV metrics with the chosen post-processing, kept inside the same JSON
    # (the only file this step writes; it is required by the final test step)
    pp_preds = {pid: remove_small_components(pred, choice["min_lesion_voxels"]) for pid, pred in preds.items()}
    pp_rows, _ = evaluate_patients(pp_preds, list(pp_preds))
    choice["cv_postprocessed_summary"] = summarize_patient_rows(pp_rows)
    write_json(POSTPROC_JSON, choice)
    print(f"Post-processing chosen on CV: min {choice['min_lesion_voxels']} voxels "
          f"(abnormal Dice {best['mean_dice']:.4f})")
    return choice


def load_postprocessing_choice():
    if not POSTPROC_JSON.exists():
        return choose_postprocessing()
    return json.loads(POSTPROC_JSON.read_text(encoding="utf-8"))


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
    model = UNetPlusPlus2D(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, base_channels=BASE_CHANNELS).to(DEVICE)
    if REQUIRE_GPU and not next(model.parameters()).is_cuda:
        raise RuntimeError("Model parameters are not on the GPU.")
    return model


def report_gpu():
    """Print the GPU in use and run a tiny CUDA operation to prove it works."""
    if DEVICE.type != "cuda":
        print("WARNING: running on CPU (--allow-cpu). Training will be very slow.")
        return
    props = torch.cuda.get_device_properties(DEVICE)
    probe = torch.ones(1, device=DEVICE) * 2          # fails here if CUDA is broken
    torch.cuda.synchronize(DEVICE)
    print(f"GPU: {props.name} | {props.total_memory / 1024**3:.1f} GB | "
          f"torch {torch.__version__} CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()} | "
          f"probe {probe.item():.0f} on {probe.device}")


def build_scheduler(optimizer, total_epochs):
    """Cosine annealing over total_epochs: EPOCHS for CV folds, final_epochs for the final model."""
    if not USE_LR_SCHEDULER:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(total_epochs), eta_min=MIN_LEARNING_RATE)


def scheduler_description(total_epochs):
    if not USE_LR_SCHEDULER:
        return "none"
    return (f"CosineAnnealingLR(T_max={int(total_epochs)}, eta_min={MIN_LEARNING_RATE}), "
            f"start lr={LEARNING_RATE}, stepped once per epoch")


def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def run_config():
    return {
        "model_name": MODEL_NAME,
        "use_distance": USE_DISTANCE,
        "context_slices": CONTEXT_SLICES,
        "in_channels": IN_CHANNELS,
        "channel_order": CHANNEL_ORDER,
        "tta": "flip (none, LR, UD, both), softmax mean" if USE_TTA else "none",
        "postprocessing": f"remove 26-connected components of labels {list(POSTPROC_TARGET_LABELS)} "
                          f"smaller than N voxels; N chosen on pooled CV from {MIN_LESION_VOXEL_CANDIDATES}",
        "class_names": CLASS_NAMES,
        "eval_targets": {k: list(v) for k, v in EVAL_TARGETS.items()},
        "selection_target": SELECTION_TARGET,
        "selection_rule": "mean per-patient 3D Dice over val patients with GT",
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "lr_scheduler_cv": scheduler_description(EPOCHS),
        "lr_scheduler_final": "CosineAnnealingLR(T_max=final_epochs = median of 5 CV best epochs), "
                              "see final_results/final_plan.json",
        "seed": RANDOM_SEED, "seed_rule": "RANDOM_SEED + fold (final model: RANDOM_SEED, as script 14)",
        "dice_loss": f"batch-level soft Dice, smooth={DICE_SMOOTH}, background excluded",
        "class_weight_power": CLASS_WEIGHT_POWER,
        "augmentation": {
            "spatial": "h-flip, v-flip, rot90 on all channels + label",
            "intensity": "scale/shift on MRI channels (all context slices), brain voxels only; DIST never",
        },
        "distance_config": (json.loads((DISTANCE_ROOT / "distance_config.json").read_text(encoding="utf-8"))
                            if USE_DISTANCE else None),
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
    scheduler = build_scheduler(optimizer, EPOCHS)          # CV: full EPOCHS
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
                         f"{MODEL_NAME} fold {fold} (best epoch {best_epoch})")

    # final evaluation of the best checkpoint
    best = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best["model"])
    # raw (no post-processing) TTA predictions; saved as NIfTI and reused for the
    # post-processing choice after all folds
    preds, val_losses = predict_volumes(model, val_loader, val_ids, criterion, tta=USE_TTA)
    rows, confusion = evaluate_patients(preds, val_ids)
    summary = summarize_patient_rows(rows)
    summary.update({"fold": fold, "best_epoch": best_epoch, "selection_score": best_score,
                    "val_loss": val_losses["loss"], "tta": USE_TTA})
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
    ax.set_title(f"{MODEL_NAME} — 5-fold CV")
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
    min_voxels = int(load_postprocessing_choice()["min_lesion_voxels"])

    for fold in range(1, 6):
        n_done = len(read_csv(LOG_ROOT / f"fold_{fold}" / "epoch_log.csv"))
        if n_done != EPOCHS:
            raise RuntimeError(f"Fold {fold} trained {n_done} epochs, expected the full {EPOCHS} before "
                               "taking the median best epoch.")
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
        "lr_scheduler_final": scheduler_description(final_epochs),
        "lr_schedule_rule": "final model anneals over its own final_epochs (T_max=final_epochs); "
                            "CV folds used T_max=EPOCHS",
        "postprocessing_min_voxels": min_voxels,
        "postprocessing_rule": "chosen on pooled CV before test; primary test result = with post-processing",
        "class_voxel_counts": counts.tolist(),
        "class_weights": weights.tolist(),
        "n_dev_patients": len(dev_ids), "n_test_patients": len(test_ids),
    })

    print(f"\n=== Final model: {len(dev_ids)} patients ({len(train_ds)} slices), {final_epochs} epochs "
          f"(CV best epochs {best_epochs}) ===")

    model = build_model()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, final_epochs)    # final: T_max = final_epochs
    criterion = CombinedLoss(torch.tensor(weights, device=DEVICE))
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None
    print(f"LR schedule (final): {scheduler_description(final_epochs)}")

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
        "model": model.state_dict(), "epochs": final_epochs, "config": run_config(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "lr_scheduler": scheduler_description(final_epochs)})
    plot_training_curves(epoch_rows, None, FINAL_LOG_DIR / "plots" / "final70_training_curves.png",
                         f"{MODEL_NAME} final model (70 patients, {final_epochs} epochs)")

    # ---- final model on its own 70 training patients (no augmentation), as script 14 ----
    dev_eval_ds = DistanceSliceDataset([fold1 / "train", fold1 / "val"], dev_ids, augment=False)
    dev_eval_loader = make_loader(dev_eval_ds, shuffle=False, seed=seed)
    dev_preds, _ = predict_volumes(model, dev_eval_loader, dev_ids, tta=USE_TTA)
    dev_preds = {pid: remove_small_components(pred, min_voxels) for pid, pred in dev_preds.items()}
    dev_rows, dev_confusion = evaluate_patients(dev_preds, dev_ids)
    train_summary = summarize_patient_rows(dev_rows)
    save_evaluation_outputs(dev_preds, dev_rows, dev_confusion, train_summary, FINAL_LOG_DIR, "train70",
                            [], save_nifti=False)
    del dev_preds

    # ---- single test evaluation ----
    test_ds = DistanceSliceDataset([SPLIT_ROOT / "test"], test_ids, augment=False)
    test_loader = make_loader(test_ds, shuffle=False, seed=seed)
    raw_preds, _ = predict_volumes(model, test_loader, test_ids, tta=USE_TTA)

    # secondary: without post-processing (only reported as a column in the final report)
    raw_rows, _ = evaluate_patients(raw_preds, test_ids)
    raw_summary = summarize_patient_rows(raw_rows)
    del raw_rows

    # primary: with the CV-chosen post-processing (pre-specified)
    preds = {pid: remove_small_components(pred, min_voxels) for pid, pred in raw_preds.items()}
    rows, confusion = evaluate_patients(preds, test_ids)
    summary = summarize_patient_rows(rows)
    summary.update({"final_epochs": final_epochs, "cv_best_epochs": best_epochs,
                    "postprocessing_min_voxels": min_voxels, "tta": USE_TTA, "role": "primary"})
    save_evaluation_outputs(preds, rows, confusion, summary, FINAL_LOG_DIR, "test30",
                            sorted(test_ids))   # PNG for every test patient

    # combined CV vs test report
    cv = json.loads((LOG_ROOT / "cv_summary" / "cv_mean_std.json").read_text(encoding="utf-8"))
    cv_pp = load_postprocessing_choice().get("cv_postprocessed_summary", {})
    report = []
    for name in EVAL_TARGETS:
        for metric in ("dice", "sensitivity", "hd95", "fp_volume_ml"):
            key = f"{name}_{metric}_mean"
            report.append({
                "target": name, "metric": metric,
                "train70_mean": train_summary.get(key, math.nan),
                "cv_mean": cv.get(key, {}).get("mean", math.nan),
                "cv_std": cv.get(key, {}).get("std", math.nan),
                "cv_pooled_postprocessed_mean": cv_pp.get(key, math.nan),
                "test_mean": summary.get(key, math.nan),
                "test_raw_mean": raw_summary.get(key, math.nan),
                "test_std": summary.get(f"{name}_{metric}_std", math.nan),
                "test_median": summary.get(f"{name}_{metric}_median", math.nan),
                "test_n_with_gt": summary[f"{name}_n_patients_with_gt"],
                "test_n_hd95_nan": summary[f"{name}_n_hd95_nan"],
            })
    write_csv(FINAL_LOG_DIR / "train_cv_test_report.csv", report)

    print(f"\n=== TEST (30 patients) — {MODEL_NAME}, TTA={USE_TTA}, post-proc min {min_voxels} voxels ===")
    for name in EVAL_TARGETS:
        print(f"{name:20s} Dice {summary[f'{name}_dice_mean']:.4f} ± {summary[f'{name}_dice_std']:.4f} "
              f"(n={summary[f'{name}_n_patients_with_gt']})  HD95 {summary[f'{name}_hd95_mean']:.2f} mm "
              f"(NaN={summary[f'{name}_n_hd95_nan']})")
    print(f"Outputs in {FINAL_LOG_DIR}")


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Device: {DEVICE}  AMP: {USE_AMP}  |  {MODEL_NAME}: {IN_CHANNELS} channels {CHANNEL_ORDER}")
    report_gpu()
    print(f"Outputs: {LOG_ROOT}")
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
            choose_postprocessing()
        elif SMOKE_TEST:
            choose_postprocessing(folds=[ONLY_FOLD])   # exercise the code path on dev data only

    if ONLY_FOLD is None and not CV_ONLY:
        train_final_and_test()


if __name__ == "__main__":
    main()
