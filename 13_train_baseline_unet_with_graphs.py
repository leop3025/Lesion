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
import traceback

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import Dataset, DataLoader

# ============================================================
# PROJECT IMPORT
# ============================================================

THIS_FILE = Path(__file__).resolve()
BASE_DIR = THIS_FILE.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# unet2d.py is expected in:
# C:\Users\Preethi\Documents\LESION\unet2d.py
from unet2d import UNet2D


# ============================================================
# BASELINE CONFIG
# ============================================================

PROJECT_DIR = Path(r"C:\Users\Preethi\Documents\LESION")

SPLIT_ROOT = PROJECT_DIR / "splits" / "baseline_2d"
PREPROCESSED_ROOT = PROJECT_DIR / "preprocessed"

LOG_ROOT = PROJECT_DIR / "logs" / "baseline_unet"
CHECKPOINT_ROOT = PROJECT_DIR / "checkpoints" / "baseline_unet"
SUMMARY_CSV = PROJECT_DIR / "logs" / "baseline_fold_metrics.csv"
FAILURE_LOG = LOG_ROOT / "fold_failures.txt"

NUM_CLASSES = 3

CLASS_NAMES = [
    "background",
    "normal_wmh",
    "abnormal_wmh",
]

CHANNEL_ORDER = [
    "T1",
    "T2",
    "FLAIR",
]

BASE_CHANNELS = 32

EPOCHS = 50
BATCH_SIZE = 8
NUM_WORKERS = 0

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# SAVE EVERY EPOCH
CHECKPOINT_EVERY = 1

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

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = (
    DEVICE.type == "cuda"
)

ONLY_FOLD = None

if "--fold" in sys.argv:
    idx = sys.argv.index("--fold")

    if idx + 1 >= len(sys.argv):
        raise RuntimeError(
            "--fold requires a value from 1 to 5"
        )

    ONLY_FOLD = int(sys.argv[idx + 1])

    if ONLY_FOLD not in {1, 2, 3, 4, 5}:
        raise RuntimeError(
            "--fold must be between 1 and 5"
        )


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# ATOMIC FILE HELPERS
# ============================================================

def atomic_write_text(path, text):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp"
    )

    os.close(fd)
    tmp = Path(tmp_name)

    try:
        tmp.write_text(
            text,
            encoding="utf-8"
        )

        os.replace(
            tmp,
            path
        )

    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_torch_save(path, payload):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp"
    )

    os.close(fd)
    tmp = Path(tmp_name)

    try:
        torch.save(
            payload,
            tmp
        )

        os.replace(
            tmp,
            path
        )

    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# ============================================================
# SPLIT / LEAKAGE SAFEGUARDS
# ============================================================

def read_patient_ids(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing split file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(
            f"Empty split file: {path}"
        )

    ids = [
        str(row["patient_id"])
        .strip()
        .zfill(3)
        for row in rows
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            f"Duplicate patient IDs in {path}"
        )

    return ids


def patient_id_from_filename(path):
    match = re.match(
        r"(\d{3})_slice_\d+\.npz$",
        path.name
    )

    if not match:
        raise RuntimeError(
            f"Unexpected slice filename: {path.name}"
        )

    return match.group(1)


def validate_fold_membership(fold):
    fold_dir = SPLIT_ROOT / f"fold_{fold}"

    train_patient_csv = (
        fold_dir
        / "train_patients.csv"
    )

    val_patient_csv = (
        fold_dir
        / "val_patients.csv"
    )

    test_patient_csv = (
        SPLIT_ROOT
        / "test_30_patients.csv"
    )

    train_ids = set(
        read_patient_ids(train_patient_csv)
    )

    val_ids = set(
        read_patient_ids(val_patient_csv)
    )

    test_ids = set(
        read_patient_ids(test_patient_csv)
    )

    if len(train_ids) != 56:
        raise RuntimeError(
            f"Fold {fold}: expected 56 train patients, "
            f"got {len(train_ids)}"
        )

    if len(val_ids) != 14:
        raise RuntimeError(
            f"Fold {fold}: expected 14 val patients, "
            f"got {len(val_ids)}"
        )

    if len(test_ids) != 30:
        raise RuntimeError(
            f"Expected 30 test patients, "
            f"got {len(test_ids)}"
        )

    if train_ids & val_ids:
        raise RuntimeError(
            f"Fold {fold}: PATIENT LEAKAGE train vs val"
        )

    if train_ids & test_ids:
        raise RuntimeError(
            f"Fold {fold}: PATIENT LEAKAGE train vs test"
        )

    if val_ids & test_ids:
        raise RuntimeError(
            f"Fold {fold}: PATIENT LEAKAGE val vs test"
        )

    train_image_dir = (
        fold_dir
        / "train"
        / "images"
    )

    val_image_dir = (
        fold_dir
        / "val"
        / "images"
    )

    train_slice_patients = {
        patient_id_from_filename(p)
        for p in train_image_dir.glob("*.npz")
    }

    val_slice_patients = {
        patient_id_from_filename(p)
        for p in val_image_dir.glob("*.npz")
    }

    if train_slice_patients != train_ids:
        missing = sorted(
            train_ids - train_slice_patients
        )

        extra = sorted(
            train_slice_patients - train_ids
        )

        raise RuntimeError(
            f"Fold {fold}: TRAIN slice membership mismatch. "
            f"missing={missing}, extra={extra}"
        )

    if val_slice_patients != val_ids:
        missing = sorted(
            val_ids - val_slice_patients
        )

        extra = sorted(
            val_slice_patients - val_ids
        )

        raise RuntimeError(
            f"Fold {fold}: VAL slice membership mismatch. "
            f"missing={missing}, extra={extra}"
        )

    if train_slice_patients & val_slice_patients:
        raise RuntimeError(
            f"Fold {fold}: SLICE-LEVEL PATIENT LEAKAGE"
        )

    if train_slice_patients & test_ids:
        raise RuntimeError(
            f"Fold {fold}: TEST patient appears in train slices"
        )

    if val_slice_patients & test_ids:
        raise RuntimeError(
            f"Fold {fold}: TEST patient appears in val slices"
        )

    print()
    print(f"Fold {fold} leakage safeguard: PASS")
    print(f"  train patients: {len(train_ids)}")
    print(f"  val patients  : {len(val_ids)}")
    print(f"  test patients : {len(test_ids)}")

    return (
        train_ids,
        val_ids,
        test_ids,
    )


# ============================================================
# DATASET
# ============================================================

class BaselineSliceDataset(Dataset):
    def __init__(
        self,
        partition_root,
        expected_patient_ids,
        augment=False
    ):
        self.partition_root = Path(
            partition_root
        )

        self.image_dir = (
            self.partition_root
            / "images"
        )

        self.label_dir = (
            self.partition_root
            / "labels"
        )

        self.expected_patient_ids = set(
            expected_patient_ids
        )

        self.augment = augment

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Missing image directory: {self.image_dir}"
            )

        if not self.label_dir.exists():
            raise FileNotFoundError(
                f"Missing label directory: {self.label_dir}"
            )

        self.image_paths = sorted(
            self.image_dir.glob("*.npz")
        )

        if not self.image_paths:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

        self.samples = []

        for image_path in self.image_paths:
            pid = patient_id_from_filename(
                image_path
            )

            if pid not in self.expected_patient_ids:
                raise RuntimeError(
                    f"STRUCTURAL LEAKAGE BLOCK: "
                    f"{image_path.name} belongs to patient {pid}, "
                    "who is not allowed in this Dataset."
                )

            label_path = (
                self.label_dir
                / image_path.name
            )

            if not label_path.exists():
                raise FileNotFoundError(
                    f"Missing matching label: {label_path}"
                )

            self.samples.append(
                (
                    image_path,
                    label_path
                )
            )

        actual_patients = {
            patient_id_from_filename(p)
            for p, _ in self.samples
        }

        if actual_patients != self.expected_patient_ids:
            raise RuntimeError(
                "Dataset patient membership does not exactly match "
                "the locked split."
            )

    def __len__(self):
        return len(self.samples)

    def _augment(
        self,
        image,
        label
    ):
        if random.random() < P_HORIZONTAL_FLIP:
            image = np.flip(
                image,
                axis=2
            ).copy()

            label = np.flip(
                label,
                axis=1
            ).copy()

        if random.random() < P_VERTICAL_FLIP:
            image = np.flip(
                image,
                axis=1
            ).copy()

            label = np.flip(
                label,
                axis=0
            ).copy()

        if random.random() < P_ROT90:
            k = random.choice(
                [1, 2, 3]
            )

            image = np.rot90(
                image,
                k=k,
                axes=(1, 2)
            ).copy()

            label = np.rot90(
                label,
                k=k,
                axes=(0, 1)
            ).copy()

        if random.random() < P_INTENSITY_SCALE:
            scale = random.uniform(
                *INTENSITY_SCALE_RANGE
            )

            image = (
                image * scale
            ).astype(
                np.float32,
                copy=False
            )

        if random.random() < P_INTENSITY_SHIFT:
            shift = random.uniform(
                *INTENSITY_SHIFT_RANGE
            )

            image = (
                image + shift
            ).astype(
                np.float32,
                copy=False
            )

        return (
            image,
            label
        )

    def __getitem__(
        self,
        index
    ):
        image_path, label_path = self.samples[index]

        with np.load(
            image_path,
            allow_pickle=False
        ) as image_npz:
            image = (
                image_npz["image"]
                .astype(
                    np.float32,
                    copy=True
                )
            )

            patient_id = str(
                image_npz["patient_id"]
            )

            slice_index = int(
                image_npz["slice_index"]
            )

            channel_order = [
                str(x)
                for x in image_npz[
                    "channel_order"
                ].tolist()
            ]

        if channel_order != CHANNEL_ORDER:
            raise RuntimeError(
                f"Channel order mismatch in {image_path}: "
                f"{channel_order}"
            )

        with np.load(
            label_path,
            allow_pickle=False
        ) as label_npz:
            label = (
                label_npz["label"]
                .astype(
                    np.int64,
                    copy=True
                )
            )

        if image.ndim != 3:
            raise RuntimeError(
                f"Expected image [3,H,W], "
                f"got {image.shape}"
            )

        if image.shape[0] != 3:
            raise RuntimeError(
                f"Expected 3 channels, "
                f"got {image.shape[0]}"
            )

        if label.ndim != 2:
            raise RuntimeError(
                f"Expected label [H,W], "
                f"got {label.shape}"
            )

        unique = set(
            np.unique(label).tolist()
        )

        if not unique.issubset(
            {0, 1, 2}
        ):
            raise RuntimeError(
                f"Unexpected labels {unique} in {label_path}"
            )

        if self.augment and USE_AUGMENTATION:
            image, label = self._augment(
                image,
                label
            )

        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "patient_id": patient_id,
            "slice_index": slice_index,
        }


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_voxel_counts(dataset):
    counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64
    )

    print()
    print(
        "Computing class counts from TRAINING FOLD ONLY..."
    )

    for _, label_path in dataset.samples:
        with np.load(
            label_path,
            allow_pickle=False
        ) as data:
            label = data["label"]

        bincount = np.bincount(
            label.ravel(),
            minlength=NUM_CLASSES
        )

        counts += bincount[:NUM_CLASSES]

    if np.any(counts == 0):
        raise RuntimeError(
            f"At least one class has zero training voxels: "
            f"{counts.tolist()}"
        )

    return counts


def class_weights_from_counts(counts):
    counts = counts.astype(
        np.float64
    )

    frequencies = (
        counts
        / counts.sum()
    )

    weights = (
        1.0
        / np.power(
            frequencies,
            CLASS_WEIGHT_POWER
        )
    )

    weights = (
        weights
        / weights.mean()
    )

    return weights.astype(
        np.float32
    )


# ============================================================
# LOSS
# ============================================================

class MulticlassDiceLoss(nn.Module):
    def __init__(
        self,
        include_background=False,
        smooth=1e-6
    ):
        super().__init__()

        self.include_background = (
            include_background
        )

        self.smooth = smooth

    def forward(
        self,
        logits,
        target
    ):
        probs = torch.softmax(
            logits,
            dim=1
        )

        one_hot = (
            torch.nn.functional.one_hot(
                target,
                num_classes=NUM_CLASSES
            )
            .permute(
                0,
                3,
                1,
                2
            )
            .float()
        )

        first_class = (
            0
            if self.include_background
            else 1
        )

        dice_losses = []

        for cls in range(
            first_class,
            NUM_CLASSES
        ):
            p = probs[:, cls]
            t = one_hot[:, cls]

            intersection = torch.sum(
                p * t,
                dim=(1, 2)
            )

            denom = (
                torch.sum(
                    p,
                    dim=(1, 2)
                )
                +
                torch.sum(
                    t,
                    dim=(1, 2)
                )
            )

            dice = (
                2.0 * intersection
                + self.smooth
            ) / (
                denom
                + self.smooth
            )

            dice_losses.append(
                1.0 - dice
            )

        return (
            torch.stack(
                dice_losses,
                dim=0
            )
            .mean()
        )


class CombinedLoss(nn.Module):
    def __init__(
        self,
        class_weights
    ):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(
            weight=class_weights
        )

        self.dice = MulticlassDiceLoss(
            include_background=INCLUDE_BACKGROUND_IN_DICE_LOSS
        )

    def forward(
        self,
        logits,
        target
    ):
        ce_loss = self.ce(
            logits,
            target
        )

        dice_loss = self.dice(
            logits,
            target
        )

        total_loss = (
            CE_LOSS_WEIGHT * ce_loss
            +
            DICE_LOSS_WEIGHT * dice_loss
        )

        return (
            total_loss,
            ce_loss,
            dice_loss
        )


# ============================================================
# FAST EPOCH METRICS
# ============================================================

def dice_from_logits(
    logits,
    target
):
    pred = torch.argmax(
        logits,
        dim=1
    )

    scores = []

    for cls in range(NUM_CLASSES):
        p = pred == cls
        t = target == cls

        intersection = torch.sum(
            p & t
        ).float()

        denom = (
            torch.sum(p).float()
            +
            torch.sum(t).float()
        )

        if float(denom) == 0.0:
            score = 1.0

        else:
            score = float(
                (
                    2.0
                    * intersection
                    / denom
                )
                .detach()
                .cpu()
            )

        scores.append(score)

    return scores


# ============================================================
# TRAIN / VAL EPOCH
# ============================================================

def run_train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler
):
    model.train()

    total_losses = []
    ce_losses = []
    dice_losses = []

    dice_values = [
        [],
        [],
        []
    ]

    for batch in loader:
        image = (
            batch["image"]
            .to(
                DEVICE,
                non_blocking=True
            )
        )

        label = (
            batch["label"]
            .to(
                DEVICE,
                non_blocking=True
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP
        ):
            logits = model(image)

            (
                loss,
                ce_loss,
                dice_loss
            ) = criterion(
                logits,
                label
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        total_losses.append(
            float(
                loss.detach().cpu()
            )
        )

        ce_losses.append(
            float(
                ce_loss.detach().cpu()
            )
        )

        dice_losses.append(
            float(
                dice_loss.detach().cpu()
            )
        )

        scores = dice_from_logits(
            logits.detach(),
            label
        )

        for cls in range(NUM_CLASSES):
            dice_values[cls].append(
                scores[cls]
            )

    return {
        "loss": float(
            np.mean(total_losses)
        ),
        "ce_loss": float(
            np.mean(ce_losses)
        ),
        "dice_loss": float(
            np.mean(dice_losses)
        ),
        "dice_background": float(
            np.mean(dice_values[0])
        ),
        "dice_normal_wmh": float(
            np.mean(dice_values[1])
        ),
        "dice_abnormal_wmh": float(
            np.mean(dice_values[2])
        ),
    }


@torch.no_grad()
def run_val_epoch(
    model,
    loader,
    criterion
):
    model.eval()

    total_losses = []
    ce_losses = []
    dice_losses = []

    dice_values = [
        [],
        [],
        []
    ]

    for batch in loader:
        image = (
            batch["image"]
            .to(
                DEVICE,
                non_blocking=True
            )
        )

        label = (
            batch["label"]
            .to(
                DEVICE,
                non_blocking=True
            )
        )

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP
        ):
            logits = model(image)

            (
                loss,
                ce_loss,
                dice_loss
            ) = criterion(
                logits,
                label
            )

        total_losses.append(
            float(
                loss.detach().cpu()
            )
        )

        ce_losses.append(
            float(
                ce_loss.detach().cpu()
            )
        )

        dice_losses.append(
            float(
                dice_loss.detach().cpu()
            )
        )

        scores = dice_from_logits(
            logits,
            label
        )

        for cls in range(NUM_CLASSES):
            dice_values[cls].append(
                scores[cls]
            )

    return {
        "loss": float(
            np.mean(total_losses)
        ),
        "ce_loss": float(
            np.mean(ce_losses)
        ),
        "dice_loss": float(
            np.mean(dice_losses)
        ),
        "dice_background": float(
            np.mean(dice_values[0])
        ),
        "dice_normal_wmh": float(
            np.mean(dice_values[1])
        ),
        "dice_abnormal_wmh": float(
            np.mean(dice_values[2])
        ),
    }


# ============================================================
# HD95 + SENSITIVITY
# ============================================================

def dice_and_sensitivity(
    pred,
    target,
    cls
):
    p = pred == cls
    t = target == cls

    tp = np.count_nonzero(
        p & t
    )

    fp = np.count_nonzero(
        p & ~t
    )

    fn = np.count_nonzero(
        ~p & t
    )

    denom = (
        2 * tp
        + fp
        + fn
    )

    if denom == 0:
        dice = 1.0

    else:
        dice = (
            2.0 * tp
            / denom
        )

    gt_positive = (
        tp + fn
    )

    if gt_positive == 0:
        sensitivity = math.nan

    else:
        sensitivity = (
            tp
            / gt_positive
        )

    return (
        float(dice),
        float(sensitivity)
    )


def hd95(
    pred_mask,
    target_mask,
    spacing
):
    pred_mask = (
        pred_mask.astype(bool)
    )

    target_mask = (
        target_mask.astype(bool)
    )

    pred_exists = bool(
        pred_mask.any()
    )

    target_exists = bool(
        target_mask.any()
    )

    if (
        not pred_exists
        and not target_exists
    ):
        return 0.0

    if pred_exists != target_exists:
        return math.nan

    structure = np.ones(
        (
            3,
            3,
            3
        ),
        dtype=bool
    )

    pred_surface = (
        pred_mask
        ^ binary_erosion(
            pred_mask,
            structure=structure,
            border_value=0
        )
    )

    target_surface = (
        target_mask
        ^ binary_erosion(
            target_mask,
            structure=structure,
            border_value=0
        )
    )

    if (
        not pred_surface.any()
        or not target_surface.any()
    ):
        return math.nan

    distance_to_target = (
        distance_transform_edt(
            ~target_surface,
            sampling=spacing
        )
    )

    distance_to_pred = (
        distance_transform_edt(
            ~pred_surface,
            sampling=spacing
        )
    )

    distances = np.concatenate(
        [
            distance_to_target[
                pred_surface
            ],
            distance_to_pred[
                target_surface
            ]
        ]
    )

    return float(
        np.percentile(
            distances,
            95
        )
    )


def find_flair_for_patient(
    patient_id
):
    patient_dir = (
        PREPROCESSED_ROOT
        / patient_id
    )

    candidates = [
        p
        for p in patient_dir.rglob("*.nii*")
        if (
            "flair" in p.name.lower()
            and "mask" not in p.name.lower()
        )
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"{patient_id}: expected exactly one FLAIR "
            f"in {patient_dir}; got {len(candidates)}"
        )

    return candidates[0]


# ============================================================
# FINAL PER-FOLD VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_fold_3d(
    model,
    val_loader,
    val_patient_ids
):
    model.eval()

    pred_volumes = {}
    target_volumes = {}
    spacing_map = {}

    for patient_id in sorted(
        val_patient_ids
    ):
        flair_path = find_flair_for_patient(
            patient_id
        )

        flair_img = nib.load(
            str(flair_path)
        )

        shape = flair_img.shape[:3]

        spacing = tuple(
            float(x)
            for x in flair_img.header.get_zooms()[:3]
        )

        pred_volumes[
            patient_id
        ] = np.zeros(
            shape,
            dtype=np.uint8
        )

        target_volumes[
            patient_id
        ] = np.zeros(
            shape,
            dtype=np.uint8
        )

        spacing_map[
            patient_id
        ] = spacing

    for batch in val_loader:
        image = (
            batch["image"]
            .to(
                DEVICE,
                non_blocking=True
            )
        )

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP
        ):
            logits = model(image)

        prediction = (
            torch.argmax(
                logits,
                dim=1
            )
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        target = (
            batch["label"]
            .numpy()
            .astype(np.uint8)
        )

        patient_ids = list(
            batch["patient_id"]
        )

        slice_indices = (
            batch["slice_index"]
        )

        if torch.is_tensor(
            slice_indices
        ):
            slice_indices = (
                slice_indices
                .cpu()
                .numpy()
                .tolist()
            )

        for i, patient_id in enumerate(
            patient_ids
        ):
            z = int(
                slice_indices[i]
            )

            pred_volumes[
                patient_id
            ][
                :,
                :,
                z
            ] = prediction[i]

            target_volumes[
                patient_id
            ][
                :,
                :,
                z
            ] = target[i]

    per_class = {
        cls: {
            "dice": [],
            "sensitivity": [],
            "hd95": []
        }
        for cls in range(NUM_CLASSES)
    }

    for patient_id in sorted(
        val_patient_ids
    ):
        pred = pred_volumes[
            patient_id
        ]

        target = target_volumes[
            patient_id
        ]

        spacing = spacing_map[
            patient_id
        ]

        for cls in range(NUM_CLASSES):
            dice, sensitivity = (
                dice_and_sensitivity(
                    pred,
                    target,
                    cls
                )
            )

            hausdorff = hd95(
                pred == cls,
                target == cls,
                spacing
            )

            per_class[
                cls
            ]["dice"].append(
                dice
            )

            per_class[
                cls
            ]["sensitivity"].append(
                sensitivity
            )

            per_class[
                cls
            ]["hd95"].append(
                hausdorff
            )

    def safe_nanmean(values):
        arr = np.asarray(
            values,
            dtype=float
        )

        finite = arr[
            np.isfinite(arr)
        ]

        if finite.size == 0:
            return math.nan

        return float(
            finite.mean()
        )

    metrics = {}

    for cls, class_name in enumerate(
        CLASS_NAMES
    ):
        metrics[
            f"dice_{class_name}"
        ] = safe_nanmean(
            per_class[
                cls
            ]["dice"]
        )

        metrics[
            f"sensitivity_{class_name}"
        ] = safe_nanmean(
            per_class[
                cls
            ]["sensitivity"]
        )

        metrics[
            f"hd95_{class_name}"
        ] = safe_nanmean(
            per_class[
                cls
            ]["hd95"]
        )

    return metrics


# ============================================================
# CSV LOGGING
# ============================================================

EPOCH_FIELDS = [
    "epoch",

    "train_loss",
    "train_ce_loss",
    "train_dice_loss",

    "val_loss",
    "val_ce_loss",
    "val_dice_loss",

    "train_dice_background",
    "train_dice_normal_wmh",
    "train_dice_abnormal_wmh",

    "val_dice_background",
    "val_dice_normal_wmh",
    "val_dice_abnormal_wmh",

    "best_val_dice_abnormal_wmh",

    "learning_rate",
]


def append_epoch_log(
    path,
    row
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    exists = path.exists()

    with path.open(
        "a",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=EPOCH_FIELDS
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# TRAINING GRAPHS
# ============================================================

def read_epoch_csv(path):
    with path.open(
        "r",
        encoding="utf-8",
        newline=""
    ) as f:
        return list(
            csv.DictReader(f)
        )


def plot_fold_graphs(
    fold_log_dir,
    fold
):
    epoch_csv = (
        fold_log_dir
        / "epochs.csv"
    )

    if not epoch_csv.exists():
        return

    rows = read_epoch_csv(
        epoch_csv
    )

    if not rows:
        return

    epochs = [
        int(r["epoch"])
        for r in rows
    ]

    # --------------------------------------------------------
    # 1. TOTAL LOSS
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        [
            float(r["train_loss"])
            for r in rows
        ],
        label="Train loss"
    )

    plt.plot(
        epochs,
        [
            float(r["val_loss"])
            for r in rows
        ],
        label="Validation loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Fold {fold} — Total loss"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        fold_log_dir
        / "loss_curve.png",
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # 2. CE LOSS
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        [
            float(r["train_ce_loss"])
            for r in rows
        ],
        label="Train CE loss"
    )

    plt.plot(
        epochs,
        [
            float(r["val_ce_loss"])
            for r in rows
        ],
        label="Validation CE loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(
        f"Fold {fold} — Cross-entropy loss"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        fold_log_dir
        / "ce_loss_curve.png",
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # 3. DICE LOSS
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        [
            float(r["train_dice_loss"])
            for r in rows
        ],
        label="Train Dice loss"
    )

    plt.plot(
        epochs,
        [
            float(r["val_dice_loss"])
            for r in rows
        ],
        label="Validation Dice loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Dice loss")
    plt.title(
        f"Fold {fold} — Dice loss"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        fold_log_dir
        / "dice_loss_curve.png",
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # 4. VALIDATION DICE ALL CLASSES
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        [
            float(r["val_dice_background"])
            for r in rows
        ],
        label="Background"
    )

    plt.plot(
        epochs,
        [
            float(r["val_dice_normal_wmh"])
            for r in rows
        ],
        label="Normal WMH"
    )

    plt.plot(
        epochs,
        [
            float(r["val_dice_abnormal_wmh"])
            for r in rows
        ],
        label="Abnormal WMH"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.ylim(
        0.0,
        1.0
    )

    plt.title(
        f"Fold {fold} — Validation Dice by class"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        fold_log_dir
        / "validation_dice_curves.png",
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # 5. nWMH TRAIN vs VAL DICE
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        [
            float(r["train_dice_normal_wmh"])
            for r in rows
        ],
        label="Train normal-WMH Dice"
    )

    plt.plot(
        epochs,
        [
            float(r["val_dice_normal_wmh"])
            for r in rows
        ],
        label="Validation normal-WMH Dice"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.ylim(
        0.0,
        1.0
    )

    plt.title(
        f"Fold {fold} — Normal-WMH Dice"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        fold_log_dir
        / "normal_wmh_dice_curve.png",
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # 6. abWMH TRAIN vs VAL DICE
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        [
            float(r["train_dice_abnormal_wmh"])
            for r in rows
        ],
        label="Train abnormal-WMH Dice"
    )

    plt.plot(
        epochs,
        [
            float(r["val_dice_abnormal_wmh"])
            for r in rows
        ],
        label="Validation abnormal-WMH Dice"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.ylim(
        0.0,
        1.0
    )

    plt.title(
        f"Fold {fold} — Abnormal-WMH Dice"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        fold_log_dir
        / "abnormal_wmh_dice_curve.png",
        dpi=180
    )

    plt.close()


def plot_cross_fold_summary():
    rows = []

    for fold in range(
        1,
        6
    ):
        path = (
            LOG_ROOT
            / f"fold_{fold}"
            / "final_metrics.json"
        )

        if not path.exists():
            continue

        metrics = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        rows.append(metrics)

    if not rows:
        return

    graph_dir = (
        LOG_ROOT
        / "graphs"
    )

    graph_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fold_labels = [
        f"Fold {int(r['fold'])}"
        for r in rows
    ]

    x = np.arange(
        len(rows)
    )

    width = 0.25

    # --------------------------------------------------------
    # 1. DICE ACROSS FOLDS
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        x - width,
        [
            float(r["dice_background"])
            for r in rows
        ],
        width,
        label="Background"
    )

    plt.bar(
        x,
        [
            float(r["dice_normal_wmh"])
            for r in rows
        ],
        width,
        label="Normal WMH"
    )

    plt.bar(
        x + width,
        [
            float(r["dice_abnormal_wmh"])
            for r in rows
        ],
        width,
        label="Abnormal WMH"
    )

    plt.xticks(
        x,
        fold_labels
    )

    plt.ylabel("Dice")
    plt.ylim(
        0.0,
        1.0
    )

    plt.title(
        "Baseline U-Net — Validation Dice across folds"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        graph_dir
        / "fold_dice_comparison.png",
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # 2. SENSITIVITY ACROSS FOLDS
    # --------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        x - width,
        [
            float(r["sensitivity_background"])
            for r in rows
        ],
        width,
        label="Background"
    )

    plt.bar(
        x,
        [
            float(r["sensitivity_normal_wmh"])
            for r in rows
        ],
        width,
        label="Normal WMH"
    )

    plt.bar(
        x + width,
        [
            float(r["sensitivity_abnormal_wmh"])
            for r in rows
        ],
        width,
        label="Abnormal WMH"
    )

    plt.xticks(
        x,
        fold_labels
    )

    plt.ylabel("Sensitivity")
    plt.ylim(
        0.0,
        1.0
    )

    plt.title(
        "Baseline U-Net — Validation sensitivity across folds"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        graph_dir
        / "fold_sensitivity_comparison.png",
        dpi=180
    )

    plt.close()

    # --------------------------------------------------------
    # 3. HD95 ACROSS FOLDS
    # --------------------------------------------------------

    hd_bg = [
        float(r["hd95_background"])
        for r in rows
    ]

    hd_n = [
        float(r["hd95_normal_wmh"])
        for r in rows
    ]

    hd_a = [
        float(r["hd95_abnormal_wmh"])
        for r in rows
    ]

    if any(
        np.isfinite(
            hd_bg
            + hd_n
            + hd_a
        )
    ):
        plt.figure(
            figsize=(9, 5)
        )

        plt.bar(
            x - width,
            hd_bg,
            width,
            label="Background"
        )

        plt.bar(
            x,
            hd_n,
            width,
            label="Normal WMH"
        )

        plt.bar(
            x + width,
            hd_a,
            width,
            label="Abnormal WMH"
        )

        plt.xticks(
            x,
            fold_labels
        )

        plt.ylabel("HD95 (mm)")

        plt.title(
            "Baseline U-Net — Validation HD95 across folds"
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            graph_dir
            / "fold_hd95_comparison.png",
            dpi=180
        )

        plt.close()


# ============================================================
# CHECKPOINTS
# ============================================================

def save_checkpoint(
    path,
    fold,
    epoch,
    model,
    optimizer,
    best_metric,
    class_counts,
    class_weights
):
    payload = {
        "fold": fold,
        "epoch": epoch,

        "best_metric": float(
            best_metric
        ),

        "model_state": (
            model.state_dict()
        ),

        "optimizer_state": (
            optimizer.state_dict()
        ),

        "class_counts": (
            class_counts.tolist()
        ),

        "class_weights": (
            class_weights.tolist()
        ),

        "channel_order": CHANNEL_ORDER,
        "num_classes": NUM_CLASSES,
    }

    atomic_torch_save(
        path,
        payload
    )


# ============================================================
# FOLD TRAINING
# ============================================================

def train_fold(fold):
    print()
    print("=" * 88)
    print(f"FOLD {fold}")
    print("=" * 88)

    fold_dir = (
        SPLIT_ROOT
        / f"fold_{fold}"
    )

    fold_log_dir = (
        LOG_ROOT
        / f"fold_{fold}"
    )

    fold_ckpt_dir = (
        CHECKPOINT_ROOT
        / f"fold_{fold}"
    )

    # ========================================================
    # FRESH START EVERY RUN — NO RESUME
    # Existing outputs for this fold are deleted first.
    # This guarantees epochs.csv, .pth files, metrics and plots
    # are overwritten from a clean Epoch 1 run.
    # ========================================================

    if fold_log_dir.exists():
        shutil.rmtree(
            fold_log_dir
        )

    if fold_ckpt_dir.exists():
        shutil.rmtree(
            fold_ckpt_dir
        )

    fold_log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fold_ckpt_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        f"Fold {fold}: fresh run started — old logs/checkpoints cleared."
    )

    done_marker = (
        fold_log_dir
        / "fold_done.json"
    )

    final_metrics_path = (
        fold_log_dir
        / "final_metrics.json"
    )

    (
        train_ids,
        val_ids,
        test_ids
    ) = validate_fold_membership(
        fold
    )

    train_dataset = (
        BaselineSliceDataset(
            partition_root=(
                fold_dir
                / "train"
            ),
            expected_patient_ids=train_ids,
            augment=True
        )
    )

    val_dataset = (
        BaselineSliceDataset(
            partition_root=(
                fold_dir
                / "val"
            ),
            expected_patient_ids=val_ids,
            augment=False
        )
    )

    print()
    print(
        f"Fold {fold} slices:"
    )

    print(
        "  train:",
        len(train_dataset)
    )

    print(
        "  val  :",
        len(val_dataset)
    )

    class_counts = (
        compute_class_voxel_counts(
            train_dataset
        )
    )

    class_weights = (
        class_weights_from_counts(
            class_counts
        )
    )

    print()
    print(
        "Training-fold class distribution:"
    )

    for cls, name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"  {name:<14} "
            f"voxels={int(class_counts[cls]):>14,d} "
            f"weight={float(class_weights[cls]):.6f}"
        )

    class_info = {
        "class_counts": {
            CLASS_NAMES[i]: int(
                class_counts[i]
            )
            for i in range(NUM_CLASSES)
        },

        "class_weights": {
            CLASS_NAMES[i]: float(
                class_weights[i]
            )
            for i in range(NUM_CLASSES)
        },

        "class_weight_power": (
            CLASS_WEIGHT_POWER
        ),

        "loss": (
            "weighted_cross_entropy + dice"
        )
    }

    atomic_write_text(
        fold_log_dir
        / "class_weights.json",
        json.dumps(
            class_info,
            indent=2
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        drop_last=False
    )

    model = UNet2D(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS
    ).to(
        DEVICE
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
    )

    class_weight_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = CombinedLoss(
        class_weight_tensor
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda"
        )
        if USE_AMP
        else None
    )

    # ========================================================
    # ALWAYS START FROM SCRATCH
    # ========================================================

    start_epoch = 1
    best_metric = -math.inf

    for epoch in range(
        1,
        EPOCHS + 1
    ):
        train_metrics = (
            run_train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler
            )
        )

        val_metrics = (
            run_val_epoch(
                model,
                val_loader,
                criterion
            )
        )

        current_metric = (
            val_metrics[
                "dice_abnormal_wmh"
            ]
        )

        improved = (
            current_metric
            > best_metric
        )

        if improved:
            best_metric = current_metric

        print()
        print(
            f"Fold {fold} | "
            f"Epoch {epoch:03d}/{EPOCHS}"
        )

        print(
            f"  train total/CE/Dice loss: "
            f"{train_metrics['loss']:.5f} / "
            f"{train_metrics['ce_loss']:.5f} / "
            f"{train_metrics['dice_loss']:.5f}"
        )

        print(
            f"  val   total/CE/Dice loss: "
            f"{val_metrics['loss']:.5f} / "
            f"{val_metrics['ce_loss']:.5f} / "
            f"{val_metrics['dice_loss']:.5f}"
        )

        print(
            f"  train Dice BG/nWMH/abWMH: "
            f"{train_metrics['dice_background']:.4f}/"
            f"{train_metrics['dice_normal_wmh']:.4f}/"
            f"{train_metrics['dice_abnormal_wmh']:.4f}"
        )

        print(
            f"  val   Dice BG/nWMH/abWMH: "
            f"{val_metrics['dice_background']:.4f}/"
            f"{val_metrics['dice_normal_wmh']:.4f}/"
            f"{val_metrics['dice_abnormal_wmh']:.4f}"
        )

        print(
            f"  best abWMH val Dice: "
            f"{best_metric:.4f}"
        )

        append_epoch_log(
            fold_log_dir
            / "epochs.csv",
            {
                "epoch": epoch,

                "train_loss": (
                    train_metrics["loss"]
                ),

                "train_ce_loss": (
                    train_metrics["ce_loss"]
                ),

                "train_dice_loss": (
                    train_metrics["dice_loss"]
                ),

                "val_loss": (
                    val_metrics["loss"]
                ),

                "val_ce_loss": (
                    val_metrics["ce_loss"]
                ),

                "val_dice_loss": (
                    val_metrics["dice_loss"]
                ),

                "train_dice_background": (
                    train_metrics[
                        "dice_background"
                    ]
                ),

                "train_dice_normal_wmh": (
                    train_metrics[
                        "dice_normal_wmh"
                    ]
                ),

                "train_dice_abnormal_wmh": (
                    train_metrics[
                        "dice_abnormal_wmh"
                    ]
                ),

                "val_dice_background": (
                    val_metrics[
                        "dice_background"
                    ]
                ),

                "val_dice_normal_wmh": (
                    val_metrics[
                        "dice_normal_wmh"
                    ]
                ),

                "val_dice_abnormal_wmh": (
                    val_metrics[
                        "dice_abnormal_wmh"
                    ]
                ),

                "best_val_dice_abnormal_wmh": (
                    best_metric
                ),

                "learning_rate": (
                    optimizer.param_groups[
                        0
                    ]["lr"]
                )
            }
        )

        # ----------------------------------------------------
        # SAVE EVERY EPOCH .PTH
        # ----------------------------------------------------

        if (
            epoch % CHECKPOINT_EVERY == 0
            or epoch == EPOCHS
        ):
            save_checkpoint(
                fold_ckpt_dir
                / f"epoch_{epoch:04d}.pth",
                fold,
                epoch,
                model,
                optimizer,
                best_metric,
                class_counts,
                class_weights
            )

        # ----------------------------------------------------
        # SAVE BEST .PTH
        # ----------------------------------------------------

        if improved:
            save_checkpoint(
                fold_ckpt_dir
                / "best.pth",
                fold,
                epoch,
                model,
                optimizer,
                best_metric,
                class_counts,
                class_weights
            )

    # ========================================================
    # ALL EPOCHS FINISHED
    # No graphs were generated during training.
    # Final validation now runs using best.pth.
    # ========================================================

    # ========================================================
    # FINAL VALIDATION USING BEST CHECKPOINT
    # ========================================================

    best_path = (
        fold_ckpt_dir
        / "best.pth"
    )

    if not best_path.exists():
        raise RuntimeError(
            f"Fold {fold}: best.pth not found"
        )

    best_checkpoint = torch.load(
        best_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state"
        ]
    )

    final_metrics = (
        evaluate_fold_3d(
            model,
            val_loader,
            val_ids
        )
    )

    final_metrics[
        "fold"
    ] = fold

    final_metrics[
        "best_epoch"
    ] = int(
        best_checkpoint[
            "epoch"
        ]
    )

    atomic_write_text(
        final_metrics_path,
        json.dumps(
            final_metrics,
            indent=2
        )
    )

    atomic_write_text(
        done_marker,
        json.dumps(
            {
                "fold": fold,
                "status": "done",
                "best_epoch": (
                    final_metrics[
                        "best_epoch"
                    ]
                )
            },
            indent=2
        )
    )

    print()
    print(
        f"Fold {fold} FINAL VALIDATION METRICS"
    )

    print(
        "-" * 72
    )

    print(
        f"Best epoch: "
        f"{final_metrics['best_epoch']}"
    )

    for key, value in final_metrics.items():
        if key in {
            "fold",
            "best_epoch"
        }:
            continue

        if (
            isinstance(
                value,
                float
            )
            and np.isfinite(value)
        ):
            print(
                f"  {key:<34}: "
                f"{value:.6f}"
            )

        else:
            print(
                f"  {key:<34}: NaN"
            )

    # --------------------------------------------------------
    # GENERATE THIS FOLD'S GRAPHS ONCE, ONLY AFTER TRAINING
    # AND FINAL VALIDATION HAVE COMPLETED.
    # --------------------------------------------------------
    print()
    print(
        f"Fold {fold}: generating final training graphs..."
    )

    plot_fold_graphs(
        fold_log_dir,
        fold
    )


# ============================================================
# SUMMARY CSV
# ============================================================

METRIC_COLUMNS = [
    f"{metric}_{class_name}"
    for metric in [
        "dice",
        "sensitivity",
        "hd95"
    ]
    for class_name in CLASS_NAMES
]


def rebuild_fold_summary():
    rows = []

    for fold in range(
        1,
        6
    ):
        path = (
            LOG_ROOT
            / f"fold_{fold}"
            / "final_metrics.json"
        )

        if not path.exists():
            continue

        metrics = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        row = {
            "fold": str(fold)
        }

        for metric in METRIC_COLUMNS:
            row[
                metric
            ] = metrics.get(
                metric,
                math.nan
            )

        rows.append(row)

    if not rows:
        return

    mean_row = {
        "fold": "mean"
    }

    std_row = {
        "fold": "std"
    }

    for metric in METRIC_COLUMNS:
        values = np.asarray(
            [
                float(
                    row[metric]
                )
                for row in rows
            ],
            dtype=float
        )

        finite = values[
            np.isfinite(values)
        ]

        if finite.size == 0:
            mean_row[
                metric
            ] = math.nan

            std_row[
                metric
            ] = math.nan

        else:
            mean_row[
                metric
            ] = float(
                np.mean(finite)
            )

            if finite.size > 1:
                std_row[
                    metric
                ] = float(
                    np.std(
                        finite,
                        ddof=1
                    )
                )

            else:
                std_row[
                    metric
                ] = 0.0

    rows.extend(
        [
            mean_row,
            std_row
        ]
    )

    SUMMARY_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with SUMMARY_CSV.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                ["fold"]
                + METRIC_COLUMNS
            )
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# FAILURE LOG
# ============================================================

def append_failure(
    fold,
    exc
):
    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    old_text = ""

    if FAILURE_LOG.exists():
        old_text = FAILURE_LOG.read_text(
            encoding="utf-8"
        )

    new_text = (
        old_text
        + "\n"
        + "=" * 88
        + "\n"
        + f"FOLD {fold} FAILED\n"
        + "=" * 88
        + "\n"
        + traceback.format_exc()
        + "\n"
    )

    atomic_write_text(
        FAILURE_LOG,
        new_text
    )


# ============================================================
# MAIN
# ============================================================

def main():
    seed_everything(
        RANDOM_SEED
    )

    print("=" * 88)
    print(
        "BASELINE PLAIN 2D U-NET — 3 CLASS"
    )
    print("=" * 88)

    print()
    print("Split root:")
    print(SPLIT_ROOT)

    print()
    print("Input channels:")
    print("  0 = T1")
    print("  1 = T2")
    print("  2 = FLAIR")

    print()
    print("Target classes:")
    print("  0 = background")
    print("  1 = normal WMH")
    print("  2 = abnormal WMH")

    print()
    print("Distance conditioning: NONE")
    print("Model: plain U-Net")
    print("Loss: weighted CE + Dice")
    print("Checkpoint: EVERY EPOCH (.pth), FRESH RUN / NO RESUME")
    print("Plots: GENERATED ONLY AT THE END OF EACH FOLD")
    print("Epoch metrics: CSV (old fold CSV is replaced on restart)")
    print("Final TEST set: SEALED / NOT EVALUATED")

    print()
    print(
        "Device:",
        DEVICE
    )

    if DEVICE.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    folds = (
        [ONLY_FOLD]
        if ONLY_FOLD is not None
        else [
            1,
            2,
            3,
            4,
            5
        ]
    )

    for fold in folds:
        try:
            train_fold(
                fold
            )

        except Exception as exc:
            print()
            print(
                f"FOLD {fold} FAILED:"
            )

            print(exc)

            append_failure(
                fold,
                exc
            )

            continue

        rebuild_fold_summary()

    # ========================================================
    # ALL REQUESTED FOLDS FINISHED
    # Build final summary and cross-fold graphs once.
    # ========================================================
    rebuild_fold_summary()

    print()
    print(
        "Generating final cross-fold graphs..."
    )

    plot_cross_fold_summary()

    print()
    print("=" * 88)
    print(
        "BASELINE TRAINING FINISHED"
    )
    print("=" * 88)

    print()
    print(
        "Fold summary CSV:"
    )
    print(
        SUMMARY_CSV
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "The 30-patient final test set was used ONLY "
        "for leakage checking and was NOT evaluated."
    )


if __name__ == "__main__":
    main()
