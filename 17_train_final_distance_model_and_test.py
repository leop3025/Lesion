from __future__ import annotations

from pathlib import Path

import contextlib
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

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
)

from torch.utils.data import (
    Dataset,
    DataLoader,
)


# ============================================================
# PROJECT IMPORT
# ============================================================

THIS_FILE = Path(__file__).resolve()
BASE_DIR = THIS_FILE.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BASE_DIR),
    )

from unet2d import UNet2D


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Preethi\Documents\LESION"
)

SPLIT_ROOT = (
    PROJECT_DIR
    / "splits"
    / "baseline_2d"
)

PREPROCESSED_ROOT = (
    PROJECT_DIR
    / "preprocessed"
)

DISTANCE_ROOT = (
    PROJECT_DIR
    / "distance_labels"
)

CV_LOG_ROOT = (
    PROJECT_DIR
    / "logs"
    / "distance_conditioned_unet"
)

FINAL_LOG_DIR = (
    PROJECT_DIR
    / "logs"
    / "distance_conditioned_unet"
    / "final_results"
)

FINAL_CKPT_DIR = (
    PROJECT_DIR
    / "checkpoints"
    / "distance_conditioned_final70"
)

FINAL_MODEL_PATH = (
    FINAL_CKPT_DIR
    / "final70_distance_model.pth"
)

TERMINAL_LOG = (
    FINAL_LOG_DIR
    / "17_final_distance_training_and_test_terminal.txt"
)

TRAINING_HISTORY_CSV = (
    FINAL_LOG_DIR
    / "final70_training_history.csv"
)

TRAIN_PATIENT_METRICS_CSV = (
    FINAL_LOG_DIR
    / "train70_per_patient_metrics.csv"
)

TRAIN_METRICS_JSON = (
    FINAL_LOG_DIR
    / "train70_final_metrics.json"
)

TRAIN_METRICS_CSV = (
    FINAL_LOG_DIR
    / "train70_final_metrics.csv"
)

TEST_PATIENT_METRICS_CSV = (
    FINAL_LOG_DIR
    / "test30_per_patient_metrics.csv"
)

TEST_METRICS_JSON = (
    FINAL_LOG_DIR
    / "test30_final_metrics.json"
)

TEST_METRICS_CSV = (
    FINAL_LOG_DIR
    / "test30_final_metrics.csv"
)

CV_METRICS_JSON = (
    FINAL_LOG_DIR
    / "overall_cv_metrics.json"
)

CV_METRICS_CSV = (
    FINAL_LOG_DIR
    / "overall_cv_metrics.csv"
)

FINAL_REPORT_CSV = (
    FINAL_LOG_DIR
    / "distance_conditioned_train_cv_test_summary.csv"
)


# ============================================================
# CONFIG
# ============================================================

NUM_CLASSES = 3

CLASS_NAMES = [
    "background",
    "normal_wmh",
    "abnormal_wmh",
]

BASE_CHANNEL_ORDER = [
    "T1",
    "T2",
    "FLAIR",
]

CHANNEL_ORDER = [
    "T1",
    "T2",
    "FLAIR",
    "DISTANCE_TO_VENTRICLE",
]

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


# ============================================================
# AUGMENTATION — SAME AS BASELINE
# ============================================================

USE_AUGMENTATION = True

P_HORIZONTAL_FLIP = 0.5
P_VERTICAL_FLIP = 0.5
P_ROT90 = 0.5

P_INTENSITY_SCALE = 0.3

INTENSITY_SCALE_RANGE = (
    0.90,
    1.10,
)

P_INTENSITY_SHIFT = 0.3

INTENSITY_SHIFT_RANGE = (
    -0.10,
    0.10,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = (
    DEVICE.type == "cuda"
)


# ============================================================
# METRICS
# ============================================================

METRIC_COLUMNS = [
    f"{metric}_{class_name}"
    for metric in [
        "dice",
        "sensitivity",
        "hd95",
    ]
    for class_name in CLASS_NAMES
]


# ============================================================
# LOGGER
# ============================================================

class Tee:

    def __init__(
        self,
        terminal,
        logfile,
    ):
        self.terminal = terminal
        self.logfile = logfile

    def write(
        self,
        message,
    ):
        self.terminal.write(
            message
        )

        self.logfile.write(
            message
        )

        self.terminal.flush()
        self.logfile.flush()

    def flush(
        self,
    ):
        self.terminal.flush()
        self.logfile.flush()


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(
    seed,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# ATOMIC FILE HELPERS
# ============================================================

def atomic_write_text(
    path,
    text,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
    )

    os.close(
        fd
    )

    tmp = Path(
        tmp_name
    )

    try:

        tmp.write_text(
            text,
            encoding="utf-8",
        )

        os.replace(
            tmp,
            path,
        )

    finally:

        if tmp.exists():
            tmp.unlink(
                missing_ok=True
            )


def atomic_torch_save(
    path,
    payload,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
    )

    os.close(
        fd
    )

    tmp = Path(
        tmp_name
    )

    try:

        torch.save(
            payload,
            tmp,
        )

        os.replace(
            tmp,
            path,
        )

    finally:

        if tmp.exists():
            tmp.unlink(
                missing_ok=True
            )


# ============================================================
# SPLIT HELPERS
# ============================================================

def read_patient_ids(
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            f"Missing split file:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(
                f
            )
        )

    if not rows:

        raise RuntimeError(
            f"Empty split file:\n{path}"
        )

    ids = [

        str(
            row["patient_id"]
        )
        .strip()
        .zfill(3)

        for row in rows
    ]

    if len(ids) != len(set(ids)):

        raise RuntimeError(
            f"Duplicate patient IDs:\n{path}"
        )

    return ids


def patient_id_from_filename(
    path,
):

    match = re.match(
        r"(\d{3})_slice_\d+\.npz$",
        path.name,
    )

    if not match:

        raise RuntimeError(
            f"Unexpected filename: {path.name}"
        )

    return match.group(
        1
    )


# ============================================================
# DISTANCE MAP
# ============================================================

def distance_path_for_patient(
    patient_id,
):

    return (
        DISTANCE_ROOT
        / patient_id
        / (
            f"{patient_id}"
            "_distance_to_ventricle_normalized.nii.gz"
        )
    )


def find_flair_for_patient(
    patient_id,
):

    patient_dir = (
        PREPROCESSED_ROOT
        / patient_id
    )

    candidates = [

        p

        for p in patient_dir.rglob(
            "*.nii*"
        )

        if (
            "flair"
            in p.name.lower()
            and
            "mask"
            not in p.name.lower()
        )
    ]

    if len(candidates) != 1:

        raise RuntimeError(
            f"{patient_id}: expected exactly "
            f"one FLAIR; got {len(candidates)}"
        )

    return candidates[
        0
    ]


def load_distance_volume(
    patient_id,
):

    path = distance_path_for_patient(
        patient_id
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing distance map:\n{path}"
        )

    distance_img = nib.load(
        str(path)
    )

    flair_img = nib.load(
        str(
            find_flair_for_patient(
                patient_id
            )
        )
    )

    distance = np.asarray(
        distance_img.dataobj,
        dtype=np.float32,
    )

    if distance.shape != flair_img.shape[:3]:

        raise RuntimeError(
            f"{patient_id}: distance/FLAIR "
            "shape mismatch."
        )

    if not np.allclose(
        distance_img.affine,
        flair_img.affine,
        atol=1e-4,
        rtol=0,
    ):

        raise RuntimeError(
            f"{patient_id}: distance/FLAIR "
            "affine mismatch."
        )

    if not np.all(
        np.isfinite(
            distance
        )
    ):

        raise RuntimeError(
            f"{patient_id}: distance map "
            "contains NaN/Inf."
        )

    if float(
        np.min(
            distance
        )
    ) < -1e-5:

        raise RuntimeError(
            f"{patient_id}: negative "
            "normalized distance."
        )

    if float(
        np.max(
            distance
        )
    ) > 1.00001:

        raise RuntimeError(
            f"{patient_id}: normalized "
            "distance > 1."
        )

    return distance


# ============================================================
# DATASET
# ============================================================

class DistanceDataset(
    Dataset
):

    def __init__(
        self,
        partition_roots,
        expected_patient_ids,
        augment=False,
    ):

        self.partition_roots = [
            Path(p)
            for p in partition_roots
        ]

        self.expected_patient_ids = set(
            expected_patient_ids
        )

        self.augment = augment

        self.samples = []

        self.distance_cache = {}

        seen_names = set()

        for partition_root in self.partition_roots:

            image_dir = (
                partition_root
                / "images"
            )

            label_dir = (
                partition_root
                / "labels"
            )

            if not image_dir.exists():

                raise FileNotFoundError(
                    f"Missing image dir:\n{image_dir}"
                )

            if not label_dir.exists():

                raise FileNotFoundError(
                    f"Missing label dir:\n{label_dir}"
                )

            image_paths = sorted(
                image_dir.glob(
                    "*.npz"
                )
            )

            for image_path in image_paths:

                pid = (
                    patient_id_from_filename(
                        image_path
                    )
                )

                if pid not in self.expected_patient_ids:

                    raise RuntimeError(
                        f"LEAKAGE BLOCK: patient {pid} "
                        "not allowed in this dataset."
                    )

                if image_path.name in seen_names:

                    raise RuntimeError(
                        "Duplicate slice across partitions: "
                        + image_path.name
                    )

                seen_names.add(
                    image_path.name
                )

                label_path = (
                    label_dir
                    / image_path.name
                )

                if not label_path.exists():

                    raise FileNotFoundError(
                        f"Missing label:\n{label_path}"
                    )

                self.samples.append(
                    (
                        image_path,
                        label_path,
                    )
                )

        actual_ids = {

            patient_id_from_filename(
                p
            )

            for p, _
            in self.samples
        }

        if actual_ids != self.expected_patient_ids:

            missing = sorted(
                self.expected_patient_ids
                - actual_ids
            )

            extra = sorted(
                actual_ids
                - self.expected_patient_ids
            )

            raise RuntimeError(
                "Patient membership mismatch. "
                f"missing={missing}, extra={extra}"
            )

        print()
        print(
            "Loading/validating distance maps..."
        )

        for patient_id in sorted(
            self.expected_patient_ids
        ):

            self.distance_cache[
                patient_id
            ] = load_distance_volume(
                patient_id
            )

        print(
            "Distance maps validated:",
            len(
                self.distance_cache
            )
        )


    def __len__(
        self,
    ):

        return len(
            self.samples
        )


    def _augment(
        self,
        image,
        label,
    ):

        # ====================================================
        # GEOMETRIC AUGMENTATION:
        # applies to all 4 channels and label
        # ====================================================

        if random.random() < P_HORIZONTAL_FLIP:

            image = np.flip(
                image,
                axis=2,
            ).copy()

            label = np.flip(
                label,
                axis=1,
            ).copy()

        if random.random() < P_VERTICAL_FLIP:

            image = np.flip(
                image,
                axis=1,
            ).copy()

            label = np.flip(
                label,
                axis=0,
            ).copy()

        if random.random() < P_ROT90:

            k = random.choice(
                [
                    1,
                    2,
                    3,
                ]
            )

            image = np.rot90(
                image,
                k=k,
                axes=(
                    1,
                    2,
                ),
            ).copy()

            label = np.rot90(
                label,
                k=k,
                axes=(
                    0,
                    1,
                ),
            ).copy()


        # ====================================================
        # INTENSITY AUGMENTATION:
        # MRI CHANNELS ONLY
        # channel 3 distance is untouched
        # ====================================================

        if random.random() < P_INTENSITY_SCALE:

            scale = random.uniform(
                *INTENSITY_SCALE_RANGE
            )

            image[
                :3
            ] *= scale

        if random.random() < P_INTENSITY_SHIFT:

            shift = random.uniform(
                *INTENSITY_SHIFT_RANGE
            )

            image[
                :3
            ] += shift

        return (
            image,
            label,
        )


    def __getitem__(
        self,
        index,
    ):

        image_path, label_path = (
            self.samples[
                index
            ]
        )

        with np.load(
            image_path,
            allow_pickle=False,
        ) as data:

            image = data[
                "image"
            ].astype(
                np.float32,
                copy=True,
            )

            patient_id = str(
                data[
                    "patient_id"
                ]
            ).zfill(
                3
            )

            slice_index = int(
                data[
                    "slice_index"
                ]
            )

            order = [
                str(x)
                for x in data[
                    "channel_order"
                ].tolist()
            ]

        if order != BASE_CHANNEL_ORDER:

            raise RuntimeError(
                f"Channel order mismatch: {order}"
            )

        if image.shape[
            0
        ] != 3:

            raise RuntimeError(
                f"Expected 3 MRI channels, "
                f"got {image.shape}"
            )

        with np.load(
            label_path,
            allow_pickle=False,
        ) as data:

            label = data[
                "label"
            ].astype(
                np.int64,
                copy=True,
            )

        unique = set(
            np.unique(
                label
            ).tolist()
        )

        if not unique.issubset(
            {
                0,
                1,
                2,
            }
        ):

            raise RuntimeError(
                f"Invalid labels: {unique}"
            )

        distance_volume = (
            self.distance_cache[
                patient_id
            ]
        )

        if slice_index >= distance_volume.shape[
            2
        ]:

            raise RuntimeError(
                f"{patient_id}: bad slice index "
                f"{slice_index}"
            )

        distance = (
            distance_volume[
                :,
                :,
                slice_index
            ]
            .astype(
                np.float32,
                copy=True,
            )
        )

        if distance.shape != label.shape:

            raise RuntimeError(
                f"{patient_id}: distance slice "
                "shape mismatch."
            )

        image = np.concatenate(
            [
                image,
                distance[
                    None,
                    ...
                ],
            ],
            axis=0,
        )

        if image.shape[
            0
        ] != 4:

            raise RuntimeError(
                "Expected four input channels."
            )

        if (
            self.augment
            and USE_AUGMENTATION
        ):

            image, label = (
                self._augment(
                    image,
                    label,
                )
            )

        return {

            "image":
                torch.from_numpy(
                    image
                ),

            "label":
                torch.from_numpy(
                    label
                ),

            "patient_id":
                patient_id,

            "slice_index":
                slice_index,
        }


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_counts(
    dataset,
):

    counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64,
    )

    for _, label_path in dataset.samples:

        with np.load(
            label_path,
            allow_pickle=False,
        ) as data:

            label = data[
                "label"
            ]

        counts += np.bincount(
            label.ravel(),
            minlength=NUM_CLASSES,
        )[
            :NUM_CLASSES
        ]

    if np.any(
        counts == 0
    ):

        raise RuntimeError(
            f"Zero class voxels: {counts.tolist()}"
        )

    return counts


def class_weights_from_counts(
    counts,
):

    frequencies = (
        counts.astype(
            np.float64
        )
        / counts.sum()
    )

    weights = (
        1.0
        /
        np.power(
            frequencies,
            CLASS_WEIGHT_POWER,
        )
    )

    weights /= weights.mean()

    return weights.astype(
        np.float32
    )


# ============================================================
# LOSS
# ============================================================

class MulticlassDiceLoss(
    nn.Module
):

    def __init__(
        self,
        include_background=False,
        smooth=1e-6,
    ):

        super().__init__()

        self.include_background = (
            include_background
        )

        self.smooth = smooth


    def forward(
        self,
        logits,
        target,
    ):

        probs = torch.softmax(
            logits,
            dim=1,
        )

        one_hot = (
            torch.nn.functional.one_hot(
                target,
                NUM_CLASSES,
            )
            .permute(
                0,
                3,
                1,
                2,
            )
            .float()
        )

        start = (
            0
            if self.include_background
            else 1
        )

        losses = []

        for cls in range(
            start,
            NUM_CLASSES,
        ):

            p = probs[
                :,
                cls,
            ]

            t = one_hot[
                :,
                cls,
            ]

            intersection = torch.sum(
                p * t,
                dim=(
                    1,
                    2,
                ),
            )

            denom = (
                torch.sum(
                    p,
                    dim=(
                        1,
                        2,
                    ),
                )
                +
                torch.sum(
                    t,
                    dim=(
                        1,
                        2,
                    ),
                )
            )

            dice = (
                2 * intersection
                + self.smooth
            ) / (
                denom
                + self.smooth
            )

            losses.append(
                1 - dice
            )

        return torch.stack(
            losses
        ).mean()


class CombinedLoss(
    nn.Module
):

    def __init__(
        self,
        class_weights,
    ):

        super().__init__()

        self.ce = nn.CrossEntropyLoss(
            weight=class_weights
        )

        self.dice = MulticlassDiceLoss(
            include_background=(
                INCLUDE_BACKGROUND_IN_DICE_LOSS
            )
        )


    def forward(
        self,
        logits,
        target,
    ):

        ce = self.ce(
            logits,
            target,
        )

        dice = self.dice(
            logits,
            target,
        )

        total = (
            CE_LOSS_WEIGHT * ce
            +
            DICE_LOSS_WEIGHT * dice
        )

        return (
            total,
            ce,
            dice,
        )


# ============================================================
# FAST DICE
# ============================================================

def dice_from_logits(
    logits,
    target,
):

    pred = torch.argmax(
        logits,
        dim=1,
    )

    scores = []

    for cls in range(
        NUM_CLASSES
    ):

        p = (
            pred == cls
        )

        t = (
            target == cls
        )

        intersection = torch.sum(
            p & t
        ).float()

        denom = (
            torch.sum(
                p
            ).float()
            +
            torch.sum(
                t
            ).float()
        )

        if float(
            denom
        ) == 0:

            score = 1.0

        else:

            score = float(
                (
                    2
                    * intersection
                    / denom
                )
                .detach()
                .cpu()
            )

        scores.append(
            score
        )

    return scores


# ============================================================
# TRAIN EPOCH
# ============================================================

def run_train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
):

    model.train()

    total_losses = []
    ce_losses = []
    dice_losses = []

    dice_values = [
        [],
        [],
        [],
    ]

    for batch in loader:

        image = (
            batch[
                "image"
            ]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        label = (
            batch[
                "label"
            ]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP,
        ):

            logits = model(
                image
            )

            (
                loss,
                ce,
                dice_loss,
            ) = criterion(
                logits,
                label,
            )

        if scaler is not None:

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

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
                ce.detach().cpu()
            )
        )

        dice_losses.append(
            float(
                dice_loss.detach().cpu()
            )
        )

        scores = dice_from_logits(
            logits.detach(),
            label,
        )

        for cls in range(
            NUM_CLASSES
        ):

            dice_values[
                cls
            ].append(
                scores[
                    cls
                ]
            )

    return {

        "loss":
            float(
                np.mean(
                    total_losses
                )
            ),

        "ce_loss":
            float(
                np.mean(
                    ce_losses
                )
            ),

        "dice_loss":
            float(
                np.mean(
                    dice_losses
                )
            ),

        "dice_background":
            float(
                np.mean(
                    dice_values[
                        0
                    ]
                )
            ),

        "dice_normal_wmh":
            float(
                np.mean(
                    dice_values[
                        1
                    ]
                )
            ),

        "dice_abnormal_wmh":
            float(
                np.mean(
                    dice_values[
                        2
                    ]
                )
            ),
    }


# ============================================================
# TRAINING HISTORY
# ============================================================

def write_history_header(
    path,
):

    fields = [

        "epoch",

        "train_loss",

        "train_ce_loss",

        "train_dice_loss",

        "dice_background",

        "dice_normal_wmh",

        "dice_abnormal_wmh",

        "learning_rate",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        csv.DictWriter(
            f,
            fieldnames=fields,
        ).writeheader()


def append_history(
    path,
    epoch,
    metrics,
    optimizer,
):

    fields = [

        "epoch",

        "train_loss",

        "train_ce_loss",

        "train_dice_loss",

        "dice_background",

        "dice_normal_wmh",

        "dice_abnormal_wmh",

        "learning_rate",
    ]

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writerow(
            {

                "epoch":
                    epoch,

                "train_loss":
                    metrics[
                        "loss"
                    ],

                "train_ce_loss":
                    metrics[
                        "ce_loss"
                    ],

                "train_dice_loss":
                    metrics[
                        "dice_loss"
                    ],

                "dice_background":
                    metrics[
                        "dice_background"
                    ],

                "dice_normal_wmh":
                    metrics[
                        "dice_normal_wmh"
                    ],

                "dice_abnormal_wmh":
                    metrics[
                        "dice_abnormal_wmh"
                    ],

                "learning_rate":
                    optimizer.param_groups[
                        0
                    ][
                        "lr"
                    ],
            }
        )


# ============================================================
# 3D METRICS
# ============================================================

def dice_and_sensitivity(
    pred,
    target,
    cls,
):

    p = (
        pred == cls
    )

    t = (
        target == cls
    )

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

    dice = (
        1.0
        if denom == 0
        else (
            2 * tp
            / denom
        )
    )

    positives = (
        tp + fn
    )

    sensitivity = (
        math.nan
        if positives == 0
        else (
            tp / positives
        )
    )

    return (
        float(
            dice
        ),
        float(
            sensitivity
        ),
    )


def hd95(
    pred_mask,
    target_mask,
    spacing,
):

    pred_mask = (
        pred_mask.astype(
            bool
        )
    )

    target_mask = (
        target_mask.astype(
            bool
        )
    )

    pred_exists = bool(
        pred_mask.any()
    )

    target_exists = bool(
        target_mask.any()
    )

    if (
        not pred_exists
        and
        not target_exists
    ):

        return 0.0

    if (
        pred_exists
        != target_exists
    ):

        return math.nan

    structure = np.ones(
        (
            3,
            3,
            3,
        ),
        dtype=bool,
    )

    pred_surface = (
        pred_mask
        ^
        binary_erosion(
            pred_mask,
            structure=structure,
            border_value=0,
        )
    )

    target_surface = (
        target_mask
        ^
        binary_erosion(
            target_mask,
            structure=structure,
            border_value=0,
        )
    )

    if (
        not pred_surface.any()
        or
        not target_surface.any()
    ):

        return math.nan

    d_target = (
        distance_transform_edt(
            ~target_surface,
            sampling=spacing,
        )
    )

    d_pred = (
        distance_transform_edt(
            ~pred_surface,
            sampling=spacing,
        )
    )

    distances = np.concatenate(
        [

            d_target[
                pred_surface
            ],

            d_pred[
                target_surface
            ],
        ]
    )

    return float(
        np.percentile(
            distances,
            95,
        )
    )


# ============================================================
# 3D EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_3d(
    model,
    loader,
    patient_ids,
):

    model.eval()

    pred_volumes = {}
    target_volumes = {}
    spacing_map = {}

    for patient_id in sorted(
        patient_ids
    ):

        flair_img = nib.load(
            str(
                find_flair_for_patient(
                    patient_id
                )
            )
        )

        shape = (
            flair_img.shape[
                :3
            ]
        )

        spacing = tuple(
            float(x)
            for x in flair_img.header.get_zooms()[
                :3
            ]
        )

        pred_volumes[
            patient_id
        ] = np.zeros(
            shape,
            dtype=np.uint8,
        )

        target_volumes[
            patient_id
        ] = np.zeros(
            shape,
            dtype=np.uint8,
        )

        spacing_map[
            patient_id
        ] = spacing


    for batch in loader:

        image = (
            batch[
                "image"
            ]
            .to(
                DEVICE,
                non_blocking=True,
            )
        )

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP,
        ):

            logits = model(
                image
            )

        prediction = (
            torch.argmax(
                logits,
                dim=1,
            )
            .cpu()
            .numpy()
            .astype(
                np.uint8
            )
        )

        target = (
            batch[
                "label"
            ]
            .numpy()
            .astype(
                np.uint8
            )
        )

        patient_batch = list(
            batch[
                "patient_id"
            ]
        )

        slice_indices = (
            batch[
                "slice_index"
            ]
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
            patient_batch
        ):

            z = int(
                slice_indices[
                    i
                ]
            )

            pred_volumes[
                patient_id
            ][
                :,
                :,
                z
            ] = prediction[
                i
            ]

            target_volumes[
                patient_id
            ][
                :,
                :,
                z
            ] = target[
                i
            ]


    rows = []

    for patient_id in sorted(
        patient_ids
    ):

        pred = (
            pred_volumes[
                patient_id
            ]
        )

        target = (
            target_volumes[
                patient_id
            ]
        )

        spacing = (
            spacing_map[
                patient_id
            ]
        )

        row = {
            "patient_id":
                patient_id
        }

        for (
            cls,
            class_name,
        ) in enumerate(
            CLASS_NAMES
        ):

            dice, sensitivity = (
                dice_and_sensitivity(
                    pred,
                    target,
                    cls,
                )
            )

            hausdorff = hd95(

                pred == cls,

                target == cls,

                spacing,
            )

            row[
                f"dice_{class_name}"
            ] = dice

            row[
                f"sensitivity_{class_name}"
            ] = sensitivity

            row[
                f"hd95_{class_name}"
            ] = hausdorff

        rows.append(
            row
        )

    return rows


# ============================================================
# STATISTICS
# ============================================================

def finite_mean_std(
    values,
):

    array = np.asarray(
        values,
        dtype=float,
    )

    finite = array[
        np.isfinite(
            array
        )
    ]

    if finite.size == 0:

        return (
            math.nan,
            math.nan,
            0,
        )

    mean = float(
        np.mean(
            finite
        )
    )

    std = (

        float(
            np.std(
                finite,
                ddof=1,
            )
        )

        if finite.size > 1

        else 0.0
    )

    return (
        mean,
        std,
        int(
            finite.size
        ),
    )


def summarize_metrics(
    rows,
):

    summary = {

        "n_patients":
            len(
                rows
            )
    }

    for metric in METRIC_COLUMNS:

        (
            mean,
            std,
            n_finite,
        ) = finite_mean_std(
            [
                float(
                    row[
                        metric
                    ]
                )
                for row in rows
            ]
        )

        summary[
            f"{metric}_mean"
        ] = mean

        summary[
            f"{metric}_std"
        ] = std

        summary[
            f"{metric}_n_finite"
        ] = n_finite

    return summary


# ============================================================
# SAVE PARTITION RESULTS
# ============================================================

def save_partition_results(
    rows,
    summary,
    per_patient_csv,
    summary_json,
    summary_csv,
):

    with per_patient_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=(
                [
                    "patient_id"
                ]
                +
                METRIC_COLUMNS
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    atomic_write_text(

        summary_json,

        json.dumps(
            summary,
            indent=2,
        ),
    )


    with summary_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=[
                "metric",
                "mean",
                "std",
                "n_finite",
            ],
        )

        writer.writeheader()

        for metric in METRIC_COLUMNS:

            writer.writerow(
                {

                    "metric":
                        metric,

                    "mean":
                        summary[
                            f"{metric}_mean"
                        ],

                    "std":
                        summary[
                            f"{metric}_std"
                        ],

                    "n_finite":
                        summary[
                            f"{metric}_n_finite"
                        ],
                }
            )


# ============================================================
# LOAD CV RESULTS
# ============================================================

def load_cv_metrics():

    rows = []

    for fold in range(
        1,
        6
    ):

        path = (
            CV_LOG_ROOT
            / f"fold_{fold}"
            / "final_metrics.json"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Missing Step-16 result:\n{path}"
            )

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if int(
            data[
                "fold"
            ]
        ) != fold:

            raise RuntimeError(
                f"Fold mismatch in {path}"
            )

        if (
            "best_epoch"
            not in data
        ):

            raise RuntimeError(
                f"best_epoch missing: {path}"
            )

        rows.append(
            data
        )

    return rows


def choose_final_epochs(
    cv_rows,
):

    epochs = np.asarray(
        [

            int(
                row[
                    "best_epoch"
                ]
            )

            for row in cv_rows
        ],
        dtype=int,
    )

    result = int(
        np.rint(
            np.median(
                epochs
            )
        )
    )

    if result < 1:

        raise RuntimeError(
            "Invalid final epoch count."
        )

    return result


def save_cv_summary(
    cv_rows,
):

    summary = {

        "n_folds":
            5,

        "best_epochs":
            [

                int(
                    row[
                        "best_epoch"
                    ]
                )

                for row in cv_rows
            ],
    }

    for metric in METRIC_COLUMNS:

        values = [

            float(
                row[
                    metric
                ]
            )

            for row in cv_rows
        ]

        (
            mean,
            std,
            count,
        ) = finite_mean_std(
            values
        )

        summary[
            f"{metric}_mean"
        ] = mean

        summary[
            f"{metric}_std"
        ] = std

        summary[
            f"{metric}_n_finite"
        ] = count


    atomic_write_text(

        CV_METRICS_JSON,

        json.dumps(
            summary,
            indent=2,
        ),
    )


    with CV_METRICS_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=[
                "metric",
                "mean",
                "std",
            ],
        )

        writer.writeheader()

        for metric in METRIC_COLUMNS:

            writer.writerow(
                {

                    "metric":
                        metric,

                    "mean":
                        summary[
                            f"{metric}_mean"
                        ],

                    "std":
                        summary[
                            f"{metric}_std"
                        ],
                }
            )

    return summary


# ============================================================
# FINAL COMBINED REPORT
# ============================================================

def save_final_report(
    train_summary,
    cv_summary,
    test_summary,
):

    with FINAL_REPORT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

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

            writer.writerow(
                {

                    "metric":
                        metric,

                    "train70_mean":
                        train_summary[
                            f"{metric}_mean"
                        ],

                    "train70_std":
                        train_summary[
                            f"{metric}_std"
                        ],

                    "cv_mean":
                        cv_summary[
                            f"{metric}_mean"
                        ],

                    "cv_std":
                        cv_summary[
                            f"{metric}_std"
                        ],

                    "test30_mean":
                        test_summary[
                            f"{metric}_mean"
                        ],

                    "test30_std":
                        test_summary[
                            f"{metric}_std"
                        ],
                }
            )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_metric_summary(
    title,
    summary,
    n_patients,
):

    print()
    print(
        "=" * 88
    )

    print(
        title
    )

    print(
        "=" * 88
    )

    for class_name in [
        "normal_wmh",
        "abnormal_wmh",
    ]:

        print()
        print(
            class_name.upper()
        )

        for metric in [
            "dice",
            "sensitivity",
            "hd95",
        ]:

            key = (
                f"{metric}_{class_name}"
            )

            mean = (
                summary[
                    f"{key}_mean"
                ]
            )

            std = (
                summary[
                    f"{key}_std"
                ]
            )

            n = (
                summary[
                    f"{key}_n_finite"
                ]
            )

            unit = (
                " mm"
                if metric == "hd95"
                else ""
            )

            print(
                f"  {metric:<12}: "
                f"{mean:.6f} ± "
                f"{std:.6f}"
                f"{unit} "
                f"(finite n={n}/{n_patients})"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(
        RANDOM_SEED
    )

    print(
        "=" * 88
    )

    print(
        "STEP 17 — FINAL DISTANCE-CONDITIONED MODEL + TEST"
    )

    print(
        "=" * 88
    )

    print()
    print(
        "Device:",
        DEVICE
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )


    # ========================================================
    # LOAD FIVE COMPLETED CV FOLDS
    # ========================================================

    cv_rows = load_cv_metrics()

    best_epochs = [

        int(
            row[
                "best_epoch"
            ]
        )

        for row in cv_rows
    ]

    final_epochs = choose_final_epochs(
        cv_rows
    )

    print()
    print(
        "Five Step-16 folds found."
    )

    print(
        "CV best epochs:",
        best_epochs
    )

    print(
        "Final training epoch count "
        "(median CV best epoch):",
        final_epochs
    )


    # ========================================================
    # PATIENT MEMBERSHIP
    # ========================================================

    fold1 = (
        SPLIT_ROOT
        / "fold_1"
    )

    train_ids = set(
        read_patient_ids(
            fold1
            / "train_patients.csv"
        )
    )

    val_ids = set(
        read_patient_ids(
            fold1
            / "val_patients.csv"
        )
    )

    test_ids = set(
        read_patient_ids(
            SPLIT_ROOT
            / "test_30_patients.csv"
        )
    )

    development_ids = (
        train_ids
        | val_ids
    )

    if train_ids & val_ids:

        raise RuntimeError(
            "Fold-1 train/val overlap."
        )

    if len(
        development_ids
    ) != 70:

        raise RuntimeError(
            f"Expected 70 development patients, "
            f"got {len(development_ids)}"
        )

    if len(
        test_ids
    ) != 30:

        raise RuntimeError(
            f"Expected 30 test patients, "
            f"got {len(test_ids)}"
        )

    if development_ids & test_ids:

        raise RuntimeError(
            "DATA LEAKAGE: development/test overlap."
        )

    print()
    print(
        "Final membership safeguard: PASS"
    )

    print(
        "  development:",
        len(
            development_ids
        )
    )

    print(
        "  test       :",
        len(
            test_ids
        )
    )

    print(
        "  overlap    : 0"
    )


    # ========================================================
    # PREPARE FINAL OUTPUT DIRECTORY
    # ========================================================

    # IMPORTANT:
    # Do NOT delete FINAL_LOG_DIR here because the terminal
    # log file inside it is already open by the __main__ block.

    FINAL_LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

    # Clean only the final checkpoint directory.
    if FINAL_CKPT_DIR.exists():

        shutil.rmtree(
            FINAL_CKPT_DIR
        )

    FINAL_CKPT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # SAVE CV METRICS
    # ========================================================

    cv_summary = save_cv_summary(
        cv_rows
    )


    # ========================================================
    # DATASETS
    # ========================================================

    development_dataset = DistanceDataset(

        partition_roots=[

            fold1
            / "train",

            fold1
            / "val",
        ],

        expected_patient_ids=(
            development_ids
        ),

        augment=True,
    )


    development_eval_dataset = DistanceDataset(

        partition_roots=[

            fold1
            / "train",

            fold1
            / "val",
        ],

        expected_patient_ids=(
            development_ids
        ),

        augment=False,
    )


    test_dataset = DistanceDataset(

        partition_roots=[

            SPLIT_ROOT
            / "test"
        ],

        expected_patient_ids=(
            test_ids
        ),

        augment=False,
    )


    print()
    print(
        "Development slices:",
        len(
            development_dataset
        )
    )

    print(
        "Test slices:",
        len(
            test_dataset
        )
    )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_counts = compute_class_counts(
        development_dataset
    )

    class_weights = class_weights_from_counts(
        class_counts
    )

    print()
    print(
        "70-patient class distribution:"
    )

    for cls, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {name:<14} "
            f"voxels="
            f"{int(class_counts[cls]):>14,d} "
            f"weight="
            f"{float(class_weights[cls]):.6f}"
        )


    # ========================================================
    # LOADERS
    # ========================================================

    development_loader = DataLoader(

        development_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        ),

        drop_last=False,
    )


    development_eval_loader = DataLoader(

        development_eval_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        ),

        drop_last=False,
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type == "cuda"
        ),

        drop_last=False,
    )


    # ========================================================
    # FINAL 4-CHANNEL MODEL
    # ========================================================

    model = UNet2D(

        in_channels=4,

        num_classes=NUM_CLASSES,

        base_channels=BASE_CHANNELS,

    ).to(
        DEVICE
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,
    )


    class_weight_tensor = torch.tensor(

        class_weights,

        dtype=torch.float32,

        device=DEVICE,
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
    # TRAIN FINAL MODEL
    # ========================================================

    write_history_header(
        TRAINING_HISTORY_CSV
    )


    print()
    print(
        "=" * 88
    )

    print(
        f"TRAINING FINAL MODEL ON ALL 70 "
        f"DEVELOPMENT PATIENTS FOR "
        f"{final_epochs} EPOCHS"
    )

    print(
        "=" * 88
    )


    for epoch in range(
        1,
        final_epochs + 1,
    ):

        metrics = run_train_epoch(

            model,

            development_loader,

            optimizer,

            criterion,

            scaler,
        )

        print()
        print(
            f"Final70 | Epoch "
            f"{epoch:03d}/{final_epochs}"
        )

        print(
            "  train total/CE/Dice loss: "
            f"{metrics['loss']:.5f} / "
            f"{metrics['ce_loss']:.5f} / "
            f"{metrics['dice_loss']:.5f}"
        )

        print(
            "  train Dice "
            "BG/nWMH/abWMH: "
            f"{metrics['dice_background']:.4f}/"
            f"{metrics['dice_normal_wmh']:.4f}/"
            f"{metrics['dice_abnormal_wmh']:.4f}"
        )

        append_history(

            TRAINING_HISTORY_CSV,

            epoch,

            metrics,

            optimizer,
        )


    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    atomic_torch_save(

        FINAL_MODEL_PATH,

        {

            "experiment":
                "distance_conditioned_unet",

            "model_state":
                model.state_dict(),

            "optimizer_state":
                optimizer.state_dict(),

            "epochs":
                final_epochs,

            "epoch_selection_rule":
                "median of 5 distance-conditioned "
                "CV best epochs",

            "cv_best_epochs":
                best_epochs,

            "class_counts":
                class_counts.tolist(),

            "class_weights":
                class_weights.tolist(),

            "channel_order":
                CHANNEL_ORDER,

            "num_input_channels":
                4,

            "num_classes":
                NUM_CLASSES,

            "development_patients":
                70,

            "test_patients":
                30,

            "distance_channel":
                "normalized physical distance "
                "to ventricle",

            "test_policy":
                "single evaluation after all "
                "training choices fixed from CV",
        },
    )


    print()
    print(
        "Final model saved:"
    )

    print(
        FINAL_MODEL_PATH
    )


    # ========================================================
    # TRAIN-70 EVALUATION
    # ========================================================

    print()
    print(
        "=" * 88
    )

    print(
        "EVALUATING FINAL MODEL ON 70 DEVELOPMENT PATIENTS"
    )

    print(
        "=" * 88
    )


    train_rows = evaluate_3d(

        model,

        development_eval_loader,

        development_ids,
    )


    train_summary = summarize_metrics(
        train_rows
    )


    train_summary[
        "partition"
    ] = "train70"


    save_partition_results(

        train_rows,

        train_summary,

        TRAIN_PATIENT_METRICS_CSV,

        TRAIN_METRICS_JSON,

        TRAIN_METRICS_CSV,
    )


    # ========================================================
    # FINAL TEST — ONE TIME
    # ========================================================

    print()
    print(
        "=" * 88
    )

    print(
        "ONE-TIME FINAL EVALUATION ON 30-PATIENT TEST SET"
    )

    print(
        "=" * 88
    )


    test_rows = evaluate_3d(

        model,

        test_loader,

        test_ids,
    )


    test_summary = summarize_metrics(
        test_rows
    )


    test_summary[
        "partition"
    ] = "test30"


    test_summary[
        "final_epochs"
    ] = final_epochs


    test_summary[
        "cv_best_epochs"
    ] = best_epochs


    test_summary[
        "test_policy"
    ] = (
        "one-time evaluation after all model "
        "and epoch choices fixed using CV"
    )


    save_partition_results(

        test_rows,

        test_summary,

        TEST_PATIENT_METRICS_CSV,

        TEST_METRICS_JSON,

        TEST_METRICS_CSV,
    )


    # ========================================================
    # COMBINED REPORT
    # ========================================================

    save_final_report(

        train_summary,

        cv_summary,

        test_summary,
    )


    # ========================================================
    # TERMINAL METRICS
    # ========================================================

    print_metric_summary(

        "FINAL 70-PATIENT TRAIN METRICS — MEAN ± SD",

        train_summary,

        70,
    )


    print_metric_summary(

        "FINAL 30-PATIENT TEST METRICS — MEAN ± SD",

        test_summary,

        30,
    )


    print()
    print(
        "=" * 88
    )

    print(
        "DISTANCE-CONDITIONED 5-FOLD CV — MEAN ± SD"
    )

    print(
        "=" * 88
    )


    for class_name in [
        "normal_wmh",
        "abnormal_wmh",
    ]:

        print()
        print(
            class_name.upper()
        )

        for metric in [
            "dice",
            "sensitivity",
            "hd95",
        ]:

            key = (
                f"{metric}_{class_name}"
            )

            mean = (
                cv_summary[
                    f"{key}_mean"
                ]
            )

            std = (
                cv_summary[
                    f"{key}_std"
                ]
            )

            unit = (
                " mm"
                if metric == "hd95"
                else ""
            )

            print(
                f"  {metric:<12}: "
                f"{mean:.6f} ± "
                f"{std:.6f}{unit}"
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print(
        "=" * 88
    )

    print(
        "STEP 17 COMPLETE"
    )

    print(
        "=" * 88
    )

    print()
    print(
        "Final model:"
    )

    print(
        FINAL_MODEL_PATH
    )

    print()
    print(
        "Training history:"
    )

    print(
        TRAINING_HISTORY_CSV
    )

    print()
    print(
        "Train metrics:"
    )

    print(
        TRAIN_METRICS_CSV
    )

    print()
    print(
        "Test per-patient metrics:"
    )

    print(
        TEST_PATIENT_METRICS_CSV
    )

    print()
    print(
        "Final test metrics:"
    )

    print(
        TEST_METRICS_CSV
    )

    print()
    print(
        "Combined Train + CV + Test:"
    )

    print(
        FINAL_REPORT_CSV
    )

    print()
    print(
        "Terminal log:"
    )

    print(
        TERMINAL_LOG
    )


# ============================================================
# RUN + TERMINAL LOG
# ============================================================

if __name__ == "__main__":

    FINAL_LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TERMINAL_LOG.open(
        "a",
        encoding="utf-8",
    ) as logfile:

        logfile.write(
            "\n\n"
            + "=" * 88
            + "\n"
        )

        logfile.write(
            "NEW STEP-17 FINAL DISTANCE MODEL RUN\n"
        )

        logfile.write(
            "=" * 88
            + "\n"
        )

        stdout_tee = Tee(
            sys.stdout,
            logfile,
        )

        stderr_tee = Tee(
            sys.stderr,
            logfile,
        )

        with (
            contextlib.redirect_stdout(
                stdout_tee
            ),
            contextlib.redirect_stderr(
                stderr_tee
            )
        ):

            try:

                main()

            except Exception:

                print()
                print(
                    "=" * 88
                )

                print(
                    "STEP 17 FAILED"
                )

                print(
                    "=" * 88
                )

                traceback.print_exc()

                raise