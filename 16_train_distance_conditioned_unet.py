# ============================================================
# 16_train_distance_conditioned_unet.py
#
# DISTANCE-CONDITIONED 2D U-NET
#
# INPUT CHANNELS
#   0 = T1
#   1 = T2
#   2 = FLAIR
#   3 = normalized distance-to-ventricle
#
# TARGET CLASSES
#   0 = background
#   1 = normal WMH
#   2 = abnormal WMH
#
# IMPORTANT:
# - SAME architecture as baseline
# - SAME folds
# - SAME train/val split
# - SAME loss
# - SAME augmentation
# - SAME optimizer
# - SAME hyperparameters
#
# ONLY experimental difference:
#     Baseline            = 3 channels
#     Distance model      = 4 channels
#
# FINAL 30-PATIENT TEST SET REMAINS SEALED.
#
# Usage:
#
#   Run one fold:
#       python 16_train_distance_conditioned_unet.py --fold 1
#
#   Run all folds:
#       python 16_train_distance_conditioned_unet.py
#
# ============================================================

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

import matplotlib.pyplot as plt
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


# ------------------------------------------------------------
# SAME U-NET AS BASELINE.
#
# unet2d.py must support configurable in_channels.
# Your current UNet2D already does.
# ------------------------------------------------------------

from unet2d import UNet2D


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Preethi\Documents\LESION"
)


# ------------------------------------------------------------
# SAME locked baseline folds
# ------------------------------------------------------------

SPLIT_ROOT = (
    PROJECT_DIR
    / "splits"
    / "baseline_2d"
)


# ------------------------------------------------------------
# Preprocessed 3D NIfTI data
# ------------------------------------------------------------

PREPROCESSED_ROOT = (
    PROJECT_DIR
    / "preprocessed"
)


# ------------------------------------------------------------
# Stage-15 distance maps
# ------------------------------------------------------------

DISTANCE_ROOT = (
    PROJECT_DIR
    / "distance_labels"
)


# ------------------------------------------------------------
# Completely separate experiment outputs
# ------------------------------------------------------------

LOG_ROOT = (
    PROJECT_DIR
    / "logs"
    / "distance_conditioned_unet"
)

CHECKPOINT_ROOT = (
    PROJECT_DIR
    / "checkpoints"
    / "distance_conditioned_unet"
)

SUMMARY_CSV = (
    LOG_ROOT
    / "distance_fold_metrics.csv"
)

FAILURE_LOG = (
    LOG_ROOT
    / "fold_failures.txt"
)

TERMINAL_LOG = (
    LOG_ROOT
    / "16_distance_conditioned_training_terminal.txt"
)


# ============================================================
# MODEL / TRAINING CONFIG
# ============================================================

NUM_CLASSES = 3

CLASS_NAMES = [
    "background",
    "normal_wmh",
    "abnormal_wmh",
]


# ------------------------------------------------------------
# Original MRI order in baseline NPZ files
# ------------------------------------------------------------

BASE_CHANNEL_ORDER = [
    "T1",
    "T2",
    "FLAIR",
]


# ------------------------------------------------------------
# New distance-conditioned order
# ------------------------------------------------------------

CHANNEL_ORDER = [
    "T1",
    "T2",
    "FLAIR",
    "DISTANCE_TO_VENTRICLE",
]


BASE_CHANNELS = 32


# ------------------------------------------------------------
# SAME baseline training settings
# ------------------------------------------------------------

EPOCHS = 50

BATCH_SIZE = 8

NUM_WORKERS = 0

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

CHECKPOINT_EVERY = 1

RANDOM_SEED = 20260911


# ------------------------------------------------------------
# SAME loss
# ------------------------------------------------------------

CE_LOSS_WEIGHT = 1.0

DICE_LOSS_WEIGHT = 1.0

CLASS_WEIGHT_POWER = 0.5

INCLUDE_BACKGROUND_IN_DICE_LOSS = False


# ------------------------------------------------------------
# SAME augmentation probabilities
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = (
    DEVICE.type
    == "cuda"
)


# ============================================================
# OPTIONAL --fold ARGUMENT
# ============================================================

ONLY_FOLD = None

if "--fold" in sys.argv:

    index = sys.argv.index(
        "--fold"
    )

    if index + 1 >= len(
        sys.argv
    ):

        raise RuntimeError(
            "--fold requires a value from 1 to 5"
        )

    ONLY_FOLD = int(
        sys.argv[
            index + 1
        ]
    )

    if ONLY_FOLD not in {
        1,
        2,
        3,
        4,
        5,
    }:

        raise RuntimeError(
            "--fold must be between 1 and 5"
        )


# ============================================================
# TERMINAL + TXT LOG
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

    path = Path(
        path
    )

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

    temporary = Path(
        tmp_name
    )

    try:

        temporary.write_text(
            text,
            encoding="utf-8",
        )

        os.replace(
            temporary,
            path,
        )

    finally:

        if temporary.exists():

            temporary.unlink(
                missing_ok=True
            )


def atomic_torch_save(
    path,
    payload,
):

    path = Path(
        path
    )

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

    temporary = Path(
        tmp_name
    )

    try:

        torch.save(
            payload,
            temporary,
        )

        os.replace(
            temporary,
            path,
        )

    finally:

        if temporary.exists():

            temporary.unlink(
                missing_ok=True
            )


# ============================================================
# SPLIT HELPERS
# ============================================================

def read_patient_ids(
    path,
):

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing split file:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        rows = list(
            csv.DictReader(
                file
            )
        )

    if not rows:

        raise RuntimeError(
            f"Empty split file:\n{path}"
        )

    patient_ids = [

        str(
            row[
                "patient_id"
            ]
        )
        .strip()
        .zfill(3)

        for row in rows
    ]

    if (
        len(patient_ids)
        != len(
            set(
                patient_ids
            )
        )
    ):

        raise RuntimeError(
            f"Duplicate patient IDs in:\n{path}"
        )

    return patient_ids


def patient_id_from_filename(
    path,
):

    match = re.match(
        r"(\d{3})_slice_\d+\.npz$",
        path.name,
    )

    if not match:

        raise RuntimeError(
            "Unexpected slice filename: "
            + path.name
        )

    return match.group(
        1
    )


# ============================================================
# DISTANCE MAP HELPERS
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


def flair_path_for_patient(
    patient_id,
):

    patient_dir = (
        PREPROCESSED_ROOT
        / patient_id
    )

    candidates = [

        path

        for path in patient_dir.rglob(
            "*.nii*"
        )

        if (
            "flair"
            in path.name.lower()
            and
            "mask"
            not in path.name.lower()
        )
    ]

    if len(
        candidates
    ) != 1:

        raise RuntimeError(
            f"{patient_id}: expected exactly one FLAIR "
            f"in {patient_dir}; found {len(candidates)}"
        )

    return candidates[
        0
    ]


def load_and_validate_distance_volume(
    patient_id,
):

    distance_path = (
        distance_path_for_patient(
            patient_id
        )
    )

    if not distance_path.exists():

        raise FileNotFoundError(
            f"{patient_id}: missing distance map:\n"
            f"{distance_path}"
        )

    flair_path = (
        flair_path_for_patient(
            patient_id
        )
    )

    distance_img = nib.load(
        str(
            distance_path
        )
    )

    flair_img = nib.load(
        str(
            flair_path
        )
    )

    distance = np.asarray(
        distance_img.dataobj,
        dtype=np.float32,
    )

    if distance.ndim != 3:

        raise RuntimeError(
            f"{patient_id}: distance map must be 3D; "
            f"got {distance.shape}"
        )

    if (
        distance.shape
        != flair_img.shape[:3]
    ):

        raise RuntimeError(
            f"{patient_id}: distance/FLAIR shape mismatch. "
            f"distance={distance.shape}, "
            f"FLAIR={flair_img.shape[:3]}"
        )

    if not np.allclose(
        distance_img.affine,
        flair_img.affine,
        atol=1e-4,
        rtol=0.0,
    ):

        raise RuntimeError(
            f"{patient_id}: distance map affine "
            "does not match preprocessed FLAIR."
        )

    if not np.all(
        np.isfinite(
            distance
        )
    ):

        raise RuntimeError(
            f"{patient_id}: distance map contains NaN/Inf."
        )

    minimum = float(
        np.min(
            distance
        )
    )

    maximum = float(
        np.max(
            distance
        )
    )

    # Stage 15 normalized map should be [0,1].
    if minimum < -1e-5:

        raise RuntimeError(
            f"{patient_id}: normalized distance contains "
            f"negative values. min={minimum}"
        )

    if maximum > 1.00001:

        raise RuntimeError(
            f"{patient_id}: normalized distance exceeds 1. "
            f"max={maximum}"
        )

    return distance


# ============================================================
# FOLD / LEAKAGE SAFEGUARDS
# ============================================================

def validate_fold_membership(
    fold,
):

    fold_dir = (
        SPLIT_ROOT
        / f"fold_{fold}"
    )

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
        read_patient_ids(
            train_patient_csv
        )
    )

    val_ids = set(
        read_patient_ids(
            val_patient_csv
        )
    )

    test_ids = set(
        read_patient_ids(
            test_patient_csv
        )
    )

    # --------------------------------------------------------
    # SAME membership assumptions as baseline training.
    # --------------------------------------------------------

    if len(
        train_ids
    ) != 56:

        raise RuntimeError(
            f"Fold {fold}: expected 56 train patients, "
            f"got {len(train_ids)}"
        )

    if len(
        val_ids
    ) != 14:

        raise RuntimeError(
            f"Fold {fold}: expected 14 validation patients, "
            f"got {len(val_ids)}"
        )

    if len(
        test_ids
    ) != 30:

        raise RuntimeError(
            f"Expected 30 test patients, got {len(test_ids)}"
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

        patient_id_from_filename(
            path
        )

        for path in train_image_dir.glob(
            "*.npz"
        )
    }

    val_slice_patients = {

        patient_id_from_filename(
            path
        )

        for path in val_image_dir.glob(
            "*.npz"
        )
    }

    if (
        train_slice_patients
        != train_ids
    ):

        missing = sorted(
            train_ids
            - train_slice_patients
        )

        extra = sorted(
            train_slice_patients
            - train_ids
        )

        raise RuntimeError(
            f"Fold {fold}: TRAIN slice membership mismatch. "
            f"missing={missing}, extra={extra}"
        )

    if (
        val_slice_patients
        != val_ids
    ):

        missing = sorted(
            val_ids
            - val_slice_patients
        )

        extra = sorted(
            val_slice_patients
            - val_ids
        )

        raise RuntimeError(
            f"Fold {fold}: VAL slice membership mismatch. "
            f"missing={missing}, extra={extra}"
        )

    if (
        train_slice_patients
        & val_slice_patients
    ):

        raise RuntimeError(
            f"Fold {fold}: SLICE-LEVEL PATIENT LEAKAGE"
        )

    if (
        train_slice_patients
        & test_ids
    ):

        raise RuntimeError(
            f"Fold {fold}: TEST patient appears "
            "in training slices"
        )

    if (
        val_slice_patients
        & test_ids
    ):

        raise RuntimeError(
            f"Fold {fold}: TEST patient appears "
            "in validation slices"
        )

    # --------------------------------------------------------
    # Distance availability safeguard
    # --------------------------------------------------------

    all_development_ids = (
        train_ids
        | val_ids
    )

    for patient_id in sorted(
        all_development_ids
    ):

        distance_path = (
            distance_path_for_patient(
                patient_id
            )
        )

        if not distance_path.exists():

            raise FileNotFoundError(
                f"Fold {fold}: missing distance map "
                f"for patient {patient_id}:\n"
                f"{distance_path}"
            )

    print()
    print(
        f"Fold {fold} leakage/data safeguard: PASS"
    )

    print(
        f"  train patients : {len(train_ids)}"
    )

    print(
        f"  val patients   : {len(val_ids)}"
    )

    print(
        f"  test patients  : {len(test_ids)}"
    )

    print(
        f"  distance maps  : {len(all_development_ids)} available"
    )

    return (
        train_ids,
        val_ids,
        test_ids,
    )


# ============================================================
# DISTANCE-CONDITIONED DATASET
# ============================================================

class DistanceConditionedSliceDataset(
    Dataset
):

    def __init__(
        self,
        partition_root,
        expected_patient_ids,
        augment=False,
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

        # ----------------------------------------------------
        # In-memory per-patient distance cache.
        #
        # NUM_WORKERS=0, so no worker duplication problem.
        # ----------------------------------------------------

        self.distance_cache = {}

        if not self.image_dir.exists():

            raise FileNotFoundError(
                f"Missing image directory:\n"
                f"{self.image_dir}"
            )

        if not self.label_dir.exists():

            raise FileNotFoundError(
                f"Missing label directory:\n"
                f"{self.label_dir}"
            )

        image_paths = sorted(
            self.image_dir.glob(
                "*.npz"
            )
        )

        if not image_paths:

            raise RuntimeError(
                f"No images found in:\n"
                f"{self.image_dir}"
            )

        self.samples = []

        for image_path in image_paths:

            patient_id = (
                patient_id_from_filename(
                    image_path
                )
            )

            if (
                patient_id
                not in self.expected_patient_ids
            ):

                raise RuntimeError(
                    "STRUCTURAL LEAKAGE BLOCK: "
                    f"{image_path.name} belongs to "
                    f"patient {patient_id}, who is "
                    "not allowed in this Dataset."
                )

            label_path = (
                self.label_dir
                / image_path.name
            )

            if not label_path.exists():

                raise FileNotFoundError(
                    f"Missing matching label:\n"
                    f"{label_path}"
                )

            self.samples.append(
                (
                    image_path,
                    label_path,
                )
            )

        actual_patients = {

            patient_id_from_filename(
                image_path
            )

            for image_path, _
            in self.samples
        }

        if (
            actual_patients
            != self.expected_patient_ids
        ):

            missing = sorted(
                self.expected_patient_ids
                - actual_patients
            )

            extra = sorted(
                actual_patients
                - self.expected_patient_ids
            )

            raise RuntimeError(
                "Dataset patient membership does not "
                "exactly match the locked split. "
                f"missing={missing}, extra={extra}"
            )

        # ----------------------------------------------------
        # Validate every patient's distance volume once before
        # training starts.
        # ----------------------------------------------------

        print()
        print(
            "Validating distance maps for Dataset..."
        )

        for patient_id in sorted(
            self.expected_patient_ids
        ):

            distance = (
                load_and_validate_distance_volume(
                    patient_id
                )
            )

            self.distance_cache[
                patient_id
            ] = distance

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
        # GEOMETRIC AUGMENTATION
        #
        # Apply to ALL FOUR input channels together:
        #
        #   T1
        #   T2
        #   FLAIR
        #   DISTANCE
        #
        # and identically to the target label.
        # ====================================================

        if (
            random.random()
            < P_HORIZONTAL_FLIP
        ):

            image = np.flip(
                image,
                axis=2,
            ).copy()

            label = np.flip(
                label,
                axis=1,
            ).copy()

        if (
            random.random()
            < P_VERTICAL_FLIP
        ):

            image = np.flip(
                image,
                axis=1,
            ).copy()

            label = np.flip(
                label,
                axis=0,
            ).copy()

        if (
            random.random()
            < P_ROT90
        ):

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
                axes=(1, 2),
            ).copy()

            label = np.rot90(
                label,
                k=k,
                axes=(0, 1),
            ).copy()

        # ====================================================
        # INTENSITY AUGMENTATION
        #
        # CRITICAL:
        # ONLY apply to MRI channels [0:3].
        #
        # Distance channel [3] remains unchanged.
        # ====================================================

        if (
            random.random()
            < P_INTENSITY_SCALE
        ):

            scale = random.uniform(
                *INTENSITY_SCALE_RANGE
            )

            image[
                0:3
            ] = (
                image[
                    0:3
                ]
                * scale
            ).astype(
                np.float32,
                copy=False,
            )

        if (
            random.random()
            < P_INTENSITY_SHIFT
        ):

            shift = random.uniform(
                *INTENSITY_SHIFT_RANGE
            )

            image[
                0:3
            ] = (
                image[
                    0:3
                ]
                + shift
            ).astype(
                np.float32,
                copy=False,
            )

        return (
            image,
            label,
        )


    def __getitem__(
        self,
        index,
    ):

        (
            image_path,
            label_path,
        ) = self.samples[
            index
        ]

        # ====================================================
        # LOAD BASELINE 3-CHANNEL MRI
        # ====================================================

        with np.load(
            image_path,
            allow_pickle=False,
        ) as image_npz:

            image = (
                image_npz[
                    "image"
                ]
                .astype(
                    np.float32,
                    copy=True,
                )
            )

            patient_id = str(
                image_npz[
                    "patient_id"
                ]
            ).zfill(
                3
            )

            slice_index = int(
                image_npz[
                    "slice_index"
                ]
            )

            baseline_channel_order = [

                str(
                    value
                )

                for value in image_npz[
                    "channel_order"
                ].tolist()
            ]

        if (
            baseline_channel_order
            != BASE_CHANNEL_ORDER
        ):

            raise RuntimeError(
                f"Baseline channel order mismatch "
                f"in {image_path}:\n"
                f"{baseline_channel_order}"
            )

        if image.ndim != 3:

            raise RuntimeError(
                "Expected MRI image "
                f"[3,H,W], got {image.shape}"
            )

        if image.shape[
            0
        ] != 3:

            raise RuntimeError(
                "Expected three MRI channels, "
                f"got {image.shape[0]}"
            )

        # ====================================================
        # LOAD ORIGINAL 3-CLASS LABEL
        # ====================================================

        with np.load(
            label_path,
            allow_pickle=False,
        ) as label_npz:

            label = (
                label_npz[
                    "label"
                ]
                .astype(
                    np.int64,
                    copy=True,
                )
            )

        if label.ndim != 2:

            raise RuntimeError(
                "Expected target label "
                f"[H,W], got {label.shape}"
            )

        unique_labels = set(
            np.unique(
                label
            ).tolist()
        )

        if not unique_labels.issubset(
            {
                0,
                1,
                2,
            }
        ):

            raise RuntimeError(
                f"Unexpected target labels "
                f"{unique_labels} in {label_path}"
            )

        # ====================================================
        # EXTRACT CORRESPONDING DISTANCE SLICE
        # ====================================================

        distance_volume = (
            self.distance_cache[
                patient_id
            ]
        )

        if (
            slice_index < 0
            or
            slice_index
            >= distance_volume.shape[
                2
            ]
        ):

            raise RuntimeError(
                f"{patient_id}: slice index "
                f"{slice_index} outside distance "
                f"volume shape {distance_volume.shape}"
            )

        distance_slice = (
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

        if (
            distance_slice.shape
            != label.shape
        ):

            raise RuntimeError(
                f"{patient_id} slice {slice_index}: "
                "distance/target shape mismatch. "
                f"distance={distance_slice.shape}, "
                f"label={label.shape}"
            )

        if (
            image.shape[
                1:
            ]
            != label.shape
        ):

            raise RuntimeError(
                f"{patient_id} slice {slice_index}: "
                "MRI/target shape mismatch. "
                f"MRI={image.shape}, "
                f"label={label.shape}"
            )

        if not np.all(
            np.isfinite(
                distance_slice
            )
        ):

            raise RuntimeError(
                f"{patient_id} slice {slice_index}: "
                "distance contains NaN/Inf."
            )

        # ====================================================
        # CONCATENATE 4TH CHANNEL
        # ====================================================

        distance_channel = (
            distance_slice[
                None,
                :,
                :,
            ]
        )

        image = np.concatenate(
            [
                image,
                distance_channel,
            ],
            axis=0,
        )

        if image.shape[
            0
        ] != 4:

            raise RuntimeError(
                "Expected four-channel input "
                f"after distance concatenation; "
                f"got {image.shape}"
            )

        # ====================================================
        # AUGMENT
        # ====================================================

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

def compute_class_voxel_counts(
    dataset,
):

    counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64,
    )

    print()
    print(
        "Computing class counts from TRAINING FOLD ONLY..."
    )

    for (
        _,
        label_path,
    ) in dataset.samples:

        with np.load(
            label_path,
            allow_pickle=False,
        ) as data:

            label = data[
                "label"
            ]

        bincount = np.bincount(
            label.ravel(),
            minlength=NUM_CLASSES,
        )

        counts += (
            bincount[
                :NUM_CLASSES
            ]
        )

    if np.any(
        counts == 0
    ):

        raise RuntimeError(
            "At least one class has zero "
            f"training voxels: {counts.tolist()}"
        )

    return counts


def class_weights_from_counts(
    counts,
):

    counts = counts.astype(
        np.float64
    )

    frequencies = (
        counts
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

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        one_hot = (
            torch.nn.functional.one_hot(
                target,
                num_classes=NUM_CLASSES,
            )
            .permute(
                0,
                3,
                1,
                2,
            )
            .float()
        )

        first_class = (
            0
            if self.include_background
            else 1
        )

        losses = []

        for class_index in range(
            first_class,
            NUM_CLASSES,
        ):

            prediction = probabilities[
                :,
                class_index,
            ]

            target_class = one_hot[
                :,
                class_index,
            ]

            intersection = torch.sum(
                prediction
                * target_class,
                dim=(
                    1,
                    2,
                ),
            )

            denominator = (
                torch.sum(
                    prediction,
                    dim=(
                        1,
                        2,
                    ),
                )
                +
                torch.sum(
                    target_class,
                    dim=(
                        1,
                        2,
                    ),
                )
            )

            dice = (
                2.0
                * intersection
                + self.smooth
            ) / (
                denominator
                + self.smooth
            )

            losses.append(
                1.0
                - dice
            )

        return (
            torch.stack(
                losses,
                dim=0,
            )
            .mean()
        )


class CombinedLoss(
    nn.Module
):

    def __init__(
        self,
        class_weights,
    ):

        super().__init__()

        self.ce = (
            nn.CrossEntropyLoss(
                weight=class_weights
            )
        )

        self.dice = (
            MulticlassDiceLoss(
                include_background=(
                    INCLUDE_BACKGROUND_IN_DICE_LOSS
                )
            )
        )


    def forward(
        self,
        logits,
        target,
    ):

        ce_loss = self.ce(
            logits,
            target,
        )

        dice_loss = self.dice(
            logits,
            target,
        )

        total_loss = (
            CE_LOSS_WEIGHT
            * ce_loss
            +
            DICE_LOSS_WEIGHT
            * dice_loss
        )

        return (
            total_loss,
            ce_loss,
            dice_loss,
        )


# ============================================================
# FAST EPOCH DICE
# ============================================================

def dice_from_logits(
    logits,
    target,
):

    prediction = torch.argmax(
        logits,
        dim=1,
    )

    scores = []

    for class_index in range(
        NUM_CLASSES
    ):

        predicted_class = (
            prediction
            == class_index
        )

        target_class = (
            target
            == class_index
        )

        intersection = torch.sum(
            predicted_class
            & target_class
        ).float()

        denominator = (
            torch.sum(
                predicted_class
            ).float()
            +
            torch.sum(
                target_class
            ).float()
        )

        if float(
            denominator
        ) == 0.0:

            score = 1.0

        else:

            score = float(
                (
                    2.0
                    * intersection
                    / denominator
                )
                .detach()
                .cpu()
            )

        scores.append(
            score
        )

    return scores


# ============================================================
# TRAINING EPOCH
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

        if image.shape[
            1
        ] != 4:

            raise RuntimeError(
                "Training batch does not have "
                f"4 channels: {image.shape}"
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
                ce_loss,
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
                loss
                .detach()
                .cpu()
            )
        )

        ce_losses.append(
            float(
                ce_loss
                .detach()
                .cpu()
            )
        )

        dice_losses.append(
            float(
                dice_loss
                .detach()
                .cpu()
            )
        )

        scores = dice_from_logits(
            logits.detach(),
            label,
        )

        for class_index in range(
            NUM_CLASSES
        ):

            dice_values[
                class_index
            ].append(
                scores[
                    class_index
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
# VALIDATION EPOCH
# ============================================================

@torch.no_grad()
def run_val_epoch(
    model,
    loader,
    criterion,
):

    model.eval()

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

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP,
        ):

            logits = model(
                image
            )

            (
                loss,
                ce_loss,
                dice_loss,
            ) = criterion(
                logits,
                label,
            )

        total_losses.append(
            float(
                loss
                .detach()
                .cpu()
            )
        )

        ce_losses.append(
            float(
                ce_loss
                .detach()
                .cpu()
            )
        )

        dice_losses.append(
            float(
                dice_loss
                .detach()
                .cpu()
            )
        )

        scores = dice_from_logits(
            logits,
            label,
        )

        for class_index in range(
            NUM_CLASSES
        ):

            dice_values[
                class_index
            ].append(
                scores[
                    class_index
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
# PATIENT-WISE METRICS
# ============================================================

def dice_and_sensitivity(
    prediction,
    target,
    class_index,
):

    predicted_class = (
        prediction
        == class_index
    )

    target_class = (
        target
        == class_index
    )

    true_positive = np.count_nonzero(
        predicted_class
        & target_class
    )

    false_positive = np.count_nonzero(
        predicted_class
        & ~target_class
    )

    false_negative = np.count_nonzero(
        ~predicted_class
        & target_class
    )

    denominator = (
        2
        * true_positive
        +
        false_positive
        +
        false_negative
    )

    if denominator == 0:

        dice = 1.0

    else:

        dice = (
            2.0
            * true_positive
            / denominator
        )

    positive_ground_truth = (
        true_positive
        + false_negative
    )

    if (
        positive_ground_truth
        == 0
    ):

        sensitivity = math.nan

    else:

        sensitivity = (
            true_positive
            /
            positive_ground_truth
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
    prediction_mask,
    target_mask,
    spacing,
):

    prediction_mask = (
        prediction_mask.astype(
            bool
        )
    )

    target_mask = (
        target_mask.astype(
            bool
        )
    )

    prediction_exists = bool(
        prediction_mask.any()
    )

    target_exists = bool(
        target_mask.any()
    )

    if (
        not prediction_exists
        and
        not target_exists
    ):

        return 0.0

    if (
        prediction_exists
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

    prediction_surface = (
        prediction_mask
        ^
        binary_erosion(
            prediction_mask,
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
        not prediction_surface.any()
        or
        not target_surface.any()
    ):

        return math.nan

    distance_to_target = (
        distance_transform_edt(
            ~target_surface,
            sampling=spacing,
        )
    )

    distance_to_prediction = (
        distance_transform_edt(
            ~prediction_surface,
            sampling=spacing,
        )
    )

    distances = np.concatenate(
        [
            distance_to_target[
                prediction_surface
            ],
            distance_to_prediction[
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
# FINAL 3D VALIDATION FOR ONE FOLD
# ============================================================

@torch.no_grad()
def evaluate_fold_3d(
    model,
    val_loader,
    val_patient_ids,
):

    model.eval()

    prediction_volumes = {}

    target_volumes = {}

    spacing_map = {}

    for patient_id in sorted(
        val_patient_ids
    ):

        flair_path = (
            flair_path_for_patient(
                patient_id
            )
        )

        flair_img = nib.load(
            str(
                flair_path
            )
        )

        shape = (
            flair_img.shape[
                :3
            ]
        )

        spacing = tuple(
            float(
                value
            )
            for value
            in flair_img.header.get_zooms()[
                :3
            ]
        )

        prediction_volumes[
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

    for batch in val_loader:

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

        patient_ids = list(
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

        for batch_index, patient_id in enumerate(
            patient_ids
        ):

            z_index = int(
                slice_indices[
                    batch_index
                ]
            )

            prediction_volumes[
                patient_id
            ][
                :,
                :,
                z_index
            ] = prediction[
                batch_index
            ]

            target_volumes[
                patient_id
            ][
                :,
                :,
                z_index
            ] = target[
                batch_index
            ]

    per_class = {

        class_index: {

            "dice": [],

            "sensitivity": [],

            "hd95": [],
        }

        for class_index
        in range(
            NUM_CLASSES
        )
    }

    for patient_id in sorted(
        val_patient_ids
    ):

        prediction = (
            prediction_volumes[
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

        for class_index in range(
            NUM_CLASSES
        ):

            (
                dice,
                sensitivity,
            ) = dice_and_sensitivity(
                prediction,
                target,
                class_index,
            )

            hausdorff = hd95(
                prediction
                == class_index,

                target
                == class_index,

                spacing,
            )

            per_class[
                class_index
            ][
                "dice"
            ].append(
                dice
            )

            per_class[
                class_index
            ][
                "sensitivity"
            ].append(
                sensitivity
            )

            per_class[
                class_index
            ][
                "hd95"
            ].append(
                hausdorff
            )

    def safe_nanmean(
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

            return math.nan

        return float(
            finite.mean()
        )

    metrics = {}

    for (
        class_index,
        class_name,
    ) in enumerate(
        CLASS_NAMES
    ):

        metrics[
            f"dice_{class_name}"
        ] = safe_nanmean(
            per_class[
                class_index
            ][
                "dice"
            ]
        )

        metrics[
            f"sensitivity_{class_name}"
        ] = safe_nanmean(
            per_class[
                class_index
            ][
                "sensitivity"
            ]
        )

        metrics[
            f"hd95_{class_name}"
        ] = safe_nanmean(
            per_class[
                class_index
            ][
                "hd95"
            ]
        )

    return metrics


# ============================================================
# EPOCH CSV
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
    row,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    exists = path.exists()

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=EPOCH_FIELDS,
        )

        if not exists:

            writer.writeheader()

        writer.writerow(
            row
        )


# ============================================================
# PLOTS
# ============================================================

def read_epoch_csv(
    path,
):

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        return list(
            csv.DictReader(
                file
            )
        )


def plot_fold_graphs(
    fold_log_dir,
    fold,
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

        int(
            row[
                "epoch"
            ]
        )

        for row in rows
    ]

    # ========================================================
    # TOTAL LOSS
    # ========================================================

    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.plot(
        epochs,
        [
            float(
                row[
                    "train_loss"
                ]
            )
            for row in rows
        ],
        label="Train loss",
    )

    plt.plot(
        epochs,
        [
            float(
                row[
                    "val_loss"
                ]
            )
            for row in rows
        ],
        label="Validation loss",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        f"Fold {fold} — Distance-conditioned total loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        fold_log_dir
        / "loss_curve.png",
        dpi=180,
    )

    plt.close()


    # ========================================================
    # CE LOSS
    # ========================================================

    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.plot(
        epochs,
        [
            float(
                row[
                    "train_ce_loss"
                ]
            )
            for row in rows
        ],
        label="Train CE",
    )

    plt.plot(
        epochs,
        [
            float(
                row[
                    "val_ce_loss"
                ]
            )
            for row in rows
        ],
        label="Validation CE",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Cross-entropy loss"
    )

    plt.title(
        f"Fold {fold} — Distance-conditioned CE loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        fold_log_dir
        / "ce_loss_curve.png",
        dpi=180,
    )

    plt.close()


    # ========================================================
    # DICE LOSS
    # ========================================================

    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.plot(
        epochs,
        [
            float(
                row[
                    "train_dice_loss"
                ]
            )
            for row in rows
        ],
        label="Train Dice loss",
    )

    plt.plot(
        epochs,
        [
            float(
                row[
                    "val_dice_loss"
                ]
            )
            for row in rows
        ],
        label="Validation Dice loss",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Dice loss"
    )

    plt.title(
        f"Fold {fold} — Distance-conditioned Dice loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        fold_log_dir
        / "dice_loss_curve.png",
        dpi=180,
    )

    plt.close()


    # ========================================================
    # VALIDATION CLASS DICE
    # ========================================================

    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.plot(
        epochs,
        [
            float(
                row[
                    "val_dice_background"
                ]
            )
            for row in rows
        ],
        label="Background",
    )

    plt.plot(
        epochs,
        [
            float(
                row[
                    "val_dice_normal_wmh"
                ]
            )
            for row in rows
        ],
        label="Normal WMH",
    )

    plt.plot(
        epochs,
        [
            float(
                row[
                    "val_dice_abnormal_wmh"
                ]
            )
            for row in rows
        ],
        label="Abnormal WMH",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Dice"
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.title(
        f"Fold {fold} — Validation Dice"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        fold_log_dir
        / "validation_dice_curves.png",
        dpi=180,
    )

    plt.close()


    # ========================================================
    # NORMAL WMH TRAIN vs VAL
    # ========================================================

    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.plot(
        epochs,
        [
            float(
                row[
                    "train_dice_normal_wmh"
                ]
            )
            for row in rows
        ],
        label="Train normal-WMH",
    )

    plt.plot(
        epochs,
        [
            float(
                row[
                    "val_dice_normal_wmh"
                ]
            )
            for row in rows
        ],
        label="Validation normal-WMH",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Dice"
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.title(
        f"Fold {fold} — Normal-WMH Dice"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        fold_log_dir
        / "normal_wmh_dice_curve.png",
        dpi=180,
    )

    plt.close()


    # ========================================================
    # ABNORMAL WMH TRAIN vs VAL
    # ========================================================

    plt.figure(
        figsize=(
            8,
            5,
        )
    )

    plt.plot(
        epochs,
        [
            float(
                row[
                    "train_dice_abnormal_wmh"
                ]
            )
            for row in rows
        ],
        label="Train abnormal-WMH",
    )

    plt.plot(
        epochs,
        [
            float(
                row[
                    "val_dice_abnormal_wmh"
                ]
            )
            for row in rows
        ],
        label="Validation abnormal-WMH",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Dice"
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.title(
        f"Fold {fold} — Abnormal-WMH Dice"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        fold_log_dir
        / "abnormal_wmh_dice_curve.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# CROSS-FOLD GRAPHS
# ============================================================

def plot_cross_fold_summary():

    rows = []

    for fold in range(
        1,
        6,
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

        rows.append(
            metrics
        )

    if not rows:

        return

    graph_dir = (
        LOG_ROOT
        / "graphs"
    )

    graph_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = [

        f"Fold {int(row['fold'])}"

        for row in rows
    ]

    x = np.arange(
        len(
            rows
        )
    )

    width = 0.25


    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    plt.bar(
        x - width,
        [
            float(
                row[
                    "dice_background"
                ]
            )
            for row in rows
        ],
        width,
        label="Background",
    )

    plt.bar(
        x,
        [
            float(
                row[
                    "dice_normal_wmh"
                ]
            )
            for row in rows
        ],
        width,
        label="Normal WMH",
    )

    plt.bar(
        x + width,
        [
            float(
                row[
                    "dice_abnormal_wmh"
                ]
            )
            for row in rows
        ],
        width,
        label="Abnormal WMH",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel(
        "Dice"
    )

    plt.ylim(
        0,
        1,
    )

    plt.title(
        "Distance-conditioned U-Net — Dice across folds"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        graph_dir
        / "fold_dice_comparison.png",
        dpi=180,
    )

    plt.close()


    # --------------------------------------------------------
    # Sensitivity
    # --------------------------------------------------------

    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    plt.bar(
        x - width,
        [
            float(
                row[
                    "sensitivity_background"
                ]
            )
            for row in rows
        ],
        width,
        label="Background",
    )

    plt.bar(
        x,
        [
            float(
                row[
                    "sensitivity_normal_wmh"
                ]
            )
            for row in rows
        ],
        width,
        label="Normal WMH",
    )

    plt.bar(
        x + width,
        [
            float(
                row[
                    "sensitivity_abnormal_wmh"
                ]
            )
            for row in rows
        ],
        width,
        label="Abnormal WMH",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel(
        "Sensitivity"
    )

    plt.ylim(
        0,
        1,
    )

    plt.title(
        "Distance-conditioned U-Net — Sensitivity across folds"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        graph_dir
        / "fold_sensitivity_comparison.png",
        dpi=180,
    )

    plt.close()


    # --------------------------------------------------------
    # HD95
    # --------------------------------------------------------

    plt.figure(
        figsize=(
            9,
            5,
        )
    )

    plt.bar(
        x - width,
        [
            float(
                row[
                    "hd95_background"
                ]
            )
            for row in rows
        ],
        width,
        label="Background",
    )

    plt.bar(
        x,
        [
            float(
                row[
                    "hd95_normal_wmh"
                ]
            )
            for row in rows
        ],
        width,
        label="Normal WMH",
    )

    plt.bar(
        x + width,
        [
            float(
                row[
                    "hd95_abnormal_wmh"
                ]
            )
            for row in rows
        ],
        width,
        label="Abnormal WMH",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel(
        "HD95 (mm)"
    )

    plt.title(
        "Distance-conditioned U-Net — HD95 across folds"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        graph_dir
        / "fold_hd95_comparison.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    fold,
    epoch,
    model,
    optimizer,
    best_metric,
    class_counts,
    class_weights,
):

    payload = {

        "experiment":
            "distance_conditioned_unet",

        "fold":
            fold,

        "epoch":
            epoch,

        "best_metric":
            float(
                best_metric
            ),

        "model_state":
            model.state_dict(),

        "optimizer_state":
            optimizer.state_dict(),

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

        "distance_channel":
            (
                "Stage-15 normalized "
                "distance-to-ventricle"
            ),

        "distance_normalization":
            (
                "physical EDT clipped at 50 mm "
                "then divided by 50"
            ),
    }

    atomic_torch_save(
        path,
        payload,
    )


# ============================================================
# FOLD TRAINING
# ============================================================

def train_fold(
    fold,
):

    print()
    print(
        "=" * 88
    )

    print(
        f"DISTANCE-CONDITIONED FOLD {fold}"
    )

    print(
        "=" * 88
    )

    fold_dir = (
        SPLIT_ROOT
        / f"fold_{fold}"
    )

    fold_log_dir = (
        LOG_ROOT
        / f"fold_{fold}"
    )

    fold_checkpoint_dir = (
        CHECKPOINT_ROOT
        / f"fold_{fold}"
    )


    # ========================================================
    # FRESH RUN — mirrors baseline experiment
    # ========================================================

    if fold_log_dir.exists():

        shutil.rmtree(
            fold_log_dir
        )

    if fold_checkpoint_dir.exists():

        shutil.rmtree(
            fold_checkpoint_dir
        )

    fold_log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"Fold {fold}: fresh distance-conditioned run."
    )


    # ========================================================
    # MEMBERSHIP VALIDATION
    # ========================================================

    (
        train_ids,
        val_ids,
        test_ids,
    ) = validate_fold_membership(
        fold
    )


    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = (
        DistanceConditionedSliceDataset(

            partition_root=(
                fold_dir
                / "train"
            ),

            expected_patient_ids=(
                train_ids
            ),

            augment=True,
        )
    )

    val_dataset = (
        DistanceConditionedSliceDataset(

            partition_root=(
                fold_dir
                / "val"
            ),

            expected_patient_ids=(
                val_ids
            ),

            augment=False,
        )
    )

    print()
    print(
        f"Fold {fold} slices:"
    )

    print(
        "  train:",
        len(
            train_dataset
        )
    )

    print(
        "  val  :",
        len(
            val_dataset
        )
    )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

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

    for (
        class_index,
        class_name,
    ) in enumerate(
        CLASS_NAMES
    ):

        print(
            f"  {class_name:<14} "
            f"voxels="
            f"{int(class_counts[class_index]):>14,d} "
            f"weight="
            f"{float(class_weights[class_index]):.6f}"
        )

    class_info = {

        "experiment":
            "distance_conditioned_unet",

        "class_counts": {

            CLASS_NAMES[
                index
            ]:
                int(
                    class_counts[
                        index
                    ]
                )

            for index
            in range(
                NUM_CLASSES
            )
        },

        "class_weights": {

            CLASS_NAMES[
                index
            ]:
                float(
                    class_weights[
                        index
                    ]
                )

            for index
            in range(
                NUM_CLASSES
            )
        },

        "class_weight_power":
            CLASS_WEIGHT_POWER,

        "loss":
            "weighted_cross_entropy + dice",

        "input_channels":
            CHANNEL_ORDER,
    }

    atomic_write_text(
        fold_log_dir
        / "class_weights.json",

        json.dumps(
            class_info,
            indent=2,
        ),
    )


    # ========================================================
    # LOADERS
    # ========================================================

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type
            == "cuda"
        ),

        drop_last=False,
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=(
            DEVICE.type
            == "cuda"
        ),

        drop_last=False,
    )


    # ========================================================
    # MODEL
    #
    # ONLY architecture-level change:
    #
    #     in_channels = 4
    #
    # Everything else remains identical.
    # ========================================================

    model = UNet2D(

        in_channels=4,

        num_classes=NUM_CLASSES,

        base_channels=BASE_CHANNELS,

    ).to(
        DEVICE
    )


    # ========================================================
    # OPTIMIZER / LOSS
    # ========================================================

    optimizer = (
        torch.optim.AdamW(

            model.parameters(),

            lr=LEARNING_RATE,

            weight_decay=WEIGHT_DECAY,
        )
    )

    class_weight_tensor = (
        torch.tensor(

            class_weights,

            dtype=torch.float32,

            device=DEVICE,
        )
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
    # TRAINING
    # ========================================================

    best_metric = (
        -math.inf
    )

    best_epoch = None

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_metrics = (
            run_train_epoch(

                model,

                train_loader,

                optimizer,

                criterion,

                scaler,
            )
        )

        val_metrics = (
            run_val_epoch(

                model,

                val_loader,

                criterion,
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

            best_metric = (
                current_metric
            )

            best_epoch = epoch

        print()
        print(
            f"Fold {fold} | "
            f"Epoch {epoch:03d}/{EPOCHS}"
        )

        print(
            "  train total/CE/Dice loss: "
            f"{train_metrics['loss']:.5f} / "
            f"{train_metrics['ce_loss']:.5f} / "
            f"{train_metrics['dice_loss']:.5f}"
        )

        print(
            "  val   total/CE/Dice loss: "
            f"{val_metrics['loss']:.5f} / "
            f"{val_metrics['ce_loss']:.5f} / "
            f"{val_metrics['dice_loss']:.5f}"
        )

        print(
            "  train Dice BG/nWMH/abWMH: "
            f"{train_metrics['dice_background']:.4f}/"
            f"{train_metrics['dice_normal_wmh']:.4f}/"
            f"{train_metrics['dice_abnormal_wmh']:.4f}"
        )

        print(
            "  val   Dice BG/nWMH/abWMH: "
            f"{val_metrics['dice_background']:.4f}/"
            f"{val_metrics['dice_normal_wmh']:.4f}/"
            f"{val_metrics['dice_abnormal_wmh']:.4f}"
        )

        print(
            "  best abWMH val Dice: "
            f"{best_metric:.4f}"
        )


        # ====================================================
        # EPOCH CSV
        # ====================================================

        append_epoch_log(

            fold_log_dir
            / "epochs.csv",

            {

                "epoch":
                    epoch,

                "train_loss":
                    train_metrics[
                        "loss"
                    ],

                "train_ce_loss":
                    train_metrics[
                        "ce_loss"
                    ],

                "train_dice_loss":
                    train_metrics[
                        "dice_loss"
                    ],

                "val_loss":
                    val_metrics[
                        "loss"
                    ],

                "val_ce_loss":
                    val_metrics[
                        "ce_loss"
                    ],

                "val_dice_loss":
                    val_metrics[
                        "dice_loss"
                    ],

                "train_dice_background":
                    train_metrics[
                        "dice_background"
                    ],

                "train_dice_normal_wmh":
                    train_metrics[
                        "dice_normal_wmh"
                    ],

                "train_dice_abnormal_wmh":
                    train_metrics[
                        "dice_abnormal_wmh"
                    ],

                "val_dice_background":
                    val_metrics[
                        "dice_background"
                    ],

                "val_dice_normal_wmh":
                    val_metrics[
                        "dice_normal_wmh"
                    ],

                "val_dice_abnormal_wmh":
                    val_metrics[
                        "dice_abnormal_wmh"
                    ],

                "best_val_dice_abnormal_wmh":
                    best_metric,

                "learning_rate":
                    optimizer.param_groups[
                        0
                    ][
                        "lr"
                    ],
            },
        )


        # ====================================================
        # SAVE EVERY EPOCH
        # ====================================================

        if (
            epoch
            % CHECKPOINT_EVERY
            == 0
            or
            epoch
            == EPOCHS
        ):

            save_checkpoint(

                fold_checkpoint_dir
                / (
                    f"epoch_{epoch:04d}.pth"
                ),

                fold,

                epoch,

                model,

                optimizer,

                best_metric,

                class_counts,

                class_weights,
            )


        # ====================================================
        # BEST CHECKPOINT
        # ====================================================

        if improved:

            save_checkpoint(

                fold_checkpoint_dir
                / "best.pth",

                fold,

                epoch,

                model,

                optimizer,

                best_metric,

                class_counts,

                class_weights,
            )


    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    best_path = (
        fold_checkpoint_dir
        / "best.pth"
    )

    if not best_path.exists():

        raise RuntimeError(
            f"Fold {fold}: best.pth was not created."
        )

    best_checkpoint = torch.load(

        best_path,

        map_location=DEVICE,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state"
        ]
    )


    # ========================================================
    # FINAL PATIENT-WISE 3D VALIDATION
    # ========================================================

    print()
    print(
        f"Fold {fold}: running final "
        "patient-wise 3D validation..."
    )

    final_metrics = (
        evaluate_fold_3d(

            model,

            val_loader,

            val_ids,
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

    final_metrics[
        "experiment"
    ] = (
        "distance_conditioned_unet"
    )

    final_metrics[
        "input_channels"
    ] = CHANNEL_ORDER


    # ========================================================
    # SAVE FINAL METRICS
    # ========================================================

    final_metrics_path = (
        fold_log_dir
        / "final_metrics.json"
    )

    atomic_write_text(

        final_metrics_path,

        json.dumps(
            final_metrics,
            indent=2,
        ),
    )


    # ========================================================
    # DONE MARKER
    # ========================================================

    atomic_write_text(

        fold_log_dir
        / "fold_done.json",

        json.dumps(

            {

                "fold":
                    fold,

                "status":
                    "done",

                "experiment":
                    "distance_conditioned_unet",

                "best_epoch":
                    int(
                        best_checkpoint[
                            "epoch"
                        ]
                    ),
            },

            indent=2,
        ),
    )


    # ========================================================
    # TERMINAL METRICS
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        f"FOLD {fold} FINAL VALIDATION METRICS"
    )

    print(
        "=" * 72
    )

    print(
        "Best epoch:",
        final_metrics[
            "best_epoch"
        ]
    )

    for (
        key,
        value,
    ) in final_metrics.items():

        if key in {
            "fold",
            "best_epoch",
            "experiment",
            "input_channels",
        }:

            continue

        if (
            isinstance(
                value,
                float,
            )
            and
            np.isfinite(
                value
            )
        ):

            print(
                f"  {key:<34}: "
                f"{value:.6f}"
            )

        else:

            print(
                f"  {key:<34}: NaN"
            )


    # ========================================================
    # GRAPHS
    # ========================================================

    print()
    print(
        f"Fold {fold}: generating graphs..."
    )

    plot_fold_graphs(
        fold_log_dir,
        fold,
    )

    print()
    print(
        f"FOLD {fold}: COMPLETE"
    )


# ============================================================
# CROSS-FOLD SUMMARY
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


def rebuild_fold_summary():

    rows = []

    for fold in range(
        1,
        6,
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

            "fold":
                str(
                    fold
                ),

            "best_epoch":
                metrics.get(
                    "best_epoch",
                    "",
                ),
        }

        for metric in METRIC_COLUMNS:

            row[
                metric
            ] = metrics.get(
                metric,
                math.nan,
            )

        rows.append(
            row
        )

    if not rows:

        return

    mean_row = {

        "fold":
            "mean",

        "best_epoch":
            "",
    }

    std_row = {

        "fold":
            "std",

        "best_epoch":
            "",
    }

    for metric in METRIC_COLUMNS:

        values = np.asarray(
            [

                float(
                    row[
                        metric
                    ]
                )

                for row in rows
            ],
            dtype=float,
        )

        finite = values[
            np.isfinite(
                values
            )
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
                np.mean(
                    finite
                )
            )

            if finite.size > 1:

                std_row[
                    metric
                ] = float(
                    np.std(
                        finite,
                        ddof=1,
                    )
                )

            else:

                std_row[
                    metric
                ] = 0.0

    rows.extend(
        [
            mean_row,
            std_row,
        ]
    )

    SUMMARY_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SUMMARY_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=(
                [
                    "fold",
                    "best_epoch",
                ]
                +
                METRIC_COLUMNS
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# FAILURE LOG
# ============================================================

def append_failure(
    fold,
    exception,
):

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    previous_text = ""

    if FAILURE_LOG.exists():

        previous_text = (
            FAILURE_LOG.read_text(
                encoding="utf-8"
            )
        )

    new_text = (

        previous_text

        + "\n"

        + "=" * 88

        + "\n"

        + f"FOLD {fold} FAILED"

        + "\n"

        + "=" * 88

        + "\n"

        + str(
            exception
        )

        + "\n\n"

        + traceback.format_exc()

        + "\n"
    )

    atomic_write_text(
        FAILURE_LOG,
        new_text,
    )


# ============================================================
# PRINT FINAL CROSS-FOLD SUMMARY
# ============================================================

def print_cross_fold_summary():

    if not SUMMARY_CSV.exists():

        return

    with SUMMARY_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        rows = list(
            csv.DictReader(
                file
            )
        )

    mean_row = next(
        (
            row
            for row in rows
            if row[
                "fold"
            ] == "mean"
        ),
        None,
    )

    std_row = next(
        (
            row
            for row in rows
            if row[
                "fold"
            ] == "std"
        ),
        None,
    )

    if (
        mean_row is None
        or std_row is None
    ):

        return

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

            mean_value = float(
                mean_row[
                    key
                ]
            )

            std_value = float(
                std_row[
                    key
                ]
            )

            unit = (
                " mm"
                if metric == "hd95"
                else ""
            )

            print(
                f"  {metric:<12}: "
                f"{mean_value:.6f} "
                f"± {std_value:.6f}"
                f"{unit}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(
        RANDOM_SEED
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 88
    )

    print(
        "STEP 16 — DISTANCE-CONDITIONED PLAIN 2D U-NET"
    )

    print(
        "=" * 88
    )

    print()
    print(
        "Experiment:"
    )

    print(
        "  Baseline architecture + "
        "normalized distance-to-ventricle channel"
    )

    print()
    print(
        "Input channels:"
    )

    print(
        "  0 = T1"
    )

    print(
        "  1 = T2"
    )

    print(
        "  2 = FLAIR"
    )

    print(
        "  3 = normalized distance-to-ventricle"
    )

    print()
    print(
        "Target classes:"
    )

    print(
        "  0 = background"
    )

    print(
        "  1 = normal WMH"
    )

    print(
        "  2 = abnormal WMH"
    )

    print()
    print(
        "Distance map root:"
    )

    print(
        DISTANCE_ROOT
    )

    print()
    print(
        "Split root:"
    )

    print(
        SPLIT_ROOT
    )

    print()
    print(
        "Hyperparameters:"
    )

    print(
        f"  Epochs       : {EPOCHS}"
    )

    print(
        f"  Batch size   : {BATCH_SIZE}"
    )

    print(
        f"  Learning rate: {LEARNING_RATE}"
    )

    print(
        f"  Weight decay : {WEIGHT_DECAY}"
    )

    print(
        f"  Base channels: {BASE_CHANNELS}"
    )

    print()
    print(
        "Loss:"
    )

    print(
        "  weighted cross entropy + Dice"
    )

    print()
    print(
        "IMPORTANT AUGMENTATION RULE:"
    )

    print(
        "  geometric transforms -> MRI + distance + labels"
    )

    print(
        "  intensity transforms -> MRI channels ONLY"
    )

    print()
    print(
        "Final test set:"
    )

    print(
        "  SEALED — used only for leakage checks"
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

        print(
            "CUDA capability:",
            torch.cuda.get_device_capability(
                0
            )
        )

    folds = (

        [
            ONLY_FOLD
        ]

        if ONLY_FOLD
        is not None

        else
        [
            1,
            2,
            3,
            4,
            5,
        ]
    )

    print()
    print(
        "Folds requested:",
        folds
    )


    # ========================================================
    # RUN FOLDS
    # ========================================================

    for fold in folds:

        try:

            train_fold(
                fold
            )

        except Exception as exception:

            print()
            print(
                "!" * 88
            )

            print(
                f"FOLD {fold} FAILED"
            )

            print(
                "!" * 88
            )

            print(
                str(
                    exception
                )
            )

            traceback.print_exc()

            append_failure(
                fold,
                exception,
            )

            continue

        rebuild_fold_summary()


    # ========================================================
    # FINAL CROSS-FOLD OUTPUTS
    # ========================================================

    rebuild_fold_summary()

    plot_cross_fold_summary()

    print_cross_fold_summary()

    print()
    print(
        "=" * 88
    )

    print(
        "STEP 16 FINISHED"
    )

    print(
        "=" * 88
    )

    print()
    print(
        "Logs:"
    )

    print(
        LOG_ROOT
    )

    print()
    print(
        "Checkpoints:"
    )

    print(
        CHECKPOINT_ROOT
    )

    print()
    print(
        "CV summary:"
    )

    print(
        SUMMARY_CSV
    )

    print()
    print(
        "Terminal log:"
    )

    print(
        TERMINAL_LOG
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The 30-patient final test set has NOT "
        "been evaluated in Step 16."
    )

    print(
        "After all 5 folds finish, use the CV best epochs "
        "to train the final 70-patient distance-conditioned "
        "model before one-time test evaluation."
    )


# ============================================================
# RUN + SAVE TERMINAL OUTPUT
# ============================================================

if __name__ == "__main__":

    LOG_ROOT.mkdir(
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
            "NEW STEP-16 RUN\n"
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
                    "FATAL STEP-16 ERROR"
                )

                print(
                    "=" * 88
                )

                traceback.print_exc()

                raise