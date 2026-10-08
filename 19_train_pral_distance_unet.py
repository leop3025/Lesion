from __future__ import annotations

from pathlib import Path

import argparse
import contextlib
import csv
import hashlib
import importlib.util
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

from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
)

from torch.utils.data import (
    Dataset,
    DataLoader,
)


# ============================================================
# PROJECT PATHS
# ============================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parent

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

UNET_FILE = (
    PROJECT_DIR
    / "unet2d.py"
)

PRAL_FILE = (
    PROJECT_DIR
    / "18_pral_loss.py"
)


# ============================================================
# IMPORT UNET
# ============================================================

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_DIR),
    )

from unet2d import UNet2D


# ============================================================
# IMPORT PRAL LOSS FROM 18_pral_loss.py
#
# Python module names cannot normally begin with a number,
# therefore importlib is used.
# ============================================================

if not PRAL_FILE.exists():
    raise FileNotFoundError(
        f"Missing PRAL module:\n{PRAL_FILE}"
    )

pral_spec = importlib.util.spec_from_file_location(
    "pral_loss_module",
    PRAL_FILE,
)

if (
    pral_spec is None
    or
    pral_spec.loader is None
):
    raise RuntimeError(
        "Could not load 18_pral_loss.py"
    )

pral_module = importlib.util.module_from_spec(
    pral_spec
)

pral_spec.loader.exec_module(
    pral_module
)

PRALLoss = pral_module.PRALLoss

PRAL_SOURCE_SHA256 = hashlib.sha256(
    PRAL_FILE.read_bytes()
).hexdigest()


# ============================================================
# CONFIGURATION
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

EPOCHS = 50
BATCH_SIZE = 8
NUM_WORKERS = 0

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

RANDOM_SEED = 20260911

CE_LOSS_WEIGHT = 1.0
DICE_LOSS_WEIGHT = 1.0

CLASS_WEIGHT_POWER = 0.5

INCLUDE_BACKGROUND_IN_DICE_LOSS = False

CHECKPOINT_EVERY = 1


# ============================================================
# AUGMENTATION
#
# IDENTICAL TO BASELINE / STEP 16
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
# METRIC COLUMN NAMES
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
# SIMPLE TEE LOGGER
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

    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
    )

    os.close(
        fd
    )

    temp_path = Path(
        temp_name
    )

    try:

        temp_path.write_text(
            text,
            encoding="utf-8",
        )

        os.replace(
            temp_path,
            path,
        )

    finally:

        if temp_path.exists():

            temp_path.unlink(
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

    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
    )

    os.close(
        fd
    )

    temp_path = Path(
        temp_name
    )

    try:

        torch.save(
            payload,
            temp_path,
        )

        os.replace(
            temp_path,
            path,
        )

    finally:

        if temp_path.exists():

            temp_path.unlink(
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
            f"Missing split CSV:\n{path}"
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
            f"Empty split CSV:\n{path}"
        )

    patient_ids = [
        str(
            row["patient_id"]
        )
        .strip()
        .zfill(3)

        for row in rows
    ]

    if (
        len(patient_ids)
        !=
        len(set(patient_ids))
    ):

        raise RuntimeError(
            f"Duplicate patient IDs:\n{path}"
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
            f"Unexpected slice filename: "
            f"{path.name}"
        )

    return match.group(
        1
    )


# ============================================================
# DISTANCE / ZONE PATHS
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


def zone_path_for_patient(
    patient_id,
):

    return (
        DISTANCE_ROOT
        / patient_id
        / (
            f"{patient_id}"
            "_pv_deep_zone_map.nii.gz"
        )
    )


# ============================================================
# FIND PREPROCESSED FLAIR
# ============================================================

def find_flair_for_patient(
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
            "flair" in path.name.lower()
            and
            "mask" not in path.name.lower()
        )
    ]

    if len(candidates) != 1:

        raise RuntimeError(
            f"{patient_id}: expected exactly "
            f"one FLAIR; found "
            f"{len(candidates)}"
        )

    return candidates[0]


# ============================================================
# LOAD + VALIDATE DISTANCE AND ZONE VOLUMES
# ============================================================

def load_spatial_volumes(
    patient_id,
):

    distance_path = (
        distance_path_for_patient(
            patient_id
        )
    )

    zone_path = (
        zone_path_for_patient(
            patient_id
        )
    )

    if not distance_path.exists():

        raise FileNotFoundError(
            f"Missing distance map:\n"
            f"{distance_path}"
        )

    if not zone_path.exists():

        raise FileNotFoundError(
            f"Missing PV/deep zone map:\n"
            f"{zone_path}"
        )

    distance_img = nib.load(
        str(
            distance_path
        )
    )

    zone_img = nib.load(
        str(
            zone_path
        )
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

    zone = np.asarray(
        zone_img.dataobj,
    ).astype(
        np.int64,
        copy=False,
    )

    expected_shape = (
        flair_img.shape[:3]
    )

    if distance.shape != expected_shape:

        raise RuntimeError(
            f"{patient_id}: distance/FLAIR "
            f"shape mismatch.\n"
            f"distance={distance.shape}\n"
            f"FLAIR={expected_shape}"
        )

    if zone.shape != expected_shape:

        raise RuntimeError(
            f"{patient_id}: zone/FLAIR "
            f"shape mismatch.\n"
            f"zone={zone.shape}\n"
            f"FLAIR={expected_shape}"
        )

    if not np.allclose(
        distance_img.affine,
        flair_img.affine,
        atol=1e-4,
        rtol=0,
    ):

        raise RuntimeError(
            f"{patient_id}: distance affine mismatch."
        )

    if not np.allclose(
        zone_img.affine,
        flair_img.affine,
        atol=1e-4,
        rtol=0,
    ):

        raise RuntimeError(
            f"{patient_id}: zone affine mismatch."
        )

    if not np.all(
        np.isfinite(
            distance
        )
    ):

        raise RuntimeError(
            f"{patient_id}: distance map "
            f"contains NaN/Inf."
        )

    if float(
        np.min(
            distance
        )
    ) < -1e-5:

        raise RuntimeError(
            f"{patient_id}: normalized "
            f"distance < 0."
        )

    if float(
        np.max(
            distance
        )
    ) > 1.00001:

        raise RuntimeError(
            f"{patient_id}: normalized "
            f"distance > 1."
        )

    unique_zones = set(
        np.unique(
            zone
        ).tolist()
    )

    if not unique_zones.issubset(
        {
            0,
            1,
            2,
        }
    ):

        raise RuntimeError(
            f"{patient_id}: unexpected "
            f"zone labels: "
            f"{sorted(unique_zones)}"
        )

    return (
        distance,
        zone,
    )


# ============================================================
# DATASET
# ============================================================

class PRALDistanceDataset(
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

        self.expected_patient_ids = set(
            expected_patient_ids
        )

        self.augment = augment

        image_dir = (
            self.partition_root
            / "images"
        )

        label_dir = (
            self.partition_root
            / "labels"
        )

        if not image_dir.exists():

            raise FileNotFoundError(
                f"Missing image directory:\n"
                f"{image_dir}"
            )

        if not label_dir.exists():

            raise FileNotFoundError(
                f"Missing label directory:\n"
                f"{label_dir}"
            )

        self.samples = []

        for image_path in sorted(
            image_dir.glob(
                "*.npz"
            )
        ):

            patient_id = (
                patient_id_from_filename(
                    image_path
                )
            )

            if (
                patient_id
                not in
                self.expected_patient_ids
            ):

                raise RuntimeError(
                    f"LEAKAGE BLOCK: patient "
                    f"{patient_id} not allowed "
                    f"in {self.partition_root}"
                )

            label_path = (
                label_dir
                / image_path.name
            )

            if not label_path.exists():

                raise FileNotFoundError(
                    f"Missing label:\n"
                    f"{label_path}"
                )

            self.samples.append(
                (
                    image_path,
                    label_path,
                )
            )

        actual_ids = {
            patient_id_from_filename(
                image_path
            )
            for image_path, _
            in self.samples
        }

        if (
            actual_ids
            !=
            self.expected_patient_ids
        ):

            missing = sorted(
                self.expected_patient_ids
                -
                actual_ids
            )

            extra = sorted(
                actual_ids
                -
                self.expected_patient_ids
            )

            raise RuntimeError(
                "Patient membership mismatch.\n"
                f"Missing: {missing}\n"
                f"Extra: {extra}"
            )

        # ====================================================
        # CACHE DISTANCE + ZONE VOLUMES
        # ====================================================

        self.distance_cache = {}
        self.zone_cache = {}

        print()
        print(
            f"Validating spatial maps for "
            f"{self.partition_root.name}..."
        )

        for patient_id in sorted(
            self.expected_patient_ids
        ):

            (
                distance,
                zone,
            ) = load_spatial_volumes(
                patient_id
            )

            self.distance_cache[
                patient_id
            ] = distance

            self.zone_cache[
                patient_id
            ] = zone

        print(
            "Spatial maps validated:",
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
        zone,
    ):

        # ====================================================
        # GEOMETRIC TRANSFORMS:
        # MRI + DISTANCE + LABEL + ZONE TOGETHER
        # ====================================================

        if (
            random.random()
            <
            P_HORIZONTAL_FLIP
        ):

            image = np.flip(
                image,
                axis=2,
            ).copy()

            label = np.flip(
                label,
                axis=1,
            ).copy()

            zone = np.flip(
                zone,
                axis=1,
            ).copy()

        if (
            random.random()
            <
            P_VERTICAL_FLIP
        ):

            image = np.flip(
                image,
                axis=1,
            ).copy()

            label = np.flip(
                label,
                axis=0,
            ).copy()

            zone = np.flip(
                zone,
                axis=0,
            ).copy()

        if (
            random.random()
            <
            P_ROT90
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

            zone = np.rot90(
                zone,
                k=k,
                axes=(
                    0,
                    1,
                ),
            ).copy()

        # ====================================================
        # INTENSITY AUGMENTATION:
        # ONLY T1/T2/FLAIR
        #
        # channel 3 = distance, untouched
        # ====================================================

        if (
            random.random()
            <
            P_INTENSITY_SCALE
        ):

            scale = random.uniform(
                *INTENSITY_SCALE_RANGE
            )

            image[
                :3
            ] *= scale

        if (
            random.random()
            <
            P_INTENSITY_SHIFT
        ):

            shift = random.uniform(
                *INTENSITY_SHIFT_RANGE
            )

            image[
                :3
            ] += shift

        return (
            image,
            label,
            zone,
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

            channel_order = [
                str(x)
                for x in data[
                    "channel_order"
                ].tolist()
            ]

        if (
            channel_order
            !=
            BASE_CHANNEL_ORDER
        ):

            raise RuntimeError(
                f"{patient_id}: unexpected "
                f"channel order: "
                f"{channel_order}"
            )

        if image.shape[0] != 3:

            raise RuntimeError(
                f"{patient_id}: expected "
                f"3 MRI channels; got "
                f"{image.shape}"
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
                f"{patient_id}: invalid "
                f"segmentation labels "
                f"{sorted(unique_labels)}"
            )

        distance_volume = (
            self.distance_cache[
                patient_id
            ]
        )

        zone_volume = (
            self.zone_cache[
                patient_id
            ]
        )

        if (
            slice_index < 0
            or
            slice_index
            >=
            distance_volume.shape[2]
        ):

            raise RuntimeError(
                f"{patient_id}: invalid "
                f"slice index "
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

        zone = (
            zone_volume[
                :,
                :,
                slice_index
            ]
            .astype(
                np.int64,
                copy=True,
            )
        )

        if (
            distance.shape
            !=
            label.shape
        ):

            raise RuntimeError(
                f"{patient_id}: distance/label "
                f"slice shape mismatch."
            )

        if (
            zone.shape
            !=
            label.shape
        ):

            raise RuntimeError(
                f"{patient_id}: zone/label "
                f"slice shape mismatch."
            )

        # ====================================================
        # APPEND NORMALIZED DISTANCE AS CHANNEL 4
        # ====================================================

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

        if image.shape[0] != 4:

            raise RuntimeError(
                f"{patient_id}: expected "
                f"4 input channels."
            )

        if (
            self.augment
            and
            USE_AUGMENTATION
        ):

            (
                image,
                label,
                zone,
            ) = self._augment(
                image,
                label,
                zone,
            )

        return {

            "image":
                torch.from_numpy(
                    image
                ).float(),

            "label":
                torch.from_numpy(
                    label
                ).long(),

            "zone":
                torch.from_numpy(
                    zone
                ).long(),

            "patient_id":
                patient_id,

            "slice_index":
                slice_index,
        }


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def compute_class_counts(
    dataset,
):

    counts = np.zeros(
        NUM_CLASSES,
        dtype=np.int64,
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
            f"Zero class voxels: "
            f"{counts.tolist()}"
        )

    return counts


def class_weights_from_counts(
    counts,
):

    frequencies = (
        counts.astype(
            np.float64
        )
        /
        counts.sum()
    )

    weights = (
        1.0
        /
        np.power(
            frequencies,
            CLASS_WEIGHT_POWER,
        )
    )

    weights /= (
        weights.mean()
    )

    return weights.astype(
        np.float32
    )


# ============================================================
# BATCH DICE
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

        pred_mask = (
            prediction
            ==
            class_index
        )

        target_mask = (
            target
            ==
            class_index
        )

        intersection = torch.sum(
            pred_mask
            &
            target_mask
        ).float()

        denominator = (
            torch.sum(
                pred_mask
            ).float()
            +
            torch.sum(
                target_mask
            ).float()
        )

        if float(
            denominator
        ) == 0:

            score = 1.0

        else:

            score = float(
                (
                    2.0
                    *
                    intersection
                    /
                    denominator
                )
                .detach()
                .cpu()
            )

        scores.append(
            score
        )

    return scores


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def run_train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
):

    model.train()

    losses = []
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

        zone = (
            batch[
                "zone"
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
                total_loss,
                ce_loss,
                dice_loss,
            ) = criterion(
                logits,
                label,
                zone,
            )

        if scaler is not None:

            scaler.scale(
                total_loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            total_loss.backward()

            optimizer.step()

        losses.append(
            float(
                total_loss
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

        batch_dice = dice_from_logits(
            logits.detach(),
            label,
        )

        for class_index in range(
            NUM_CLASSES
        ):

            dice_values[
                class_index
            ].append(
                batch_dice[
                    class_index
                ]
            )

    return {

        "loss":
            float(
                np.mean(
                    losses
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
# VALIDATION ONE EPOCH
# ============================================================

@torch.no_grad()
def run_validation_epoch(
    model,
    loader,
    criterion,
):

    model.eval()

    losses = []
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

        zone = (
            batch[
                "zone"
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
                total_loss,
                ce_loss,
                dice_loss,
            ) = criterion(
                logits,
                label,
                zone,
            )

        losses.append(
            float(
                total_loss
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

        batch_dice = dice_from_logits(
            logits,
            label,
        )

        for class_index in range(
            NUM_CLASSES
        ):

            dice_values[
                class_index
            ].append(
                batch_dice[
                    class_index
                ]
            )

    return {

        "loss":
            float(
                np.mean(
                    losses
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
# HISTORY CSV
# ============================================================

HISTORY_FIELDS = [

    "epoch",

    "train_loss",
    "train_ce_loss",
    "train_dice_loss",

    "train_dice_background",
    "train_dice_normal_wmh",
    "train_dice_abnormal_wmh",

    "val_loss",
    "val_ce_loss",
    "val_dice_loss",

    "val_dice_background",
    "val_dice_normal_wmh",
    "val_dice_abnormal_wmh",

    "best_val_abnormal_wmh_dice",

    "learning_rate",
]


def write_history_header(
    path,
):

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=HISTORY_FIELDS,
        )

        writer.writeheader()


def append_history(
    path,
    epoch,
    train_metrics,
    val_metrics,
    best_val_dice,
    optimizer,
):

    row = {

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

        "best_val_abnormal_wmh_dice":
            best_val_dice,

        "learning_rate":
            optimizer.param_groups[
                0
            ][
                "lr"
            ],
    }

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=HISTORY_FIELDS,
        )

        writer.writerow(
            row
        )


# ============================================================
# 3D METRIC FUNCTIONS
# ============================================================

def dice_and_sensitivity(
    prediction,
    target,
    class_index,
):

    pred_mask = (
        prediction
        ==
        class_index
    )

    target_mask = (
        target
        ==
        class_index
    )

    true_positive = np.count_nonzero(
        pred_mask
        &
        target_mask
    )

    false_positive = np.count_nonzero(
        pred_mask
        &
        ~target_mask
    )

    false_negative = np.count_nonzero(
        ~pred_mask
        &
        target_mask
    )

    dice_denominator = (
        2
        *
        true_positive
        +
        false_positive
        +
        false_negative
    )

    if dice_denominator == 0:

        dice = 1.0

    else:

        dice = (
            2
            *
            true_positive
            /
            dice_denominator
        )

    target_positive = (
        true_positive
        +
        false_negative
    )

    if target_positive == 0:

        sensitivity = math.nan

    else:

        sensitivity = (
            true_positive
            /
            target_positive
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
        !=
        target_exists
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
# PATIENT-WISE 3D VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_3d(
    model,
    loader,
    patient_ids,
):

    model.eval()

    prediction_volumes = {}
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
            flair_img.shape[:3]
        )

        spacing = tuple(
            float(value)
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

        for index, patient_id in enumerate(
            patient_batch
        ):

            slice_index = int(
                slice_indices[
                    index
                ]
            )

            prediction_volumes[
                patient_id
            ][
                :,
                :,
                slice_index
            ] = prediction[
                index
            ]

            target_volumes[
                patient_id
            ][
                :,
                :,
                slice_index
            ] = target[
                index
            ]

    rows = []

    for patient_id in sorted(
        patient_ids
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

        row = {
            "patient_id":
                patient_id
        }

        for (
            class_index,
            class_name,
        ) in enumerate(
            CLASS_NAMES
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
                ==
                class_index,
                target
                ==
                class_index,
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

    values = np.asarray(
        values,
        dtype=float,
    )

    finite_values = values[
        np.isfinite(
            values
        )
    ]

    if finite_values.size == 0:

        return (
            math.nan,
            math.nan,
            0,
        )

    mean = float(
        np.mean(
            finite_values
        )
    )

    if finite_values.size > 1:

        std = float(
            np.std(
                finite_values,
                ddof=1,
            )
        )

    else:

        std = 0.0

    return (
        mean,
        std,
        int(
            finite_values.size
        ),
    )


def summarize_patient_metrics(
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
                row[
                    metric
                ]
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
# SAVE PATIENT METRICS
# ============================================================

def save_patient_metrics_csv(
    rows,
    path,
):

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "patient_id"
            ]
            +
            METRIC_COLUMNS,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

def save_summary_csv(
    summary,
    path,
):

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
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
# SAFE TORCH LOAD
# ============================================================

def load_checkpoint(
    path,
):

    try:

        return torch.load(
            path,
            map_location=DEVICE,
            weights_only=False,
        )

    except TypeError:

        return torch.load(
            path,
            map_location=DEVICE,
        )


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_fold(
    fold,
    lambda_pv,
):

    seed_everything(
        RANDOM_SEED
        +
        fold
    )

    lambda_name = (
        f"lambda_{lambda_pv:g}"
        .replace(
            ".",
            "p",
        )
    )

    log_root = (
        PROJECT_DIR
        / "logs"
        / "pral_distance_unet"
        / lambda_name
    )

    checkpoint_root = (
        PROJECT_DIR
        / "checkpoints"
        / "pral_distance_unet"
        / lambda_name
    )

    fold_log_dir = (
        log_root
        / f"fold_{fold}"
    )

    fold_checkpoint_dir = (
        checkpoint_root
        / f"fold_{fold}"
    )

    # ========================================================
    # CLEAN ONLY CURRENT FOLD
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

    terminal_log_path = (
        fold_log_dir
        / "terminal.txt"
    )

    history_csv = (
        fold_log_dir
        / "epoch_metrics.csv"
    )

    per_patient_csv = (
        fold_log_dir
        / "per_patient_metrics.csv"
    )

    summary_csv = (
        fold_log_dir
        / "final_metrics.csv"
    )

    final_metrics_json = (
        fold_log_dir
        / "final_metrics.json"
    )

    best_checkpoint_path = (
        fold_checkpoint_dir
        / "best.pth"
    )

    fold_root = (
        SPLIT_ROOT
        / f"fold_{fold}"
    )

    train_csv = (
        fold_root
        / "train_patients.csv"
    )

    val_csv = (
        fold_root
        / "val_patients.csv"
    )

    test_csv = (
        SPLIT_ROOT
        / "test_30_patients.csv"
    )

    train_ids = set(
        read_patient_ids(
            train_csv
        )
    )

    val_ids = set(
        read_patient_ids(
            val_csv
        )
    )

    test_ids = set(
        read_patient_ids(
            test_csv
        )
    )

    # ========================================================
    # LEAKAGE SAFEGUARDS
    # ========================================================

    if len(train_ids) != 56:

        raise RuntimeError(
            f"Fold {fold}: expected "
            f"56 training patients; "
            f"got {len(train_ids)}"
        )

    if len(val_ids) != 14:

        raise RuntimeError(
            f"Fold {fold}: expected "
            f"14 validation patients; "
            f"got {len(val_ids)}"
        )

    if len(test_ids) != 30:

        raise RuntimeError(
            f"Expected 30 test patients; "
            f"got {len(test_ids)}"
        )

    if train_ids & val_ids:

        raise RuntimeError(
            "DATA LEAKAGE: "
            "train/validation overlap."
        )

    if train_ids & test_ids:

        raise RuntimeError(
            "DATA LEAKAGE: "
            "train/test overlap."
        )

    if val_ids & test_ids:

        raise RuntimeError(
            "DATA LEAKAGE: "
            "validation/test overlap."
        )

    # ========================================================
    # OPEN LOG ONLY AFTER DIRECTORY CLEANUP
    # ========================================================

    with terminal_log_path.open(
        "w",
        encoding="utf-8",
    ) as logfile:

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

            print()
            print(
                "=" * 88
            )

            print(
                "STEP 19 — PRAL DISTANCE-CONDITIONED U-NET"
            )

            print(
                "=" * 88
            )

            print(
                "Fold:",
                fold
            )

            print(
                "lambda_pv:",
                lambda_pv
            )

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

            print()
            print(
                "INPUT CHANNELS:"
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
                "  3 = normalized distance "
                "to ventricle"
            )

            print()
            print(
                "PRAL WEIGHT MAP:"
            )

            print(
                f"  PV <=10 mm = "
                f"{lambda_pv}"
            )

            print(
                "  outside PV  = 1.0"
            )

            print()
            print(
                "Patient membership:"
            )

            print(
                "  Train:",
                len(
                    train_ids
                )
            )

            print(
                "  Val  :",
                len(
                    val_ids
                )
            )

            print(
                "  Test :",
                len(
                    test_ids
                ),
                "(sealed)"
            )

            print(
                "  Overlap: 0"
            )

            # =================================================
            # DATASETS
            # =================================================

            train_dataset = (
                PRALDistanceDataset(

                    partition_root=(
                        fold_root
                        / "train"
                    ),

                    expected_patient_ids=(
                        train_ids
                    ),

                    augment=True,
                )
            )

            val_dataset = (
                PRALDistanceDataset(

                    partition_root=(
                        fold_root
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
                "Training slices:",
                len(
                    train_dataset
                )
            )

            print(
                "Validation slices:",
                len(
                    val_dataset
                )
            )

            # =================================================
            # CLASS WEIGHTS — TRAIN ONLY
            # =================================================

            class_counts = compute_class_counts(
                train_dataset
            )

            class_weights = (
                class_weights_from_counts(
                    class_counts
                )
            )

            print()
            print(
                "Training class distribution:"
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

            # =================================================
            # LOADERS
            # =================================================

            train_loader = DataLoader(

                train_dataset,

                batch_size=BATCH_SIZE,

                shuffle=True,

                num_workers=NUM_WORKERS,

                pin_memory=(
                    DEVICE.type
                    ==
                    "cuda"
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
                    ==
                    "cuda"
                ),

                drop_last=False,
            )

            # =================================================
            # MODEL
            # =================================================

            model = UNet2D(

                in_channels=4,

                num_classes=NUM_CLASSES,

                base_channels=BASE_CHANNELS,

            ).to(
                DEVICE
            )

            # =================================================
            # OPTIMIZER
            # =================================================

            optimizer = torch.optim.AdamW(

                model.parameters(),

                lr=LEARNING_RATE,

                weight_decay=WEIGHT_DECAY,
            )

            # =================================================
            # CLASS WEIGHT TENSOR
            # =================================================

            class_weight_tensor = torch.tensor(

                class_weights,

                dtype=torch.float32,

                device=DEVICE,
            )

            # =================================================
            # PRAL LOSS
            # =================================================

            criterion = PRALLoss(

                class_weights=(
                    class_weight_tensor
                ),

                lambda_pv=(
                    lambda_pv
                ),

                ce_weight=(
                    CE_LOSS_WEIGHT
                ),

                dice_weight=(
                    DICE_LOSS_WEIGHT
                ),

                include_background_in_dice=(
                    INCLUDE_BACKGROUND_IN_DICE_LOSS
                ),
            )

            # =================================================
            # AMP
            # =================================================

            scaler = (

                torch.amp.GradScaler(
                    "cuda"
                )

                if USE_AMP

                else None
            )

            # =================================================
            # HISTORY
            # =================================================

            write_history_header(
                history_csv
            )

            best_val_abnormal_dice = (
                -math.inf
            )

            best_epoch = None

            print()
            print(
                "=" * 88
            )

            print(
                f"TRAINING FOLD {fold}"
            )

            print(
                "=" * 88
            )

            # =================================================
            # EPOCH LOOP
            # =================================================

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
                    run_validation_epoch(

                        model,

                        val_loader,

                        criterion,
                    )
                )

                current_abnormal_dice = (
                    val_metrics[
                        "dice_abnormal_wmh"
                    ]
                )

                improved = (
                    current_abnormal_dice
                    >
                    best_val_abnormal_dice
                )

                if improved:

                    best_val_abnormal_dice = (
                        current_abnormal_dice
                    )

                    best_epoch = (
                        epoch
                    )

                print()
                print(
                    f"Fold {fold} | "
                    f"Epoch "
                    f"{epoch:03d}/{EPOCHS}"
                )

                print(
                    "  train loss "
                    "total/CE/Dice: "
                    f"{train_metrics['loss']:.5f} / "
                    f"{train_metrics['ce_loss']:.5f} / "
                    f"{train_metrics['dice_loss']:.5f}"
                )

                print(
                    "  val   loss "
                    "total/CE/Dice: "
                    f"{val_metrics['loss']:.5f} / "
                    f"{val_metrics['ce_loss']:.5f} / "
                    f"{val_metrics['dice_loss']:.5f}"
                )

                print(
                    "  train Dice "
                    "BG/nWMH/abWMH: "
                    f"{train_metrics['dice_background']:.4f}/"
                    f"{train_metrics['dice_normal_wmh']:.4f}/"
                    f"{train_metrics['dice_abnormal_wmh']:.4f}"
                )

                print(
                    "  val   Dice "
                    "BG/nWMH/abWMH: "
                    f"{val_metrics['dice_background']:.4f}/"
                    f"{val_metrics['dice_normal_wmh']:.4f}/"
                    f"{val_metrics['dice_abnormal_wmh']:.4f}"
                )

                print(
                    "  best val abWMH Dice:",
                    f"{best_val_abnormal_dice:.6f}",
                    f"(epoch {best_epoch})"
                )

                # =============================================
                # SAVE EPOCH CHECKPOINT
                # =============================================

                checkpoint_payload = {

                    "experiment":
                        "pral_distance_unet",

                    "pral_loss_sha256":
                        PRAL_SOURCE_SHA256,

                    "fold":
                        fold,

                    "epoch":
                        epoch,

                    "lambda_pv":
                        lambda_pv,

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

                    "base_channels":
                        BASE_CHANNELS,

                    "learning_rate":
                        LEARNING_RATE,

                    "weight_decay":
                        WEIGHT_DECAY,

                    "batch_size":
                        BATCH_SIZE,

                    "ce_loss_weight":
                        CE_LOSS_WEIGHT,

                    "dice_loss_weight":
                        DICE_LOSS_WEIGHT,

                    "best_val_abnormal_wmh_dice":
                        best_val_abnormal_dice,

                    "best_epoch":
                        best_epoch,

                    "random_seed":
                        RANDOM_SEED,

                    "pv_zone_definition":
                        ">0 mm and <=10 mm "
                        "from ventricle",

                    "deep_zone_definition":
                        ">10 mm from ventricle",
                }

                if (
                    epoch
                    %
                    CHECKPOINT_EVERY
                    ==
                    0
                ):

                    epoch_path = (
                        fold_checkpoint_dir
                        /
                        f"epoch_{epoch:03d}.pth"
                    )

                    atomic_torch_save(
                        epoch_path,
                        checkpoint_payload,
                    )

                if improved:

                    atomic_torch_save(
                        best_checkpoint_path,
                        checkpoint_payload,
                    )

                append_history(

                    history_csv,

                    epoch,

                    train_metrics,

                    val_metrics,

                    best_val_abnormal_dice,

                    optimizer,
                )

            # =================================================
            # LOAD BEST CHECKPOINT
            # =================================================

            if not best_checkpoint_path.exists():

                raise RuntimeError(
                    "Best checkpoint was not created."
                )

            best_checkpoint = (
                load_checkpoint(
                    best_checkpoint_path
                )
            )

            best_epoch = int(
                best_checkpoint[
                    "best_epoch"
                ]
            )

            best_val_abnormal_dice = float(
                best_checkpoint[
                    "best_val_abnormal_wmh_dice"
                ]
            )

            model.load_state_dict(
                best_checkpoint[
                    "model_state"
                ]
            )

            print()
            print(
                "=" * 88
            )

            print(
                "BEST CHECKPOINT"
            )

            print(
                "=" * 88
            )

            print(
                "Best epoch:",
                best_epoch
            )

            print(
                "Best validation abWMH Dice:",
                best_val_abnormal_dice
            )

            # =================================================
            # FINAL PATIENT-WISE 3D VALIDATION
            # =================================================

            print()
            print(
                "=" * 88
            )

            print(
                "FINAL PATIENT-WISE 3D "
                "VALIDATION"
            )

            print(
                "=" * 88
            )

            validation_rows = (
                evaluate_3d(

                    model,

                    val_loader,

                    val_ids,
                )
            )

            summary = (
                summarize_patient_metrics(
                    validation_rows
                )
            )

            save_patient_metrics_csv(
                validation_rows,
                per_patient_csv,
            )

            save_summary_csv(
                summary,
                summary_csv,
            )

            # =================================================
            # JSON FORMAT COMPATIBLE WITH FUTURE FINAL SCRIPT
            # =================================================

            final_result = {

                "experiment":
                    "pral_distance_unet",

                "pral_loss_sha256":
                    PRAL_SOURCE_SHA256,

                "fold":
                    fold,

                "lambda_pv":
                    lambda_pv,

                "best_epoch":
                    best_epoch,

                "best_val_abnormal_wmh_dice":
                    best_val_abnormal_dice,

                "n_train_patients":
                    len(
                        train_ids
                    ),

                "n_val_patients":
                    len(
                        val_ids
                    ),

                "n_test_patients_sealed":
                    len(
                        test_ids
                    ),

                "input_channels":
                    CHANNEL_ORDER,

                "loss":
                    "PRAL weighted CE + "
                    "PRAL weighted Dice",

                "pv_weight":
                    lambda_pv,

                "outside_pv_weight":
                    1.0,
            }

            # Direct metric keys are intentionally
            # included for compatibility with Step 17 style.

            for metric in METRIC_COLUMNS:

                final_result[
                    metric
                ] = summary[
                    f"{metric}_mean"
                ]

                final_result[
                    f"{metric}_std"
                ] = summary[
                    f"{metric}_std"
                ]

                final_result[
                    f"{metric}_n_finite"
                ] = summary[
                    f"{metric}_n_finite"
                ]

            atomic_write_text(

                final_metrics_json,

                json.dumps(
                    final_result,
                    indent=2,
                ),
            )

            # =================================================
            # PRINT FINAL RESULTS
            # =================================================

            print()
            print(
                "=" * 88
            )

            print(
                f"FOLD {fold} FINAL "
                f"3D METRICS"
            )

            print(
                "=" * 88
            )

            print()
            print(
                "NORMAL WMH"
            )

            for metric in [
                "dice",
                "sensitivity",
                "hd95",
            ]:

                key = (
                    f"{metric}_normal_wmh"
                )

                unit = (
                    " mm"
                    if metric == "hd95"
                    else ""
                )

                print(
                    f"  {metric:<12}: "
                    f"{summary[f'{key}_mean']:.6f} "
                    f"± "
                    f"{summary[f'{key}_std']:.6f}"
                    f"{unit}"
                )

            print()
            print(
                "ABNORMAL WMH"
            )

            for metric in [
                "dice",
                "sensitivity",
                "hd95",
            ]:

                key = (
                    f"{metric}_abnormal_wmh"
                )

                unit = (
                    " mm"
                    if metric == "hd95"
                    else ""
                )

                print(
                    f"  {metric:<12}: "
                    f"{summary[f'{key}_mean']:.6f} "
                    f"± "
                    f"{summary[f'{key}_std']:.6f}"
                    f"{unit}"
                )

            print()
            print(
                "=" * 88
            )

            print(
                f"FOLD {fold} COMPLETE"
            )

            print(
                "=" * 88
            )

            print()
            print(
                "Best checkpoint:"
            )

            print(
                best_checkpoint_path
            )

            print()
            print(
                "Metrics JSON:"
            )

            print(
                final_metrics_json
            )

            print()
            print(
                "Terminal log:"
            )

            print(
                terminal_log_path
            )

    return final_result


# ============================================================
# CROSS-FOLD SUMMARY
# ============================================================

def build_cross_fold_summary(
    lambda_pv,
):

    lambda_name = (
        f"lambda_{lambda_pv:g}"
        .replace(
            ".",
            "p",
        )
    )

    log_root = (
        PROJECT_DIR
        / "logs"
        / "pral_distance_unet"
        / lambda_name
    )

    rows = []

    for fold in range(
        1,
        6
    ):

        metrics_path = (
            log_root
            / f"fold_{fold}"
            / "final_metrics.json"
        )

        if not metrics_path.exists():

            return

        data = json.loads(
            metrics_path.read_text(
                encoding="utf-8"
            )
        )

        rows.append(
            data
        )

    summary_csv = (
        log_root
        / "pral_fold_metrics.csv"
    )

    summary_json = (
        log_root
        / "pral_cv_summary.json"
    )

    csv_fields = [

        "fold",
        "lambda_pv",
        "best_epoch",
        "best_val_abnormal_wmh_dice",
    ] + METRIC_COLUMNS

    with summary_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    field:
                        row[
                            field
                        ]
                    for field in csv_fields
                }
            )

    cv_summary = {

        "experiment":
            "pral_distance_unet",

        "pral_loss_sha256":
            PRAL_SOURCE_SHA256,

        "lambda_pv":
            lambda_pv,

        "n_folds":
            5,

        "best_epochs":
            [
                int(
                    row[
                        "best_epoch"
                    ]
                )
                for row in rows
            ],
    }

    for metric in METRIC_COLUMNS:

        (
            mean,
            std,
            count,
        ) = finite_mean_std(
            [
                row[
                    metric
                ]
                for row in rows
            ]
        )

        cv_summary[
            f"{metric}_mean"
        ] = mean

        cv_summary[
            f"{metric}_std"
        ] = std

        cv_summary[
            f"{metric}_n_finite"
        ] = count

    atomic_write_text(

        summary_json,

        json.dumps(
            cv_summary,
            indent=2,
        ),
    )

    print()
    print(
        "=" * 88
    )

    print(
        "PRAL 5-FOLD CROSS-VALIDATION COMPLETE"
    )

    print(
        "=" * 88
    )

    print(
        "lambda_pv:",
        lambda_pv
    )

    print(
        "Best epochs:",
        cv_summary[
            "best_epochs"
        ]
    )

    print()
    print(
        "NORMAL WMH"
    )

    for metric in [
        "dice",
        "sensitivity",
        "hd95",
    ]:

        key = (
            f"{metric}_normal_wmh"
        )

        unit = (
            " mm"
            if metric == "hd95"
            else ""
        )

        print(
            f"  {metric:<12}: "
            f"{cv_summary[f'{key}_mean']:.6f} "
            f"± "
            f"{cv_summary[f'{key}_std']:.6f}"
            f"{unit}"
        )

    print()
    print(
        "ABNORMAL WMH"
    )

    for metric in [
        "dice",
        "sensitivity",
        "hd95",
    ]:

        key = (
            f"{metric}_abnormal_wmh"
        )

        unit = (
            " mm"
            if metric == "hd95"
            else ""
        )

        print(
            f"  {metric:<12}: "
            f"{cv_summary[f'{key}_mean']:.6f} "
            f"± "
            f"{cv_summary[f'{key}_std']:.6f}"
            f"{unit}"
        )

    print()
    print(
        "Cross-fold CSV:"
    )

    print(
        summary_csv
    )

    print()
    print(
        "Cross-fold JSON:"
    )

    print(
        summary_json
    )

    return cv_summary


# ============================================================
# LAMBDA-SWEEP / MODEL-SELECTION HELPERS
# ============================================================

SELECTION_ROOT = (
    PROJECT_DIR
    / "logs"
    / "pral_distance_unet"
)

LAMBDA_SELECTION_CSV = (
    SELECTION_ROOT
    / "lambda_selection.csv"
)

SELECTED_LAMBDA_JSON = (
    SELECTION_ROOT
    / "selected_lambda.json"
)


def lambda_name_from_value(
    lambda_pv,
):

    return (
        f"lambda_{lambda_pv:g}"
        .replace(
            ".",
            "p",
        )
    )


def lambda_log_root(
    lambda_pv,
):

    return (
        SELECTION_ROOT
        / lambda_name_from_value(
            lambda_pv
        )
    )


def lambda_cv_is_complete_and_current(
    lambda_pv,
):

    """
    A lambda CV result is reusable only if all five fold JSON files:
      1) exist,
      2) were generated with the requested lambda,
      3) were generated with the CURRENT 18_pral_loss.py source.

    This prevents stale results from the previous PRAL implementation
    from being silently reused after Step 18 is edited.
    """

    log_root = lambda_log_root(
        lambda_pv
    )

    for fold in range(
        1,
        6
    ):

        metrics_path = (
            log_root
            / f"fold_{fold}"
            / "final_metrics.json"
        )

        if not metrics_path.exists():

            return False

        try:

            data = json.loads(
                metrics_path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:

            return False

        if not math.isclose(
            float(
                data.get(
                    "lambda_pv",
                    math.nan,
                )
            ),
            float(
                lambda_pv
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):

            return False

        if (
            data.get(
                "pral_loss_sha256"
            )
            !=
            PRAL_SOURCE_SHA256
        ):

            return False

    return True


def load_current_cv_summary(
    lambda_pv,
):

    summary_path = (
        lambda_log_root(
            lambda_pv
        )
        / "pral_cv_summary.json"
    )

    if not summary_path.exists():

        return None

    try:

        data = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:

        return None

    if not math.isclose(
        float(
            data.get(
                "lambda_pv",
                math.nan,
            )
        ),
        float(
            lambda_pv
        ),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):

        return None

    if (
        data.get(
            "pral_loss_sha256"
        )
        !=
        PRAL_SOURCE_SHA256
    ):

        return None

    return data


def ensure_cv_for_lambda(
    lambda_pv,
    force=False,
):

    """
    Ensure that a complete, CURRENT 5-fold CV exists for one lambda.

    If Step 18 has changed since an older run, the SHA256 guard makes
    that older run stale and the five folds are retrained automatically.
    """

    if (
        not force
        and
        lambda_cv_is_complete_and_current(
            lambda_pv
        )
    ):

        summary = load_current_cv_summary(
            lambda_pv
        )

        if summary is None:

            summary = build_cross_fold_summary(
                lambda_pv=lambda_pv
            )

        print()
        print(
            "=" * 88
        )
        print(
            "REUSING COMPLETE CURRENT CV"
        )
        print(
            "=" * 88
        )
        print(
            "lambda_pv:",
            lambda_pv
        )
        print(
            "PRAL source SHA256:",
            PRAL_SOURCE_SHA256
        )

        return summary

    print()
    print(
        "=" * 88
    )
    print(
        "RUNNING 5-FOLD CV FOR LAMBDA"
    )
    print(
        "=" * 88
    )
    print(
        "lambda_pv:",
        lambda_pv
    )
    print(
        "PRAL source SHA256:",
        PRAL_SOURCE_SHA256
    )

    for fold in range(
        1,
        6
    ):

        train_fold(
            fold=fold,
            lambda_pv=lambda_pv,
        )

    summary = build_cross_fold_summary(
        lambda_pv=lambda_pv
    )

    if summary is None:

        raise RuntimeError(
            f"Could not build completed CV summary "
            f"for lambda={lambda_pv}"
        )

    return summary


def select_best_lambda(
    lambda_values,
):

    """
    Select lambda using DEVELOPMENT CV ONLY.

    Pre-specified selection metric:
        mean patient-wise 3D abnormal-WMH Dice across the five folds.

    Higher is better.

    Exact ties are resolved by choosing the smaller lambda, which keeps
    the spatial reweighting as conservative as possible.

    The sealed 30-patient test set is NEVER used for lambda selection.
    """

    lambda_values = sorted(
        {
            float(value)
            for value in lambda_values
        }
    )

    if not lambda_values:

        raise ValueError(
            "No lambda values were supplied "
            "for selection."
        )

    rows = []

    for lambda_pv in lambda_values:

        if not lambda_cv_is_complete_and_current(
            lambda_pv
        ):

            raise RuntimeError(
                f"lambda={lambda_pv:g} does not have "
                f"a complete CURRENT 5-fold CV. "
                f"Run the sweep first."
            )

        summary = load_current_cv_summary(
            lambda_pv
        )

        if summary is None:

            summary = build_cross_fold_summary(
                lambda_pv=lambda_pv
            )

        if summary is None:

            raise RuntimeError(
                f"Could not load CV summary "
                f"for lambda={lambda_pv:g}"
            )

        best_epochs = [
            int(value)
            for value
            in summary[
                "best_epochs"
            ]
        ]

        median_best_epoch = float(
            np.median(
                best_epochs
            )
        )

        recommended_final_epochs = int(
            round(
                median_best_epoch
            )
        )

        row = {

            "lambda_pv":
                float(
                    lambda_pv
                ),

            "dice_abnormal_wmh_mean":
                float(
                    summary[
                        "dice_abnormal_wmh_mean"
                    ]
                ),

            "dice_abnormal_wmh_std":
                float(
                    summary[
                        "dice_abnormal_wmh_std"
                    ]
                ),

            "dice_normal_wmh_mean":
                float(
                    summary[
                        "dice_normal_wmh_mean"
                    ]
                ),

            "dice_normal_wmh_std":
                float(
                    summary[
                        "dice_normal_wmh_std"
                    ]
                ),

            "sensitivity_abnormal_wmh_mean":
                float(
                    summary[
                        "sensitivity_abnormal_wmh_mean"
                    ]
                ),

            "sensitivity_abnormal_wmh_std":
                float(
                    summary[
                        "sensitivity_abnormal_wmh_std"
                    ]
                ),

            "hd95_abnormal_wmh_mean":
                float(
                    summary[
                        "hd95_abnormal_wmh_mean"
                    ]
                ),

            "hd95_abnormal_wmh_std":
                float(
                    summary[
                        "hd95_abnormal_wmh_std"
                    ]
                ),

            "best_epochs":
                best_epochs,

            "median_best_epoch":
                median_best_epoch,

            "recommended_final_epochs":
                recommended_final_epochs,
        }

        rows.append(
            row
        )

    # Primary criterion:
    # maximum mean abnormal-WMH Dice.
    #
    # Exact numerical ties:
    # smaller lambda wins.
    ranked = sorted(
        rows,
        key=lambda row: (
            -row[
                "dice_abnormal_wmh_mean"
            ],
            row[
                "lambda_pv"
            ],
        ),
    )

    for rank, row in enumerate(
        ranked,
        start=1,
    ):

        row[
            "rank"
        ] = rank

    selected = ranked[
        0
    ]

    SELECTION_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_fields = [
        "rank",
        "lambda_pv",
        "dice_abnormal_wmh_mean",
        "dice_abnormal_wmh_std",
        "dice_normal_wmh_mean",
        "dice_normal_wmh_std",
        "sensitivity_abnormal_wmh_mean",
        "sensitivity_abnormal_wmh_std",
        "hd95_abnormal_wmh_mean",
        "hd95_abnormal_wmh_std",
        "median_best_epoch",
        "recommended_final_epochs",
        "best_epochs",
    ]

    with LAMBDA_SELECTION_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=selection_fields,
        )

        writer.writeheader()

        for row in ranked:

            csv_row = dict(
                row
            )

            csv_row[
                "best_epochs"
            ] = ",".join(
                str(value)
                for value
                in row[
                    "best_epochs"
                ]
            )

            writer.writerow(
                {
                    field:
                        csv_row[
                            field
                        ]
                    for field in selection_fields
                }
            )

    selected_payload = {

        "experiment":
            "pral_distance_unet",

        "selection_status":
            "LOCKED_FROM_DEVELOPMENT_CV",

        "selection_metric":
            "dice_abnormal_wmh_mean",

        "selection_rule":
            (
                "Choose the lambda with the highest "
                "mean patient-wise 3D abnormal-WMH Dice "
                "across the five development CV folds. "
                "If exactly tied, choose the smaller lambda."
            ),

        "test_set_used_for_selection":
            False,

        "candidate_lambdas":
            [
                float(value)
                for value
                in lambda_values
            ],

        "selected_lambda_pv":
            float(
                selected[
                    "lambda_pv"
                ]
            ),

        "selected_cv_abnormal_wmh_dice_mean":
            float(
                selected[
                    "dice_abnormal_wmh_mean"
                ]
            ),

        "selected_cv_abnormal_wmh_dice_std":
            float(
                selected[
                    "dice_abnormal_wmh_std"
                ]
            ),

        "selected_best_epochs":
            [
                int(value)
                for value
                in selected[
                    "best_epochs"
                ]
            ],

        "selected_median_best_epoch":
            float(
                selected[
                    "median_best_epoch"
                ]
            ),

        "recommended_final70_epochs":
            int(
                selected[
                    "recommended_final_epochs"
                ]
            ),

        "pral_loss_sha256":
            PRAL_SOURCE_SHA256,

        "input_channels":
            CHANNEL_ORDER,

        "num_input_channels":
            4,

        "num_classes":
            NUM_CLASSES,

        "base_channels":
            BASE_CHANNELS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "batch_size":
            BATCH_SIZE,

        "ce_loss_weight":
            CE_LOSS_WEIGHT,

        "dice_loss_weight":
            DICE_LOSS_WEIGHT,

        "pv_zone_definition":
            ">0 mm and <=10 mm from ventricle",

        "deep_zone_definition":
            ">10 mm from ventricle",
    }

    atomic_write_text(
        SELECTED_LAMBDA_JSON,
        json.dumps(
            selected_payload,
            indent=2,
        ),
    )

    print()
    print(
        "=" * 88
    )
    print(
        "PRAL LAMBDA SELECTION COMPLETE"
    )
    print(
        "=" * 88
    )

    print()
    print(
        "Selection used DEVELOPMENT CV only."
    )
    print(
        "Sealed 30-patient test set used:",
        "NO"
    )

    print()
    print(
        f"{'Rank':<6}"
        f"{'Lambda':<10}"
        f"{'abWMH Dice':<22}"
        f"{'nWMH Dice':<22}"
        f"{'abWMH Sens':<22}"
        f"{'abWMH HD95':<22}"
        f"{'Final epochs':<14}"
    )

    print(
        "-" * 118
    )

    for row in ranked:

        print(
            f"{row['rank']:<6}"
            f"{row['lambda_pv']:<10g}"
            f"{row['dice_abnormal_wmh_mean']:.6f} "
            f"± {row['dice_abnormal_wmh_std']:.6f}   "
            f"{row['dice_normal_wmh_mean']:.6f} "
            f"± {row['dice_normal_wmh_std']:.6f}   "
            f"{row['sensitivity_abnormal_wmh_mean']:.6f} "
            f"± {row['sensitivity_abnormal_wmh_std']:.6f}   "
            f"{row['hd95_abnormal_wmh_mean']:.6f} "
            f"± {row['hd95_abnormal_wmh_std']:.6f}   "
            f"{row['recommended_final_epochs']:<14d}"
        )

    print()
    print(
        "SELECTED lambda_pv:",
        selected_payload[
            "selected_lambda_pv"
        ]
    )

    print(
        "Selected CV abnormal-WMH Dice:",
        (
            f"{selected_payload['selected_cv_abnormal_wmh_dice_mean']:.6f}"
            f" ± "
            f"{selected_payload['selected_cv_abnormal_wmh_dice_std']:.6f}"
        )
    )

    print(
        "Selected fold best epochs:",
        selected_payload[
            "selected_best_epochs"
        ]
    )

    print(
        "Recommended FINAL70 epochs:",
        selected_payload[
            "recommended_final70_epochs"
        ]
    )

    print()
    print(
        "Selection CSV:"
    )
    print(
        LAMBDA_SELECTION_CSV
    )

    print()
    print(
        "LOCKED selection JSON for Step 20:"
    )
    print(
        SELECTED_LAMBDA_JSON
    )

    return selected_payload


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(

        description=(
            "Train distance-conditioned U-Net with PRAL, "
            "run a development-only lambda sweep, and lock "
            "the lambda/epoch count for FINAL70 training."
        )
    )

    parser.add_argument(

        "--fold",

        type=int,

        choices=[
            1,
            2,
            3,
            4,
            5,
        ],

        default=None,

        help=(
            "Train one fold for --lambda-pv. "
            "Cannot be combined with --sweep."
        ),
    )

    parser.add_argument(

        "--lambda-pv",

        type=float,

        default=3.0,

        help=(
            "Single PRAL PV weight. "
            "Used when --sweep is not supplied."
        ),
    )

    parser.add_argument(

        "--sweep",

        type=float,

        nargs="+",

        default=None,

        help=(
            "Run/collect full 5-fold CV for each lambda, "
            "then automatically choose the best lambda "
            "using mean 3D abnormal-WMH Dice. "
            "Example: --sweep 2 3 4 5"
        ),
    )

    parser.add_argument(

        "--select-only",

        action="store_true",

        help=(
            "Do not train. Read already-completed CURRENT "
            "CV results for the lambdas in --sweep and "
            "create the locked lambda-selection files."
        ),
    )

    parser.add_argument(

        "--force",

        action="store_true",

        help=(
            "Force retraining even if a current complete "
            "5-fold CV already exists for a lambda."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print()
    print(
        "=" * 88
    )
    print(
        "STEP 19 — PRAL DISTANCE-CONDITIONED U-NET"
    )
    print(
        "=" * 88
    )

    print(
        "PyTorch:",
        torch.__version__
    )

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
        "PRAL source SHA256:",
        PRAL_SOURCE_SHA256
    )

    # ========================================================
    # LAMBDA SWEEP + AUTOMATIC SELECTION
    # ========================================================

    if args.sweep is not None:

        if args.fold is not None:

            raise ValueError(
                "--fold cannot be combined "
                "with --sweep."
            )

        lambda_values = sorted(
            {
                float(value)
                for value
                in args.sweep
            }
        )

        if any(
            value < 1.0
            for value
            in lambda_values
        ):

            raise ValueError(
                "Every sweep lambda must be >= 1.0"
            )

        print()
        print(
            "Lambda candidates:",
            lambda_values
        )

        print(
            "Selection metric:",
            "mean patient-wise 3D abnormal-WMH Dice"
        )

        print(
            "Test set used for selection:",
            "NO"
        )

        if args.select_only:

            print()
            print(
                "Mode: SELECT ONLY "
                "(no training)"
            )

        else:

            print()
            print(
                "Mode: CV SWEEP + SELECT"
            )

            for lambda_pv in lambda_values:

                ensure_cv_for_lambda(
                    lambda_pv=lambda_pv,
                    force=args.force,
                )

        select_best_lambda(
            lambda_values=lambda_values
        )

        return

    # ========================================================
    # SINGLE-LAMBDA MODE
    # ========================================================

    if args.select_only:

        raise ValueError(
            "--select-only requires --sweep."
        )

    if args.lambda_pv < 1.0:

        raise ValueError(
            "--lambda-pv must be >= 1.0"
        )

    print(
        "lambda_pv:",
        args.lambda_pv
    )

    # ========================================================
    # ONE-FOLD MODE
    # ========================================================

    if args.fold is not None:

        train_fold(
            fold=args.fold,
            lambda_pv=args.lambda_pv,
        )

        build_cross_fold_summary(
            lambda_pv=args.lambda_pv
        )

        return

    # ========================================================
    # ONE LAMBDA — ALL FIVE FOLDS
    # ========================================================

    ensure_cv_for_lambda(
        lambda_pv=args.lambda_pv,
        force=args.force,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception:

        print()
        print(
            "=" * 88
        )
        print(
            "STEP 19 FAILED"
        )
        print(
            "=" * 88
        )

        traceback.print_exc()

        raise
