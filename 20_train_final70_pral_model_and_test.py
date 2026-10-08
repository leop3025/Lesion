from __future__ import annotations

from pathlib import Path

import contextlib
import csv
import importlib.util
import json
import math
import os
import random
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
    ConcatDataset,
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

STEP19_FILE = (
    PROJECT_DIR
    / "19_train_pral_distance_unet.py"
)

SELECTION_JSON = (
    PROJECT_DIR
    / "logs"
    / "pral_distance_unet"
    / "selected_lambda.json"
)

FINAL_LOG_DIR = (
    PROJECT_DIR
    / "logs"
    / "pral_distance_unet"
    / "final_results"
)

FINAL_CHECKPOINT_DIR = (
    PROJECT_DIR
    / "checkpoints"
    / "pral_final70"
)

FINAL_CHECKPOINT = (
    FINAL_CHECKPOINT_DIR
    / "final70_pral_model.pth"
)

TERMINAL_LOG = (
    FINAL_LOG_DIR
    / "terminal.txt"
)

TRAIN_HISTORY_CSV = (
    FINAL_LOG_DIR
    / "training_history.csv"
)

TRAIN70_METRICS_CSV = (
    FINAL_LOG_DIR
    / "train70_patient_metrics.csv"
)

TEST30_METRICS_CSV = (
    FINAL_LOG_DIR
    / "test30_patient_metrics.csv"
)

FINAL_SUMMARY_JSON = (
    FINAL_LOG_DIR
    / "final70_test30_summary.json"
)

FINAL_SUMMARY_CSV = (
    FINAL_LOG_DIR
    / "final70_test30_summary.csv"
)


# ============================================================
# IMPORT STEP 19
# ============================================================

if not STEP19_FILE.exists():

    raise FileNotFoundError(
        f"Missing Step 19 script:\n"
        f"{STEP19_FILE}"
    )


spec = importlib.util.spec_from_file_location(
    "step19_module",
    STEP19_FILE,
)

if (
    spec is None
    or
    spec.loader is None
):

    raise RuntimeError(
        "Could not import "
        "19_train_pral_distance_unet.py"
    )


step19 = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    step19
)


# ============================================================
# REUSE LOCKED STEP-19 COMPONENTS
# ============================================================

PRALDistanceDataset = (
    step19.PRALDistanceDataset
)

UNet2D = (
    step19.UNet2D
)

PRALLoss = (
    step19.PRALLoss
)

read_patient_ids = (
    step19.read_patient_ids
)

compute_class_counts = (
    step19.compute_class_counts
)

class_weights_from_counts = (
    step19.class_weights_from_counts
)

find_flair_for_patient = (
    step19.find_flair_for_patient
)

atomic_torch_save = (
    step19.atomic_torch_save
)

atomic_write_text = (
    step19.atomic_write_text
)


# ============================================================
# LOCKED CONFIGURATION
# ============================================================

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

INCLUDE_BACKGROUND_IN_DICE_LOSS = False

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = (
    DEVICE.type == "cuda"
)


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
# LOAD SELECTED LAMBDA
# ============================================================

def load_locked_selection():

    if not SELECTION_JSON.exists():

        raise FileNotFoundError(
            "Missing selected lambda file:\n"
            f"{SELECTION_JSON}\n\n"
            "Run Step 19 lambda selection first."
        )

    data = json.loads(
        SELECTION_JSON.read_text(
            encoding="utf-8"
        )
    )

    if (
        "selected_lambda_pv"
        not in data
    ):

        raise RuntimeError(
            "selected_lambda_pv missing "
            "from selected_lambda.json"
        )

    if (
        "recommended_final70_epochs"
        not in data
    ):

        raise RuntimeError(
            "recommended_final70_epochs "
            "missing from selected_lambda.json"
        )

    lambda_pv = float(
        data[
            "selected_lambda_pv"
        ]
    )

    final_epochs = int(
        data[
            "recommended_final70_epochs"
        ]
    )

    if lambda_pv < 1:

        raise RuntimeError(
            "Invalid selected lambda."
        )

    if final_epochs < 1:

        raise RuntimeError(
            "Invalid final epoch count."
        )

    return (
        data,
        lambda_pv,
        final_epochs,
    )


# ============================================================
# FINAL70 DATASET WRAPPER
#
# Combines Fold-1 train + Fold-1 val.
# These are exactly the 70 development patients.
# ============================================================

def build_final70_dataset(
    train_ids,
    val_ids,
):

    fold1_root = (
        SPLIT_ROOT
        / "fold_1"
    )

    dataset_a = PRALDistanceDataset(

        partition_root=(
            fold1_root
            / "train"
        ),

        expected_patient_ids=(
            train_ids
        ),

        augment=True,
    )

    dataset_b = PRALDistanceDataset(

        partition_root=(
            fold1_root
            / "val"
        ),

        expected_patient_ids=(
            val_ids
        ),

        augment=True,
    )

    return (
        dataset_a,
        dataset_b,
        ConcatDataset(
            [
                dataset_a,
                dataset_b,
            ]
        ),
    )


# ============================================================
# COMPUTE CLASS COUNTS OVER ALL 70 PATIENTS
# ============================================================

def compute_final70_class_counts(
    dataset_a,
    dataset_b,
):

    counts_a = compute_class_counts(
        dataset_a
    )

    counts_b = compute_class_counts(
        dataset_b
    )

    return (
        counts_a
        +
        counts_b
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
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

        total_losses.append(
            float(
                total_loss.detach().cpu()
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
    }


# ============================================================
# METRICS
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

    tp = np.count_nonzero(
        pred_mask
        &
        target_mask
    )

    fp = np.count_nonzero(
        pred_mask
        &
        ~target_mask
    )

    fn = np.count_nonzero(
        ~pred_mask
        &
        target_mask
    )

    denominator = (
        2 * tp
        +
        fp
        +
        fn
    )

    if denominator == 0:

        dice = 1.0

    else:

        dice = (
            2 * tp
            /
            denominator
        )

    positive_gt = (
        tp
        +
        fn
    )

    if positive_gt == 0:

        sensitivity = math.nan

    else:

        sensitivity = (
            tp
            /
            positive_gt
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

    pred_exists = bool(
        prediction_mask.any()
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

    pred_surface = (
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
        not pred_surface.any()
        or
        not target_surface.any()
    ):

        return math.nan

    dist_to_target = (
        distance_transform_edt(
            ~target_surface,
            sampling=spacing,
        )
    )

    dist_to_pred = (
        distance_transform_edt(
            ~pred_surface,
            sampling=spacing,
        )
    )

    distances = np.concatenate(
        [
            dist_to_target[
                pred_surface
            ],
            dist_to_pred[
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
# PATIENT-WISE 3D EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_patientwise_3d(
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
            float(x)
            for x
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

        for class_index, class_name in enumerate(
            CLASS_NAMES
        ):

            dice, sensitivity = (
                dice_and_sensitivity(
                    prediction,
                    target,
                    class_index,
                )
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
# SUMMARIZE METRICS
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


def finite_mean_std(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:

        return (
            math.nan,
            math.nan,
            0,
        )

    mean = float(
        np.mean(
            values
        )
    )

    if values.size > 1:

        std = float(
            np.std(
                values,
                ddof=1,
            )
        )

    else:

        std = 0.0

    return (
        mean,
        std,
        int(
            values.size
        ),
    )


def summarize_rows(
    rows,
):

    summary = {
        "n_patients":
            len(
                rows
            )
    }

    for metric in METRIC_COLUMNS:

        values = [
            row[
                metric
            ]
            for row in rows
        ]

        mean, std, n_finite = (
            finite_mean_std(
                values
            )
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

def save_patient_metrics(
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
# PRINT SUMMARY
# ============================================================

def print_summary(
    title,
    summary,
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


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything(
        RANDOM_SEED
    )

    # ========================================================
    # LOAD LOCKED CV DECISION
    # ========================================================

    (
        selection_data,
        lambda_pv,
        final_epochs,
    ) = load_locked_selection()

    print()
    print(
        "=" * 88
    )

    print(
        "STEP 20 — FINAL70 PRAL MODEL + SEALED TEST"
    )

    print(
        "=" * 88
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
        "LOCKED FROM DEVELOPMENT CV"
    )

    print(
        "Selected lambda_pv:",
        lambda_pv
    )

    print(
        "Selected fold best epochs:",
        selection_data.get(
            "selected_best_epochs"
        )
    )

    print(
        "FINAL70 epochs:",
        final_epochs
    )

    print(
        "Selection metric:",
        selection_data.get(
            "selection_metric",
            "dice_abnormal_wmh_mean",
        )
    )

    print(
        "Test used during lambda selection:",
        selection_data.get(
            "test_set_used_for_selection",
            False,
        )
    )

    # ========================================================
    # PATIENT SPLITS
    # ========================================================

    fold1_root = (
        SPLIT_ROOT
        / "fold_1"
    )

    train56_ids = set(
        read_patient_ids(
            fold1_root
            / "train_patients.csv"
        )
    )

    val14_ids = set(
        read_patient_ids(
            fold1_root
            / "val_patients.csv"
        )
    )

    test30_ids = set(
        read_patient_ids(
            SPLIT_ROOT
            / "test_30_patients.csv"
        )
    )

    development70_ids = (
        train56_ids
        |
        val14_ids
    )

    # ========================================================
    # LEAKAGE SAFEGUARDS
    # ========================================================

    if len(
        train56_ids
    ) != 56:

        raise RuntimeError(
            "Expected 56 Fold-1 "
            "training patients."
        )

    if len(
        val14_ids
    ) != 14:

        raise RuntimeError(
            "Expected 14 Fold-1 "
            "validation patients."
        )

    if len(
        development70_ids
    ) != 70:

        raise RuntimeError(
            "Expected exactly 70 "
            "development patients."
        )

    if len(
        test30_ids
    ) != 30:

        raise RuntimeError(
            "Expected exactly 30 "
            "test patients."
        )

    overlap = (
        development70_ids
        &
        test30_ids
    )

    if overlap:

        raise RuntimeError(
            "DATA LEAKAGE: development/"
            f"test overlap: {sorted(overlap)}"
        )

    print()
    print(
        "FINAL MEMBERSHIP SAFEGUARD: PASS"
    )

    print(
        "  development:",
        len(
            development70_ids
        )
    )

    print(
        "  test       :",
        len(
            test30_ids
        )
    )

    print(
        "  overlap    :",
        len(
            overlap
        )
    )

    # ========================================================
    # PREPARE OUTPUT DIRECTORIES
    #
    # Important:
    # Do directory deletion BEFORE terminal.txt is opened.
    # ========================================================

    if FINAL_LOG_DIR.exists():

        shutil.rmtree(
            FINAL_LOG_DIR
        )

    if FINAL_CHECKPOINT_DIR.exists():

        shutil.rmtree(
            FINAL_CHECKPOINT_DIR
        )

    FINAL_LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # TERMINAL LOGGER
    # ========================================================

    with TERMINAL_LOG.open(
        "w",
        encoding="utf-8",
    ) as logfile:

        tee_stdout = Tee(
            sys.stdout,
            logfile,
        )

        tee_stderr = Tee(
            sys.stderr,
            logfile,
        )

        with (
            contextlib.redirect_stdout(
                tee_stdout
            ),
            contextlib.redirect_stderr(
                tee_stderr
            )
        ):

            print()
            print(
                "=" * 88
            )

            print(
                "BUILDING FINAL70 DATASET"
            )

            print(
                "=" * 88
            )

            # =================================================
            # FINAL70 TRAIN DATASET
            # =================================================

            (
                dataset56,
                dataset14,
                final70_dataset,
            ) = build_final70_dataset(

                train56_ids,
                val14_ids,
            )

            print()
            print(
                "Development patients:",
                len(
                    development70_ids
                )
            )

            print(
                "Development slices:",
                len(
                    final70_dataset
                )
            )

            # =================================================
            # CLASS COUNTS OVER ALL 70
            # =================================================

            class_counts = (
                compute_final70_class_counts(
                    dataset56,
                    dataset14,
                )
            )

            class_weights = (
                class_weights_from_counts(
                    class_counts
                )
            )

            print()
            print(
                "FINAL70 CLASS DISTRIBUTION"
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
            # TRAIN LOADER
            # =================================================

            train_loader = DataLoader(

                final70_dataset,

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

            scaler = (

                torch.amp.GradScaler(
                    "cuda"
                )

                if USE_AMP

                else None
            )

            # =================================================
            # TRAINING HISTORY
            # =================================================

            with TRAIN_HISTORY_CSV.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as file:

                writer = csv.DictWriter(

                    file,

                    fieldnames=[
                        "epoch",
                        "total_loss",
                        "ce_loss",
                        "dice_loss",
                    ],
                )

                writer.writeheader()

            # =================================================
            # FINAL70 TRAINING
            # =================================================

            print()
            print(
                "=" * 88
            )

            print(
                "FINAL70 TRAINING"
            )

            print(
                "=" * 88
            )

            print(
                "lambda_pv:",
                lambda_pv
            )

            print(
                "epochs:",
                final_epochs
            )

            for epoch in range(
                1,
                final_epochs + 1,
            ):

                metrics = train_one_epoch(

                    model,

                    train_loader,

                    optimizer,

                    criterion,

                    scaler,
                )

                print()
                print(
                    f"Epoch "
                    f"{epoch:03d}/"
                    f"{final_epochs}"
                )

                print(
                    "  total loss:",
                    f"{metrics['loss']:.6f}"
                )

                print(
                    "  weighted CE:",
                    f"{metrics['ce_loss']:.6f}"
                )

                print(
                    "  weighted Dice:",
                    f"{metrics['dice_loss']:.6f}"
                )

                with TRAIN_HISTORY_CSV.open(
                    "a",
                    encoding="utf-8",
                    newline="",
                ) as file:

                    writer = csv.DictWriter(

                        file,

                        fieldnames=[
                            "epoch",
                            "total_loss",
                            "ce_loss",
                            "dice_loss",
                        ],
                    )

                    writer.writerow(
                        {

                            "epoch":
                                epoch,

                            "total_loss":
                                metrics[
                                    "loss"
                                ],

                            "ce_loss":
                                metrics[
                                    "ce_loss"
                                ],

                            "dice_loss":
                                metrics[
                                    "dice_loss"
                                ],
                        }
                    )

            # =================================================
            # SAVE LOCKED FINAL MODEL
            # =================================================

            checkpoint = {

                "experiment":
                    "final70_pral_distance_unet",

                "model_state":
                    model.state_dict(),

                "optimizer_state":
                    optimizer.state_dict(),

                "selected_lambda_pv":
                    lambda_pv,

                "final_epochs":
                    final_epochs,

                "selected_best_epochs":
                    selection_data.get(
                        "selected_best_epochs"
                    ),

                "selection_metric":
                    selection_data.get(
                        "selection_metric",
                        "dice_abnormal_wmh_mean",
                    ),

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

                "random_seed":
                    RANDOM_SEED,

                "n_development_patients":
                    70,

                "n_test_patients":
                    30,

                "pv_zone_definition":
                    ">0 mm and <=10 mm "
                    "from ventricle",

                "deep_zone_definition":
                    ">10 mm from ventricle",
            }

            atomic_torch_save(
                FINAL_CHECKPOINT,
                checkpoint,
            )

            print()
            print(
                "=" * 88
            )

            print(
                "FINAL70 MODEL SAVED"
            )

            print(
                "=" * 88
            )

            print(
                FINAL_CHECKPOINT
            )

            # =================================================
            # NON-AUGMENTED DEVELOPMENT DATASETS FOR EVALUATION
            # =================================================

            dev56_eval = PRALDistanceDataset(

                partition_root=(
                    fold1_root
                    / "train"
                ),

                expected_patient_ids=(
                    train56_ids
                ),

                augment=False,
            )

            dev14_eval = PRALDistanceDataset(

                partition_root=(
                    fold1_root
                    / "val"
                ),

                expected_patient_ids=(
                    val14_ids
                ),

                augment=False,
            )

            development_eval_dataset = (
                ConcatDataset(
                    [
                        dev56_eval,
                        dev14_eval,
                    ]
                )
            )

            development_eval_loader = (
                DataLoader(

                    development_eval_dataset,

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
            )

            # =================================================
            # TRAIN70 EVALUATION
            # =================================================

            print()
            print(
                "=" * 88
            )

            print(
                "EVALUATING FINAL MODEL ON "
                "70 DEVELOPMENT PATIENTS"
            )

            print(
                "=" * 88
            )

            train70_rows = (
                evaluate_patientwise_3d(

                    model,

                    development_eval_loader,

                    development70_ids,
                )
            )

            train70_summary = (
                summarize_rows(
                    train70_rows
                )
            )

            save_patient_metrics(
                train70_rows,
                TRAIN70_METRICS_CSV,
            )

            print_summary(
                "FINAL70 DEVELOPMENT METRICS",
                train70_summary,
            )

            # =================================================
            # SEALED TEST DATASET
            #
            # This is the first and only place the test set
            # affects reported final performance.
            # No tuning follows this evaluation.
            # =================================================

            print()
            print(
                "=" * 88
            )

            print(
                "OPENING SEALED 30-PATIENT TEST SET"
            )

            print(
                "=" * 88
            )

            test_dataset = PRALDistanceDataset(

                partition_root=(
                    SPLIT_ROOT
                    / "test"
                ),

                expected_patient_ids=(
                    test30_ids
                ),

                augment=False,
            )

            test_loader = DataLoader(

                test_dataset,

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

            print()
            print(
                "Test patients:",
                len(
                    test30_ids
                )
            )

            print(
                "Test slices:",
                len(
                    test_dataset
                )
            )

            # =================================================
            # ONE-TIME SEALED TEST EVALUATION
            # =================================================

            test30_rows = (
                evaluate_patientwise_3d(

                    model,

                    test_loader,

                    test30_ids,
                )
            )

            test30_summary = (
                summarize_rows(
                    test30_rows
                )
            )

            save_patient_metrics(
                test30_rows,
                TEST30_METRICS_CSV,
            )

            print_summary(
                "SEALED 30-PATIENT TEST RESULTS",
                test30_summary,
            )

            # =================================================
            # FINAL JSON
            # =================================================

            final_summary = {

                "experiment":
                    "final70_pral_distance_unet",

                "lambda_selection_source":
                    str(
                        SELECTION_JSON
                    ),

                "selected_lambda_pv":
                    lambda_pv,

                "selected_cv_abnormal_wmh_dice":
                    selection_data.get(
                        "selected_cv_abnormal_wmh_dice"
                    ),

                "selected_best_epochs":
                    selection_data.get(
                        "selected_best_epochs"
                    ),

                "final70_epochs":
                    final_epochs,

                "selection_metric":
                    selection_data.get(
                        "selection_metric",
                        "dice_abnormal_wmh_mean",
                    ),

                "test_set_used_for_lambda_selection":
                    False,

                "development_patients":
                    70,

                "test_patients":
                    30,

                "checkpoint":
                    str(
                        FINAL_CHECKPOINT
                    ),

                "train70":
                    train70_summary,

                "test30":
                    test30_summary,
            }

            atomic_write_text(

                FINAL_SUMMARY_JSON,

                json.dumps(
                    final_summary,
                    indent=2,
                ),
            )

            # =================================================
            # SUMMARY CSV
            # =================================================

            with FINAL_SUMMARY_CSV.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as file:

                writer = csv.DictWriter(

                    file,

                    fieldnames=[
                        "dataset",
                        "metric",
                        "mean",
                        "std",
                        "n_finite",
                    ],
                )

                writer.writeheader()

                for (
                    dataset_name,
                    summary,
                ) in [

                    (
                        "development70",
                        train70_summary,
                    ),

                    (
                        "test30",
                        test30_summary,
                    ),
                ]:

                    for metric in METRIC_COLUMNS:

                        writer.writerow(
                            {

                                "dataset":
                                    dataset_name,

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

            # =================================================
            # COMPLETE
            # =================================================

            print()
            print(
                "=" * 88
            )

            print(
                "STEP 20 COMPLETE"
            )

            print(
                "=" * 88
            )

            print()
            print(
                "LOCKED MODEL"
            )

            print(
                "lambda_pv:",
                lambda_pv
            )

            print(
                "epochs:",
                final_epochs
            )

            print()
            print(
                "Checkpoint:"
            )

            print(
                FINAL_CHECKPOINT
            )

            print()
            print(
                "Test patient metrics:"
            )

            print(
                TEST30_METRICS_CSV
            )

            print()
            print(
                "Final summary:"
            )

            print(
                FINAL_SUMMARY_JSON
            )

            print()
            print(
                "IMPORTANT:"
            )

            print(
                "The sealed test result must "
                "not be used to change lambda, "
                "epochs, architecture, loss, "
                "or other hyperparameters."
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
            "STEP 20 FAILED"
        )

        print(
            "=" * 88
        )

        traceback.print_exc()

        raise