from __future__ import annotations

from pathlib import Path

import argparse
import math
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Preethi\Documents\LESION"
)

DISTANCE_ROOT = (
    PROJECT_DIR
    / "distance_labels"
)

PREPROCESSED_ROOT = (
    PROJECT_DIR
    / "preprocessed"
)


# ============================================================
# LABEL CONFIGURATION
# ============================================================

NUM_CLASSES = 3

CLASS_NAMES = [
    "background",
    "normal_wmh",
    "abnormal_wmh",
]


# ============================================================
# PV / DEEP ZONE LABELS FROM STAGE 15
# ============================================================
#
# Stage 15:
#
#   0 = ventricle
#   1 = periventricular zone
#       (>0 mm and <=10 mm from ventricle)
#   2 = deep zone
#       (>10 mm)
#
# PRAL weighting:
#
#   PV zone       -> lambda_pv
#   everything else -> 1.0
#
# ============================================================

VENTRICLE_ZONE_LABEL = 0
PV_ZONE_LABEL = 1
DEEP_ZONE_LABEL = 2


# ============================================================
# DEFAULT LOSS SETTINGS
# ============================================================

DEFAULT_LAMBDA_PV = 3.0

CE_LOSS_WEIGHT = 1.0
DICE_LOSS_WEIGHT = 1.0

INCLUDE_BACKGROUND_IN_DICE = False

SMOOTH = 1e-6


# ============================================================
# FILE HELPERS
# ============================================================

def pv_zone_path_for_patient(
    patient_id: str,
) -> Path:

    patient_id = str(
        patient_id
    ).zfill(3)

    return (
        DISTANCE_ROOT
        / patient_id
        / (
            f"{patient_id}"
            "_pv_deep_zone_map.nii.gz"
        )
    )


def find_flair_for_patient(
    patient_id: str,
) -> Path:

    patient_id = str(
        patient_id
    ).zfill(3)

    patient_dir = (
        PREPROCESSED_ROOT
        / patient_id
    )

    if not patient_dir.exists():

        raise FileNotFoundError(
            f"Missing patient directory:\n"
            f"{patient_dir}"
        )

    candidates = [

        p

        for p in patient_dir.rglob(
            "*.nii*"
        )

        if (
            "flair" in p.name.lower()
            and
            "mask" not in p.name.lower()
        )
    ]

    if len(candidates) != 1:

        raise RuntimeError(
            f"{patient_id}: expected exactly "
            f"one preprocessed FLAIR volume, "
            f"found {len(candidates)}."
        )

    return candidates[0]


# ============================================================
# LOAD STAGE-15 PV/DEEP ZONE MAP
# ============================================================

def load_pv_deep_zone_map(
    patient_id: str,
) -> tuple[np.ndarray, nib.Nifti1Image]:

    patient_id = str(
        patient_id
    ).zfill(3)

    zone_path = pv_zone_path_for_patient(
        patient_id
    )

    if not zone_path.exists():

        raise FileNotFoundError(
            f"Missing Stage-15 PV/deep zone map:\n"
            f"{zone_path}"
        )

    zone_img = nib.load(
        str(
            zone_path
        )
    )

    zone_map = np.asarray(
        zone_img.dataobj
    )

    if not np.all(
        np.isfinite(
            zone_map
        )
    ):

        raise RuntimeError(
            f"{patient_id}: zone map contains "
            f"NaN or Inf."
        )

    zone_map = zone_map.astype(
        np.int16,
        copy=False,
    )

    unique = set(
        np.unique(
            zone_map
        ).tolist()
    )

    expected = {
        VENTRICLE_ZONE_LABEL,
        PV_ZONE_LABEL,
        DEEP_ZONE_LABEL,
    }

    if not unique.issubset(
        expected
    ):

        raise RuntimeError(
            f"{patient_id}: unexpected zone labels "
            f"{sorted(unique)}. "
            f"Expected subset of "
            f"{sorted(expected)}."
        )

    # ========================================================
    # VERIFY AGAINST PREPROCESSED FLAIR
    # ========================================================

    flair_path = find_flair_for_patient(
        patient_id
    )

    flair_img = nib.load(
        str(
            flair_path
        )
    )

    if zone_map.shape != flair_img.shape[:3]:

        raise RuntimeError(
            f"{patient_id}: zone-map / FLAIR "
            f"shape mismatch.\n"
            f"Zone:  {zone_map.shape}\n"
            f"FLAIR: {flair_img.shape[:3]}"
        )

    if not np.allclose(
        zone_img.affine,
        flair_img.affine,
        atol=1e-4,
        rtol=0,
    ):

        raise RuntimeError(
            f"{patient_id}: zone-map / FLAIR "
            f"affine mismatch."
        )

    return (
        zone_map,
        zone_img,
    )


# ============================================================
# CREATE PRAL WEIGHT MAP — NUMPY VERSION
# ============================================================

def build_pral_weight_map_numpy(
    zone_map: np.ndarray,
    lambda_pv: float = DEFAULT_LAMBDA_PV,
) -> np.ndarray:

    if lambda_pv < 1.0:

        raise ValueError(
            "lambda_pv must be >= 1.0."
        )

    zone_map = np.asarray(
        zone_map
    )

    weights = np.ones(
        zone_map.shape,
        dtype=np.float32,
    )

    weights[
        zone_map == PV_ZONE_LABEL
    ] = float(
        lambda_pv
    )

    return weights


# ============================================================
# CREATE PRAL WEIGHT MAP — TORCH VERSION
# ============================================================

def build_pral_weight_map(
    zone_map: torch.Tensor,
    lambda_pv: float = DEFAULT_LAMBDA_PV,
) -> torch.Tensor:
    """
    Convert Stage-15 zone labels into voxel weights.

    Input
    -----
    zone_map:
        [B, H, W]
        or
        [H, W]

        Labels:
            0 = ventricle
            1 = periventricular
            2 = deep

    Output
    ------
    weights:
        Same spatial shape as zone_map.

        PV voxels:
            lambda_pv

        All other voxels:
            1.0
    """

    if lambda_pv < 1.0:

        raise ValueError(
            "lambda_pv must be >= 1.0."
        )

    weights = torch.ones_like(
        zone_map,
        dtype=torch.float32,
    )

    pv_mask = (
        zone_map
        == PV_ZONE_LABEL
    )

    weights = torch.where(

        pv_mask,

        torch.as_tensor(
            lambda_pv,
            dtype=torch.float32,
            device=zone_map.device,
        ),

        weights,
    )

    return weights


# ============================================================
# WEIGHTED CROSS ENTROPY
# ============================================================

class PRALCrossEntropyLoss(
    nn.Module
):
    """
    Periventricular-region-aware cross entropy.

    Cross entropy is calculated with reduction='none',
    multiplied voxelwise by the PRAL weight map,
    and only then reduced.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        lambda_pv: float = DEFAULT_LAMBDA_PV,
    ):

        super().__init__()

        self.lambda_pv = float(
            lambda_pv
        )

        if class_weights is not None:

            self.register_buffer(
                "class_weights",
                class_weights.detach().clone(),
            )

        else:

            self.class_weights = None


    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        zone_map: torch.Tensor,
    ) -> torch.Tensor:

        if logits.ndim != 4:

            raise ValueError(
                "logits must have shape "
                "[B, C, H, W]."
            )

        if target.ndim != 3:

            raise ValueError(
                "target must have shape "
                "[B, H, W]."
            )

        if zone_map.ndim != 3:

            raise ValueError(
                "zone_map must have shape "
                "[B, H, W]."
            )

        if target.shape != zone_map.shape:

            raise ValueError(
                "target and zone_map must have "
                "identical shapes."
            )

        if logits.shape[
            0
        ] != target.shape[
            0
        ]:

            raise ValueError(
                "Batch size mismatch."
            )

        if logits.shape[
            2:
        ] != target.shape[
            1:
        ]:

            raise ValueError(
                "Spatial shape mismatch."
            )

        # ====================================================
        # STANDARD CE BUT KEEP EVERY VOXEL
        # ====================================================

        ce_voxel = F.cross_entropy(

            logits,

            target,

            weight=self.class_weights,

            reduction="none",
        )

        # ce_voxel:
        # [B, H, W]

        spatial_weights = (
            build_pral_weight_map(
                zone_map,
                lambda_pv=self.lambda_pv,
            )
        )

        # ====================================================
        # APPLY PV WEIGHT
        # ====================================================

        weighted_ce = (
            ce_voxel
            * spatial_weights
        )



        if self.class_weights is not None:

            target_class_weights = (
                self.class_weights[
                    target
                ]
        )

            effective_weights = (
                target_class_weights
                * spatial_weights
            )

        else:
            effective_weights = (
             spatial_weights
         )

        loss = (
            weighted_ce.sum()
         /
            effective_weights.sum().clamp_min(
                1e-8
            )
        )



        return loss


# ============================================================
# WEIGHTED MULTICLASS DICE LOSS
# ============================================================

class PRALDiceLoss(
    nn.Module
):
    """
    Region-aware multiclass soft Dice.

    The PRAL voxel weights are inserted into both
    the Dice intersection and denominator.

    This is preferable to multiplying a scalar Dice
    loss after Dice has already been reduced.
    """

    def __init__(
        self,
        lambda_pv: float = DEFAULT_LAMBDA_PV,
        include_background: bool = False,
        smooth: float = SMOOTH,
    ):

        super().__init__()

        self.lambda_pv = float(
            lambda_pv
        )

        self.include_background = bool(
            include_background
        )

        self.smooth = float(
            smooth
        )


    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        zone_map: torch.Tensor,
    ) -> torch.Tensor:

        if logits.ndim != 4:

            raise ValueError(
                "logits must have shape "
                "[B, C, H, W]."
            )

        if target.ndim != 3:

            raise ValueError(
                "target must have shape "
                "[B, H, W]."
            )

        if zone_map.ndim != 3:

            raise ValueError(
                "zone_map must have shape "
                "[B, H, W]."
            )

        if target.shape != zone_map.shape:

            raise ValueError(
                "target and zone_map shapes differ."
            )

        probs = torch.softmax(
            logits,
            dim=1,
        )

        one_hot = (
            F.one_hot(
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

        spatial_weights = (
            build_pral_weight_map(
                zone_map,
                lambda_pv=self.lambda_pv,
            )
            .unsqueeze(
                1
            )
        )

        # spatial_weights:
        # [B, 1, H, W]

        start_class = (
            0
            if self.include_background
            else 1
        )

        losses = []

        for cls in range(
            start_class,
            NUM_CLASSES,
        ):

            p = probs[
                :,
                cls:
                cls + 1,
                :,
                :,
            ]

            t = one_hot[
                :,
                cls:
                cls + 1,
                :,
                :,
            ]

            weighted_intersection = torch.sum(

                spatial_weights
                * p
                * t,

                dim=(
                    2,
                    3,
                ),
            )

            weighted_prediction = torch.sum(

                spatial_weights
                * p,

                dim=(
                    2,
                    3,
                ),
            )

            weighted_target = torch.sum(

                spatial_weights
                * t,

                dim=(
                    2,
                    3,
                ),
            )

            dice = (

                2.0
                * weighted_intersection
                + self.smooth

            ) / (

                weighted_prediction
                + weighted_target
                + self.smooth
            )

            class_loss = (
                1.0
                - dice
            )

            losses.append(
                class_loss
            )

        if not losses:

            raise RuntimeError(
                "No classes selected for Dice loss."
            )

        return torch.cat(
            losses,
            dim=1,
        ).mean()


# ============================================================
# COMPLETE PRAL LOSS
# ============================================================

class PRALLoss(
    nn.Module
):
    """
    Periventricular Region-Aware Loss (PRAL)

    total =
        CE_LOSS_WEIGHT   * weighted CE
        +
        DICE_LOSS_WEIGHT * weighted Dice

    Both components use the Stage-15 PV-zone map.

    Inputs
    ------
    logits:
        [B, 3, H, W]

    target:
        [B, H, W]
        0 background
        1 normal WMH
        2 abnormal WMH

    zone_map:
        [B, H, W]
        0 ventricle
        1 PV zone
        2 deep zone
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        lambda_pv: float = DEFAULT_LAMBDA_PV,
        ce_weight: float = CE_LOSS_WEIGHT,
        dice_weight: float = DICE_LOSS_WEIGHT,
        include_background_in_dice: bool = (
            INCLUDE_BACKGROUND_IN_DICE
        ),
    ):

        super().__init__()

        if lambda_pv < 1.0:

            raise ValueError(
                "lambda_pv must be >= 1.0."
            )

        if ce_weight < 0:

            raise ValueError(
                "ce_weight must be >= 0."
            )

        if dice_weight < 0:

            raise ValueError(
                "dice_weight must be >= 0."
            )

        if (
            ce_weight == 0
            and
            dice_weight == 0
        ):

            raise ValueError(
                "At least one loss component "
                "must have non-zero weight."
            )

        self.lambda_pv = float(
            lambda_pv
        )

        self.ce_weight = float(
            ce_weight
        )

        self.dice_weight = float(
            dice_weight
        )

        self.ce = PRALCrossEntropyLoss(

            class_weights=class_weights,

            lambda_pv=lambda_pv,
        )

        self.dice = PRALDiceLoss(

            lambda_pv=lambda_pv,

            include_background=(
                include_background_in_dice
            ),
        )


    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        zone_map: torch.Tensor,
    ):

        ce_loss = self.ce(

            logits,

            target,

            zone_map,
        )

        dice_loss = self.dice(

            logits,

            target,

            zone_map,
        )

        total = (

            self.ce_weight
            * ce_loss

            +

            self.dice_weight
            * dice_loss
        )

        return (
            total,
            ce_loss,
            dice_loss,
        )


# ============================================================
# SANITY TEST — ACTUAL PATIENT
# ============================================================

def sanity_check_patient(
    patient_id: str,
    lambda_pv: float,
):

    patient_id = str(
        patient_id
    ).zfill(3)

    print()
    print(
        "=" * 80
    )

    print(
        "PRAL SANITY CHECK"
    )

    print(
        "=" * 80
    )

    print(
        "Patient:",
        patient_id,
    )

    print(
        "lambda_pv:",
        lambda_pv,
    )

    print()

    zone_map, zone_img = (
        load_pv_deep_zone_map(
            patient_id
        )
    )

    weights = build_pral_weight_map_numpy(

        zone_map,

        lambda_pv=lambda_pv,
    )

    # ========================================================
    # COUNTS
    # ========================================================

    vent_mask = (
        zone_map
        == VENTRICLE_ZONE_LABEL
    )

    pv_mask = (
        zone_map
        == PV_ZONE_LABEL
    )

    deep_mask = (
        zone_map
        == DEEP_ZONE_LABEL
    )

    n_total = int(
        zone_map.size
    )

    n_vent = int(
        np.count_nonzero(
            vent_mask
        )
    )

    n_pv = int(
        np.count_nonzero(
            pv_mask
        )
    )

    n_deep = int(
        np.count_nonzero(
            deep_mask
        )
    )

    print(
        "Zone-map shape:",
        zone_map.shape,
    )

    print(
        "Voxel spacing:",
        tuple(
            float(x)
            for x in zone_img.header.get_zooms()[
                :3
            ]
        ),
    )

    print()

    print(
        "Voxel counts:"
    )

    print(
        f"  total      : {n_total:,}"
    )

    print(
        f"  ventricle  : {n_vent:,}"
    )

    print(
        f"  PV <=10 mm : {n_pv:,}"
    )

    print(
        f"  deep >10 mm: {n_deep:,}"
    )

    print()

    print(
        "Zone percentages:"
    )

    print(
        f"  PV   : "
        f"{100.0 * n_pv / n_total:.3f}%"
    )

    print(
        f"  deep : "
        f"{100.0 * n_deep / n_total:.3f}%"
    )

    # ========================================================
    # WEIGHT VALIDATION
    # ========================================================

    if n_pv == 0:

        raise RuntimeError(
            "No PV voxels found."
        )

    if n_deep == 0:

        raise RuntimeError(
            "No deep-zone voxels found."
        )

    pv_weights = weights[
        pv_mask
    ]

    deep_weights = weights[
        deep_mask
    ]

    vent_weights = weights[
        vent_mask
    ]

    pv_mean = float(
        np.mean(
            pv_weights
        )
    )

    deep_mean = float(
        np.mean(
            deep_weights
        )
    )

    vent_mean = (

        float(
            np.mean(
                vent_weights
            )
        )

        if n_vent > 0

        else math.nan
    )

    print()
    print(
        "Weight statistics:"
    )

    print(
        f"  PV mean weight       : "
        f"{pv_mean:.6f}"
    )

    print(
        f"  deep mean weight     : "
        f"{deep_mean:.6f}"
    )

    print(
        f"  ventricle mean weight: "
        f"{vent_mean:.6f}"
    )

    print(
        f"  global min weight    : "
        f"{float(weights.min()):.6f}"
    )

    print(
        f"  global max weight    : "
        f"{float(weights.max()):.6f}"
    )

    # ========================================================
    # HARD ASSERTIONS
    # ========================================================

    if not np.allclose(

        pv_weights,

        lambda_pv,
    ):

        raise AssertionError(
            "PV voxels do not all have "
            "lambda_pv weight."
        )

    if not np.allclose(

        deep_weights,

        1.0,
    ):

        raise AssertionError(
            "Deep voxels do not all have "
            "weight 1."
        )

    if (
        n_vent > 0
        and
        not np.allclose(
            vent_weights,
            1.0,
        )
    ):

        raise AssertionError(
            "Ventricle voxels do not have "
            "weight 1."
        )

    if not (
        pv_mean
        >
        deep_mean
    ):

        raise AssertionError(
            "PV mean weight is not greater "
            "than deep mean weight."
        )

    # ========================================================
    # TORCH UNIT TEST
    # ========================================================

    print()
    print(
        "Running PyTorch unit test..."
    )

    # Find a slice containing PV voxels.

    pv_per_slice = np.count_nonzero(
        pv_mask,
        axis=(
            0,
            1,
        ),
    )

    slice_index = int(
        np.argmax(
            pv_per_slice
        )
    )

    zone_slice = (
        zone_map[
            :,
            :,
            slice_index
        ]
        .astype(
            np.int64
        )
    )

    H, W = zone_slice.shape

    torch.manual_seed(
        12345
    )

    logits = torch.randn(
        1,
        NUM_CLASSES,
        H,
        W,
        dtype=torch.float32,
        requires_grad=True,
    )

    target = torch.randint(
        low=0,
        high=NUM_CLASSES,
        size=(
            1,
            H,
            W,
        ),
        dtype=torch.long,
    )

    zone_tensor = torch.from_numpy(
        zone_slice
    ).unsqueeze(
        0
    )

    criterion = PRALLoss(
        lambda_pv=lambda_pv,
    )

    (
        total_loss,
        ce_loss,
        dice_loss,
    ) = criterion(

        logits,

        target,

        zone_tensor,
    )

    if not torch.isfinite(
        total_loss
    ):

        raise AssertionError(
            "Total PRAL loss is non-finite."
        )

    if not torch.isfinite(
        ce_loss
    ):

        raise AssertionError(
            "PRAL CE loss is non-finite."
        )

    if not torch.isfinite(
        dice_loss
    ):

        raise AssertionError(
            "PRAL Dice loss is non-finite."
        )

    total_loss.backward()

    if logits.grad is None:

        raise AssertionError(
            "No gradient produced."
        )

    if not torch.all(
        torch.isfinite(
            logits.grad
        )
    ):

        raise AssertionError(
            "Gradient contains NaN/Inf."
        )

    torch_weights = build_pral_weight_map(
        zone_tensor,
        lambda_pv=lambda_pv,
    )

    pv_tensor = (
        zone_tensor
        == PV_ZONE_LABEL
    )

    non_pv_tensor = (
        zone_tensor
        != PV_ZONE_LABEL
    )

    pv_torch_mean = float(
        torch_weights[
            pv_tensor
        ].mean()
    )

    outside_torch_mean = float(
        torch_weights[
            non_pv_tensor
        ].mean()
    )

    print(
        f"Selected slice: {slice_index}"
    )

    print(
        f"PV voxels in slice: "
        f"{int(pv_tensor.sum()):,}"
    )

    print(
        f"PV tensor mean weight: "
        f"{pv_torch_mean:.6f}"
    )

    print(
        f"Outside-PV mean weight: "
        f"{outside_torch_mean:.6f}"
    )

    print()

    print(
        "Synthetic forward/backward:"
    )

    print(
        f"  total loss : "
        f"{float(total_loss.detach()):.6f}"
    )

    print(
        f"  weighted CE: "
        f"{float(ce_loss.detach()):.6f}"
    )

    print(
        f"  weighted Dice: "
        f"{float(dice_loss.detach()):.6f}"
    )

    print(
        f"  gradient finite: YES"
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        "PRAL SANITY CHECK: PASS"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Confirmed:"
    )

    print(
        f"  PV <=10 mm weight = {lambda_pv}"
    )

    print(
        "  Outside PV weight = 1.0"
    )

    print(
        "  Weight map geometry = valid"
    )

    print(
        "  PRAL CE forward = valid"
    )

    print(
        "  PRAL Dice forward = valid"
    )

    print(
        "  Backpropagation = valid"
    )

    print()

    print(
        "The module is ready to be imported "
        "into the training script."
    )


# ============================================================
# SIMPLE SYNTHETIC UNIT TEST
# ============================================================

def synthetic_unit_test(
    lambda_pv: float,
):

    print()
    print(
        "=" * 80
    )

    print(
        "SYNTHETIC PRAL UNIT TEST"
    )

    print(
        "=" * 80
    )

    zone_map = torch.tensor(
        [
            [
                [0, 1, 1, 2],
                [0, 1, 2, 2],
                [2, 2, 2, 2],
                [1, 1, 2, 0],
            ]
        ],
        dtype=torch.long,
    )

    weights = build_pral_weight_map(
        zone_map,
        lambda_pv=lambda_pv,
    )

    expected = torch.tensor(
        [
            [
                [1, lambda_pv, lambda_pv, 1],
                [1, lambda_pv, 1, 1],
                [1, 1, 1, 1],
                [lambda_pv, lambda_pv, 1, 1],
            ]
        ],
        dtype=torch.float32,
    )

    if not torch.allclose(
        weights,
        expected,
    ):

        raise AssertionError(
            "Synthetic PRAL weight map failed."
        )

    logits = torch.randn(
        1,
        NUM_CLASSES,
        4,
        4,
        requires_grad=True,
    )

    target = torch.randint(
        0,
        NUM_CLASSES,
        (
            1,
            4,
            4,
        ),
        dtype=torch.long,
    )

    criterion = PRALLoss(
        lambda_pv=lambda_pv,
    )

    total, ce, dice = criterion(

        logits,

        target,

        zone_map,
    )

    if not torch.isfinite(
        total
    ):

        raise AssertionError(
            "Synthetic PRAL total loss "
            "is non-finite."
        )

    total.backward()

    if logits.grad is None:

        raise AssertionError(
            "Synthetic PRAL backward failed."
        )

    print()
    print(
        "Weight map:"
    )

    print(
        weights
    )

    print()

    print(
        f"Total loss : "
        f"{float(total.detach()):.6f}"
    )

    print(
        f"CE loss    : "
        f"{float(ce.detach()):.6f}"
    )

    print(
        f"Dice loss  : "
        f"{float(dice.detach()):.6f}"
    )

    print()

    print(
        "SYNTHETIC UNIT TEST: PASS"
    )
    print()
print(
    "Testing lambda=1 CE equivalence..."
)

torch.manual_seed(
    999
)

test_logits = torch.randn(
    2,
    NUM_CLASSES,
    8,
    8,
    requires_grad=True,
)

test_target = torch.randint(
    0,
    NUM_CLASSES,
    (
        2,
        8,
        8,
    ),
    dtype=torch.long,
)

test_zone = torch.randint(
    0,
    3,
    (
        2,
        8,
        8,
    ),
    dtype=torch.long,
)

test_class_weights = torch.tensor(
    [
        0.10,
        1.60,
        1.30,
    ],
    dtype=torch.float32,
)

baseline_ce = F.cross_entropy(
    test_logits,
    test_target,
    weight=test_class_weights,
    reduction="mean",
)

pral_ce_lambda1 = PRALCrossEntropyLoss(
    class_weights=test_class_weights,
    lambda_pv=1.0,
)

pral_ce_value = pral_ce_lambda1(
    test_logits,
    test_target,
    test_zone,
)

difference = abs(
    float(
        baseline_ce.detach()
        -
        pral_ce_value.detach()
    )
)

print(
    f"Baseline weighted CE : "
    f"{float(baseline_ce.detach()):.8f}"
)

print(
    f"PRAL CE lambda=1    : "
    f"{float(pral_ce_value.detach()):.8f}"
)

print(
    f"Absolute difference : "
    f"{difference:.12f}"
)

if not torch.allclose(
    baseline_ce,
    pral_ce_value,
    atol=1e-6,
    rtol=1e-6,
):

    raise AssertionError(
        "lambda=1 PRAL CE does not match "
        "baseline weighted CE."
    )

print(
    "lambda=1 CE equivalence: PASS"
)


# ============================================================
# COMMAND LINE
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(

        description=(
            "Standalone Periventricular "
            "Region-Aware Loss (PRAL) module "
            "and Stage-15 sanity test."
        )
    )

    parser.add_argument(

        "--patient",

        type=str,

        default=None,

        help=(
            "Patient ID for real Stage-15 "
            "sanity check, e.g. 001."
        ),
    )

    parser.add_argument(

        "--lambda-pv",

        type=float,

        default=DEFAULT_LAMBDA_PV,

        help=(
            "Weight assigned to voxels "
            "inside the <=10 mm PV band. "
            "Default: 3.0"
        ),
    )

    parser.add_argument(

        "--synthetic",

        action="store_true",

        help=(
            "Run the small synthetic "
            "unit test."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    if args.lambda_pv < 1.0:

        raise ValueError(
            "--lambda-pv must be >= 1."
        )

    ran_anything = False

    if args.synthetic:

        synthetic_unit_test(
            lambda_pv=args.lambda_pv
        )

        ran_anything = True

    if args.patient is not None:

        sanity_check_patient(

            patient_id=args.patient,

            lambda_pv=args.lambda_pv,
        )

        ran_anything = True

    if not ran_anything:

        print()
        print(
            "PRAL module imported successfully."
        )

        print()
        print(
            "Run a synthetic test:"
        )

        print(
            "python 18_pral_loss.py "
            "--synthetic "
            "--lambda-pv 3"
        )

        print()
        print(
            "Run a real-patient sanity check:"
        )

        print(
            "python 18_pral_loss.py "
            "--patient 001 "
            "--lambda-pv 3"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print()
        print(
            "=" * 80
        )

        print(
            "PRAL TEST FAILED"
        )

        print(
            "=" * 80
        )

        print(
            repr(
                exc
            )
        )

        raise