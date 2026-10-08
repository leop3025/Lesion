from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import re
import traceback

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt


# ============================================================
# STAGE 15 CONFIG
# ============================================================

PROJECT_DIR = Path(r"C:\Users\Preethi\Documents\LESION")

# Uses the SAME preprocessed 100-patient data already used by the baseline.
PREPROCESSED_ROOT = PROJECT_DIR / "preprocessed"

# New outputs only — the existing baseline data/results are NOT changed.
OUTPUT_ROOT = PROJECT_DIR / "distance_labels"
LOG_DIR = PROJECT_DIR / "logs" / "distance_labels"

SUMMARY_CSV = LOG_DIR / "pv_deep_generation_summary.csv"
CONFIG_JSON = LOG_DIR / "pv_deep_generation_config.json"
FAILURE_LOG = LOG_DIR / "pv_deep_generation_failures.txt"

# Fixed-rule reference threshold.
# IMPORTANT: because EDT uses sampling=voxel spacing, this is 10 millimetres,
# not 10 voxels.
PV_THRESHOLD_MM = 10.0

# Continuous distance channel is clipped before normalization.
# 0 mm -> 0.0; >=50 mm -> 1.0
DISTANCE_CLIP_MM = 50.0

EXPECTED_PATIENTS = 100

# Numerical tolerances for geometry checks.
AFFINE_ATOL = 1e-4
SPACING_ATOL = 1e-5


# ============================================================
# FILE DISCOVERY
# ============================================================

def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def find_exactly_one(
    patient_dir: Path,
    include_tokens,
    exclude_tokens=(),
    description="file",
):
    """
    Robustly locate one NIfTI using filename tokens.
    Search is recursive within a patient folder.
    """
    include_tokens = [t.lower() for t in include_tokens]
    exclude_tokens = [t.lower() for t in exclude_tokens]

    candidates = []

    for path in patient_dir.rglob("*"):
        if not path.is_file() or not is_nifti(path):
            continue

        name = path.name.lower()

        if all(token in name for token in include_tokens) and not any(
            token in name for token in exclude_tokens
        ):
            candidates.append(path)

    # Prefer filenames containing "mask" for annotation files.
    if len(candidates) > 1 and description != "FLAIR":
        mask_candidates = [
            p for p in candidates
            if "mask" in p.name.lower()
        ]
        if len(mask_candidates) == 1:
            candidates = mask_candidates

    if len(candidates) != 1:
        found = [str(p) for p in candidates]
        raise RuntimeError(
            f"{patient_dir.name}: expected exactly one {description}; "
            f"found {len(candidates)}.\nCandidates: {found}"
        )

    return candidates[0]


def find_patient_files(patient_dir: Path):
    """
    Adapted to the naming convention already used in the project.

    If your preprocessed masks use slightly different names, only this
    function should need adjustment.
    """
    vent = find_exactly_one(
        patient_dir,
        include_tokens=["vent"],
        description="ventricle mask",
    )

    # IMPORTANT:
    # search abnormal first and exclude 'ab' from normal search so that
    # 'abWMH' is never accidentally selected as nWMH.
    abwmh = find_exactly_one(
        patient_dir,
        include_tokens=["abwmh"],
        description="abnormal-WMH mask",
    )

    try:
        nwmh = find_exactly_one(
            patient_dir,
            include_tokens=["nwmh"],
            exclude_tokens=["abwmh"],
            description="normal-WMH mask",
        )
    except RuntimeError:
        # Fallback for filenames spelling out "normal".
        nwmh = find_exactly_one(
            patient_dir,
            include_tokens=["normal", "wmh"],
            exclude_tokens=["abnormal"],
            description="normal-WMH mask",
        )

    return {
        "vent": vent,
        "nwmh": nwmh,
        "abwmh": abwmh,
    }


# ============================================================
# GEOMETRY / NIFTI HELPERS
# ============================================================

def load_binary_mask(path: Path):
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)

    if data.ndim != 3:
        raise RuntimeError(
            f"{path}: expected 3D mask, got shape {data.shape}"
        )

    mask = data > 0

    return img, mask


def get_spacing_mm(img):
    spacing = tuple(
        float(v)
        for v in img.header.get_zooms()[:3]
    )

    if len(spacing) != 3 or any(
        (not np.isfinite(v)) or v <= 0
        for v in spacing
    ):
        raise RuntimeError(
            f"Invalid voxel spacing: {spacing}"
        )

    return spacing


def validate_same_geometry(
    reference_img,
    other_img,
    reference_name,
    other_name,
):
    if reference_img.shape[:3] != other_img.shape[:3]:
        raise RuntimeError(
            f"Geometry mismatch: {reference_name} shape "
            f"{reference_img.shape[:3]} != {other_name} shape "
            f"{other_img.shape[:3]}"
        )

    if not np.allclose(
        reference_img.affine,
        other_img.affine,
        atol=AFFINE_ATOL,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"Geometry mismatch: affine differs between "
            f"{reference_name} and {other_name}"
        )

    ref_spacing = np.asarray(
        reference_img.header.get_zooms()[:3],
        dtype=float,
    )

    other_spacing = np.asarray(
        other_img.header.get_zooms()[:3],
        dtype=float,
    )

    if not np.allclose(
        ref_spacing,
        other_spacing,
        atol=SPACING_ATOL,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"Geometry mismatch: spacing differs between "
            f"{reference_name} {tuple(ref_spacing)} and "
            f"{other_name} {tuple(other_spacing)}"
        )


def save_like_reference(
    array,
    reference_img,
    output_path: Path,
    dtype,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = reference_img.header.copy()
    header.set_data_dtype(dtype)

    out_img = nib.Nifti1Image(
        np.asarray(array, dtype=dtype),
        affine=reference_img.affine,
        header=header,
    )

    # Preserve qform/sform when possible.
    qform, qcode = reference_img.get_qform(coded=True)
    sform, scode = reference_img.get_sform(coded=True)

    if qform is not None:
        out_img.set_qform(qform, int(qcode))

    if sform is not None:
        out_img.set_sform(sform, int(scode))

    nib.save(
        out_img,
        str(output_path),
    )


# ============================================================
# DISTANCE + LABEL ENGINEERING
# ============================================================

def compute_distance_to_ventricle_mm(
    vent_mask,
    spacing_mm,
):
    """
    Distance from every non-ventricle voxel to the nearest ventricle voxel.

    CRITICAL:
    sampling=spacing_mm makes the EDT physically correct in millimetres.
    Without this, a threshold of 10 would mean 10 voxels instead of 10 mm.
    """
    if not vent_mask.any():
        raise RuntimeError(
            "Ventricle mask is empty; distance transform is undefined "
            "for this use."
        )

    # distance_transform_edt calculates distance for non-zero elements
    # to the nearest zero. Therefore ~vent_mask gives:
    #   ventricle voxels = 0
    #   everything else  = 1
    distance_mm = distance_transform_edt(
        ~vent_mask,
        sampling=spacing_mm,
    )

    return distance_mm.astype(
        np.float32,
        copy=False,
    )


def normalize_distance_map(
    distance_mm,
    clip_mm,
):
    clipped = np.clip(
        distance_mm,
        0.0,
        float(clip_mm),
    )

    normalized = (
        clipped / float(clip_mm)
    )

    return normalized.astype(
        np.float32,
        copy=False,
    )


def create_pv_deep_zone_map(
    vent_mask,
    distance_mm,
    threshold_mm,
):
    """
    Region map independent of lesion class.

    Labels:
      0 = ventricle
      1 = PV zone   (>0 mm and <= threshold)
      2 = deep zone (> threshold)

    NOTE:
    This is a distance-defined spatial zone map, not a true anatomical
    whole-white-matter segmentation because this project does not provide
    a separate whole-WM mask.
    """
    region = np.zeros(
        vent_mask.shape,
        dtype=np.uint8,
    )

    pv = (
        (~vent_mask)
        & (distance_mm > 0.0)
        & (distance_mm <= threshold_mm)
    )

    deep = (
        (~vent_mask)
        & (distance_mm > threshold_mm)
    )

    region[pv] = 1
    region[deep] = 2

    return region


def create_wmh_pv_deep_map(
    vent_mask,
    nwmh_mask,
    abwmh_mask,
    distance_mm,
    threshold_mm,
):
    """
    Labels only WMH voxels by region.

    Labels:
      0 = non-WMH / ventricle
      1 = PV WMH
      2 = deep WMH
    """
    wmh = (
        nwmh_mask
        | abwmh_mask
    ) & (~vent_mask)

    result = np.zeros(
        vent_mask.shape,
        dtype=np.uint8,
    )

    result[
        wmh
        & (distance_mm <= threshold_mm)
    ] = 1

    result[
        wmh
        & (distance_mm > threshold_mm)
    ] = 2

    return result


def create_6class_label_map(
    vent_mask,
    nwmh_mask,
    abwmh_mask,
    distance_mm,
    threshold_mm,
):
    """
    Six-class spatially resolved target map.

    Labels:
      0 = background
      1 = ventricle
      2 = normal-WMH PV
      3 = normal-WMH deep
      4 = abnormal-WMH PV
      5 = abnormal-WMH deep

    If nWMH and abWMH overlap, abnormal-WMH is given priority.
    Ventricle is finally assigned last so ventricle always has priority.
    """
    label = np.zeros(
        vent_mask.shape,
        dtype=np.uint8,
    )

    valid_nwmh = (
        nwmh_mask
        & (~vent_mask)
    )

    valid_abwmh = (
        abwmh_mask
        & (~vent_mask)
    )

    pv = distance_mm <= threshold_mm
    deep = distance_mm > threshold_mm

    label[
        valid_nwmh & pv
    ] = 2

    label[
        valid_nwmh & deep
    ] = 3

    # abnormal-WMH overwrites any nWMH overlap
    label[
        valid_abwmh & pv
    ] = 4

    label[
        valid_abwmh & deep
    ] = 5

    # ventricle has final priority
    label[
        vent_mask
    ] = 1

    return label


# ============================================================
# PER-PATIENT PROCESSING
# ============================================================

def process_patient(
    patient_id,
    patient_dir,
):
    files = find_patient_files(
        patient_dir
    )

    vent_img, vent = load_binary_mask(
        files["vent"]
    )

    nwmh_img, nwmh = load_binary_mask(
        files["nwmh"]
    )

    abwmh_img, abwmh = load_binary_mask(
        files["abwmh"]
    )

    validate_same_geometry(
        vent_img,
        nwmh_img,
        "ventricle mask",
        "normal-WMH mask",
    )

    validate_same_geometry(
        vent_img,
        abwmh_img,
        "ventricle mask",
        "abnormal-WMH mask",
    )

    spacing_mm = get_spacing_mm(
        vent_img
    )

    distance_mm = (
        compute_distance_to_ventricle_mm(
            vent,
            spacing_mm,
        )
    )

    distance_normalized = (
        normalize_distance_map(
            distance_mm,
            DISTANCE_CLIP_MM,
        )
    )

    pv_deep_zone = (
        create_pv_deep_zone_map(
            vent,
            distance_mm,
            PV_THRESHOLD_MM,
        )
    )

    wmh_pv_deep = (
        create_wmh_pv_deep_map(
            vent,
            nwmh,
            abwmh,
            distance_mm,
            PV_THRESHOLD_MM,
        )
    )

    labels_6class = (
        create_6class_label_map(
            vent,
            nwmh,
            abwmh,
            distance_mm,
            PV_THRESHOLD_MM,
        )
    )

    out_dir = (
        OUTPUT_ROOT
        / patient_id
    )

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    save_like_reference(
        distance_mm,
        vent_img,
        out_dir
        / f"{patient_id}_distance_to_ventricle_mm.nii.gz",
        np.float32,
    )

    save_like_reference(
        distance_normalized,
        vent_img,
        out_dir
        / f"{patient_id}_distance_to_ventricle_normalized.nii.gz",
        np.float32,
    )

    save_like_reference(
        pv_deep_zone,
        vent_img,
        out_dir
        / f"{patient_id}_pv_deep_zone_map.nii.gz",
        np.uint8,
    )

    save_like_reference(
        wmh_pv_deep,
        vent_img,
        out_dir
        / f"{patient_id}_wmh_pv_deep_labels.nii.gz",
        np.uint8,
    )

    save_like_reference(
        labels_6class,
        vent_img,
        out_dir
        / f"{patient_id}_6class_pv_deep_labels.nii.gz",
        np.uint8,
    )

    # --------------------------------------------------------
    # COUNTS / QC
    # --------------------------------------------------------

    nwmh_abwmh_overlap = int(
        np.count_nonzero(
            nwmh & abwmh
        )
    )

    vent_wmh_overlap = int(
        np.count_nonzero(
            vent
            & (
                nwmh
                | abwmh
            )
        )
    )

    counts_6 = {
        cls: int(
            np.count_nonzero(
                labels_6class == cls
            )
        )
        for cls in range(6)
    }

    max_distance_mm = float(
        np.max(
            distance_mm
        )
    )

    row = {
        "patient_id": patient_id,

        "spacing_x_mm": spacing_mm[0],
        "spacing_y_mm": spacing_mm[1],
        "spacing_z_mm": spacing_mm[2],

        "ventricle_voxels": int(
            np.count_nonzero(
                vent
            )
        ),

        "nwmh_voxels_original": int(
            np.count_nonzero(
                nwmh
            )
        ),

        "abwmh_voxels_original": int(
            np.count_nonzero(
                abwmh
            )
        ),

        "nwmh_abwmh_overlap_voxels": (
            nwmh_abwmh_overlap
        ),

        "ventricle_wmh_overlap_voxels": (
            vent_wmh_overlap
        ),

        "max_distance_mm": (
            max_distance_mm
        ),

        "class0_background_voxels": (
            counts_6[0]
        ),

        "class1_ventricle_voxels": (
            counts_6[1]
        ),

        "class2_nwmh_pv_voxels": (
            counts_6[2]
        ),

        "class3_nwmh_deep_voxels": (
            counts_6[3]
        ),

        "class4_abwmh_pv_voxels": (
            counts_6[4]
        ),

        "class5_abwmh_deep_voxels": (
            counts_6[5]
        ),

        "pv_threshold_mm": (
            PV_THRESHOLD_MM
        ),

        "distance_clip_mm": (
            DISTANCE_CLIP_MM
        ),

        "status": "PASS",
    }

    return row


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 88)
    print(
        "STAGE 15 — GENERATE DISTANCE-TO-VENTRICLE + PV/DEEP LABELS"
    )
    print("=" * 88)

    print()
    print(
        "Input:",
        PREPROCESSED_ROOT,
    )

    print(
        "Output:",
        OUTPUT_ROOT,
    )

    print()
    print(
        f"PV threshold: {PV_THRESHOLD_MM:.1f} mm"
    )

    print(
        f"Normalized distance clip: {DISTANCE_CLIP_MM:.1f} mm"
    )

    print()
    print(
        "IMPORTANT: EDT uses physical voxel spacing, "
        "so the 10 mm threshold is truly 10 mm."
    )

    if not PREPROCESSED_ROOT.exists():
        raise FileNotFoundError(
            f"Preprocessed root not found: {PREPROCESSED_ROOT}"
        )

    patient_dirs = sorted(
        [
            p
            for p in PREPROCESSED_ROOT.iterdir()
            if (
                p.is_dir()
                and re.fullmatch(
                    r"\d{3}",
                    p.name,
                )
            )
        ],
        key=lambda p: p.name,
    )

    if len(patient_dirs) != EXPECTED_PATIENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_PATIENTS} patient folders "
            f"under {PREPROCESSED_ROOT}, found {len(patient_dirs)}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "stage": 15,

        "input_root": str(
            PREPROCESSED_ROOT
        ),

        "output_root": str(
            OUTPUT_ROOT
        ),

        "expected_patients": (
            EXPECTED_PATIENTS
        ),

        "pv_threshold_mm": (
            PV_THRESHOLD_MM
        ),

        "distance_clip_mm": (
            DISTANCE_CLIP_MM
        ),

        "edt_sampling": (
            "physical voxel spacing in mm "
            "from the ventricle NIfTI header"
        ),

        "outputs_per_patient": {
            "distance_mm": (
                "<id>_distance_to_ventricle_mm.nii.gz"
            ),

            "distance_normalized": (
                "<id>_distance_to_ventricle_normalized.nii.gz"
            ),

            "pv_deep_zone_map": {
                "0": "ventricle",
                "1": (
                    f"PV zone: >0 to <= "
                    f"{PV_THRESHOLD_MM:g} mm"
                ),
                "2": (
                    f"deep zone: > "
                    f"{PV_THRESHOLD_MM:g} mm"
                ),
            },

            "wmh_pv_deep_labels": {
                "0": "non-WMH / ventricle",
                "1": "PV WMH",
                "2": "deep WMH",
            },

            "six_class_labels": {
                "0": "background",
                "1": "ventricle",
                "2": "normal-WMH PV",
                "3": "normal-WMH deep",
                "4": "abnormal-WMH PV",
                "5": "abnormal-WMH deep",
            },
        },

        "note": (
            "PV/deep zones are distance-defined spatial zones. "
            "They are not a whole-white-matter anatomical segmentation "
            "because no independent whole-WM mask is used."
        ),
    }

    CONFIG_JSON.write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = []
    failures = []

    for index, patient_dir in enumerate(
        patient_dirs,
        start=1,
    ):
        patient_id = patient_dir.name

        try:
            row = process_patient(
                patient_id,
                patient_dir,
            )

            rows.append(
                row
            )

            print(
                f"[{index:03d}/{len(patient_dirs):03d}] "
                f"{patient_id} PASS | "
                f"nWMH-PV={row['class2_nwmh_pv_voxels']:,} | "
                f"nWMH-deep={row['class3_nwmh_deep_voxels']:,} | "
                f"abWMH-PV={row['class4_abwmh_pv_voxels']:,} | "
                f"abWMH-deep={row['class5_abwmh_deep_voxels']:,}"
            )

        except Exception as exc:
            failures.append(
                {
                    "patient_id": patient_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

            print(
                f"[{index:03d}/{len(patient_dirs):03d}] "
                f"{patient_id} FAILED: {exc}"
            )

    # --------------------------------------------------------
    # SUMMARY CSV
    # --------------------------------------------------------

    if rows:
        fieldnames = list(
            rows[0].keys()
        )

        with SUMMARY_CSV.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

    # --------------------------------------------------------
    # FAILURE LOG
    # --------------------------------------------------------

    if failures:
        failure_text = "\n\n".join(
            [
                (
                    "=" * 88
                    + "\n"
                    + f"PATIENT {item['patient_id']} FAILED\n"
                    + "=" * 88
                    + "\n"
                    + item["error"]
                    + "\n\n"
                    + item["traceback"]
                )
                for item in failures
            ]
        )

        FAILURE_LOG.write_text(
            failure_text,
            encoding="utf-8",
        )

    elif FAILURE_LOG.exists():
        FAILURE_LOG.unlink()

    print()
    print("=" * 88)
    print("STAGE 15 COMPLETE")
    print("=" * 88)

    print()
    print(
        f"Successful patients: {len(rows)}/{len(patient_dirs)}"
    )

    print(
        f"Failed patients    : {len(failures)}"
    )

    print()
    print(
        "Summary CSV:"
    )

    print(
        SUMMARY_CSV
    )

    print()
    print(
        "Configuration:"
    )

    print(
        CONFIG_JSON
    )

    if failures:
        print()
        print(
            "Failure log:"
        )

        print(
            FAILURE_LOG
        )

        raise RuntimeError(
            f"Stage 15 finished with {len(failures)} failed patient(s)."
        )

    if len(rows) != EXPECTED_PATIENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_PATIENTS} successful patients, "
            f"got {len(rows)}"
        )

    print()
    print(
        "All patients passed. "
        "You can now build the distance-conditioned model inputs "
        "and PV/deep regional evaluation from these outputs."
    )


if __name__ == "__main__":
    main()
