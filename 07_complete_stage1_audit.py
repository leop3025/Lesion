# ============================================================
# 07_complete_stage1_audit.py
#
# FINAL STAGE-1 AUDIT
#
# Outputs:
#   data/audit/patient_001.json
#   ...
#   data/audit/patient_100.json
#
#   data/audit/summary.csv
#   manifest.csv
#
# Terminal output is ALSO saved to:
#   logs/07_complete_stage1_audit_terminal.txt
#
# This script DOES NOT modify the MRI or mask NIfTI files.
# ============================================================

from pathlib import Path
from datetime import datetime
from collections import Counter
import contextlib
import json
import os
import sys
import traceback

import nibabel as nib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Preethi\Documents\LESION"
)

OUTPUT_ROOT = PROJECT_ROOT / "output"

AUDIT_DIR = PROJECT_ROOT / "data" / "audit"

SUMMARY_CSV = AUDIT_DIR / "summary.csv"

MANIFEST_CSV = PROJECT_ROOT / "manifest.csv"

LOG_DIR = PROJECT_ROOT / "logs"

TERMINAL_LOG = (
    LOG_DIR /
    "07_complete_stage1_audit_terminal.txt"
)

PATIENT_IDS = [
    f"{i:03d}"
    for i in range(1, 101)
]


# ------------------------------------------------------------
# Known high-confidence locations
# ------------------------------------------------------------

PREPROCESSED_ROOT = (
    OUTPUT_ROOT /
    "MS_100_patient_preprocessed" /
    "MS_100_patient_preprocessed"
)

MASK_ROOT = (
    OUTPUT_ROOT /
    "MS_100_patient_masks" /
    "MS_100_patient_masks"
)


# ============================================================
# TERMINAL + TXT LOGGING
# ============================================================

class Tee:
    """
    Print simultaneously to terminal and log file.
    """

    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

        self.terminal.flush()
        self.logfile.flush()

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


# ============================================================
# GENERAL HELPERS
# ============================================================

def now_string():
    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def make_directories():

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def safe_json_value(value):

    if isinstance(
        value,
        (
            np.integer,
        )
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
        )
    ):
        return float(value)

    if isinstance(
        value,
        np.ndarray
    ):
        return value.tolist()

    return value


def atomic_write_json(
    data,
    destination
):

    destination = Path(destination)

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            default=safe_json_value
        )

    os.replace(
        temporary,
        destination
    )


def atomic_write_dataframe(
    dataframe,
    destination
):

    destination = Path(destination)

    temporary = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    dataframe.to_csv(
        temporary,
        index=False
    )

    os.replace(
        temporary,
        destination
    )


# ============================================================
# NIFTI UTILITIES
# ============================================================

def get_direction_cosines(affine):
    """
    Derive unit direction vectors from NIfTI affine.
    """

    matrix = np.asarray(
        affine[:3, :3],
        dtype=float
    )

    spacing = np.sqrt(
        np.sum(
            matrix ** 2,
            axis=0
        )
    )

    direction = np.zeros_like(
        matrix
    )

    for i in range(3):

        if spacing[i] != 0:

            direction[:, i] = (
                matrix[:, i] /
                spacing[i]
            )

    return direction


def inspect_nifti(
    path,
    is_mask=False
):

    path = Path(path)

    result = {
        "path": str(path),
        "exists": path.exists(),
        "readable": False,
        "error": None,
    }

    if not path.exists():

        result["error"] = (
            "File does not exist"
        )

        return result

    try:

        image = nib.load(
            str(path)
        )

        header = image.header

        shape = tuple(
            int(v)
            for v in image.shape
        )

        zooms = tuple(
            float(v)
            for v in header.get_zooms()[:3]
        )

        orientation = tuple(
            str(v)
            for v in nib.aff2axcodes(
                image.affine
            )
        )

        origin = [
            float(v)
            for v in image.affine[:3, 3]
        ]

        direction = (
            get_direction_cosines(
                image.affine
            )
        )

        dtype = str(
            header.get_data_dtype()
        )

        result.update(
            {
                "readable": True,
                "shape": list(shape),
                "spacing": list(zooms),
                "orientation": list(
                    orientation
                ),
                "origin": origin,
                "direction_cosines": (
                    direction.tolist()
                ),
                "affine": (
                    np.asarray(
                        image.affine
                    ).tolist()
                ),
                "dtype": dtype,
            }
        )

        if is_mask:

            data = np.asanyarray(
                image.dataobj
            )

            values, counts = (
                np.unique(
                    data,
                    return_counts=True
                )
            )

            result[
                "unique_label_values"
            ] = [
                float(v)
                for v in values
            ]

            result[
                "voxel_counts_per_label"
            ] = {
                str(float(v)): int(c)
                for v, c
                in zip(
                    values,
                    counts
                )
            }

            nonzero = int(
                np.count_nonzero(
                    data
                )
            )

            result[
                "nonzero_voxels"
            ] = nonzero

            voxel_volume = float(
                np.prod(
                    zooms
                )
            )

            result[
                "voxel_volume_mm3"
            ] = voxel_volume

            result[
                "nonzero_volume_mm3"
            ] = (
                nonzero *
                voxel_volume
            )

        return result

    except Exception as error:

        result["error"] = str(
            error
        )

        return result


# ============================================================
# PREPROCESSING HEURISTICS
# ============================================================

def preprocessing_heuristics(
    path
):
    """
    IMPORTANT:
    These are only automated clues.

    They must NOT be interpreted as proof of skull stripping
    or N4 correction.
    """

    result = {
        "skull_stripping_best_guess": (
            "uncertain"
        ),
        "n4_best_guess": (
            "uncertain"
        ),
        "needs_visual_check": True,
        "notes": [],
    }

    if path is None:

        result["notes"].append(
            "No image available."
        )

        return result

    path = Path(path)

    name_lower = str(
        path
    ).lower()

    # --------------------------------------------------------
    # Filename/path-based clues
    # --------------------------------------------------------

    skull_keywords = [
        "brain",
        "brainmask",
        "brain_mask",
        "skullstrip",
        "skull_strip",
        "bet",
    ]

    n4_keywords = [
        "n4",
        "biascorrect",
        "bias_correct",
        "bias-correct",
        "biasfield",
        "bias_field",
    ]

    if any(
        word in name_lower
        for word in skull_keywords
    ):

        result[
            "skull_stripping_best_guess"
        ] = (
            "possibly already applied "
            "(filename/path clue)"
        )

        result["notes"].append(
            "Path contains a term commonly associated "
            "with skull-stripped data."
        )

    if any(
        word in name_lower
        for word in n4_keywords
    ):

        result[
            "n4_best_guess"
        ] = (
            "possibly already applied "
            "(filename/path clue)"
        )

        result["notes"].append(
            "Path contains a term commonly associated "
            "with N4/bias correction."
        )

    # --------------------------------------------------------
    # Intensity/background clues
    # --------------------------------------------------------

    try:

        image = nib.load(
            str(path)
        )

        data = np.asarray(
            image.dataobj,
            dtype=np.float32
        )

        finite = np.isfinite(
            data
        )

        if not np.any(
            finite
        ):

            result["notes"].append(
                "No finite voxels available "
                "for intensity heuristic."
            )

            return result

        data = data.copy()

        data[
            ~finite
        ] = 0

        # Sample if very large
        flat = data.ravel()

        if flat.size > 1_000_000:

            step = max(
                1,
                flat.size // 1_000_000
            )

            sample = flat[
                ::step
            ]

        else:

            sample = flat

        zero_fraction = float(
            np.mean(
                np.isclose(
                    sample,
                    0,
                    atol=1e-6
                )
            )
        )

        result[
            "zero_voxel_fraction"
        ] = zero_fraction

        # ----------------------------------------------------
        # Edge/background test
        # ----------------------------------------------------

        if data.ndim >= 3:

            edge_values = np.concatenate(
                [
                    data[0, :, :].ravel(),
                    data[-1, :, :].ravel(),
                    data[:, 0, :].ravel(),
                    data[:, -1, :].ravel(),
                ]
            )

            edge_zero_fraction = float(
                np.mean(
                    np.isclose(
                        edge_values,
                        0,
                        atol=1e-6
                    )
                )
            )

            result[
                "edge_zero_fraction"
            ] = edge_zero_fraction

            if (
                edge_zero_fraction >
                0.80
            ):

                result[
                    "skull_stripping_best_guess"
                ] = (
                    "possibly skull-stripped; "
                    "large zero-valued image boundary"
                )

                result["notes"].append(
                    "A large fraction of boundary voxels "
                    "are zero. This can occur after skull "
                    "stripping/cropping, but this is NOT proof."
                )

            else:

                result["notes"].append(
                    "Boundary voxels are not predominantly "
                    "zero; automated inspection cannot "
                    "confirm skull stripping."
                )

        # N4 cannot be reliably established from intensities
        result["notes"].append(
            "N4 bias correction cannot be confirmed reliably "
            "from NIfTI intensities alone."
        )

    except Exception as error:

        result["notes"].append(
            "Intensity heuristic failed: "
            + str(error)
        )

    return result


# ============================================================
# FILE DISCOVERY
# ============================================================

def all_nifti_files():

    if not OUTPUT_ROOT.exists():

        raise FileNotFoundError(
            f"OUTPUT_ROOT not found:\n"
            f"{OUTPUT_ROOT}"
        )

    files = []

    for extension in (
        "*.nii",
        "*.nii.gz",
    ):

        files.extend(
            OUTPUT_ROOT.rglob(
                extension
            )
        )

    # Remove duplicates and sort
    files = sorted(
        set(files),
        key=lambda p: str(p).lower()
    )

    return files


def belongs_to_patient(
    path,
    patient_id
):
    """
    Avoid patient 001 matching 010, 100 etc.
    """

    path = Path(path)

    name = path.name.lower()

    parts = [
        part.lower()
        for part in path.parts
    ]

    pid = patient_id.lower()

    # Explicit patient directory
    if pid in parts:
        return True

    # filename starts with patient id
    if (
        name.startswith(
            pid + "_"
        )
        or
        name.startswith(
            pid + "-"
        )
        or
        name.startswith(
            pid + "."
        )
    ):
        return True

    return False


def classify_file(
    path
):

    text = str(
        path
    ).lower()

    name = path.name.lower()

    # --------------------------------------------------------
    # Masks FIRST
    # --------------------------------------------------------

    if (
        "vent_mask" in name
        or "ventricle" in text
        or "vent_masks" in text
    ):
        return "ventricle_mask"

    if (
        "nwmh" in text
        or "normal_wmh" in text
        or "normalwmh" in text
    ):
        return "normal_wmh_mask"

    if (
        "abwmh" in text
        or "abnormal_wmh" in text
        or "abnormalwmh" in text
        or "lesion_mask" in text
    ):
        return "abnormal_wmh_mask"

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    if "flair" in text:

        sagittal_words = [
            "sag",
            "sagittal",
        ]

        axial_words = [
            "ax",
            "axial",
        ]

        if any(
            word in text
            for word in sagittal_words
        ):
            return "flair_sagittal"

        if any(
            word in text
            for word in axial_words
        ):
            return "flair_axial"

        # The currently validated preprocessed FLAIR
        # is axial.
        if (
            "ms_100_patient_preprocessed"
            in text
        ):
            return "flair_axial"

        return "flair_unknown"

    if (
        "t1" in name
        or "_t1_" in text
        or "-t1-" in text
    ):
        return "t1"

    if (
        "t2" in name
        or "_t2_" in text
        or "-t2-" in text
    ):
        return "t2"

    return "other"


def source_folder_name(
    path
):

    path = Path(path)

    try:

        relative = path.relative_to(
            OUTPUT_ROOT
        )

        if len(
            relative.parts
        ) > 0:

            return relative.parts[0]

    except Exception:
        pass

    return "unknown"


def discovery_for_patient(
    patient_id,
    nifti_files
):

    categories = {
        "t1": [],
        "t2": [],
        "flair_axial": [],
        "flair_sagittal": [],
        "flair_unknown": [],
        "ventricle_mask": [],
        "normal_wmh_mask": [],
        "abnormal_wmh_mask": [],
        "other": [],
    }

    patient_files = [
        path
        for path in nifti_files
        if belongs_to_patient(
            path,
            patient_id
        )
    ]

    for path in patient_files:

        category = classify_file(
            path
        )

        categories[
            category
        ].append(
            path
        )

    return categories


# ============================================================
# HIGH-CONFIDENCE PATHS
# ============================================================

def known_paths(
    patient_id
):

    return {

        "flair_axial":
            PREPROCESSED_ROOT /
            patient_id /
            f"{patient_id}_FLAIR.nii.gz",

        "ventricle_mask":
            MASK_ROOT /
            "Vent_Masks" /
            patient_id /
            f"{patient_id}_Vent_Mask.nii.gz",

        "normal_wmh_mask":
            MASK_ROOT /
            "nWMH_Masks" /
            patient_id /
            f"{patient_id}_nWMH_Mask.nii.gz",

        "abnormal_wmh_mask":
            MASK_ROOT /
            "abWMH_Masks" /
            patient_id /
            f"{patient_id}_abWMH_Mask.nii.gz",
    }


# ============================================================
# SELECT BEST FILE
# ============================================================

def choose_best_candidate(
    category,
    candidates,
    patient_id
):

    known = known_paths(
        patient_id
    )

    if (
        category in known
        and
        known[
            category
        ].exists()
    ):

        return known[
            category
        ]

    if not candidates:

        return None

    # Prefer filenames beginning directly with patient ID
    direct = [
        path
        for path in candidates
        if path.name.lower().startswith(
            patient_id.lower() + "_"
        )
    ]

    if direct:

        candidates = direct

    # Prefer .nii.gz
    gz = [
        path
        for path in candidates
        if str(path).lower().endswith(
            ".nii.gz"
        )
    ]

    if gz:

        candidates = gz

    # For axial FLAIR avoid sag files
    if category == "flair_axial":

        nonsag = [
            path
            for path in candidates
            if "sag" not in str(
                path
            ).lower()
        ]

        if nonsag:

            candidates = nonsag

    return sorted(
        candidates,
        key=lambda p: (
            len(str(p)),
            str(p).lower()
        )
    )[0]


# ============================================================
# GEOMETRY COMPARISON
# ============================================================

def compare_geometry(
    reference,
    moving
):

    comparison = {
        "shape_match": False,
        "spacing_match": False,
        "orientation_match": False,
        "origin_match": False,
        "direction_cosines_match": False,
        "affine_match": False,
        "all_geometry_match": False,
    }

    if not (
        reference
        and
        moving
        and
        reference.get(
            "readable"
        )
        and
        moving.get(
            "readable"
        )
    ):
        return comparison

    comparison[
        "shape_match"
    ] = (
        tuple(
            reference["shape"]
        )
        ==
        tuple(
            moving["shape"]
        )
    )

    comparison[
        "spacing_match"
    ] = bool(
        np.allclose(
            reference[
                "spacing"
            ],
            moving[
                "spacing"
            ],
            rtol=0,
            atol=1e-5
        )
    )

    comparison[
        "orientation_match"
    ] = (
        tuple(
            reference[
                "orientation"
            ]
        )
        ==
        tuple(
            moving[
                "orientation"
            ]
        )
    )

    comparison[
        "origin_match"
    ] = bool(
        np.allclose(
            reference[
                "origin"
            ],
            moving[
                "origin"
            ],
            rtol=0,
            atol=1e-3
        )
    )

    comparison[
        "direction_cosines_match"
    ] = bool(
        np.allclose(
            np.asarray(
                reference[
                    "direction_cosines"
                ]
            ),
            np.asarray(
                moving[
                    "direction_cosines"
                ]
            ),
            rtol=0,
            atol=1e-5
        )
    )

    comparison[
        "affine_match"
    ] = bool(
        np.allclose(
            np.asarray(
                reference[
                    "affine"
                ]
            ),
            np.asarray(
                moving[
                    "affine"
                ]
            ),
            rtol=0,
            atol=1e-3
        )
    )

    comparison[
        "all_geometry_match"
    ] = all(
        [
            comparison[
                "shape_match"
            ],
            comparison[
                "spacing_match"
            ],
            comparison[
                "orientation_match"
            ],
            comparison[
                "origin_match"
            ],
            comparison[
                "direction_cosines_match"
            ],
            comparison[
                "affine_match"
            ],
        ]
    )

    return comparison


# ============================================================
# FOLDER USABILITY
# ============================================================

def inspect_patient_folders(
    patient_id,
    categories
):

    folder_records = {}

    all_paths = []

    for paths in categories.values():

        all_paths.extend(
            paths
        )

    for path in all_paths:

        folder = source_folder_name(
            path
        )

        if folder not in folder_records:

            folder_records[
                folder
            ] = {
                "files_found": 0,
                "readable_files": 0,
                "corrupt_files": 0,
                "usable": False,
                "files": [],
            }

        folder_records[
            folder
        ][
            "files_found"
        ] += 1

        record = {
            "path": str(path),
            "readable": False,
        }

        try:

            nib.load(
                str(path)
            )

            record[
                "readable"
            ] = True

            folder_records[
                folder
            ][
                "readable_files"
            ] += 1

        except Exception as error:

            record[
                "error"
            ] = str(error)

            folder_records[
                folder
            ][
                "corrupt_files"
            ] += 1

        folder_records[
            folder
        ][
            "files"
        ].append(
            record
        )

    for folder in folder_records:

        folder_records[
            folder
        ][
            "usable"
        ] = (
            folder_records[
                folder
            ][
                "readable_files"
            ] > 0
        )

    return folder_records


# ============================================================
# PATIENT AUDIT
# ============================================================

def audit_patient(
    patient_id,
    nifti_files
):

    categories = discovery_for_patient(
        patient_id,
        nifti_files
    )

    selected = {}

    wanted = [
        "t1",
        "t2",
        "flair_axial",
        "flair_sagittal",
        "ventricle_mask",
        "normal_wmh_mask",
        "abnormal_wmh_mask",
    ]

    for category in wanted:

        selected[
            category
        ] = choose_best_candidate(
            category,
            categories[
                category
            ],
            patient_id
        )

    # --------------------------------------------------------
    # If no axial FLAIR selected but unknown FLAIR exists,
    # use unknown FLAIR as last resort.
    # --------------------------------------------------------

    if (
        selected[
            "flair_axial"
        ]
        is None
        and
        categories[
            "flair_unknown"
        ]
    ):

        selected[
            "flair_axial"
        ] = choose_best_candidate(
            "flair_axial",
            categories[
                "flair_unknown"
            ],
            patient_id
        )

    # --------------------------------------------------------
    # Inspect selected files
    # --------------------------------------------------------

    images = {}

    for category in [
        "t1",
        "t2",
        "flair_axial",
        "flair_sagittal",
    ]:

        path = selected[
            category
        ]

        if path is None:

            images[
                category
            ] = {
                "present": False,
                "readable": False,
                "path": None,
            }

        else:

            info = inspect_nifti(
                path,
                is_mask=False
            )

            info[
                "present"
            ] = True

            info[
                "preprocessing_heuristics"
            ] = (
                preprocessing_heuristics(
                    path
                )
            )

            images[
                category
            ] = info

    masks = {}

    for category in [
        "ventricle_mask",
        "normal_wmh_mask",
        "abnormal_wmh_mask",
    ]:

        path = selected[
            category
        ]

        if path is None:

            masks[
                category
            ] = {
                "present": False,
                "readable": False,
                "path": None,
            }

        else:

            info = inspect_nifti(
                path,
                is_mask=True
            )

            info[
                "present"
            ] = True

            masks[
                category
            ] = info

    # --------------------------------------------------------
    # Compare all masks against axial FLAIR
    # --------------------------------------------------------

    flair = images[
        "flair_axial"
    ]

    geometry = {}

    for category in [
        "ventricle_mask",
        "normal_wmh_mask",
        "abnormal_wmh_mask",
    ]:

        geometry[
            category
        ] = compare_geometry(
            flair,
            masks[
                category
            ]
        )

    # --------------------------------------------------------
    # Folder usability
    # --------------------------------------------------------

    folder_usability = (
        inspect_patient_folders(
            patient_id,
            categories
        )
    )

    # --------------------------------------------------------
    # Issues
    # --------------------------------------------------------

    issues = []

    required = [
        (
            "axial FLAIR",
            images[
                "flair_axial"
            ]
        ),
        (
            "ventricle mask",
            masks[
                "ventricle_mask"
            ]
        ),
        (
            "normal WMH mask",
            masks[
                "normal_wmh_mask"
            ]
        ),
        (
            "abnormal WMH mask",
            masks[
                "abnormal_wmh_mask"
            ]
        ),
    ]

    for name, info in required:

        if not info.get(
            "present",
            False
        ):

            issues.append(
                f"missing {name}"
            )

        elif not info.get(
            "readable",
            False
        ):

            issues.append(
                f"corrupt/unreadable {name}"
            )

    for category, comparison in geometry.items():

        if (
            masks[
                category
            ].get(
                "readable",
                False
            )
            and
            flair.get(
                "readable",
                False
            )
            and
            not comparison[
                "all_geometry_match"
            ]
        ):

            issues.append(
                f"{category} geometry mismatch"
            )

    # Missing optional modalities are NOT failures.
    optional_missing = []

    for modality in [
        "t1",
        "t2",
        "flair_sagittal",
    ]:

        if not images[
            modality
        ].get(
            "present",
            False
        ):

            optional_missing.append(
                modality
            )

    required_geometry_ok = all(
        geometry[
            mask
        ][
            "all_geometry_match"
        ]
        for mask in [
            "ventricle_mask",
            "normal_wmh_mask",
            "abnormal_wmh_mask",
        ]
    )

    core_files_ok = all(
        info.get(
            "readable",
            False
        )
        for _, info
        in required
    )

    stage1_core_pass = bool(
        core_files_ok
        and
        required_geometry_ok
    )

    usable_folders = [
        folder
        for folder, record
        in folder_usability.items()
        if record[
            "usable"
        ]
    ]

    audit = {

        "patient_id":
            patient_id,

        "audit_timestamp":
            now_string(),

        "images":
            images,

        "masks":
            masks,

        "mask_vs_axial_flair_geometry":
            geometry,

        "folder_usability":
            folder_usability,

        "usable_folders":
            usable_folders,

        "optional_modalities_missing":
            optional_missing,

        "issues":
            issues,

        "stage1_core_pass":
            stage1_core_pass,

        "needs_visual_check":
            True,

        "visual_check_reason":
            (
                "Automated image statistics cannot "
                "reliably prove skull stripping or "
                "N4 bias-field correction."
            ),

        "all_discovered_candidates":
            {
                category: [
                    str(path)
                    for path in paths
                ]
                for category, paths
                in categories.items()
            },
    }

    return audit


# ============================================================
# CREATE SUMMARY ROW
# ============================================================

def summary_row(
    audit
):

    pid = audit[
        "patient_id"
    ]

    images = audit[
        "images"
    ]

    masks = audit[
        "masks"
    ]

    geometry = audit[
        "mask_vs_axial_flair_geometry"
    ]

    flair = images[
        "flair_axial"
    ]

    abnormal = masks[
        "abnormal_wmh_mask"
    ]

    row = {

        "patient_id":
            pid,

        "stage1_core_pass":
            audit[
                "stage1_core_pass"
            ],

        "t1_present":
            images[
                "t1"
            ].get(
                "readable",
                False
            ),

        "t2_present":
            images[
                "t2"
            ].get(
                "readable",
                False
            ),

        "flair_axial_present":
            images[
                "flair_axial"
            ].get(
                "readable",
                False
            ),

        "flair_sagittal_present":
            images[
                "flair_sagittal"
            ].get(
                "readable",
                False
            ),

        "ventricle_mask_present":
            masks[
                "ventricle_mask"
            ].get(
                "readable",
                False
            ),

        "normal_wmh_mask_present":
            masks[
                "normal_wmh_mask"
            ].get(
                "readable",
                False
            ),

        "abnormal_wmh_mask_present":
            masks[
                "abnormal_wmh_mask"
            ].get(
                "readable",
                False
            ),

        "ventricle_geometry_match":
            geometry[
                "ventricle_mask"
            ][
                "all_geometry_match"
            ],

        "normal_wmh_geometry_match":
            geometry[
                "normal_wmh_mask"
            ][
                "all_geometry_match"
            ],

        "abnormal_wmh_geometry_match":
            geometry[
                "abnormal_wmh_mask"
            ][
                "all_geometry_match"
            ],

        "flair_shape":
            (
                str(
                    flair.get(
                        "shape"
                    )
                )
                if flair.get(
                    "readable"
                )
                else ""
            ),

        "flair_spacing":
            (
                str(
                    flair.get(
                        "spacing"
                    )
                )
                if flair.get(
                    "readable"
                )
                else ""
            ),

        "flair_orientation":
            (
                str(
                    flair.get(
                        "orientation"
                    )
                )
                if flair.get(
                    "readable"
                )
                else ""
            ),

        "abnormal_wmh_voxels":
            abnormal.get(
                "nonzero_voxels",
                ""
            ),

        "abnormal_wmh_volume_mm3":
            abnormal.get(
                "nonzero_volume_mm3",
                ""
            ),

        "usable_folders":
            "; ".join(
                audit[
                    "usable_folders"
                ]
            ),

        "optional_modalities_missing":
            "; ".join(
                audit[
                    "optional_modalities_missing"
                ]
            ),

        "issues":
            "; ".join(
                audit[
                    "issues"
                ]
            ),

        "needs_visual_check":
            True,

        "audit_timestamp":
            audit[
                "audit_timestamp"
            ],
    }

    return row


# ============================================================
# MANIFEST UPDATE
# ============================================================

def update_manifest(
    summary_df
):

    if MANIFEST_CSV.exists():

        manifest = pd.read_csv(
            MANIFEST_CSV,
            dtype=str,
            keep_default_na=False
        )

    else:

        manifest = pd.DataFrame(
            {
                "patient_id":
                    PATIENT_IDS
            }
        )

    if "patient_id" not in manifest.columns:

        raise ValueError(
            "manifest.csv has no patient_id column."
        )

    manifest[
        "patient_id"
    ] = (
        manifest[
            "patient_id"
        ]
        .astype(str)
        .str.zfill(3)
    )

    required_columns = [
        "audit_done",
        "audit_timestamp",
        "audit_error",
    ]

    for column in required_columns:

        if column not in manifest.columns:

            manifest[
                column
            ] = ""

    summary_lookup = (
        summary_df
        .set_index(
            "patient_id"
        )
        .to_dict(
            orient="index"
        )
    )

    for index, row in manifest.iterrows():

        pid = str(
            row[
                "patient_id"
            ]
        ).zfill(3)

        if pid not in summary_lookup:
            continue

        result = summary_lookup[
            pid
        ]

        passed = bool(
            result[
                "stage1_core_pass"
            ]
        )

        manifest.at[
            index,
            "audit_done"
        ] = str(
            passed
        )

        manifest.at[
            index,
            "audit_timestamp"
        ] = result[
            "audit_timestamp"
        ]

        manifest.at[
            index,
            "audit_error"
        ] = (
            ""
            if passed
            else result[
                "issues"
            ]
        )

    atomic_write_dataframe(
        manifest,
        MANIFEST_CSV
    )


# ============================================================
# FINAL REPORT
# ============================================================

def print_final_report(
    summary_df
):

    total = len(
        summary_df
    )

    passed = int(
        summary_df[
            "stage1_core_pass"
        ].sum()
    )

    failed = (
        total -
        passed
    )

    print()
    print("=" * 72)
    print(
        "FINAL STAGE-1 AUDIT SUMMARY"
    )
    print("=" * 72)

    print(
        f"Total patients audited : {total}"
    )

    print(
        f"Core Stage-1 PASS      : {passed}"
    )

    print(
        f"Core Stage-1 FAIL      : {failed}"
    )

    print()

    for column, label in [

        (
            "t1_present",
            "T1 available"
        ),

        (
            "t2_present",
            "T2 available"
        ),

        (
            "flair_axial_present",
            "Axial FLAIR available"
        ),

        (
            "flair_sagittal_present",
            "Sagittal FLAIR available"
        ),

        (
            "ventricle_mask_present",
            "Ventricle masks"
        ),

        (
            "normal_wmh_mask_present",
            "Normal WMH masks"
        ),

        (
            "abnormal_wmh_mask_present",
            "Abnormal WMH masks"
        ),

    ]:

        count = int(
            summary_df[
                column
            ].sum()
        )

        print(
            f"{label:<25}: "
            f"{count}/{total}"
        )

    print()
    print(
        "MASK ↔ AXIAL FLAIR GEOMETRY"
    )

    print("-" * 72)

    for column, label in [

        (
            "ventricle_geometry_match",
            "Ventricle"
        ),

        (
            "normal_wmh_geometry_match",
            "Normal WMH"
        ),

        (
            "abnormal_wmh_geometry_match",
            "Abnormal WMH"
        ),

    ]:

        count = int(
            summary_df[
                column
            ].sum()
        )

        print(
            f"{label:<25}: "
            f"{count}/{total} match"
        )

    # --------------------------------------------------------
    # Most common issues
    # --------------------------------------------------------

    issue_counter = Counter()

    for text in summary_df[
        "issues"
    ]:

        if not text:
            continue

        for issue in str(
            text
        ).split(";"):

            issue = issue.strip()

            if issue:

                issue_counter[
                    issue
                ] += 1

    print()
    print(
        "MOST COMMON ISSUES"
    )

    print("-" * 72)

    if issue_counter:

        for issue, count in (
            issue_counter.most_common(
                10
            )
        ):

            print(
                f"{count:>3}  {issue}"
            )

    else:

        print(
            "No core audit issues detected."
        )

    # --------------------------------------------------------
    # Optional modalities
    # --------------------------------------------------------

    print()
    print(
        "OPTIONAL MODALITY AVAILABILITY"
    )

    print("-" * 72)

    t1_count = int(
        summary_df[
            "t1_present"
        ].sum()
    )

    t2_count = int(
        summary_df[
            "t2_present"
        ].sum()
    )

    sag_count = int(
        summary_df[
            "flair_sagittal_present"
        ].sum()
    )

    print(
        f"T1              : {t1_count}/{total}"
    )

    print(
        f"T2              : {t2_count}/{total}"
    )

    print(
        f"Sagittal FLAIR  : {sag_count}/{total}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Skull stripping and N4 status are heuristic only."
    )

    print(
        "They remain flagged as NEEDS VISUAL CHECK."
    )

    print()
    print(
        f"Per-patient JSONs:"
    )
    print(
        AUDIT_DIR
    )

    print()
    print(
        f"Summary CSV:"
    )
    print(
        SUMMARY_CSV
    )

    print()
    print(
        f"Manifest:"
    )
    print(
        MANIFEST_CSV
    )

    print()
    print(
        f"Terminal log:"
    )
    print(
        TERMINAL_LOG
    )

    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():

    make_directories()

    print("=" * 72)
    print(
        "MS3SEG FINAL STAGE-1 DATA AUDIT"
    )
    print("=" * 72)

    print(
        f"Started: {now_string()}"
    )

    print(
        f"Project root:"
    )

    print(
        PROJECT_ROOT
    )

    print(
        f"\nScanning:"
    )

    print(
        OUTPUT_ROOT
    )

    print()

    # --------------------------------------------------------
    # Discover files ONCE
    # --------------------------------------------------------

    print(
        "Building NIfTI file index..."
    )

    nifti_files = (
        all_nifti_files()
    )

    print(
        f"Found {len(nifti_files)} NIfTI files."
    )

    print()

    summary_rows = []

    successful = 0

    failed = 0

    # --------------------------------------------------------
    # Patient loop
    # --------------------------------------------------------

    for number, patient_id in enumerate(
        PATIENT_IDS,
        start=1
    ):

        print("=" * 72)

        print(
            f"PATIENT {patient_id} "
            f"({number}/100)"
        )

        print("=" * 72)

        json_path = (
            AUDIT_DIR /
            f"patient_{patient_id}.json"
        )

        try:

            # ------------------------------------------------
            # Resume/checkpoint behavior
            #
            # If JSON already exists and is readable,
            # reuse it instead of repeating the audit.
            # ------------------------------------------------

            if json_path.exists():

                try:

                    with open(
                        json_path,
                        "r",
                        encoding="utf-8"
                    ) as file:

                        audit = json.load(
                            file
                        )

                    print(
                        "Existing audit JSON found."
                    )

                    print(
                        "Checkpoint reused."
                    )

                except Exception:

                    print(
                        "Existing JSON could not be read."
                    )

                    print(
                        "Re-running patient audit..."
                    )

                    audit = audit_patient(
                        patient_id,
                        nifti_files
                    )

                    atomic_write_json(
                        audit,
                        json_path
                    )

            else:

                audit = audit_patient(
                    patient_id,
                    nifti_files
                )

                # IMPORTANT:
                # Save output FIRST.
                # Manifest is updated only after all patients.
                atomic_write_json(
                    audit,
                    json_path
                )

            row = summary_row(
                audit
            )

            summary_rows.append(
                row
            )

            # ------------------------------------------------
            # Print concise patient summary
            # ------------------------------------------------

            print(
                "Axial FLAIR:",
                "FOUND"
                if row[
                    "flair_axial_present"
                ]
                else "MISSING"
            )

            print(
                "T1:",
                "FOUND"
                if row[
                    "t1_present"
                ]
                else "NOT FOUND"
            )

            print(
                "T2:",
                "FOUND"
                if row[
                    "t2_present"
                ]
                else "NOT FOUND"
            )

            print(
                "Sagittal FLAIR:",
                "FOUND"
                if row[
                    "flair_sagittal_present"
                ]
                else "NOT FOUND"
            )

            print(
                "Ventricle geometry:",
                "PASS"
                if row[
                    "ventricle_geometry_match"
                ]
                else "FAIL"
            )

            print(
                "Normal WMH geometry:",
                "PASS"
                if row[
                    "normal_wmh_geometry_match"
                ]
                else "FAIL"
            )

            print(
                "Abnormal WMH geometry:",
                "PASS"
                if row[
                    "abnormal_wmh_geometry_match"
                ]
                else "FAIL"
            )

            print(
                "Usable folders:"
            )

            if row[
                "usable_folders"
            ]:

                print(
                    "  "
                    + row[
                        "usable_folders"
                    ]
                )

            else:

                print(
                    "  None detected"
                )

            if row[
                "optional_modalities_missing"
            ]:

                print(
                    "Optional modalities missing:"
                )

                print(
                    "  "
                    + row[
                        "optional_modalities_missing"
                    ]
                )

            if row[
                "issues"
            ]:

                print(
                    "Issues:"
                )

                print(
                    "  "
                    + row[
                        "issues"
                    ]
                )

            else:

                print(
                    "Core issues: NONE"
                )

            if row[
                "stage1_core_pass"
            ]:

                successful += 1

                print(
                    f"\nPATIENT {patient_id}: PASS"
                )

            else:

                failed += 1

                print(
                    f"\nPATIENT {patient_id}: FAIL"
                )

        except Exception as error:

            failed += 1

            print(
                f"\nPATIENT {patient_id}: ERROR"
            )

            print(
                str(error)
            )

            traceback.print_exc()

            summary_rows.append(
                {
                    "patient_id":
                        patient_id,

                    "stage1_core_pass":
                        False,

                    "t1_present":
                        False,

                    "t2_present":
                        False,

                    "flair_axial_present":
                        False,

                    "flair_sagittal_present":
                        False,

                    "ventricle_mask_present":
                        False,

                    "normal_wmh_mask_present":
                        False,

                    "abnormal_wmh_mask_present":
                        False,

                    "ventricle_geometry_match":
                        False,

                    "normal_wmh_geometry_match":
                        False,

                    "abnormal_wmh_geometry_match":
                        False,

                    "flair_shape":
                        "",

                    "flair_spacing":
                        "",

                    "flair_orientation":
                        "",

                    "abnormal_wmh_voxels":
                        "",

                    "abnormal_wmh_volume_mm3":
                        "",

                    "usable_folders":
                        "",

                    "optional_modalities_missing":
                        "",

                    "issues":
                        "audit script error: "
                        + str(error),

                    "needs_visual_check":
                        True,

                    "audit_timestamp":
                        now_string(),
                }
            )

    # ========================================================
    # WRITE SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df[
        "patient_id"
    ] = (
        summary_df[
            "patient_id"
        ]
        .astype(str)
        .str.zfill(3)
    )

    summary_df = summary_df.sort_values(
        "patient_id"
    ).reset_index(
        drop=True
    )

    atomic_write_dataframe(
        summary_df,
        SUMMARY_CSV
    )

    # ========================================================
    # UPDATE MANIFEST
    # ========================================================

    update_manifest(
        summary_df
    )

    # ========================================================
    # REPORT
    # ========================================================

    print_final_report(
        summary_df
    )

    print()
    print(
        f"Finished: {now_string()}"
    )


# ============================================================
# RUN WITH LOG CAPTURE
# ============================================================

if __name__ == "__main__":

    make_directories()

    with open(
        TERMINAL_LOG,
        "a",
        encoding="utf-8"
    ) as logfile:

        logfile.write(
            "\n\n"
            + "=" * 72
            + "\n"
        )

        logfile.write(
            "NEW AUDIT RUN\n"
        )

        logfile.write(
            now_string()
            + "\n"
        )

        logfile.write(
            "=" * 72
            + "\n"
        )

        tee_stdout = Tee(
            sys.stdout,
            logfile
        )

        tee_stderr = Tee(
            sys.stderr,
            logfile
        )

        with (
            contextlib.redirect_stdout(
                tee_stdout
            ),
            contextlib.redirect_stderr(
                tee_stderr
            )
        ):

            try:

                main()

            except Exception:

                print()
                print(
                    "FATAL ERROR"
                )
                print(
                    "=" * 72
                )

                traceback.print_exc()

                raise