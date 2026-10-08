# ============================================================
# 10_preprocess_dataset.py
#
# MS3SEG - STAGE 2 PREPROCESSING
#
# IMPORTANT:
#   Stage-1 audit/QC has already been completed.
#
# Current approved preprocessing plan:
#   Skull stripping : patient-specific from preprocess_plan.csv
#   N4 correction   : patient-specific from preprocess_plan.csv
#   Registration    : do NOT repeat unless explicitly requested
#   Resampling      : common FLAIR reference grid
#   Images          : linear interpolation
#   Masks           : nearest-neighbor interpolation
#   Normalization   : configurable in config.yaml
#
# OUTPUT:
#
# preprocessed/
# └── 001/
#     ├── 001_T1.nii.gz
#     ├── 001_T2.nii.gz
#     ├── 001_FLAIR.nii.gz
#     ├── 001_Vent_Mask.nii.gz
#     ├── 001_nWMH_Mask.nii.gz
#     └── 001_abWMH_Mask.nii.gz
#
# LOGS:
#   logs/10_preprocess_terminal.txt
#   logs/preprocess_errors.log
#
# MANIFEST:
#   preprocess_done
#   preprocess_timestamp
#   preprocess_error
#   final_shape
#   final_spacing
#
# EXAMPLES:
#
#   Test one patient:
#       python 10_preprocess_dataset.py --patient 001
#
#   Re-run one patient:
#       python 10_preprocess_dataset.py --patient 001 --force
#
#   Full cohort:
#       python 10_preprocess_dataset.py
#
# ============================================================


from pathlib import Path
from datetime import datetime

import argparse
import contextlib
import os
import sys
import traceback

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Preethi\Documents\LESION"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "output"
)


# ------------------------------------------------------------
# Existing registered/preprocessed MRI source
# ------------------------------------------------------------

SOURCE_IMAGE_ROOT = (
    OUTPUT_ROOT
    / "MS_100_patient_preprocessed"
    / "MS_100_patient_preprocessed"
)


# ------------------------------------------------------------
# Existing masks
# ------------------------------------------------------------

SOURCE_MASK_ROOT = (
    OUTPUT_ROOT
    / "MS_100_patient_masks"
    / "MS_100_patient_masks"
)


# ------------------------------------------------------------
# New Stage-2 output
# ------------------------------------------------------------

FINAL_ROOT = (
    PROJECT_ROOT
    / "preprocessed"
)


# ------------------------------------------------------------
# Audit information
# ------------------------------------------------------------

AUDIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "audit"
)

PLAN_CSV = (
    AUDIT_DIR
    / "preprocess_plan.csv"
)


# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------

MANIFEST_CSV = (
    PROJECT_ROOT
    / "manifest.csv"
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CONFIG_FILE = (
    PROJECT_ROOT
    / "config.yaml"
)


# ------------------------------------------------------------
# Logs
# ------------------------------------------------------------

LOG_DIR = (
    PROJECT_ROOT
    / "logs"
)

TERMINAL_LOG = (
    LOG_DIR
    / "10_preprocess_terminal.txt"
)

ERROR_LOG = (
    LOG_DIR
    / "preprocess_errors.log"
)


# ============================================================
# PATIENT LIST
# ============================================================

PATIENT_IDS = [
    f"{i:03d}"
    for i in range(1, 101)
]


# ============================================================
# TERMINAL + TXT LOGGER
# ============================================================

class Tee:

    def __init__(
        self,
        terminal,
        logfile
    ):

        self.terminal = terminal
        self.logfile = logfile


    def write(
        self,
        message
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
        self
    ):

        self.terminal.flush()
        self.logfile.flush()


# ============================================================
# GENERAL HELPERS
# ============================================================

def now():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def ensure_directories():

    FINAL_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def yes(
    value
):

    return (
        str(value)
        .strip()
        .upper()
        == "YES"
    )


def normalize_patient_id(
    value
):

    return str(
        value
    ).strip().zfill(3)


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "MS3SEG Stage-2 resumable preprocessing"
        )
    )

    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help=(
            "Process only one patient. "
            "Example: --patient 001"
        )
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore completed checkpoint and "
            "reprocess selected patient(s)."
        )
    )

    return parser.parse_args()


# ============================================================
# CONFIGURATION
# ============================================================

def default_config():

    return {

        "preprocessing": {

            "target_spacing_mm": {
                "x": 1.0,
                "y": 1.0,
                "z": None,
            },

            "interpolation": {
                "images": "linear",
                "masks": "nearest_neighbor",
            },

            "intensity_normalization": {

                "method": "zscore",

                "nonzero_only": True,

                "percentile_clip": {
                    "enabled": True,
                    "lower": 0.5,
                    "upper": 99.5,
                },
            },

            "n4": {
                "convergence_iterations": [
                    50,
                    50,
                    30,
                    20,
                ]
            }
        }
    }


def load_config():

    defaults = default_config()

    if not CONFIG_FILE.exists():

        print(
            "WARNING:"
        )

        print(
            f"config.yaml not found:\n{CONFIG_FILE}"
        )

        print(
            "Using safe built-in defaults."
        )

        return defaults

    with open(
        CONFIG_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        loaded = (
            yaml.safe_load(
                file
            )
            or {}
        )

    # --------------------------------------------------------
    # Guarantee sections exist
    # --------------------------------------------------------

    loaded.setdefault(
        "preprocessing",
        {}
    )

    loaded[
        "preprocessing"
    ].setdefault(
        "target_spacing_mm",
        defaults[
            "preprocessing"
        ][
            "target_spacing_mm"
        ]
    )

    loaded[
        "preprocessing"
    ].setdefault(
        "interpolation",
        defaults[
            "preprocessing"
        ][
            "interpolation"
        ]
    )

    loaded[
        "preprocessing"
    ].setdefault(
        "intensity_normalization",
        defaults[
            "preprocessing"
        ][
            "intensity_normalization"
        ]
    )

    loaded[
        "preprocessing"
    ].setdefault(
        "n4",
        defaults[
            "preprocessing"
        ][
            "n4"
        ]
    )

    return loaded


# ============================================================
# SOURCE FILE PATHS
# ============================================================

def source_paths(
    patient_id
):

    image_dir = (
        SOURCE_IMAGE_ROOT
        / patient_id
    )

    return {

        "t1":
            image_dir
            / f"{patient_id}_T1WI_reg.nii.gz",

        "t2":
            image_dir
            / f"{patient_id}_T2WI_reg.nii.gz",

        "flair":
            image_dir
            / f"{patient_id}_FLAIR.nii.gz",

        "ventricle":
            SOURCE_MASK_ROOT
            / "Vent_Masks"
            / patient_id
            / f"{patient_id}_Vent_Mask.nii.gz",

        "nwmh":
            SOURCE_MASK_ROOT
            / "nWMH_Masks"
            / patient_id
            / f"{patient_id}_nWMH_Mask.nii.gz",

        "abwmh":
            SOURCE_MASK_ROOT
            / "abWMH_Masks"
            / patient_id
            / f"{patient_id}_abWMH_Mask.nii.gz",
    }


# ============================================================
# FINAL OUTPUT PATHS
# ============================================================

def final_paths(
    patient_id
):

    patient_dir = (
        FINAL_ROOT
        / patient_id
    )

    return {

        "directory":
            patient_dir,

        "t1":
            patient_dir
            / f"{patient_id}_T1.nii.gz",

        "t2":
            patient_dir
            / f"{patient_id}_T2.nii.gz",

        "flair":
            patient_dir
            / f"{patient_id}_FLAIR.nii.gz",

        "ventricle":
            patient_dir
            / f"{patient_id}_Vent_Mask.nii.gz",

        "nwmh":
            patient_dir
            / f"{patient_id}_nWMH_Mask.nii.gz",

        "abwmh":
            patient_dir
            / f"{patient_id}_abWMH_Mask.nii.gz",
    }


# ============================================================
# CHECK SOURCE FILES
# ============================================================

def validate_source_files(
    patient_id
):

    paths = source_paths(
        patient_id
    )

    missing = []

    for name, path in paths.items():

        if not path.exists():

            missing.append(
                f"{name}: {path}"
            )

    if missing:

        message = (
            "Missing source files:\n"
            + "\n".join(
                missing
            )
        )

        raise FileNotFoundError(
            message
        )


# ============================================================
# CHECKPOINT OUTPUT VERIFICATION
# ============================================================

def patient_outputs_complete(
    patient_id
):

    paths = final_paths(
        patient_id
    )

    required = [

        paths[
            "t1"
        ],

        paths[
            "t2"
        ],

        paths[
            "flair"
        ],

        paths[
            "ventricle"
        ],

        paths[
            "nwmh"
        ],

        paths[
            "abwmh"
        ],
    ]

    for path in required:

        if not path.exists():

            return False

        try:

            image = sitk.ReadImage(
                str(path)
            )

            if image.GetDimension() != 3:

                return False

        except Exception:

            return False

    return True


# ============================================================
# READ IMAGE
# ============================================================

def read_image(
    path
):

    if not path.exists():

        raise FileNotFoundError(
            str(path)
        )

    image = sitk.ReadImage(
        str(path)
    )

    if image.GetDimension() != 3:

        raise ValueError(
            f"Expected 3-D image:\n{path}"
        )

    return image


# ============================================================
# IMAGE INFORMATION
# ============================================================

def print_image_info(
    label,
    image
):

    print(
        f"{label:<12} "
        f"size={tuple(image.GetSize())} "
        f"spacing={tuple(round(float(x), 6) for x in image.GetSpacing())}"
    )


# ============================================================
# MASK BINARIZATION
# ============================================================

def binarize_mask(
    image
):

    # --------------------------------------------------------
    # Original masks may contain:
    #   0 / 1
    #   0 / 255
    #   floating approximations such as 254.9999999
    #
    # Since each mask is stored separately, Stage-2
    # standardizes every foreground voxel to 1.
    # --------------------------------------------------------

    binary = (
        image > 0
    )

    binary = sitk.Cast(
        binary,
        sitk.sitkUInt8
    )

    return binary


# ============================================================
# SKULL STRIPPING
# ============================================================

def apply_skull_strip(
    image
):

    print(
        "  Applying requested skull/foreground extraction..."
    )

    image_float = sitk.Cast(
        image,
        sitk.sitkFloat32
    )

    # --------------------------------------------------------
    # Conservative fallback.
    #
    # Stage-1 currently says this is not required.
    # This branch should therefore normally NOT execute.
    # --------------------------------------------------------

    mask = sitk.OtsuThreshold(
        image_float,
        0,
        1,
        200
    )

    mask = sitk.Cast(
        mask,
        sitk.sitkUInt8
    )

    connected = sitk.ConnectedComponent(
        mask
    )

    relabeled = sitk.RelabelComponent(
        connected,
        sortByObjectSize=True
    )

    largest = sitk.Cast(
        relabeled == 1,
        sitk.sitkUInt8
    )

    stripped = sitk.Mask(
        image_float,
        largest
    )

    return stripped


# ============================================================
# N4 BIAS FIELD CORRECTION
# ============================================================

def apply_n4(
    image,
    config
):

    print(
        "  Applying N4 bias correction..."
    )

    image_float = sitk.Cast(
        image,
        sitk.sitkFloat32
    )

    mask = sitk.Cast(
        image_float > 0,
        sitk.sitkUInt8
    )

    n4_filter = (
        sitk.N4BiasFieldCorrectionImageFilter()
    )

    iterations = (
        config
        .get(
            "preprocessing",
            {}
        )
        .get(
            "n4",
            {}
        )
        .get(
            "convergence_iterations",
            [50, 50, 30, 20]
        )
    )

    n4_filter.SetMaximumNumberOfIterations(
        [
            int(value)
            for value in iterations
        ]
    )

    corrected = n4_filter.Execute(
        image_float,
        mask
    )

    return corrected


# ============================================================
# CREATE FINAL REFERENCE GRID
# ============================================================

def create_reference_grid(
    flair,
    config
):

    settings = (
        config
        .get(
            "preprocessing",
            {}
        )
        .get(
            "target_spacing_mm",
            {}
        )
    )

    original_spacing = np.asarray(
        flair.GetSpacing(),
        dtype=np.float64
    )

    original_size = np.asarray(
        flair.GetSize(),
        dtype=np.int64
    )

    x_config = settings.get(
        "x",
        original_spacing[0]
    )

    y_config = settings.get(
        "y",
        original_spacing[1]
    )

    z_config = settings.get(
        "z",
        None
    )

    target_spacing = np.asarray(
        [
            (
                original_spacing[0]
                if x_config is None
                else float(
                    x_config
                )
            ),

            (
                original_spacing[1]
                if y_config is None
                else float(
                    y_config
                )
            ),

            (
                original_spacing[2]
                if z_config is None
                else float(
                    z_config
                )
            ),
        ],
        dtype=np.float64
    )

    if np.any(
        target_spacing <= 0
    ):

        raise ValueError(
            "Target spacing must be positive."
        )

    # --------------------------------------------------------
    # Preserve physical extent:
    #
    # extent = (size - 1) * spacing
    #
    # new_size = round(extent/new_spacing) + 1
    # --------------------------------------------------------

    physical_extent = (
        (original_size - 1)
        * original_spacing
    )

    target_size = (
        np.round(
            physical_extent
            / target_spacing
        ).astype(
            np.int64
        )
        + 1
    )

    target_size = np.maximum(
        target_size,
        1
    )

    reference = sitk.Image(
        [
            int(value)
            for value in target_size
        ],
        sitk.sitkFloat32
    )

    reference.SetSpacing(
        tuple(
            float(value)
            for value in target_spacing
        )
    )

    reference.SetOrigin(
        flair.GetOrigin()
    )

    reference.SetDirection(
        flair.GetDirection()
    )

    return reference


# ============================================================
# INTERPOLATOR SELECTION
# ============================================================

def image_interpolator(
    config
):

    value = str(
        config
        .get(
            "preprocessing",
            {}
        )
        .get(
            "interpolation",
            {}
        )
        .get(
            "images",
            "linear"
        )
    ).strip().lower()

    if value == "linear":

        return sitk.sitkLinear

    if value in [
        "bspline",
        "b-spline",
        "b_spline",
    ]:

        return sitk.sitkBSpline

    raise ValueError(
        f"Unsupported image interpolator: {value}"
    )


# ============================================================
# RESAMPLING
# ============================================================

def resample_to_reference(
    image,
    reference,
    is_mask,
    config
):

    if is_mask:

        interpolator = (
            sitk.sitkNearestNeighbor
        )

        output_type = (
            sitk.sitkUInt8
        )

    else:

        interpolator = (
            image_interpolator(
                config
            )
        )

        output_type = (
            sitk.sitkFloat32
        )

    identity = sitk.Transform(
        3,
        sitk.sitkIdentity
    )

    result = sitk.Resample(
        image,
        reference,
        identity,
        interpolator,
        0.0,
        output_type
    )

    return result


# ============================================================
# INTENSITY NORMALIZATION
# ============================================================

def normalize_image(
    image,
    config
):

    norm_config = (
        config
        .get(
            "preprocessing",
            {}
        )
        .get(
            "intensity_normalization",
            {}
        )
    )

    method = str(
        norm_config.get(
            "method",
            "zscore"
        )
    ).strip().lower()

    if method == "none":

        return sitk.Cast(
            image,
            sitk.sitkFloat32
        )

    array = sitk.GetArrayFromImage(
        image
    ).astype(
        np.float32
    )

    finite = np.isfinite(
        array
    )

    # Replace invalid values before proceeding.
    array[
        ~finite
    ] = 0.0

    nonzero_only = bool(
        norm_config.get(
            "nonzero_only",
            True
        )
    )

    if nonzero_only:

        valid = (
            finite
            &
            (array != 0)
        )

    else:

        valid = finite

    if not np.any(
        valid
    ):

        raise ValueError(
            "No valid voxels available for normalization."
        )

    # --------------------------------------------------------
    # Optional percentile clipping
    # --------------------------------------------------------

    clipping = norm_config.get(
        "percentile_clip",
        {}
    )

    clipping_enabled = bool(
        clipping.get(
            "enabled",
            True
        )
    )

    if clipping_enabled:

        lower_percentile = float(
            clipping.get(
                "lower",
                0.5
            )
        )

        upper_percentile = float(
            clipping.get(
                "upper",
                99.5
            )
        )

        if (
            lower_percentile
            >= upper_percentile
        ):

            raise ValueError(
                "Percentile lower bound must "
                "be smaller than upper bound."
            )

        low_value = float(
            np.percentile(
                array[
                    valid
                ],
                lower_percentile
            )
        )

        high_value = float(
            np.percentile(
                array[
                    valid
                ],
                upper_percentile
            )
        )

        if high_value > low_value:

            array[
                valid
            ] = np.clip(
                array[
                    valid
                ],
                low_value,
                high_value
            )

    # --------------------------------------------------------
    # Z-score
    # --------------------------------------------------------

    if method == "zscore":

        mean_value = float(
            np.mean(
                array[
                    valid
                ]
            )
        )

        std_value = float(
            np.std(
                array[
                    valid
                ]
            )
        )

        if (
            not np.isfinite(
                std_value
            )
            or std_value < 1e-8
        ):

            raise ValueError(
                "Intensity standard deviation "
                "is zero or invalid."
            )

        normalized = np.zeros_like(
            array,
            dtype=np.float32
        )

        normalized[
            valid
        ] = (
            array[
                valid
            ]
            - mean_value
        ) / std_value

    else:

        raise ValueError(
            f"Unsupported normalization method: {method}"
        )

    output = sitk.GetImageFromArray(
        normalized
    )

    output.CopyInformation(
        image
    )

    return output


# ============================================================
# ATOMIC NIFTI WRITE
# ============================================================

def safe_write(
    image,
    destination
):

    destination = Path(
        destination
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary = (
        destination.parent
        / (
            destination.name.replace(
                ".nii.gz",
                ".tmp.nii.gz"
            )
        )
    )

    # Remove stale temporary file.
    if temporary.exists():

        temporary.unlink()

    sitk.WriteImage(
        image,
        str(temporary),
        True
    )

    # Confirm temporary output can be reopened.
    sitk.ReadImage(
        str(temporary)
    )

    os.replace(
        temporary,
        destination
    )


# ============================================================
# GEOMETRY COMPARISON
# ============================================================

def same_geometry(
    reference,
    moving
):

    size_match = (
        tuple(
            reference.GetSize()
        )
        ==
        tuple(
            moving.GetSize()
        )
    )

    spacing_match = np.allclose(
        np.asarray(
            reference.GetSpacing()
        ),
        np.asarray(
            moving.GetSpacing()
        ),
        atol=1e-6,
        rtol=0
    )

    origin_match = np.allclose(
        np.asarray(
            reference.GetOrigin()
        ),
        np.asarray(
            moving.GetOrigin()
        ),
        atol=1e-4,
        rtol=0
    )

    direction_match = np.allclose(
        np.asarray(
            reference.GetDirection()
        ),
        np.asarray(
            moving.GetDirection()
        ),
        atol=1e-6,
        rtol=0
    )

    return bool(
        size_match
        and spacing_match
        and origin_match
        and direction_match
    )


# ============================================================
# MASK VALIDATION
# ============================================================

def validate_binary_mask(
    image,
    name
):

    array = sitk.GetArrayViewFromImage(
        image
    )

    unique_values = np.unique(
        array
    )

    allowed = {
        0,
        1
    }

    actual = set(
        int(value)
        for value in unique_values
    )

    if not actual.issubset(
        allowed
    ):

        raise ValueError(
            f"{name} is not binary after preprocessing. "
            f"Labels found: {unique_values.tolist()}"
        )

    return unique_values.tolist()


# ============================================================
# FINAL OUTPUT VALIDATION
# ============================================================

def validate_final_outputs(
    patient_id
):

    paths = final_paths(
        patient_id
    )

    flair = read_image(
        paths[
            "flair"
        ]
    )

    t1 = read_image(
        paths[
            "t1"
        ]
    )

    t2 = read_image(
        paths[
            "t2"
        ]
    )

    vent = read_image(
        paths[
            "ventricle"
        ]
    )

    nwmh = read_image(
        paths[
            "nwmh"
        ]
    )

    abwmh = read_image(
        paths[
            "abwmh"
        ]
    )

    volumes = {

        "T1":
            t1,

        "T2":
            t2,

        "Ventricle":
            vent,

        "Normal WMH":
            nwmh,

        "Abnormal WMH":
            abwmh,
    }

    geometry_failures = []

    for name, image in volumes.items():

        if not same_geometry(
            flair,
            image
        ):

            geometry_failures.append(
                name
            )

    if geometry_failures:

        raise RuntimeError(
            "Final geometry mismatch against FLAIR: "
            + ", ".join(
                geometry_failures
            )
        )

    vent_labels = validate_binary_mask(
        vent,
        "Ventricle mask"
    )

    nwmh_labels = validate_binary_mask(
        nwmh,
        "Normal WMH mask"
    )

    abwmh_labels = validate_binary_mask(
        abwmh,
        "Abnormal WMH mask"
    )

    final_shape = tuple(
        int(value)
        for value in flair.GetSize()
    )

    final_spacing = tuple(
        round(
            float(value),
            6
        )
        for value in flair.GetSpacing()
    )

    return {

        "shape":
            final_shape,

        "spacing":
            final_spacing,

        "vent_labels":
            vent_labels,

        "nwmh_labels":
            nwmh_labels,

        "abwmh_labels":
            abwmh_labels,
    }


# ============================================================
# MANIFEST
# ============================================================

def load_manifest():

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

    if (
        "patient_id"
        not in manifest.columns
    ):

        raise ValueError(
            "manifest.csv does not contain patient_id."
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

        "preprocess_done",

        "preprocess_timestamp",

        "preprocess_error",

        "final_shape",

        "final_spacing",
    ]

    for column in required_columns:

        if column not in manifest.columns:

            manifest[
                column
            ] = ""

    # Ensure all patients exist.
    existing = set(
        manifest[
            "patient_id"
        ]
    )

    missing_rows = []

    for patient_id in PATIENT_IDS:

        if patient_id not in existing:

            missing_rows.append(
                {
                    "patient_id":
                        patient_id
                }
            )

    if missing_rows:

        manifest = pd.concat(
            [
                manifest,
                pd.DataFrame(
                    missing_rows
                )
            ],
            ignore_index=True
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

    return manifest


# ============================================================
# ATOMIC MANIFEST WRITE
# ============================================================

def atomic_save_manifest(
    manifest
):

    temporary = (
        MANIFEST_CSV.parent
        / "manifest.tmp.csv"
    )

    manifest.to_csv(
        temporary,
        index=False
    )

    os.replace(
        temporary,
        MANIFEST_CSV
    )


# ============================================================
# UPDATE ONE PATIENT IN MANIFEST
# ============================================================

def update_manifest_patient(
    manifest,
    patient_id,
    success,
    shape="",
    spacing="",
    error=""
):

    patient_id = normalize_patient_id(
        patient_id
    )

    mask = (
        manifest[
            "patient_id"
        ]
        ==
        patient_id
    )

    if not mask.any():

        new_row = {
            column: ""
            for column
            in manifest.columns
        }

        new_row[
            "patient_id"
        ] = patient_id

        manifest.loc[
            len(
                manifest
            )
        ] = new_row

        mask = (
            manifest[
                "patient_id"
            ]
            ==
            patient_id
        )

    manifest.loc[
        mask,
        "preprocess_done"
    ] = str(
        bool(
            success
        )
    )

    manifest.loc[
        mask,
        "preprocess_timestamp"
    ] = now()

    manifest.loc[
        mask,
        "preprocess_error"
    ] = str(
        error
    )

    manifest.loc[
        mask,
        "final_shape"
    ] = str(
        shape
    )

    manifest.loc[
        mask,
        "final_spacing"
    ] = str(
        spacing
    )

    # --------------------------------------------------------
    # Atomic checkpoint:
    # patient outputs have already been written before this.
    # --------------------------------------------------------

    atomic_save_manifest(
        manifest
    )


# ============================================================
# CHECK MANIFEST STATUS
# ============================================================

def manifest_patient_done(
    manifest,
    patient_id
):

    rows = manifest[
        manifest[
            "patient_id"
        ]
        ==
        patient_id
    ]

    if rows.empty:

        return False

    value = str(
        rows.iloc[
            0
        ].get(
            "preprocess_done",
            ""
        )
    ).strip().lower()

    return value in [
        "true",
        "1",
        "yes",
    ]


# ============================================================
# ERROR LOG
# ============================================================

def log_error(
    patient_id,
    error
):

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        ERROR_LOG,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n"
            + "=" * 72
            + "\n"
        )

        file.write(
            f"{now()} | PATIENT {patient_id}\n"
        )

        file.write(
            "=" * 72
            + "\n"
        )

        file.write(
            str(
                error
            )
            + "\n\n"
        )

        file.write(
            traceback.format_exc()
        )

        file.write(
            "\n"
        )


# ============================================================
# PREPROCESS ONE PATIENT
# ============================================================

def preprocess_patient(
    patient_id,
    plan,
    config
):

    print()
    print(
        "-" * 72
    )

    print(
        f"PATIENT {patient_id}"
    )

    print(
        "-" * 72
    )

    # ========================================================
    # SOURCE VALIDATION
    # ========================================================

    validate_source_files(
        patient_id
    )

    src = source_paths(
        patient_id
    )

    dst = final_paths(
        patient_id
    )

    dst[
        "directory"
    ].mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "SOURCE FILES"
    )

    print(
        f"T1        : {src['t1']}"
    )

    print(
        f"T2        : {src['t2']}"
    )

    print(
        f"FLAIR     : {src['flair']}"
    )

    print(
        f"Ventricle : {src['ventricle']}"
    )

    print(
        f"nWMH      : {src['nwmh']}"
    )

    print(
        f"abWMH     : {src['abwmh']}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    print()
    print(
        "Loading source volumes..."
    )

    t1 = read_image(
        src[
            "t1"
        ]
    )

    t2 = read_image(
        src[
            "t2"
        ]
    )

    flair = read_image(
        src[
            "flair"
        ]
    )

    vent = read_image(
        src[
            "ventricle"
        ]
    )

    nwmh = read_image(
        src[
            "nwmh"
        ]
    )

    abwmh = read_image(
        src[
            "abwmh"
        ]
    )

    print()
    print(
        "ORIGINAL GEOMETRY"
    )

    print_image_info(
        "T1",
        t1
    )

    print_image_info(
        "T2",
        t2
    )

    print_image_info(
        "FLAIR",
        flair
    )

    print_image_info(
        "Ventricle",
        vent
    )

    print_image_info(
        "nWMH",
        nwmh
    )

    print_image_info(
        "abWMH",
        abwmh
    )

    # ========================================================
    # STANDARDIZE MASKS TO 0/1
    # ========================================================

    vent = binarize_mask(
        vent
    )

    nwmh = binarize_mask(
        nwmh
    )

    abwmh = binarize_mask(
        abwmh
    )

    # ========================================================
    # SKULL STRIPPING
    # ========================================================

    print()
    print(
        "PREPROCESSING DECISIONS"
    )

    if yes(
        plan.get(
            "apply_skull_strip",
            "NO"
        )
    ):

        print(
            "Skull stripping: APPLY"
        )

        t1 = apply_skull_strip(
            t1
        )

        t2 = apply_skull_strip(
            t2
        )

        flair = apply_skull_strip(
            flair
        )

    else:

        print(
            "Skull stripping: SKIP"
        )

    # ========================================================
    # N4 CORRECTION
    # ========================================================

    if yes(
        plan.get(
            "apply_n4",
            "NO"
        )
    ):

        print(
            "N4 correction: APPLY"
        )

        t1 = apply_n4(
            t1,
            config
        )

        t2 = apply_n4(
            t2,
            config
        )

        flair = apply_n4(
            flair,
            config
        )

    else:

        print(
            "N4 correction: SKIP"
        )

    # ========================================================
    # REGISTRATION DECISION
    # ========================================================

    registration_requested = any(
        [

            yes(
                plan.get(
                    "register_t1_to_flair",
                    "NO"
                )
            ),

            yes(
                plan.get(
                    "register_t2_to_flair",
                    "NO"
                )
            ),

            yes(
                plan.get(
                    "register_masks_to_flair",
                    "NO"
                )
            ),
        ]
    )

    if registration_requested:

        raise NotImplementedError(
            "This patient is marked for additional registration. "
            "Stage-1 indicated the cohort was already registered. "
            "Review this patient separately before applying a "
            "new registration transform."
        )

    print(
        "Additional registration: SKIP "
        "(existing registered/validated volumes used)"
    )

    # ========================================================
    # CREATE COMMON FINAL GRID
    # ========================================================

    print()
    print(
        "Creating final FLAIR reference grid..."
    )

    reference = create_reference_grid(
        flair,
        config
    )

    print(
        "Reference size    :",
        tuple(
            reference.GetSize()
        )
    )

    print(
        "Reference spacing :",
        tuple(
            round(
                float(value),
                6
            )
            for value
            in reference.GetSpacing()
        )
    )

    print(
        "Reference origin  :",
        tuple(
            round(
                float(value),
                6
            )
            for value
            in reference.GetOrigin()
        )
    )

    # ========================================================
    # RESAMPLE
    # ========================================================

    if yes(
        plan.get(
            "resample_to_reference",
            "YES"
        )
    ):

        print()
        print(
            "Resampling images to final reference..."
        )

        t1 = resample_to_reference(
            t1,
            reference,
            is_mask=False,
            config=config
        )

        t2 = resample_to_reference(
            t2,
            reference,
            is_mask=False,
            config=config
        )

        flair = resample_to_reference(
            flair,
            reference,
            is_mask=False,
            config=config
        )

        print(
            "Resampling masks using nearest-neighbor..."
        )

        vent = resample_to_reference(
            vent,
            reference,
            is_mask=True,
            config=config
        )

        nwmh = resample_to_reference(
            nwmh,
            reference,
            is_mask=True,
            config=config
        )

        abwmh = resample_to_reference(
            abwmh,
            reference,
            is_mask=True,
            config=config
        )

        # Ensure interpolation cannot introduce
        # unexpected label values.
        vent = binarize_mask(
            vent
        )

        nwmh = binarize_mask(
            nwmh
        )

        abwmh = binarize_mask(
            abwmh
        )

    else:

        print(
            "Resampling: SKIP"
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    print()

    if yes(
        plan.get(
            "normalize_t1",
            "YES"
        )
    ):

        print(
            "Normalizing T1..."
        )

        t1 = normalize_image(
            t1,
            config
        )

    else:

        print(
            "T1 normalization: SKIP"
        )

    if yes(
        plan.get(
            "normalize_t2",
            "YES"
        )
    ):

        print(
            "Normalizing T2..."
        )

        t2 = normalize_image(
            t2,
            config
        )

    else:

        print(
            "T2 normalization: SKIP"
        )

    if yes(
        plan.get(
            "normalize_flair",
            "YES"
        )
    ):

        print(
            "Normalizing FLAIR..."
        )

        flair = normalize_image(
            flair,
            config
        )

    else:

        print(
            "FLAIR normalization: SKIP"
        )

    # ========================================================
    # PRE-SAVE GEOMETRY CHECK
    # ========================================================

    print()
    print(
        "Checking final in-memory geometry..."
    )

    final_objects = {

        "T1":
            t1,

        "T2":
            t2,

        "Ventricle":
            vent,

        "Normal WMH":
            nwmh,

        "Abnormal WMH":
            abwmh,
    }

    for name, image in final_objects.items():

        if not same_geometry(
            flair,
            image
        ):

            raise RuntimeError(
                f"{name} geometry does not match "
                f"final FLAIR geometry."
            )

    print(
        "All final geometries match."
    )

    # ========================================================
    # MASK LABEL CHECK
    # ========================================================

    vent_labels = validate_binary_mask(
        vent,
        "Ventricle"
    )

    nwmh_labels = validate_binary_mask(
        nwmh,
        "Normal WMH"
    )

    abwmh_labels = validate_binary_mask(
        abwmh,
        "Abnormal WMH"
    )

    print(
        "Final ventricle labels:",
        vent_labels
    )

    print(
        "Final nWMH labels:",
        nwmh_labels
    )

    print(
        "Final abWMH labels:",
        abwmh_labels
    )

    # ========================================================
    # SAVE
    # ========================================================

    print()
    print(
        "Saving compressed NIfTI outputs..."
    )

    safe_write(
        t1,
        dst[
            "t1"
        ]
    )

    safe_write(
        t2,
        dst[
            "t2"
        ]
    )

    safe_write(
        flair,
        dst[
            "flair"
        ]
    )

    safe_write(
        vent,
        dst[
            "ventricle"
        ]
    )

    safe_write(
        nwmh,
        dst[
            "nwmh"
        ]
    )

    safe_write(
        abwmh,
        dst[
            "abwmh"
        ]
    )

    # ========================================================
    # REOPEN AND VALIDATE SAVED OUTPUTS
    # ========================================================

    print()
    print(
        "Reopening outputs for final validation..."
    )

    validation = validate_final_outputs(
        patient_id
    )

    if not patient_outputs_complete(
        patient_id
    ):

        raise RuntimeError(
            "One or more expected output files "
            "are absent or unreadable."
        )

    print()
    print(
        "FINAL OUTPUT"
    )

    print(
        "Final shape   :",
        validation[
            "shape"
        ]
    )

    print(
        "Final spacing :",
        validation[
            "spacing"
        ]
    )

    print(
        "Ventricle labels:",
        validation[
            "vent_labels"
        ]
    )

    print(
        "nWMH labels      :",
        validation[
            "nwmh_labels"
        ]
    )

    print(
        "abWMH labels     :",
        validation[
            "abwmh_labels"
        ]
    )

    print()
    print(
        f"Output directory:\n"
        f"{dst['directory']}"
    )

    print()
    print(
        f"PATIENT {patient_id}: COMPLETE"
    )

    return (
        validation[
            "shape"
        ],
        validation[
            "spacing"
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_directories()

    args = parse_args()

    # ========================================================
    # DETERMINE PATIENTS TO PROCESS
    # ========================================================

    if (
        args.patient
        is not None
    ):

        requested = normalize_patient_id(
            args.patient
        )

        if requested not in PATIENT_IDS:

            raise ValueError(
                f"Invalid patient ID: {requested}"
            )

        patients_to_process = [
            requested
        ]

    else:

        patients_to_process = (
            PATIENT_IDS
        )

    # ========================================================
    # START
    # ========================================================

    print(
        "=" * 72
    )

    print(
        "MS3SEG STAGE 2 PREPROCESSING"
    )

    print(
        "=" * 72
    )

    print(
        "Started:",
        now()
    )

    print(
        "Patients requested:",
        len(
            patients_to_process
        )
    )

    if args.patient:

        print(
            "Single-patient mode:",
            patients_to_process[
                0
            ]
        )

    else:

        print(
            "Mode: FULL COHORT"
        )

    print(
        "Force reprocessing:",
        args.force
    )

    print()

    # ========================================================
    # VERIFY PLAN EXISTS
    # ========================================================

    if not PLAN_CSV.exists():

        raise FileNotFoundError(
            "Preprocessing plan not found:\n"
            + str(
                PLAN_CSV
            )
        )

    # ========================================================
    # LOAD PLAN
    # ========================================================

    plan_df = pd.read_csv(
        PLAN_CSV,
        dtype=str,
        keep_default_na=False
    )

    if (
        "patient_id"
        not in plan_df.columns
    ):

        raise ValueError(
            "preprocess_plan.csv has no patient_id column."
        )

    plan_df[
        "patient_id"
    ] = (
        plan_df[
            "patient_id"
        ]
        .astype(str)
        .str.zfill(3)
    )

    plan_lookup = (
        plan_df
        .set_index(
            "patient_id"
        )
        .to_dict(
            orient="index"
        )
    )

    # ========================================================
    # LOAD CONFIG
    # ========================================================

    config = load_config()

    target_cfg = (
        config
        .get(
            "preprocessing",
            {}
        )
        .get(
            "target_spacing_mm",
            {}
        )
    )

    print(
        "Target spacing configuration:"
    )

    print(
        "  X:",
        target_cfg.get(
            "x",
            "original"
        )
    )

    print(
        "  Y:",
        target_cfg.get(
            "y",
            "original"
        )
    )

    print(
        "  Z:",
        target_cfg.get(
            "z",
            "original"
        )
    )

    norm_cfg = (
        config
        .get(
            "preprocessing",
            {}
        )
        .get(
            "intensity_normalization",
            {}
        )
    )

    print(
        "Normalization:",
        norm_cfg.get(
            "method",
            "zscore"
        )
    )

    print()

    # ========================================================
    # LOAD MANIFEST
    # ========================================================

    manifest = load_manifest()

    # ========================================================
    # COUNTERS
    # ========================================================

    completed = 0

    skipped = 0

    failed = 0

    not_approved = 0

    # ========================================================
    # PATIENT LOOP
    # ========================================================

    for counter, patient_id in enumerate(
        patients_to_process,
        start=1
    ):

        print()
        print(
            "=" * 72
        )

        print(
            f"{counter}/{len(patients_to_process)} "
            f"| PATIENT {patient_id}"
        )

        print(
            "=" * 72
        )

        # ====================================================
        # PLAN CHECK
        # ====================================================

        if (
            patient_id
            not in plan_lookup
        ):

            message = (
                "No preprocessing plan found "
                "for this patient."
            )

            print(
                "FAILED:",
                message
            )

            failed += 1

            update_manifest_patient(
                manifest,
                patient_id,
                False,
                error=message
            )

            continue

        plan = plan_lookup[
            patient_id
        ]

        plan_status = str(
            plan.get(
                "plan_status",
                ""
            )
        ).strip().upper()

        if (
            plan_status
            != "APPROVED"
        ):

            print(
                "Preprocessing plan status:",
                plan_status
            )

            print(
                "Patient is not approved -> SKIP"
            )

            not_approved += 1
            skipped += 1

            continue

        # ====================================================
        # CHECKPOINT CHECK
        # ====================================================

        done_in_manifest = (
            manifest_patient_done(
                manifest,
                patient_id
            )
        )

        outputs_ok = (
            patient_outputs_complete(
                patient_id
            )
        )

        if (
            not args.force
            and
            done_in_manifest
            and
            outputs_ok
        ):

            print(
                "Checkpoint already complete."
            )

            print(
                "Manifest preprocess_done=True"
            )

            print(
                "All six output files readable."
            )

            print(
                "-> SKIP"
            )

            skipped += 1

            continue

        # ====================================================
        # FORCE MESSAGE
        # ====================================================

        if args.force:

            print(
                "FORCE enabled -> reprocessing patient."
            )

        elif (
            done_in_manifest
            and
            not outputs_ok
        ):

            print(
                "Manifest says complete but output "
                "checkpoint is incomplete."
            )

            print(
                "Patient will be rebuilt."
            )

        # ====================================================
        # PROCESS
        # ====================================================

        try:

            final_shape, final_spacing = (
                preprocess_patient(
                    patient_id,
                    plan,
                    config
                )
            )

            # ------------------------------------------------
            # IMPORTANT CHECKPOINT ORDER
            #
            # 1. preprocessing complete
            # 2. output files written
            # 3. output files reopened/validated
            # 4. THEN manifest preprocess_done=True
            # ------------------------------------------------

            update_manifest_patient(
                manifest,
                patient_id,
                True,
                shape=final_shape,
                spacing=final_spacing,
                error=""
            )

            completed += 1

        except Exception as error:

            failed += 1

            print()
            print(
                "!" * 72
            )

            print(
                f"PATIENT {patient_id}: FAILED"
            )

            print(
                "!" * 72
            )

            print(
                str(
                    error
                )
            )

            traceback.print_exc()

            log_error(
                patient_id,
                error
            )

            update_manifest_patient(
                manifest,
                patient_id,
                False,
                shape="",
                spacing="",
                error=str(
                    error
                )
            )

            # ------------------------------------------------
            # CRITICAL:
            # One failure must NOT terminate cohort.
            # ------------------------------------------------

            continue

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "STAGE 2 PREPROCESSING SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        f"Patients requested : "
        f"{len(patients_to_process)}"
    )

    print(
        f"Newly completed    : "
        f"{completed}"
    )

    print(
        f"Checkpoint skips   : "
        f"{skipped}"
    )

    print(
        f"Not approved       : "
        f"{not_approved}"
    )

    print(
        f"Failed             : "
        f"{failed}"
    )

    print()

    print(
        "Preprocessed output:"
    )

    print(
        FINAL_ROOT
    )

    print()

    print(
        "Manifest:"
    )

    print(
        MANIFEST_CSV
    )

    print()

    print(
        "Error log:"
    )

    print(
        ERROR_LOG
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
        "Finished:",
        now()
    )

    print(
        "=" * 72
    )


# ============================================================
# RUN + SAVE ALL TERMINAL OUTPUT TO TXT
# ============================================================

if __name__ == "__main__":

    ensure_directories()

    with open(
        TERMINAL_LOG,
        "a",
        encoding="utf-8"
    ) as logfile:

        logfile.write(
            "\n\n"
        )

        logfile.write(
            "=" * 72
            + "\n"
        )

        logfile.write(
            "NEW STAGE-2 PREPROCESSING RUN\n"
        )

        logfile.write(
            now()
            + "\n"
        )

        logfile.write(
            "=" * 72
            + "\n"
        )

        stdout_tee = Tee(
            sys.stdout,
            logfile
        )

        stderr_tee = Tee(
            sys.stderr,
            logfile
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
                    "=" * 72
                )

                print(
                    "FATAL STAGE-2 ERROR"
                )

                print(
                    "=" * 72
                )

                traceback.print_exc()

                raise