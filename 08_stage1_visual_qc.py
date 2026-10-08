from pathlib import Path
from datetime import datetime
import contextlib
import sys
import traceback

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Preethi\Documents\LESION"
)

OUTPUT_ROOT = PROJECT_ROOT / "output"

AUDIT_DIR = PROJECT_ROOT / "data" / "audit"

VISUAL_QC_DIR = AUDIT_DIR / "visual_qc"

REVIEW_CSV = (
    AUDIT_DIR /
    "visual_qc_review.csv"
)

LOG_DIR = PROJECT_ROOT / "logs"

TERMINAL_LOG = (
    LOG_DIR /
    "08_stage1_visual_qc_terminal.txt"
)

PATIENT_IDS = [
    f"{i:03d}"
    for i in range(1, 101)
]


# ============================================================
# LOGGING
# ============================================================

class Tee:

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
# HELPERS
# ============================================================

def timestamp():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def make_directories():

    VISUAL_QC_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# FILE DISCOVERY
# ============================================================

def belongs_to_patient(
    path,
    patient_id
):

    name = path.name.lower()

    pid = patient_id.lower()

    parts = [
        part.lower()
        for part in path.parts
    ]

    if pid in parts:
        return True

    if name.startswith(
        pid + "_"
    ):
        return True

    if name.startswith(
        pid + "-"
    ):
        return True

    return False


def all_nifti_files():

    files = []

    for pattern in [
        "*.nii",
        "*.nii.gz",
    ]:

        files.extend(
            OUTPUT_ROOT.rglob(
                pattern
            )
        )

    return sorted(
        set(files),
        key=lambda x: str(x).lower()
    )


def find_modality(
    patient_id,
    modality,
    all_files
):

    patient_files = [
        path
        for path in all_files
        if belongs_to_patient(
            path,
            patient_id
        )
    ]

    candidates = []

    for path in patient_files:

        text = str(
            path
        ).lower()

        name = path.name.lower()

        # Skip mask files
        if any(
            word in text
            for word in [
                "mask",
                "vent",
                "nwmh",
                "abwmh",
            ]
        ):
            continue

        # --------------------------------------------
        # T1
        # --------------------------------------------

        if modality == "t1":

            if "t1" in name:
                candidates.append(
                    path
                )

        # --------------------------------------------
        # T2
        # --------------------------------------------

        elif modality == "t2":

            if (
                "t2" in name
                and
                "flair" not in name
            ):
                candidates.append(
                    path
                )

        # --------------------------------------------
        # FLAIR
        # --------------------------------------------

        elif modality == "flair":

            if "flair" in name:

                if (
                    "sag" not in text
                    and
                    "sagittal" not in text
                ):
                    candidates.append(
                        path
                    )

    if not candidates:
        return None

    # Prefer preprocessed version
    preprocessed = [
        path
        for path in candidates
        if (
            "patient_preprocessed"
            in str(path).lower()
        )
    ]

    if preprocessed:
        candidates = preprocessed

    # Prefer .nii.gz
    compressed = [
        path
        for path in candidates
        if str(path).lower().endswith(
            ".nii.gz"
        )
    ]

    if compressed:
        candidates = compressed

    return sorted(
        candidates,
        key=lambda x: (
            len(str(x)),
            str(x).lower()
        )
    )[0]


# ============================================================
# IMAGE LOADING
# ============================================================

def load_volume(
    path
):

    if path is None:
        return None

    try:

        image = nib.load(
            str(path)
        )

        data = np.asarray(
            image.dataobj,
            dtype=np.float32
        )

        data[
            ~np.isfinite(data)
        ] = 0

        return data

    except Exception:

        return None


# ============================================================
# DISPLAY NORMALIZATION
# ============================================================

def normalize_for_display(
    image
):

    if image is None:
        return None

    values = image[
        np.isfinite(
            image
        )
    ]

    if values.size == 0:
        return image

    nonzero = values[
        values != 0
    ]

    if nonzero.size == 0:
        return image

    low = np.percentile(
        nonzero,
        1
    )

    high = np.percentile(
        nonzero,
        99
    )

    if high <= low:
        return image

    image = np.clip(
        image,
        low,
        high
    )

    image = (
        image - low
    ) / (
        high - low
    )

    return image


# ============================================================
# GET REPRESENTATIVE SLICES
# ============================================================

def representative_slices(
    volume
):

    if volume is None:
        return [
            None,
            None,
            None
        ]

    if volume.ndim != 3:

        return [
            None,
            None,
            None
        ]

    z = volume.shape[2]

    indices = [
        int(z * 0.35),
        int(z * 0.50),
        int(z * 0.65),
    ]

    slices = []

    for index in indices:

        index = max(
            0,
            min(
                z - 1,
                index
            )
        )

        slices.append(
            volume[
                :,
                :,
                index
            ]
        )

    return slices


# ============================================================
# SIMPLE AUTOMATED HEURISTICS
# ============================================================

def calculate_visual_hints(
    volume
):

    result = {
        "zero_fraction": None,
        "edge_zero_fraction": None,
        "skull_strip_hint": "uncertain",
        "n4_hint": "uncertain",
        "needs_visual_check": True,
    }

    if volume is None:

        return result

    finite = np.isfinite(
        volume
    )

    if not finite.any():

        return result

    sample = volume[
        finite
    ]

    result[
        "zero_fraction"
    ] = float(
        np.mean(
            np.isclose(
                sample,
                0,
                atol=1e-6
            )
        )
    )

    edges = np.concatenate(
        [
            volume[0, :, :].ravel(),
            volume[-1, :, :].ravel(),
            volume[:, 0, :].ravel(),
            volume[:, -1, :].ravel(),
        ]
    )

    result[
        "edge_zero_fraction"
    ] = float(
        np.mean(
            np.isclose(
                edges,
                0,
                atol=1e-6
            )
        )
    )

    if (
        result[
            "edge_zero_fraction"
        ] > 0.90
    ):

        result[
            "skull_strip_hint"
        ] = (
            "possibly skull-stripped"
        )

    elif (
        result[
            "edge_zero_fraction"
        ] < 0.50
    ):

        result[
            "skull_strip_hint"
        ] = (
            "possibly not skull-stripped"
        )

    else:

        result[
            "skull_strip_hint"
        ] = "uncertain"

    # N4 cannot be reliably established automatically
    result[
        "n4_hint"
    ] = (
        "cannot confirm automatically"
    )

    return result


# ============================================================
# CREATE QC FIGURE
# ============================================================

def create_qc_figure(
    patient_id,
    paths
):

    modalities = [
        ("T1", paths["t1"]),
        ("T2", paths["t2"]),
        ("Axial FLAIR", paths["flair"]),
    ]

    loaded = {}

    for label, path in modalities:

        volume = load_volume(
            path
        )

        loaded[
            label
        ] = (
            normalize_for_display(
                volume
            )
        )

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12, 12)
    )

    for row_index, (
        modality_name,
        _
    ) in enumerate(
        modalities
    ):

        volume = loaded[
            modality_name
        ]

        slices = (
            representative_slices(
                volume
            )
        )

        for column_index, slice_2d in enumerate(
            slices
        ):

            axis = axes[
                row_index,
                column_index
            ]

            axis.axis(
                "off"
            )

            if slice_2d is None:

                axis.text(
                    0.5,
                    0.5,
                    "NOT AVAILABLE",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=axis.transAxes
                )

            else:

                # Rotate only for easier viewing.
                # This does NOT modify source data.
                display = np.rot90(
                    slice_2d
                )

                axis.imshow(
                    display,
                    cmap="gray"
                )

            if column_index == 0:

                axis.set_title(
                    modality_name
                )

    fig.suptitle(
        (
            f"Patient {patient_id}\n"
            "Stage-1 Visual QC: "
            "Skull stripping / intensity bias review"
        ),
        fontsize=14
    )

    fig.tight_layout()

    destination = (
        VISUAL_QC_DIR /
        f"{patient_id}_visual_qc.png"
    )

    fig.savefig(
        destination,
        dpi=140,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    return destination


# ============================================================
# REVIEW CSV
# ============================================================

def create_review_row(
    patient_id,
    paths
):

    flair_volume = load_volume(
        paths[
            "flair"
        ]
    )

    hints = calculate_visual_hints(
        flair_volume
    )

    return {

        "patient_id":
            patient_id,

        "t1_path":
            (
                str(paths["t1"])
                if paths["t1"]
                else ""
            ),

        "t2_path":
            (
                str(paths["t2"])
                if paths["t2"]
                else ""
            ),

        "flair_path":
            (
                str(paths["flair"])
                if paths["flair"]
                else ""
            ),

        "automated_skull_strip_hint":
            hints[
                "skull_strip_hint"
            ],

        "automated_n4_hint":
            hints[
                "n4_hint"
            ],

        "zero_fraction":
            hints[
                "zero_fraction"
            ],

        "edge_zero_fraction":
            hints[
                "edge_zero_fraction"
            ],

        # ----------------------------------------------------
        # YOU fill these columns after viewing the PNGs
        # ----------------------------------------------------

        "visual_skull_stripped":
            "",

        "visual_bias_field":
            "",

        "visual_n4_needed":
            "",

        "visual_registration_issue":
            "",

        "visual_other_issue":
            "",

        "review_status":
            "PENDING",

        "review_notes":
            "",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    make_directories()

    print("=" * 72)

    print(
        "MS3SEG STAGE-1 VISUAL QC"
    )

    print("=" * 72)

    print(
        f"Started: {timestamp()}"
    )

    print()

    print(
        "Scanning NIfTI files..."
    )

    files = all_nifti_files()

    print(
        f"NIfTI files indexed: {len(files)}"
    )

    print()

    rows = []

    generated = 0
    reused = 0
    errors = 0

    for index, patient_id in enumerate(
        PATIENT_IDS,
        start=1
    ):

        print(
            "-" * 72
        )

        print(
            f"PATIENT {patient_id} "
            f"({index}/100)"
        )

        paths = {

            "t1":
                find_modality(
                    patient_id,
                    "t1",
                    files
                ),

            "t2":
                find_modality(
                    patient_id,
                    "t2",
                    files
                ),

            "flair":
                find_modality(
                    patient_id,
                    "flair",
                    files
                ),
        }

        print(
            "T1    :",
            paths["t1"]
            if paths["t1"]
            else "NOT FOUND"
        )

        print(
            "T2    :",
            paths["t2"]
            if paths["t2"]
            else "NOT FOUND"
        )

        print(
            "FLAIR :",
            paths["flair"]
            if paths["flair"]
            else "NOT FOUND"
        )

        output_png = (
            VISUAL_QC_DIR /
            f"{patient_id}_visual_qc.png"
        )

        try:

            # ------------------------------------------------
            # Resume/checkpoint behavior
            # ------------------------------------------------

            if output_png.exists():

                print(
                    "QC image already exists -> SKIP"
                )

                reused += 1

            else:

                output_png = (
                    create_qc_figure(
                        patient_id,
                        paths
                    )
                )

                print(
                    "QC image created:"
                )

                print(
                    output_png
                )

                generated += 1

            rows.append(
                create_review_row(
                    patient_id,
                    paths
                )
            )

        except Exception as error:

            errors += 1

            print(
                "ERROR:"
            )

            print(
                str(error)
            )

            traceback.print_exc()

    # ========================================================
    # SAVE REVIEW TABLE
    # ========================================================

    new_df = pd.DataFrame(
        rows
    )

    # Preserve your manual review if CSV already exists
    if REVIEW_CSV.exists():

        old_df = pd.read_csv(
            REVIEW_CSV,
            dtype={
                "patient_id": str
            },
            keep_default_na=False
        )

        old_df[
            "patient_id"
        ] = (
            old_df[
                "patient_id"
            ]
            .astype(str)
            .str.zfill(3)
        )

        manual_columns = [
            "visual_skull_stripped",
            "visual_bias_field",
            "visual_n4_needed",
            "visual_registration_issue",
            "visual_other_issue",
            "review_status",
            "review_notes",
        ]

        old_lookup = (
            old_df
            .set_index(
                "patient_id"
            )
        )

        for index, row in new_df.iterrows():

            pid = row[
                "patient_id"
            ]

            if pid not in old_lookup.index:
                continue

            for column in manual_columns:

                if column in old_lookup.columns:

                    previous = old_lookup.at[
                        pid,
                        column
                    ]

                    if str(
                        previous
                    ).strip():

                        new_df.at[
                            index,
                            column
                        ] = previous

    new_df.to_csv(
        REVIEW_CSV,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 72)

    print(
        "VISUAL QC GENERATION SUMMARY"
    )

    print("=" * 72)

    print(
        f"Patients            : {len(PATIENT_IDS)}"
    )

    print(
        f"New PNGs generated  : {generated}"
    )

    print(
        f"Existing PNGs reused: {reused}"
    )

    print(
        f"Errors              : {errors}"
    )

    print()

    print(
        "QC images:"
    )

    print(
        VISUAL_QC_DIR
    )

    print()

    print(
        "Review CSV:"
    )

    print(
        REVIEW_CSV
    )

    print()

    print(
        "NEXT ACTION:"
    )

    print(
        "Open the QC PNGs and fill the manual review "
        "columns in visual_qc_review.csv."
    )

    print()

    print(
        "Recommended entries:"
    )

    print(
        "visual_skull_stripped : YES / NO / UNCERTAIN"
    )

    print(
        "visual_bias_field     : NONE / MILD / OBVIOUS / UNCERTAIN"
    )

    print(
        "visual_n4_needed      : YES / NO / UNCERTAIN"
    )

    print(
        "review_status         : REVIEWED"
    )

    print()

    print(
        f"Finished: {timestamp()}"
    )

    print("=" * 72)


# ============================================================
# RUN AND SAVE TERMINAL OUTPUT
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
            "NEW VISUAL QC RUN\n"
        )

        logfile.write(
            timestamp()
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

                print(
                    "FATAL ERROR"
                )

                traceback.print_exc()

                raise