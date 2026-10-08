from pathlib import Path
import sys

import nibabel as nib
import numpy as np


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(
    r"C:\Users\Preethi\Documents\LESION"
)

PREPROCESSED_ROOT = (
    BASE_DIR
    / "preprocessed"
)

REPORT_PATH = (
    BASE_DIR
    / "logs"
    / "11_validate_preprocessed_terminal.txt"
)

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# TERMINAL + TXT OUTPUT
# =========================================================

class Tee:
    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


report_file = open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
)

sys.stdout = Tee(
    sys.__stdout__,
    report_file
)


# =========================================================
# EXPECTED FILE TYPES
# =========================================================

EXPECTED_KEYWORDS = {
    "FLAIR": "flair",
    "T1": "t1",
    "T2": "t2",
    "Ventricle Mask": "vent",
    "Normal WMH Mask": "nwmh",
    "Abnormal WMH Mask": "abwmh"
}


# =========================================================
# COUNTERS
# =========================================================

total = 100

passed = 0
failed = 0

missing = 0
corrupt = 0

nan_files = 0
inf_files = 0

geometry_mismatches = 0
unexpected_mask_labels = 0


# =========================================================
# HELPER
# =========================================================

def find_file(patient_dir, keyword):

    candidates = []

    for path in patient_dir.rglob("*.nii*"):

        name = path.name.lower()

        if keyword.lower() in name:
            candidates.append(path)

    if len(candidates) == 0:
        return None

    return candidates[0]


# =========================================================
# START
# =========================================================

print("=" * 75)
print("PREPROCESSED DATA VALIDATION")
print("=" * 75)

print("\nRoot:")
print(PREPROCESSED_ROOT)


# =========================================================
# PATIENT LOOP
# =========================================================

for i in range(1, 101):

    patient = f"{i:03d}"

    patient_dir = (
        PREPROCESSED_ROOT
        / patient
    )

    patient_ok = True

    print("\n" + "=" * 65)
    print(f"PATIENT {patient}")
    print("=" * 65)


    # -----------------------------------------------------
    # PATIENT FOLDER
    # -----------------------------------------------------

    if not patient_dir.exists():

        print("PATIENT FOLDER: MISSING")

        missing += 1
        failed += 1

        continue


    # =====================================================
    # FIND EXPECTED FILES
    # =====================================================

    files = {}

    for label, keyword in EXPECTED_KEYWORDS.items():

        path = find_file(
            patient_dir,
            keyword
        )

        files[label] = path

        print(f"\n{label}:")

        if path is None:

            print("  MISSING")

            missing += 1
            patient_ok = False

        else:

            print(
                " ",
                path.name
            )


    # =====================================================
    # LOAD FLAIR FIRST AS REFERENCE
    # =====================================================

    flair_path = files["FLAIR"]

    flair_img = None

    if flair_path is not None:

        try:

            flair_img = nib.load(
                flair_path
            )

        except Exception as e:

            print(
                "\nFLAIR could not be loaded:"
            )

            print(e)

            corrupt += 1
            patient_ok = False


    # =====================================================
    # CHECK EACH FILE
    # =====================================================

    for label, path in files.items():

        if path is None:
            continue


        print("\n" + "-" * 50)
        print(label)
        print("-" * 50)


        # -------------------------------------------------
        # LOAD
        # -------------------------------------------------

        try:

            img = nib.load(
                path
            )

            data = np.asanyarray(
                img.dataobj
            )

        except Exception as e:

            print(
                "Status: CORRUPT / UNREADABLE"
            )

            print(
                "Error:",
                e
            )

            corrupt += 1
            patient_ok = False

            continue


        # -------------------------------------------------
        # BASIC INFORMATION
        # -------------------------------------------------

        print(
            "Shape:",
            img.shape
        )

        print(
            "Spacing:",
            tuple(
                float(x)
                for x in img.header.get_zooms()[:3]
            )
        )

        print(
            "Orientation:",
            nib.aff2axcodes(
                img.affine
            )
        )

        print(
            "Dtype:",
            img.get_data_dtype()
        )


        # -------------------------------------------------
        # NAN CHECK
        # -------------------------------------------------

        has_nan = np.isnan(
            data
        ).any()

        print(
            "Contains NaN:",
            has_nan
        )

        if has_nan:

            nan_files += 1
            patient_ok = False


        # -------------------------------------------------
        # INF CHECK
        # -------------------------------------------------

        has_inf = np.isinf(
            data
        ).any()

        print(
            "Contains Inf:",
            has_inf
        )

        if has_inf:

            inf_files += 1
            patient_ok = False


        # -------------------------------------------------
        # INTENSITY RANGE
        # -------------------------------------------------

        finite_data = data[
            np.isfinite(data)
        ]

        if finite_data.size > 0:

            print(
                "Min:",
                float(np.min(finite_data))
            )

            print(
                "Max:",
                float(np.max(finite_data))
            )

            print(
                "Mean:",
                float(np.mean(finite_data))
            )


        # =================================================
        # MASK CHECKS
        # =================================================

        if "Mask" in label:

            labels, counts = np.unique(
                data,
                return_counts=True
            )

            print(
                "Unique labels:",
                labels.tolist()
            )

            print(
                "Voxel counts:"
            )

            for mask_label, count in zip(
                labels,
                counts
            ):

                print(
                    f"  {mask_label}: "
                    f"{int(count)}"
                )


            # ---------------------------------------------
            # LABEL SANITY
            # ---------------------------------------------

            allowed = {
                0,
                1
            }

            actual = set(
                float(x)
                for x in labels
            )

            if not actual.issubset(
                allowed
            ):

                print(
                    "WARNING: unexpected mask labels"
                )

                unexpected_mask_labels += 1
                patient_ok = False


        # =================================================
        # GEOMETRY AGAINST FLAIR
        # =================================================

        if (
            flair_img is not None
            and label != "FLAIR"
        ):

            shape_ok = (
                img.shape[:3]
                ==
                flair_img.shape[:3]
            )

            spacing_ok = np.allclose(
                img.header.get_zooms()[:3],
                flair_img.header.get_zooms()[:3],
                atol=1e-4
            )

            orientation_ok = (
                nib.aff2axcodes(
                    img.affine
                )
                ==
                nib.aff2axcodes(
                    flair_img.affine
                )
            )

            affine_ok = np.allclose(
                img.affine,
                flair_img.affine,
                atol=1e-4
            )


            print("\nGeometry vs FLAIR:")

            print(
                "  Shape match:",
                shape_ok
            )

            print(
                "  Spacing match:",
                spacing_ok
            )

            print(
                "  Orientation match:",
                orientation_ok
            )

            print(
                "  Affine match:",
                affine_ok
            )


            if not (
                shape_ok
                and spacing_ok
                and orientation_ok
                and affine_ok
            ):

                geometry_mismatches += 1
                patient_ok = False


    # =====================================================
    # PATIENT RESULT
    # =====================================================

    if patient_ok:

        print(
            f"\nPATIENT {patient}: PASS"
        )

        passed += 1

    else:

        print(
            f"\nPATIENT {patient}: FAIL"
        )

        failed += 1


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 75)
print("FINAL SUMMARY")
print("=" * 75)

print(
    "Total patients:",
    total
)

print(
    "Passed patients:",
    passed
)

print(
    "Failed patients:",
    failed
)

print(
    "Missing files/folders:",
    missing
)

print(
    "Corrupt files:",
    corrupt
)

print(
    "Files containing NaN:",
    nan_files
)

print(
    "Files containing Inf:",
    inf_files
)

print(
    "Geometry mismatches:",
    geometry_mismatches
)

print(
    "Unexpected mask labels:",
    unexpected_mask_labels
)

print("\nValidation complete.")

print("\nReport saved to:")
print(REPORT_PATH)


# =========================================================
# CLOSE LOG
# =========================================================

report_file.flush()
report_file.close()