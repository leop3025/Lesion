from pathlib import Path
import nibabel as nib
import numpy as np
import sys


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(
    r"C:\Users\Preethi\Documents\LESION"
)

MASK_ROOT = (
    BASE_DIR
    / "output"
    / "MS_100_patient_masks"
    / "MS_100_patient_masks"
)

FLAIR_ROOT = (
    BASE_DIR
    / "output"
    / "MS_100_patient_preprocessed"
    / "MS_100_patient_preprocessed"
)

REPORT_PATH = (
    BASE_DIR
    / "06_mask_flair_geometry_report.txt"
)


# =========================================================
# SAVE TERMINAL OUTPUT TO TXT ALSO
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
# MASK TYPES
# =========================================================

MASK_TYPES = {

    "Ventricle": (
        "Vent_Masks",
        "Vent"
    ),

    "Normal WMH": (
        "nWMH_Masks",
        "nWMH"
    ),

    "Abnormal WMH": (
        "abWMH_Masks",
        "abWMH"
    )
}


# =========================================================
# COUNTERS
# =========================================================

total_patients = 100

passed_patients = 0
failed_patients = 0

missing_files = 0
corrupt_files = 0

shape_mismatches = 0
spacing_mismatches = 0
orientation_mismatches = 0
affine_mismatches = 0


# =========================================================
# START REPORT
# =========================================================

print("=" * 70)
print("MS3SEG MASK VS FLAIR GEOMETRY VALIDATION")
print("=" * 70)


# =========================================================
# PATIENT LOOP
# =========================================================

for i in range(1, 101):

    patient = f"{i:03d}"

    patient_ok = True

    print("\n" + "=" * 60)
    print(f"PATIENT {patient}")
    print("=" * 60)


    # =====================================================
    # LOAD FLAIR
    # =====================================================

    flair_path = (
        FLAIR_ROOT
        / patient
        / f"{patient}_FLAIR.nii.gz"
    )

    print("\nFLAIR:")
    print(flair_path)

    if not flair_path.exists():

        print("FLAIR STATUS: MISSING")

        patient_ok = False
        missing_files += 1

        print(
            f"\nPATIENT {patient}: FAIL"
        )

        failed_patients += 1

        continue


    try:

        flair = nib.load(
            flair_path
        )

    except Exception as e:

        print(
            "FLAIR STATUS: CORRUPT"
        )

        print(
            "Error:",
            e
        )

        corrupt_files += 1
        failed_patients += 1

        continue


    print(
        "FLAIR Shape:",
        flair.shape
    )

    print(
        "FLAIR spacing:",
        tuple(
            float(x)
            for x in flair.header.get_zooms()[:3]
        )
    )

    print(
        "FLAIR orientation:",
        nib.aff2axcodes(
            flair.affine
        )
    )


    # =====================================================
    # MASK LOOP
    # =====================================================

    for mask_name, (
        folder_name,
        short_name
    ) in MASK_TYPES.items():


        mask_path = (
            MASK_ROOT
            / folder_name
            / patient
            / f"{patient}_{short_name}_Mask.nii.gz"
        )


        print("\n" + "-" * 50)
        print(mask_name)
        print("-" * 50)

        print(
            "Mask path:",
            mask_path
        )


        # -------------------------------------------------
        # EXISTS
        # -------------------------------------------------

        if not mask_path.exists():

            print(
                "Status: MISSING"
            )

            missing_files += 1
            patient_ok = False

            continue


        # -------------------------------------------------
        # LOAD
        # -------------------------------------------------

        try:

            mask = nib.load(
                mask_path
            )

            data = np.asanyarray(
                mask.dataobj
            )

        except Exception as e:

            print(
                "Status: CORRUPT"
            )

            print(
                "Error:",
                e
            )

            corrupt_files += 1
            patient_ok = False

            continue


        # =================================================
        # MASK INFORMATION
        # =================================================

        mask_shape = (
            mask.shape[:3]
        )

        mask_spacing = (
            mask.header.get_zooms()[:3]
        )

        mask_orientation = (
            nib.aff2axcodes(
                mask.affine
            )
        )

        mask_origin = (
            mask.affine[:3, 3]
        )

        dtype = (
            mask.get_data_dtype()
        )


        labels, counts = np.unique(
            data,
            return_counts=True
        )


        print(
            "Shape:",
            mask_shape
        )

        print(
            "Spacing:",
            tuple(
                float(x)
                for x in mask_spacing
            )
        )

        print(
            "Orientation:",
            mask_orientation
        )

        print(
            "Origin:",
            mask_origin.tolist()
        )

        print(
            "Dtype:",
            dtype
        )

        print(
            "Labels:",
            labels.tolist()
        )


        print(
            "Voxel counts:"
        )

        for label, count in zip(
            labels,
            counts
        ):

            print(
                f"  Label {label}: "
                f"{int(count)}"
            )


        # =================================================
        # GEOMETRY COMPARISON
        # =================================================

        flair_shape = (
            flair.shape[:3]
        )

        flair_spacing = (
            flair.header.get_zooms()[:3]
        )

        flair_orientation = (
            nib.aff2axcodes(
                flair.affine
            )
        )


        # -------------------------------------------------
        # SHAPE
        # -------------------------------------------------

        shape_ok = (
            flair_shape
            ==
            mask_shape
        )


        # -------------------------------------------------
        # SPACING
        # -------------------------------------------------

        spacing_ok = (
            np.allclose(
                flair_spacing,
                mask_spacing,
                atol=1e-4
            )
        )


        # -------------------------------------------------
        # ORIENTATION
        # -------------------------------------------------

        orientation_ok = (
            flair_orientation
            ==
            mask_orientation
        )


        # -------------------------------------------------
        # AFFINE
        # -------------------------------------------------

        affine_ok = (
            np.allclose(
                flair.affine,
                mask.affine,
                atol=1e-4
            )
        )


        # =================================================
        # PRINT COMPARISON
        # =================================================

        print("\nGeometry comparison:")

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


        # =================================================
        # COUNT MISMATCHES
        # =================================================

        if not shape_ok:

            shape_mismatches += 1
            patient_ok = False


        if not spacing_ok:

            spacing_mismatches += 1
            patient_ok = False


        if not orientation_ok:

            orientation_mismatches += 1
            patient_ok = False


        if not affine_ok:

            affine_mismatches += 1
            patient_ok = False


        # =================================================
        # MASK RESULT
        # =================================================

        if (
            shape_ok
            and spacing_ok
            and orientation_ok
            and affine_ok
        ):

            print(
                f"{mask_name}: PASS"
            )

        else:

            print(
                f"{mask_name}: FAIL"
            )


    # =====================================================
    # PATIENT RESULT
    # =====================================================

    if patient_ok:

        print(
            f"\nPATIENT {patient}: PASS"
        )

        passed_patients += 1

    else:

        print(
            f"\nPATIENT {patient}: FAIL"
        )

        failed_patients += 1


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    "Total patients:",
    total_patients
)

print(
    "Passed patients:",
    passed_patients
)

print(
    "Failed patients:",
    failed_patients
)

print(
    "Missing files:",
    missing_files
)

print(
    "Corrupt files:",
    corrupt_files
)

print(
    "Shape mismatches:",
    shape_mismatches
)

print(
    "Spacing mismatches:",
    spacing_mismatches
)

print(
    "Orientation mismatches:",
    orientation_mismatches
)

print(
    "Affine mismatches:",
    affine_mismatches
)


print("\nValidation complete.")

print(
    "\nReport saved to:"
)

print(
    REPORT_PATH
)


# =========================================================
# CLOSE REPORT
# =========================================================

report_file.flush()
report_file.close()