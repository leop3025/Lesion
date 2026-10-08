from pathlib import Path
import math

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# PATIENT
# =========================================================

PATIENT = "022"

ROOT = (
    Path(r"C:\Users\Preethi\Documents\LESION")
    / "distance_labels"
    / PATIENT
)


# =========================================================
# FILES
# =========================================================

FILES = [
    f"{PATIENT}_6class_pv_deep_labels.nii.gz",
    f"{PATIENT}_distance_to_ventricle_mm.nii.gz",
    f"{PATIENT}_distance_to_ventricle_normalized.nii.gz",
    f"{PATIENT}_pv_deep_zone_map.nii.gz",
    f"{PATIENT}_wmh_pv_deep_labels.nii.gz",
]


# =========================================================
# OPEN EACH FILE
# =========================================================

for filename in FILES:

    path = ROOT / filename

    if not path.exists():
        print("MISSING:", path)
        continue

    # Load NIfTI
    img = nib.load(path)
    data = np.asarray(img.dataobj)

    print("\n" + "=" * 70)
    print("FILE:", filename)
    print("Shape:", data.shape)
    print("Dtype:", data.dtype)
    print("Min:", float(np.min(data)))
    print("Max:", float(np.max(data)))

    # Useful for label maps
    unique_values = np.unique(data)

    if len(unique_values) <= 20:
        print("Unique values:", unique_values)

    print("=" * 70)


    # =====================================================
    # CREATE GRID FOR ALL SLICES
    # =====================================================

    num_slices = data.shape[2]

    cols = 5
    rows = math.ceil(num_slices / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(15, rows * 3)
    )

    axes = np.asarray(axes).reshape(-1)


    # =====================================================
    # DISPLAY SLICES
    # =====================================================

    for z in range(num_slices):

        slice_img = np.rot90(
            data[:, :, z]
        )

        axes[z].imshow(
            slice_img,
            cmap="viridis",
            interpolation="nearest"
        )

        axes[z].set_title(
            f"Slice {z}"
        )

        axes[z].axis("off")


    # =====================================================
    # REMOVE EMPTY BOXES
    # =====================================================

    for z in range(
        num_slices,
        len(axes)
    ):
        axes[z].axis("off")


    # =====================================================
    # TITLE
    # =====================================================

    fig.suptitle(
        f"Patient {PATIENT}\n{filename}",
        fontsize=15
    )

    plt.tight_layout()

    plt.show()