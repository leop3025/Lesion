from __future__ import annotations

from pathlib import Path
import csv
import math

import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(r"C:\Users\Preethi\Documents\LESION")

MANIFEST_PATH = PROJECT_DIR / "manifest.csv"

OUTPUT_DIR = PROJECT_DIR / "data" / "audit" / "stage1_full_audit"
OVERLAY_DIR = OUTPUT_DIR / "mask_overlays"

AUDIT_CSV = OUTPUT_DIR / "stage1_audit.csv"
SUMMARY_TXT = OUTPUT_DIR / "stage1_audit_summary.txt"

REGISTERED_ROOT = (
    PROJECT_DIR
    / "output"
    / "MS_100_patient_registered"
    / "MS_100_patient_registered"
)

# If True, regenerate PNGs even if they already exist.
OVERWRITE_PNG = False

# Axial slice positions as fractions of the z-axis.
SLICE_FRACTIONS = (0.35, 0.50, 0.65)

# Geometry tolerance.
AFFINE_ATOL = 1e-4
SPACING_ATOL = 1e-5

# Overlay transparency.
ALPHA = 0.55


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def discover_registered_files(patient_id: str) -> dict[str, Path | None]:
    pdir = REGISTERED_ROOT / patient_id

    result = {
        "t1": None,
        "t2": None,
        "brain_mask": None,
    }

    if not pdir.exists():
        return result

    result["t1"] = find_first_existing(
        [
            pdir / f"{patient_id}_T1WI_reg.nii.gz",
            pdir / f"{patient_id}_T1_reg.nii.gz",
            pdir / f"{patient_id}_T1WI.nii.gz",
            pdir / f"{patient_id}_T1.nii.gz",
        ]
    )

    result["t2"] = find_first_existing(
        [
            pdir / f"{patient_id}_T2WI_reg.nii.gz",
            pdir / f"{patient_id}_T2_reg.nii.gz",
            pdir / f"{patient_id}_T2WI.nii.gz",
            pdir / f"{patient_id}_T2.nii.gz",
        ]
    )

    result["brain_mask"] = find_first_existing(
        [
            pdir / f"{patient_id}_brain_mask.nii.gz",
            pdir / f"{patient_id}_Brain_Mask.nii.gz",
        ]
    )

    # Fallback keyword search if exact names differ.
    files = list(pdir.glob("*.nii*"))

    if result["t1"] is None:
        for f in files:
            name = f.name.lower()
            if "t1" in name and "mask" not in name:
                result["t1"] = f
                break

    if result["t2"] is None:
        for f in files:
            name = f.name.lower()
            if "t2" in name and "mask" not in name:
                result["t2"] = f
                break

    if result["brain_mask"] is None:
        for f in files:
            name = f.name.lower()
            if "brain" in name and "mask" in name:
                result["brain_mask"] = f
                break

    return result


# ============================================================
# NIFTI HELPERS
# ============================================================

def load_nifti(path: Path):
    return nib.load(str(path))


def array3d(img) -> np.ndarray:
    arr = np.asanyarray(img.dataobj)
    if arr.ndim > 3:
        arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape}")
    return arr


def shape3d(img) -> tuple[int, int, int]:
    return tuple(int(x) for x in img.shape[:3])


def spacing3d(img) -> tuple[float, float, float]:
    return tuple(float(x) for x in img.header.get_zooms()[:3])


def orientation_codes(img) -> str:
    try:
        return "".join(nib.aff2axcodes(img.affine))
    except Exception:
        return "UNKNOWN"


def affine_matches(a, b) -> bool:
    return bool(np.allclose(a, b, atol=AFFINE_ATOL))


def spacing_matches(a, b) -> bool:
    return bool(np.allclose(spacing3d(a), spacing3d(b), atol=SPACING_ATOL))


def binary_mask_stats(mask: np.ndarray) -> tuple[int, str]:
    finite = mask[np.isfinite(mask)]
    if finite.size == 0:
        return 0, "NO_FINITE_VALUES"

    uniq = np.unique(finite)
    count = int(np.count_nonzero(finite))

    if uniq.size <= 10:
        values = ",".join(str(float(x)) for x in uniq.tolist())
    else:
        values = f"{uniq.size}_UNIQUE_VALUES"

    return count, values


def intensity_stats(image: np.ndarray, brain_mask: np.ndarray | None) -> dict[str, float | int]:
    finite = np.isfinite(image)

    if brain_mask is not None and brain_mask.shape == image.shape:
        use = finite & (brain_mask > 0)
        if np.count_nonzero(use) == 0:
            use = finite
    else:
        use = finite

    vals = image[use]

    if vals.size == 0:
        return {
            "min": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "std": math.nan,
            "p01": math.nan,
            "p99": math.nan,
        }

    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "p01": float(np.percentile(vals, 1)),
        "p99": float(np.percentile(vals, 99)),
    }


def masked_zscore_like(mean: float, std: float) -> bool:
    """
    Conservative heuristic only.
    It does NOT prove normalization was performed.
    """
    if not np.isfinite(mean) or not np.isfinite(std):
        return False
    return abs(mean) < 0.15 and 0.75 <= std <= 1.25


# ============================================================
# VISUALIZATION
# ============================================================

VENT_CMAP = ListedColormap([(0, 0, 0, 0), (0.0, 1.0, 1.0, 1.0)])   # cyan
NWMH_CMAP = ListedColormap([(0, 0, 0, 0), (1.0, 1.0, 0.0, 1.0)])   # yellow
ABWMH_CMAP = ListedColormap([(0, 0, 0, 0), (1.0, 0.0, 0.0, 1.0)])  # red


def display_range(image: np.ndarray, brain_mask: np.ndarray | None):
    finite = np.isfinite(image)

    if brain_mask is not None and brain_mask.shape == image.shape:
        valid = finite & (brain_mask > 0)
        if np.count_nonzero(valid) < 100:
            valid = finite
    else:
        valid = finite

    vals = image[valid]
    if vals.size == 0:
        return 0.0, 1.0

    lo = float(np.percentile(vals, 1))
    hi = float(np.percentile(vals, 99))

    if hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))

    if hi <= lo:
        hi = lo + 1.0

    return lo, hi


def make_overlay_png(
    patient_id: str,
    flair: np.ndarray,
    vent: np.ndarray,
    nwmh: np.ndarray,
    lesion: np.ndarray,
    brain_mask: np.ndarray | None,
    out_path: Path,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not OVERWRITE_PNG:
        return

    zmax = flair.shape[2] - 1
    z_indices = [
        max(0, min(zmax, int(round(zmax * frac))))
        for frac in SLICE_FRACTIONS
    ]

    vmin, vmax = display_range(flair, brain_mask)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(11, 13),
        constrained_layout=True,
    )

    row_labels = [
        "FLAIR",
        "Ventricle overlay (cyan)",
        "WMH overlays",
        "Combined masks",
    ]

    for col, z in enumerate(z_indices):
        base = np.rot90(flair[:, :, z])

        # Row 1: FLAIR only
        ax = axes[0, col]
        ax.imshow(base, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"Axial slice z={z}")
        ax.axis("off")

        # Row 2: Ventricle
        ax = axes[1, col]
        ax.imshow(base, cmap="gray", vmin=vmin, vmax=vmax)
        ax.imshow(
            np.rot90((vent[:, :, z] > 0).astype(np.uint8)),
            cmap=VENT_CMAP,
            vmin=0,
            vmax=1,
            alpha=ALPHA,
            interpolation="nearest",
        )
        ax.axis("off")

        # Row 3: nWMH + abWMH
        ax = axes[2, col]
        ax.imshow(base, cmap="gray", vmin=vmin, vmax=vmax)
        ax.imshow(
            np.rot90((nwmh[:, :, z] > 0).astype(np.uint8)),
            cmap=NWMH_CMAP,
            vmin=0,
            vmax=1,
            alpha=ALPHA,
            interpolation="nearest",
        )
        ax.imshow(
            np.rot90((lesion[:, :, z] > 0).astype(np.uint8)),
            cmap=ABWMH_CMAP,
            vmin=0,
            vmax=1,
            alpha=ALPHA,
            interpolation="nearest",
        )
        ax.axis("off")

        # Row 4: all masks
        ax = axes[3, col]
        ax.imshow(base, cmap="gray", vmin=vmin, vmax=vmax)
        ax.imshow(
            np.rot90((vent[:, :, z] > 0).astype(np.uint8)),
            cmap=VENT_CMAP,
            vmin=0,
            vmax=1,
            alpha=ALPHA,
            interpolation="nearest",
        )
        ax.imshow(
            np.rot90((nwmh[:, :, z] > 0).astype(np.uint8)),
            cmap=NWMH_CMAP,
            vmin=0,
            vmax=1,
            alpha=ALPHA,
            interpolation="nearest",
        )
        ax.imshow(
            np.rot90((lesion[:, :, z] > 0).astype(np.uint8)),
            cmap=ABWMH_CMAP,
            vmin=0,
            vmax=1,
            alpha=ALPHA,
            interpolation="nearest",
        )
        ax.axis("off")

    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=11)

    fig.suptitle(
        f"Patient {patient_id} — Stage 1 mask alignment QC\n"
        "Cyan = ventricle | Yellow = normal WMH | Red = abnormal WMH",
        fontsize=14,
    )

    fig.savefig(
        out_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# MAIN AUDIT
# ============================================================

def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    audit_rows = []

    print("=" * 88)
    print("STAGE 1 FULL AUDIT — GEOMETRY, INTENSITY, MASKS, COLORED OVERLAYS")
    print("=" * 88)

    for idx, row in enumerate(rows, start=1):
        patient_id = row["patient_id"]

        print()
        print("-" * 88)
        print(f"PATIENT {patient_id} ({idx}/{len(rows)})")

        flair_path = Path(row["flair"]) if row.get("flair") else None
        vent_path = Path(row["ventricle_mask"]) if row.get("ventricle_mask") else None
        nwmh_path = Path(row["normal_wmh_mask"]) if row.get("normal_wmh_mask") else None
        lesion_path = Path(row["lesion_mask"]) if row.get("lesion_mask") else None

        discovered = discover_registered_files(patient_id)
        t1_path = discovered["t1"]
        t2_path = discovered["t2"]
        brain_mask_path = discovered["brain_mask"]

        problems = []

        required = {
            "FLAIR": flair_path,
            "VENT": vent_path,
            "nWMH": nwmh_path,
            "abWMH": lesion_path,
        }

        missing_required = [
            name
            for name, path in required.items()
            if path is None or not path.exists()
        ]

        if missing_required:
            problems.append("Missing: " + ", ".join(missing_required))

        # Defaults for CSV
        rec = {
            "patient_id": patient_id,
            "flair_path": str(flair_path or ""),
            "t1_path": str(t1_path or ""),
            "t2_path": str(t2_path or ""),
            "brain_mask_path": str(brain_mask_path or ""),
            "ventricle_mask_path": str(vent_path or ""),
            "normal_wmh_mask_path": str(nwmh_path or ""),
            "abwmh_mask_path": str(lesion_path or ""),
            "shape_x": "",
            "shape_y": "",
            "shape_z": "",
            "spacing_x": "",
            "spacing_y": "",
            "spacing_z": "",
            "orientation": "",
            "geometry_ok": False,
            "nan_count_flair": "",
            "inf_count_flair": "",
            "flair_min": "",
            "flair_max": "",
            "flair_mean_brain": "",
            "flair_std_brain": "",
            "flair_p01_brain": "",
            "flair_p99_brain": "",
            "looks_zscore_normalized": False,
            "brain_mask_available": bool(brain_mask_path and brain_mask_path.exists()),
            "ventricle_voxels": "",
            "nwmh_voxels": "",
            "abwmh_voxels": "",
            "ventricle_values": "",
            "nwmh_values": "",
            "abwmh_values": "",
            "all_masks_nonempty": False,
            "overlay_png": "",
            "audit_status": "FAIL",
            "issues": "",
        }

        try:
            if missing_required:
                raise FileNotFoundError("Required input missing")

            flair_img = load_nifti(flair_path)
            vent_img = load_nifti(vent_path)
            nwmh_img = load_nifti(nwmh_path)
            lesion_img = load_nifti(lesion_path)

            flair = array3d(flair_img)
            vent = array3d(vent_img)
            nwmh = array3d(nwmh_img)
            lesion = array3d(lesion_img)

            brain_mask = None
            brain_mask_img = None

            if brain_mask_path and brain_mask_path.exists():
                try:
                    brain_mask_img = load_nifti(brain_mask_path)
                    brain_mask = array3d(brain_mask_img)
                except Exception as e:
                    problems.append(f"Brain mask load error: {e}")
                    brain_mask = None
                    brain_mask_img = None

            rec["shape_x"], rec["shape_y"], rec["shape_z"] = shape3d(flair_img)
            rec["spacing_x"], rec["spacing_y"], rec["spacing_z"] = spacing3d(flair_img)
            rec["orientation"] = orientation_codes(flair_img)

            geometry_ok = True

            for name, img in [
                ("ventricle", vent_img),
                ("nWMH", nwmh_img),
                ("abWMH", lesion_img),
            ]:
                if shape3d(img) != shape3d(flair_img):
                    geometry_ok = False
                    problems.append(
                        f"{name} shape {shape3d(img)} != FLAIR {shape3d(flair_img)}"
                    )

                if not spacing_matches(img, flair_img):
                    geometry_ok = False
                    problems.append(
                        f"{name} spacing {spacing3d(img)} != FLAIR {spacing3d(flair_img)}"
                    )

                if not affine_matches(img.affine, flair_img.affine):
                    geometry_ok = False
                    problems.append(f"{name} affine mismatch")

            if brain_mask_img is not None:
                if shape3d(brain_mask_img) != shape3d(flair_img):
                    problems.append(
                        f"brain mask shape {shape3d(brain_mask_img)} != "
                        f"FLAIR {shape3d(flair_img)}"
                    )

            rec["geometry_ok"] = geometry_ok

            # FLAIR data validity
            rec["nan_count_flair"] = int(np.count_nonzero(np.isnan(flair)))
            rec["inf_count_flair"] = int(np.count_nonzero(np.isinf(flair)))

            if rec["nan_count_flair"] > 0:
                problems.append("FLAIR contains NaN values")
            if rec["inf_count_flair"] > 0:
                problems.append("FLAIR contains Inf values")

            stats = intensity_stats(flair, brain_mask)
            rec["flair_min"] = stats["min"]
            rec["flair_max"] = stats["max"]
            rec["flair_mean_brain"] = stats["mean"]
            rec["flair_std_brain"] = stats["std"]
            rec["flair_p01_brain"] = stats["p01"]
            rec["flair_p99_brain"] = stats["p99"]
            rec["looks_zscore_normalized"] = masked_zscore_like(
                stats["mean"],
                stats["std"],
            )

            # Masks
            vent_count, vent_values = binary_mask_stats(vent)
            nwmh_count, nwmh_values = binary_mask_stats(nwmh)
            lesion_count, lesion_values = binary_mask_stats(lesion)

            rec["ventricle_voxels"] = vent_count
            rec["nwmh_voxels"] = nwmh_count
            rec["abwmh_voxels"] = lesion_count

            rec["ventricle_values"] = vent_values
            rec["nwmh_values"] = nwmh_values
            rec["abwmh_values"] = lesion_values

            rec["all_masks_nonempty"] = (
                vent_count > 0
                and nwmh_count > 0
                and lesion_count > 0
            )

            if vent_count == 0:
                problems.append("Ventricle mask empty")
            if nwmh_count == 0:
                problems.append("nWMH mask empty")
            if lesion_count == 0:
                problems.append("abWMH mask empty")

            # Colored overlay
            png_path = OVERLAY_DIR / f"{patient_id}_mask_overlay.png"

            if (
                flair.shape == vent.shape
                == nwmh.shape
                == lesion.shape
            ):
                make_overlay_png(
                    patient_id=patient_id,
                    flair=flair,
                    vent=vent,
                    nwmh=nwmh,
                    lesion=lesion,
                    brain_mask=brain_mask,
                    out_path=png_path,
                )
                rec["overlay_png"] = str(png_path)
            else:
                problems.append("Overlay skipped because array shapes differ")

            rec["audit_status"] = "PASS" if len(problems) == 0 else "REVIEW"
            rec["issues"] = " | ".join(problems)

            print(f"FLAIR shape     : {shape3d(flair_img)}")
            print(f"FLAIR spacing   : {spacing3d(flair_img)}")
            print(f"Orientation     : {rec['orientation']}")
            print(f"T1 found        : {bool(t1_path)}")
            print(f"T2 found        : {bool(t2_path)}")
            print(f"Brain mask found: {rec['brain_mask_available']}")
            print(f"Geometry OK     : {rec['geometry_ok']}")
            print(
                "Mask voxels     : "
                f"vent={vent_count}, "
                f"nWMH={nwmh_count}, "
                f"abWMH={lesion_count}"
            )
            print(
                "FLAIR brain stats: "
                f"mean={stats['mean']:.4f}, "
                f"std={stats['std']:.4f}, "
                f"p01={stats['p01']:.4f}, "
                f"p99={stats['p99']:.4f}"
            )
            print(
                "Looks z-scored   : "
                f"{rec['looks_zscore_normalized']} "
                "(heuristic only)"
            )
            print(f"Audit status     : {rec['audit_status']}")

            if problems:
                for p in problems:
                    print(f"  - {p}")

        except Exception as e:
            if not problems:
                problems.append(str(e))
            rec["issues"] = " | ".join(problems)
            rec["audit_status"] = "FAIL"
            print(f"Audit status     : FAIL")
            for p in problems:
                print(f"  - {p}")

        audit_rows.append(rec)

    # ========================================================
    # SAVE CSV
    # ========================================================

    fieldnames = list(audit_rows[0].keys())

    with AUDIT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(audit_rows)
    passes = sum(r["audit_status"] == "PASS" for r in audit_rows)
    reviews = sum(r["audit_status"] == "REVIEW" for r in audit_rows)
    fails = sum(r["audit_status"] == "FAIL" for r in audit_rows)

    geometry_ok = sum(bool(r["geometry_ok"]) for r in audit_rows)
    brain_masks = sum(bool(r["brain_mask_available"]) for r in audit_rows)
    zscore_like = sum(bool(r["looks_zscore_normalized"]) for r in audit_rows)

    summary = f"""STAGE 1 FULL AUDIT SUMMARY
========================================

Patients total           : {total}
PASS                     : {passes}
REVIEW                   : {reviews}
FAIL                     : {fails}

Geometry OK              : {geometry_ok}/{total}
Brain mask available     : {brain_masks}/{total}
Z-score-like FLAIR       : {zscore_like}/{total}

Important:
- 'Z-score-like' is only a numerical heuristic. It does NOT prove that
  intensity normalization was intentionally performed.
- Visual overlays must still be reviewed for anatomical correctness.
- This script does not modify any NIfTI file.
- This script does not run skull stripping, N4, registration, resampling,
  or normalization.

Audit CSV:
{AUDIT_CSV}

Colored overlays:
{OVERLAY_DIR}
"""

    SUMMARY_TXT.write_text(summary, encoding="utf-8")

    print()
    print("=" * 88)
    print("FINAL SUMMARY")
    print("=" * 88)
    print(summary)


if __name__ == "__main__":
    main()
