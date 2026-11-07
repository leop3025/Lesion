from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt


# ============================================================
# STAGE 15a — 3D DISTANCE-TO-VENTRICLE MAPS
# ============================================================
#
# For each patient:
#   preprocessed/<id>/<id>_Vent_Mask.nii.gz  (binary, FLAIR grid)
#       -> 3D EDT in mm (sampling = voxel spacing), inside ventricles = 0
#       -> distance_map/<id>/<id>_VentDist_mm.nii.gz
#       -> clip at D_max, divide by D_max
#       -> distance_map/<id>/<id>_VentDist_norm.nii.gz
#
# D_max = 99th percentile of abWMH voxel distance in the 70
# development patients only (rounded up to nearest 5 mm).
# ============================================================

PROJECT = Path(__file__).resolve().parent
PRE = PROJECT / "preprocessed"
OUT = PROJECT / "distance_map"
QC_DIR = OUT / "qc"
SPLIT = PROJECT / "splits" / "baseline_2d"

PIDS = [f"{i:03d}" for i in range(1, 101)]
PV_THRESHOLD_MM = 10.0
N_QC_PNG = 10


def load(path):
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj), img


def save(arr, ref_img, path):
    hdr = ref_img.header.copy()
    hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), ref_img.affine, hdr), str(path))


def read_dev_patients():
    with open(SPLIT / "train_70_patients.csv", newline="") as f:
        return [r["patient_id"].zfill(3) for r in csv.DictReader(f)]


def save_qc_png(pid, d_mm, d_max):
    flair, _ = load(PRE / pid / f"{pid}_FLAIR.nii.gz")
    les, _ = load(PRE / pid / f"{pid}_abWMH_Mask.nii.gz")
    vent, _ = load(PRE / pid / f"{pid}_Vent_Mask.nii.gz")

    # pick slice with most lesion voxels (fallback: most ventricle voxels)
    counts = (les > 0).sum(axis=(0, 1))
    if counts.max() == 0:
        counts = (vent > 0).sum(axis=(0, 1))
    z = int(np.argmax(counts))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(np.rot90(flair[:, :, z]), cmap="gray")
    ax[0].contour(np.rot90(vent[:, :, z] > 0), levels=[0.5], colors="cyan", linewidths=0.8)
    ax[0].contour(np.rot90(les[:, :, z] > 0), levels=[0.5], colors="red", linewidths=0.8)
    ax[0].set_title(f"{pid} z={z}  FLAIR (cyan=vent, red=abWMH)")

    im = ax[1].imshow(np.rot90(d_mm[:, :, z]), cmap="magma")
    ax[1].contour(np.rot90(d_mm[:, :, z]), levels=[PV_THRESHOLD_MM], colors="white", linewidths=0.8)
    ax[1].set_title("distance (mm), white = 10 mm")
    fig.colorbar(im, ax=ax[1], fraction=0.046)

    ax[2].imshow(np.rot90(flair[:, :, z]), cmap="gray")
    ax[2].imshow(np.rot90(np.clip(d_mm[:, :, z], 0, d_max) / d_max), cmap="jet", alpha=0.4, vmin=0, vmax=1)
    ax[2].contour(np.rot90(les[:, :, z] > 0), levels=[0.5], colors="white", linewidths=0.8)
    ax[2].set_title(f"normalized overlay (D_max={d_max:g} mm)")

    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(QC_DIR / f"{pid}_qc.png", dpi=110)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. raw mm maps + per-patient QC stats ----
    qc_rows = []
    for pid in PIDS:
        vent, vimg = load(PRE / pid / f"{pid}_Vent_Mask.nii.gz")
        vent = vent > 0
        if not vent.any():
            raise RuntimeError(f"{pid}: empty ventricle mask")

        spacing = tuple(float(s) for s in vimg.header.get_zooms()[:3])
        d_mm = distance_transform_edt(~vent, sampling=spacing).astype(np.float32)
        if not np.isfinite(d_mm).all():
            raise RuntimeError(f"{pid}: non-finite distances")
        if d_mm[vent].any():
            raise RuntimeError(f"{pid}: non-zero distance inside ventricles")

        (OUT / pid).mkdir(parents=True, exist_ok=True)
        save(d_mm, vimg, OUT / pid / f"{pid}_VentDist_mm.nii.gz")

        les, _ = load(PRE / pid / f"{pid}_abWMH_Mask.nii.gz")
        les = les > 0
        n_les = int(les.sum())
        les_d = d_mm[les]
        qc_rows.append({
            "patient_id": pid,
            "shape": "x".join(map(str, d_mm.shape)),
            "spacing_mm": "x".join(f"{s:.3f}" for s in spacing),
            "vent_voxels": int(vent.sum()),
            "dist_min_mm": round(float(d_mm.min()), 3),
            "dist_max_mm": round(float(d_mm.max()), 3),
            "dist_mean_mm": round(float(d_mm.mean()), 3),
            "abwmh_voxels": n_les,
            "abwmh_in_vent": int((les & vent).sum()),
            "abwmh_pv_pct": round(100.0 * float(((les_d > 0) & (les_d <= PV_THRESHOLD_MM)).sum()) / n_les, 2) if n_les else "",
            "abwmh_deep_pct": round(100.0 * float((les_d > PV_THRESHOLD_MM).sum()) / n_les, 2) if n_les else "",
        })
        print(f"{pid}: max={d_mm.max():6.1f} mm  abWMH={n_les:6d}  in_vent={qc_rows[-1]['abwmh_in_vent']}")

    with open(OUT / "distance_qc.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(qc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(qc_rows)

    # ---- 2. choose D_max from DEVELOPMENT patients only ----
    dev = read_dev_patients()
    if len(dev) != 70:
        raise RuntimeError(f"Expected 70 development patients, found {len(dev)}")

    lesion_d = []
    for pid in dev:
        d_mm, _ = load(OUT / pid / f"{pid}_VentDist_mm.nii.gz")
        les, _ = load(PRE / pid / f"{pid}_abWMH_Mask.nii.gz")
        lesion_d.append(d_mm[les > 0])
    lesion_d = np.concatenate(lesion_d)
    p99 = float(np.percentile(lesion_d, 99))
    d_max = float(np.ceil(p99 / 5.0) * 5.0)
    print(f"\n99th pct abWMH distance (dev 70) = {p99:.2f} mm -> D_max = {d_max:g} mm")

    # ---- 3. normalized maps ----
    for pid in PIDS:
        d_mm, img = load(OUT / pid / f"{pid}_VentDist_mm.nii.gz")
        d_norm = np.clip(d_mm, 0, d_max) / d_max
        save(d_norm, img, OUT / pid / f"{pid}_VentDist_norm.nii.gz")

    # ---- 4. QC overlays: highest-burden patients + a few others ----
    by_burden = sorted(qc_rows, key=lambda r: r["abwmh_voxels"], reverse=True)
    qc_pids = [r["patient_id"] for r in by_burden[:N_QC_PNG // 2]]
    rest = [r["patient_id"] for r in by_burden[N_QC_PNG // 2:]]
    step = max(1, len(rest) // (N_QC_PNG - len(qc_pids)))
    qc_pids += rest[::step][:N_QC_PNG - len(qc_pids)]
    for pid in qc_pids:
        d_mm, _ = load(OUT / pid / f"{pid}_VentDist_mm.nii.gz")
        save_qc_png(pid, d_mm, d_max)

    # ---- 5. config ----
    total_in_vent = sum(r["abwmh_in_vent"] for r in qc_rows)
    config = {
        "D_max_mm": d_max,
        "p99_lesion_distance_mm": p99,
        "selection": "99th percentile of abWMH voxel distance, development 70 only, rounded up to 5 mm",
        "n_dev_lesion_voxels": int(lesion_d.size),
        "inside_ventricle_value": 0.0,
        "edt": "scipy.ndimage.distance_transform_edt(~vent, sampling=voxel spacing), full 3D volume",
        "normalization": "min(d, D_max) / D_max",
        "pv_threshold_mm": PV_THRESHOLD_MM,
        "total_abwmh_voxels_inside_vent_mask": int(total_in_vent),
        "qc_png_patients": qc_pids,
        "script": Path(__file__).name,
    }
    with open(OUT / "distance_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"abWMH voxels inside ventricle mask (all patients): {total_in_vent}")
    print(f"Done. Outputs in {OUT}")


if __name__ == "__main__":
    main()
