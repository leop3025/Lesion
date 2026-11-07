from pathlib import Path
import csv
import json

import nibabel as nib
import numpy as np


# ============================================================
# STAGE 15b — FIVE-CLASS GROUND-TRUTH MASKS
# (saved in six_class_masks/ for path compatibility)
# ============================================================
#
#   0 = Background
#   1 = Ventricle
#   2 = Normal WMH          (nWMH, PV + deep merged)
#   3 = PV Abnormal WMH     (abWMH, d <= 10 mm)
#   4 = Deep Abnormal WMH   (abWMH, d >  10 mm)
#
# Deep normal WMH was too rare (4,660 voxels, absent in 33/100
# patients) to be a separate class, so normal WMH is not split.
#
# d = raw 3D distance to ventricles (distance_map/<id>/<id>_VentDist_mm).
#
# Overlap rules (applied in this order, later wins):
#   ventricle -> nWMH -> abWMH
#   - abWMH has precedence over nWMH (same as script 12).
#   - WMH has precedence over ventricle, so lesion voxels overlapping
#     the ventricle mask are kept (d = 0 -> counted as PV).
# ============================================================

PROJECT = Path(__file__).resolve().parent
PRE = PROJECT / "preprocessed"
DIST = PROJECT / "distance_map"
OUT = PROJECT / "six_class_masks"

PIDS = [f"{i:03d}" for i in range(1, 101)]
PV_THRESHOLD_MM = 10.0

CLASS_NAMES = {
    0: "Background",
    1: "Ventricle",
    2: "Normal WMH",
    3: "PV Abnormal WMH",
    4: "Deep Abnormal WMH",
}
N_CLASSES = len(CLASS_NAMES)


def load(path):
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj), img


def build_six_class(vent, nwmh, abwmh, d_mm):
    pv = d_mm <= PV_THRESHOLD_MM
    label = np.zeros(vent.shape, dtype=np.uint8)
    label[vent] = 1
    label[nwmh] = 2
    label[abwmh & pv] = 3
    label[abwmh & ~pv] = 4
    return label


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for pid in PIDS:
        vent, vimg = load(PRE / pid / f"{pid}_Vent_Mask.nii.gz")
        nwmh, _ = load(PRE / pid / f"{pid}_nWMH_Mask.nii.gz")
        abwmh, _ = load(PRE / pid / f"{pid}_abWMH_Mask.nii.gz")
        d_mm, dimg = load(DIST / pid / f"{pid}_VentDist_mm.nii.gz")

        vent, nwmh, abwmh = vent > 0, nwmh > 0, abwmh > 0
        if not (vent.shape == nwmh.shape == abwmh.shape == d_mm.shape):
            raise RuntimeError(f"{pid}: shape mismatch")
        if not np.allclose(vimg.affine, dimg.affine, atol=1e-4):
            raise RuntimeError(f"{pid}: affine mismatch between mask and distance map")

        label = build_six_class(vent, nwmh, abwmh, d_mm)

        # sanity: every WMH voxel is labelled as some WMH class
        assert np.array_equal(label >= 2, nwmh | abwmh)
        assert np.array_equal(label >= 3, abwmh)

        hdr = vimg.header.copy()
        hdr.set_data_dtype(np.uint8)
        (OUT / pid).mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(label, vimg.affine, hdr),
                 str(OUT / pid / f"{pid}_SixClass_Mask.nii.gz"))

        counts = np.bincount(label.ravel(), minlength=N_CLASSES)
        row = {"patient_id": pid}
        row.update({f"c{k}_{CLASS_NAMES[k].replace(' ', '_')}": int(counts[k]) for k in range(N_CLASSES)})
        row["nwmh_abwmh_overlap"] = int((nwmh & abwmh).sum())
        row["nwmh_in_vent"] = int((nwmh & vent).sum())
        row["abwmh_in_vent"] = int((abwmh & vent).sum())
        rows.append(row)
        print(pid, " ".join(f"{k}:{counts[k]}" for k in range(N_CLASSES)))

    with open(OUT / "six_class_counts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    totals = {k: sum(r[k] for r in rows) for k in rows[0] if k != "patient_id"}
    patients_with = {CLASS_NAMES[k]: sum(1 for r in rows if list(r.values())[1 + k] > 0) for k in range(N_CLASSES)}

    config = {
        "classes": CLASS_NAMES,
        "pv_rule": f"0 <= d <= {PV_THRESHOLD_MM} mm (raw 3D distance to ventricle mask)",
        "deep_rule": f"d > {PV_THRESHOLD_MM} mm",
        "pv_deep_split_applies_to": "abnormal WMH only",
        "overlap_priority": "ventricle < nWMH < abWMH (later overwrites earlier)",
        "distance_source": "distance_map/<id>/<id>_VentDist_mm.nii.gz",
        "voxel_totals": totals,
        "patients_with_class": patients_with,
        "script": Path(__file__).name,
    }
    with open(OUT / "six_class_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nTotals:", json.dumps(totals, indent=2))
    print("Patients with class:", json.dumps(patients_with, indent=2))
    print(f"Done. Outputs in {OUT}")


if __name__ == "__main__":
    main()
