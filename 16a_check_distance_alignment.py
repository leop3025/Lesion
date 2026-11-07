from pathlib import Path
import re

import nibabel as nib
import numpy as np


# ============================================================
# STAGE 16a — CHECK THAT EACH SLICE GETS ITS OWN PATIENT'S
#             DISTANCE MAP AND LABEL (same patient, same slice)
# ============================================================
#
# For every slice .npz in splits/baseline_2d (all folds + test):
#   1. filename patient ID == patient_id stored inside the .npz
#   2. FLAIR channel in the .npz == preprocessed/<id>/<id>_FLAIR at
#      slice_index  (proves patient + slice index are right)
#   3. distance map, label and FLAIR share shape and affine
#   4. distance slice is exactly 0 where that patient's ventricle
#      mask is 1, and > 0 elsewhere (proves the map came from
#      this patient's ventricles)
#   5. 5-class label slice == labels rebuilt from that patient's
#      own masks at the same slice
#   6. negative control: distance map of a DIFFERENT patient does
#      NOT match the ventricle mask (shows the check can fail)
# ============================================================

PROJECT = Path(__file__).resolve().parent
SPLIT = PROJECT / "splits" / "baseline_2d"
PRE = PROJECT / "preprocessed"
DIST = PROJECT / "distance_map"
LABEL = PROJECT / "six_class_masks"

cache = {}


def vols(pid):
    if pid not in cache:
        load = lambda p: nib.load(str(p))
        flair = load(PRE / pid / f"{pid}_FLAIR.nii.gz")
        vent = load(PRE / pid / f"{pid}_Vent_Mask.nii.gz")
        dmm = load(DIST / pid / f"{pid}_VentDist_mm.nii.gz")
        dnorm = load(DIST / pid / f"{pid}_VentDist_norm.nii.gz")
        lab = load(LABEL / pid / f"{pid}_SixClass_Mask.nii.gz")
        for name, img in [("dist_mm", dmm), ("dist_norm", dnorm), ("label", lab), ("vent", vent)]:
            if img.shape != flair.shape or not np.allclose(img.affine, flair.affine, atol=1e-4):
                raise AssertionError(f"{pid}: {name} geometry differs from FLAIR")
        cache[pid] = {
            "flair": np.asanyarray(flair.dataobj).astype(np.float32),
            "vent": np.asanyarray(vent.dataobj) > 0,
            "dmm": np.asanyarray(dmm.dataobj).astype(np.float32),
            "dnorm": np.asanyarray(dnorm.dataobj).astype(np.float32),
            "label": np.asanyarray(lab.dataobj).astype(np.uint8),
        }
    return cache[pid]


def main():
    paths = sorted(SPLIT.rglob("images/*.npz"))
    print(f"Checking {len(paths)} slice files ...")
    errors = []
    patients = set()

    for path in paths:
        fname_pid = re.match(r"(\d{3})_slice_\d+\.npz$", path.name).group(1)
        with np.load(path, allow_pickle=False) as z:
            image = z["image"]
            pid = str(z["patient_id"]).zfill(3)
            zi = int(z["slice_index"])
        if pid != fname_pid:
            errors.append(f"{path}: filename pid {fname_pid} != stored pid {pid}")
            continue
        v = vols(pid)
        patients.add(pid)

        if not np.allclose(image[2], v["flair"][:, :, zi], atol=1e-5):
            errors.append(f"{path}: FLAIR channel does not match {pid} slice {zi}")

        d = v["dmm"][:, :, zi]
        vent = v["vent"][:, :, zi]
        if np.any(d[vent] != 0) or np.any(d[~vent] <= 0):
            errors.append(f"{path}: distance zeros do not match {pid} ventricle mask at slice {zi}")

    # label rebuilt from the patient's own masks (whole volume, once per patient)
    for pid in sorted(patients):
        v = vols(pid)
        nw = np.asanyarray(nib.load(str(PRE / pid / f"{pid}_nWMH_Mask.nii.gz")).dataobj) > 0
        ab = np.asanyarray(nib.load(str(PRE / pid / f"{pid}_abWMH_Mask.nii.gz")).dataobj) > 0
        pv = v["dmm"] <= 10.0
        ref = np.zeros(v["vent"].shape, np.uint8)
        ref[v["vent"]] = 1
        ref[nw] = 2
        ref[ab & pv] = 3
        ref[ab & ~pv] = 4
        if not np.array_equal(ref, v["label"]):
            errors.append(f"{pid}: 5-class label does not match the patient's own masks")
        if not np.allclose(v["dnorm"], np.clip(v["dmm"], 0, 35.0) / 35.0, atol=1e-6):
            errors.append(f"{pid}: normalized map != clip(mm, 0, 35) / 35")

    # negative control: pair each patient with the next patient's map
    ids = sorted(patients)
    mismatches = 0
    for a, b in zip(ids, ids[1:] + ids[:1]):
        va, vb = vols(a), vols(b)
        if va["vent"].shape != vb["vent"].shape:
            mismatches += 1
            continue
        if np.any(vb["dmm"][va["vent"]] != 0):
            mismatches += 1
    print(f"Negative control: {mismatches}/{len(ids)} wrong-patient pairings detected as mismatches")

    print(f"Patients checked: {len(patients)}")
    if errors:
        print(f"FAILED: {len(errors)} problems")
        for e in errors[:30]:
            print("  ", e)
    else:
        print("PASS: every slice uses its own patient's distance map, label and slice index.")


if __name__ == "__main__":
    main()
