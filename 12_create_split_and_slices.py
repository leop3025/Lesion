from pathlib import Path
import csv
import json
import os
import shutil
import sys
import tempfile

import nibabel as nib
import numpy as np


# ============================================================
# STAGE 12 — LOCKED 70/30 PATIENT SPLIT + 5-FOLD CV + 2D SLICES
# ============================================================
#
# What this script does:
#
#   100 patients
#       |
#       +-- 70 development patients
#       |      |
#       |      +-- 5-fold patient-level CV
#       |             Fold 1: 56 train / 14 val
#       |             Fold 2: 56 train / 14 val
#       |             Fold 3: 56 train / 14 val
#       |             Fold 4: 56 train / 14 val
#       |             Fold 5: 56 train / 14 val
#       |
#       +-- 30 FINAL TEST patients
#
# IMPORTANT:
#   - Split is done by PATIENT, never by slice.
#   - Slicing happens only AFTER patient membership is locked.
#   - Test patients never appear in any fold train/val.
#   - Each development patient appears in validation exactly once.
#   - Stratification uses abnormal-WMH lesion burden tertile per patient.
#   - Existing output is never regenerated unless --force is supplied.
#
# Saved layout:
#
# splits/
#   baseline_2d/
#       master_split.csv
#       train_70_patients.csv
#       test_30_patients.csv
#       split_summary.csv
#       split_config.json
#
#       fold_1/
#           train_patients.csv
#           val_patients.csv
#           train/
#               images/*.npz
#               labels/*.npz
#           val/
#               images/*.npz
#               labels/*.npz
#
#       ...
#
#       test/
#           images/*.npz
#           labels/*.npz
#
# Each image .npz contains:
#   image : float32 [3,H,W]  (T1,T2,FLAIR)
#   patient_id
#   slice_index
#
# Each label .npz contains:
#   label : uint8 [H,W]
#           0 background
#           1 normal WMH
#           2 abnormal WMH
#
# Ventricle voxels are folded into background for the plain baseline.
# ============================================================


# ============================================================
# PATHS / CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\Preethi\Documents\LESION")
PREPROCESSED_ROOT = BASE_DIR / "preprocessed"

SPLIT_ROOT = BASE_DIR / "splits" / "baseline_2d"
MASTER_SPLIT = SPLIT_ROOT / "master_split.csv"
TRAIN70_CSV = SPLIT_ROOT / "train_70_patients.csv"
TEST30_CSV = SPLIT_ROOT / "test_30_patients.csv"
SUMMARY_CSV = SPLIT_ROOT / "split_summary.csv"
CONFIG_JSON = SPLIT_ROOT / "split_config.json"

LOG_DIR = BASE_DIR / "logs"
LOG_PATH = LOG_DIR / "12_patient_split_and_slice_terminal.txt"

EXPECTED_PATIENTS = 100
TRAIN_COUNT = 70
TEST_COUNT = 30
N_FOLDS = 5

RANDOM_SEED = 20260911

CHANNEL_ORDER = ("T1", "T2", "FLAIR")

# Exclude completely empty slices after preprocessing/skull stripping.
EXCLUDE_EMPTY_SLICES = True
NONZERO_EPS = 1e-8

FORCE = "--force" in sys.argv


# ============================================================
# TERMINAL + LOG
# ============================================================

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


LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_PATH.open("w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)


# ============================================================
# ATOMIC HELPERS
# ============================================================

def atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)

    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)

    try:
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        os.replace(tmp, path)

    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_save_npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=path.stem + ".",
        suffix=".tmp.npz"
    )
    os.close(fd)
    tmp = Path(tmp_name)

    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# ============================================================
# NIFTI DISCOVERY
# ============================================================

def all_nifti(patient_dir: Path):
    return sorted(patient_dir.rglob("*.nii*"))


def find_unique(patient_dir: Path, include_tokens, exclude_tokens=()):
    matches = []

    for path in all_nifti(patient_dir):
        name = path.name.lower()

        if not all(tok.lower() in name for tok in include_tokens):
            continue

        if any(tok.lower() in name for tok in exclude_tokens):
            continue

        matches.append(path)

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No NIfTI matched include={include_tokens}, "
            f"exclude={exclude_tokens} in {patient_dir}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous NIfTI match in {patient_dir} "
            f"for include={include_tokens}:\n"
            + "\n".join(f"  {p.name}" for p in matches)
        )

    return matches[0]


def discover_patient_files(patient_id: str):
    patient_dir = PREPROCESSED_ROOT / patient_id

    if not patient_dir.exists():
        raise FileNotFoundError(
            f"Missing preprocessed patient folder: {patient_dir}"
        )

    return {
        "T1": find_unique(
            patient_dir,
            ["t1"],
            ["mask", "label"]
        ),
        "T2": find_unique(
            patient_dir,
            ["t2"],
            ["mask", "label"]
        ),
        "FLAIR": find_unique(
            patient_dir,
            ["flair"],
            ["mask", "label"]
        ),
        "ventricle": find_unique(
            patient_dir,
            ["vent"],
            ["distance"]
        ),
        "nWMH": find_unique(
            patient_dir,
            ["nwmh"],
            ["abwmh"]
        ),
        "abWMH": find_unique(
            patient_dir,
            ["abwmh"]
        ),
    }


# ============================================================
# LOAD / GEOMETRY
# ============================================================

def load_3d(path: Path):
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)

    if data.ndim > 3:
        data = np.squeeze(data)

    if data.ndim != 3:
        raise RuntimeError(
            f"Expected 3D image: {path}; got shape {data.shape}"
        )

    if not np.all(np.isfinite(data)):
        raise RuntimeError(
            f"NaN/Inf found in {path}"
        )

    return data, img


def validate_same_geometry(patient_id, arrays, images):
    shapes = {
        key: tuple(arr.shape)
        for key, arr in arrays.items()
    }

    if len(set(shapes.values())) != 1:
        raise RuntimeError(
            f"{patient_id}: shape mismatch: {shapes}"
        )

    ref = images["FLAIR"]

    for key, img in images.items():
        if key == "FLAIR":
            continue

        if not np.allclose(
            img.header.get_zooms()[:3],
            ref.header.get_zooms()[:3],
            atol=1e-4
        ):
            raise RuntimeError(
                f"{patient_id}: spacing mismatch for {key}"
            )

        if nib.aff2axcodes(img.affine) != nib.aff2axcodes(ref.affine):
            raise RuntimeError(
                f"{patient_id}: orientation mismatch for {key}"
            )

        if not np.allclose(
            img.affine,
            ref.affine,
            atol=1e-4
        ):
            raise RuntimeError(
                f"{patient_id}: affine mismatch for {key}"
            )


# ============================================================
# PATIENT INVENTORY + LESION BURDEN
# ============================================================

def collect_patient_inventory():
    patients = []

    for i in range(1, EXPECTED_PATIENTS + 1):
        pid = f"{i:03d}"

        files = discover_patient_files(pid)

        arrays = {}
        images = {}

        for key, path in files.items():
            arrays[key], images[key] = load_3d(path)

        validate_same_geometry(
            pid,
            arrays,
            images
        )

        abwmh = arrays["abWMH"]

        lesion_voxels = int(
            np.count_nonzero(abwmh > 0)
        )

        spacing = images["abWMH"].header.get_zooms()[:3]
        voxel_volume_mm3 = float(np.prod(spacing))

        lesion_volume_ml = (
            lesion_voxels
            * voxel_volume_mm3
            / 1000.0
        )

        patients.append(
            {
                "patient_id": pid,
                "lesion_voxels": lesion_voxels,
                "lesion_volume_ml": lesion_volume_ml,
                "files": files
            }
        )

        print(
            f"{pid}: lesion voxels={lesion_voxels:,} | "
            f"lesion volume={lesion_volume_ml:.4f} mL"
        )

    return patients


# ============================================================
# BURDEN TERTILES
# ============================================================

def assign_tertiles(patients):
    ordered = sorted(
        patients,
        key=lambda r: (
            r["lesion_volume_ml"],
            r["patient_id"]
        )
    )

    n = len(ordered)

    labels = ["low", "medium", "high"]

    for rank, row in enumerate(ordered):
        idx = min(
            2,
            (3 * rank) // n
        )

        row["burden_tertile"] = labels[idx]
        row["burden_rank"] = rank + 1

    return patients


# ============================================================
# STRATIFIED 70/30 SPLIT
# ============================================================

def grouped_by_tertile(rows):
    groups = {
        "low": [],
        "medium": [],
        "high": []
    }

    for row in rows:
        groups[row["burden_tertile"]].append(row)

    return groups


def shuffled(rows, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    return [rows[int(i)] for i in idx]


def make_70_30_split(patients):
    """
    Exact test balance:
        low    10
        medium 10
        high   10

    Remaining 70 automatically become the development pool.
    """
    groups = grouped_by_tertile(patients)

    train = []
    test = []

    for offset, label in enumerate(["low", "medium", "high"]):
        group = shuffled(
            groups[label],
            RANDOM_SEED + 100 + offset
        )

        test.extend(group[:10])
        train.extend(group[10:])

    train = sorted(train, key=lambda r: r["patient_id"])
    test = sorted(test, key=lambda r: r["patient_id"])

    if len(train) != TRAIN_COUNT:
        raise RuntimeError(
            f"Expected {TRAIN_COUNT} train patients, got {len(train)}"
        )

    if len(test) != TEST_COUNT:
        raise RuntimeError(
            f"Expected {TEST_COUNT} test patients, got {len(test)}"
        )

    return train, test


# ============================================================
# 5-FOLD STRATIFIED CV INSIDE 70
# ============================================================

def make_five_folds(train70):
    """
    Produces 5 folds of:
        56 train
        14 val

    Patient-level only.

    Each tertile is shuffled independently and distributed round-robin
    across validation folds. Extra patients are rotated so fold burden
    distributions stay as even as mathematically possible.
    """
    groups = grouped_by_tertile(train70)

    fold_val = [[] for _ in range(N_FOLDS)]

    offset = 0

    for group_i, label in enumerate(["low", "medium", "high"]):
        group = shuffled(
            groups[label],
            RANDOM_SEED + 200 + group_i
        )

        for i, row in enumerate(group):
            fold_index = (i + offset) % N_FOLDS
            fold_val[fold_index].append(row)

        offset = (
            offset + len(group)
        ) % N_FOLDS

    all_ids = {
        row["patient_id"]
        for row in train70
    }

    folds = []

    for fold_idx in range(N_FOLDS):
        val_rows = sorted(
            fold_val[fold_idx],
            key=lambda r: r["patient_id"]
        )

        val_ids = {
            row["patient_id"]
            for row in val_rows
        }

        train_rows = sorted(
            [
                row
                for row in train70
                if row["patient_id"] not in val_ids
            ],
            key=lambda r: r["patient_id"]
        )

        if len(val_rows) != 14:
            raise RuntimeError(
                f"Fold {fold_idx + 1}: expected 14 val patients, "
                f"got {len(val_rows)}"
            )

        if len(train_rows) != 56:
            raise RuntimeError(
                f"Fold {fold_idx + 1}: expected 56 train patients, "
                f"got {len(train_rows)}"
            )

        if (
            {r["patient_id"] for r in train_rows}
            & val_ids
        ):
            raise RuntimeError(
                f"Fold {fold_idx + 1}: train/val patient leakage."
            )

        if (
            {r["patient_id"] for r in train_rows}
            | val_ids
        ) != all_ids:
            raise RuntimeError(
                f"Fold {fold_idx + 1}: incomplete patient coverage."
            )

        folds.append(
            {
                "train": train_rows,
                "val": val_rows
            }
        )

    # Every development patient must be val exactly once.
    counts = {}

    for fold in folds:
        for row in fold["val"]:
            pid = row["patient_id"]
            counts[pid] = counts.get(pid, 0) + 1

    bad = {
        pid: count
        for pid, count in counts.items()
        if count != 1
    }

    if (
        len(counts) != TRAIN_COUNT
        or bad
    ):
        raise RuntimeError(
            f"5-fold validation coverage failed: {bad}"
        )

    return folds


# ============================================================
# GLOBAL LEAKAGE CHECKS
# ============================================================

def ids(rows):
    return {
        row["patient_id"]
        for row in rows
    }


def validate_splits(train70, test30, folds):
    train_ids = ids(train70)
    test_ids = ids(test30)

    if train_ids & test_ids:
        raise RuntimeError(
            "LEAKAGE: patient appears in both train70 and test30."
        )

    if len(train_ids | test_ids) != EXPECTED_PATIENTS:
        raise RuntimeError(
            "70/30 split does not cover all 100 patients exactly once."
        )

    for fold_no, fold in enumerate(folds, start=1):
        fold_train = ids(fold["train"])
        fold_val = ids(fold["val"])

        if fold_train & fold_val:
            raise RuntimeError(
                f"Fold {fold_no}: train/val leakage."
            )

        if fold_train & test_ids:
            raise RuntimeError(
                f"Fold {fold_no}: TEST patient entered training."
            )

        if fold_val & test_ids:
            raise RuntimeError(
                f"Fold {fold_no}: TEST patient entered validation."
            )

        if (
            fold_train | fold_val
        ) != train_ids:
            raise RuntimeError(
                f"Fold {fold_no}: train+val != development 70."
            )

    print()
    print("=" * 82)
    print("PATIENT-LEVEL LEAKAGE CHECK: PASS")
    print("=" * 82)
    print("✓ Split performed by patient ID only.")
    print("✓ 70 development and 30 final-test patients are disjoint.")
    print("✓ Each fold has 56 train / 14 validation patients.")
    print("✓ Test patients never appear in any fold.")
    print("✓ Every development patient is validation exactly once.")


# ============================================================
# SUMMARY
# ============================================================

def burden_counts(rows):
    counts = {
        "low": 0,
        "medium": 0,
        "high": 0
    }

    for row in rows:
        counts[row["burden_tertile"]] += 1

    return counts


def print_split_summary(train70, test30, folds):
    print()
    print("=" * 94)
    print("SPLIT SUMMARY")
    print("=" * 94)

    print(
        f"{'GROUP':<18}"
        f"{'N':>6}"
        f"{'LOW':>8}"
        f"{'MED':>8}"
        f"{'HIGH':>8}"
    )
    print("-" * 48)

    groups = [
        ("TRAIN_70", train70),
        ("TEST_30", test30)
    ]

    for fold_no, fold in enumerate(folds, start=1):
        groups.append(
            (f"FOLD_{fold_no}_TRAIN", fold["train"])
        )
        groups.append(
            (f"FOLD_{fold_no}_VAL", fold["val"])
        )

    summary_rows = []

    for name, rows in groups:
        counts = burden_counts(rows)

        print(
            f"{name:<18}"
            f"{len(rows):>6}"
            f"{counts['low']:>8}"
            f"{counts['medium']:>8}"
            f"{counts['high']:>8}"
        )

        summary_rows.append(
            {
                "group": name,
                "patient_count": len(rows),
                "low": counts["low"],
                "medium": counts["medium"],
                "high": counts["high"]
            }
        )

    return summary_rows


# ============================================================
# PATIENT CSV HELPERS
# ============================================================

PATIENT_FIELDS = [
    "patient_id",
    "burden_tertile",
    "burden_rank",
    "lesion_voxels",
    "lesion_volume_ml"
]


def strip_files(rows):
    output = []

    for row in rows:
        output.append(
            {
                "patient_id": row["patient_id"],
                "burden_tertile": row["burden_tertile"],
                "burden_rank": row["burden_rank"],
                "lesion_voxels": row["lesion_voxels"],
                "lesion_volume_ml": round(
                    float(row["lesion_volume_ml"]),
                    6
                )
            }
        )

    return output


# ============================================================
# 3-CLASS LABEL
# ============================================================

def build_label(vent, nwmh, abwmh):
    label = np.zeros(
        nwmh.shape,
        dtype=np.uint8
    )

    # 1 = normal WMH
    label[nwmh > 0] = 1

    # 2 = abnormal WMH
    # abnormal gets precedence if any overlap exists.
    label[abwmh > 0] = 2

    # Baseline decision:
    # ventricle is NOT a target class.
    label[vent > 0] = 0

    return label


# ============================================================
# BUILD AXIAL SLICES FOR ONE PATIENT
# ============================================================

def load_patient_arrays(patient_row):
    pid = patient_row["patient_id"]
    files = patient_row["files"]

    arrays = {}
    images = {}

    for key, path in files.items():
        arrays[key], images[key] = load_3d(path)

    validate_same_geometry(
        pid,
        arrays,
        images
    )

    print()
    print(f"{pid} channel order confirmed:")
    print("  0 =", files["T1"].name)
    print("  1 =", files["T2"].name)
    print("  2 =", files["FLAIR"].name)

    return arrays


def make_patient_slice_cache(patient_row):
    """
    Generates in-memory slice tuples once for the patient.

    Returns list of:
        (z, image[3,H,W], label[H,W])
    """
    arrays = load_patient_arrays(patient_row)

    t1 = arrays["T1"].astype(np.float32, copy=False)
    t2 = arrays["T2"].astype(np.float32, copy=False)
    flair = arrays["FLAIR"].astype(np.float32, copy=False)

    vent = arrays["ventricle"]
    nwmh = arrays["nWMH"]
    abwmh = arrays["abWMH"]

    label = build_label(
        vent,
        nwmh,
        abwmh
    )

    slices = []

    for z in range(flair.shape[2]):
        image = np.stack(
            [
                t1[:, :, z],
                t2[:, :, z],
                flair[:, :, z]
            ],
            axis=0
        ).astype(np.float32, copy=False)

        target = label[:, :, z]

        if EXCLUDE_EMPTY_SLICES:
            if not np.any(
                np.abs(image) > NONZERO_EPS
            ):
                continue

        slices.append(
            (
                z,
                image,
                target.astype(np.uint8, copy=False)
            )
        )

    print(
        f"{patient_row['patient_id']}: "
        f"kept {len(slices)}/{flair.shape[2]} axial slices"
    )

    return slices


# ============================================================
# SAVE SLICE TO A PARTITION
# ============================================================

def save_slice_pair(
    root: Path,
    patient_id: str,
    z: int,
    image: np.ndarray,
    label: np.ndarray
):
    image_dir = root / "images"
    label_dir = root / "labels"

    image_name = f"{patient_id}_slice_{z:03d}.npz"
    label_name = f"{patient_id}_slice_{z:03d}.npz"

    image_path = image_dir / image_name
    label_path = label_dir / label_name

    atomic_save_npz(
        image_path,
        image=image,
        patient_id=np.array(patient_id),
        slice_index=np.array(z, dtype=np.int16),
        channel_order=np.array(CHANNEL_ORDER)
    )

    atomic_save_npz(
        label_path,
        label=label,
        patient_id=np.array(patient_id),
        slice_index=np.array(z, dtype=np.int16)
    )


# ============================================================
# MATERIALIZE FOLD FOLDERS
# ============================================================

def materialize_slices(patients, test30, folds):
    """
    Physical folder structure is created exactly as requested:
        fold_X/train/images
        fold_X/train/labels
        fold_X/val/images
        fold_X/val/labels
        test/images
        test/labels

    Because a patient belongs to train in four of the five folds,
    their slice files are copied into each relevant fold directory.
    This uses more disk space, but makes the train/val/test folders
    explicit and easy to inspect.
    """

    patient_lookup = {
        row["patient_id"]: row
        for row in patients
    }

    # Map patient -> destinations.
    destinations = {
        row["patient_id"]: []
        for row in patients
    }

    for fold_no, fold in enumerate(folds, start=1):
        for row in fold["train"]:
            destinations[row["patient_id"]].append(
                SPLIT_ROOT
                / f"fold_{fold_no}"
                / "train"
            )

        for row in fold["val"]:
            destinations[row["patient_id"]].append(
                SPLIT_ROOT
                / f"fold_{fold_no}"
                / "val"
            )

    for row in test30:
        destinations[row["patient_id"]].append(
            SPLIT_ROOT / "test"
        )

    total_slice_pairs = 0

    for patient_id in sorted(patient_lookup):
        row = patient_lookup[patient_id]

        slices = make_patient_slice_cache(
            row
        )

        patient_destinations = destinations[
            patient_id
        ]

        if not patient_destinations:
            raise RuntimeError(
                f"{patient_id}: no split destination assigned."
            )

        for z, image, label in slices:
            for dest in patient_destinations:
                save_slice_pair(
                    dest,
                    patient_id,
                    z,
                    image,
                    label
                )

                total_slice_pairs += 1

    return total_slice_pairs


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 88)
    print("STAGE 12 — 70/30 PATIENT SPLIT + 5-FOLD CV + 2D SLICES")
    print("=" * 88)
    print()
    print("Preprocessed root:")
    print(PREPROCESSED_ROOT)
    print()
    print("Output split root:")
    print(SPLIT_ROOT)

    # --------------------------------------------------------
    # Reproducibility guard
    # --------------------------------------------------------

    if SPLIT_ROOT.exists():
        if not FORCE:
            raise RuntimeError(
                "\nSPLIT OUTPUT ALREADY EXISTS.\n"
                "Nothing was regenerated.\n\n"
                f"Existing folder:\n  {SPLIT_ROOT}\n\n"
                "To intentionally replace it, run:\n"
                "  python 12_create_split_and_slices.py --force\n"
            )

        print()
        print("FORCE MODE: deleting old baseline_2d split folder...")
        shutil.rmtree(SPLIT_ROOT)

    SPLIT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Inventory + lesion burden
    # --------------------------------------------------------

    print()
    print("=" * 88)
    print("STEP 1 — PATIENT INVENTORY + ABNORMAL-WMH BURDEN")
    print("=" * 88)

    patients = collect_patient_inventory()

    if len(patients) != EXPECTED_PATIENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_PATIENTS} patients, got {len(patients)}"
        )

    patients = assign_tertiles(
        patients
    )

    # --------------------------------------------------------
    # 70/30 split
    # --------------------------------------------------------

    print()
    print("=" * 88)
    print("STEP 2 — STRATIFIED 70/30 PATIENT SPLIT")
    print("=" * 88)

    train70, test30 = make_70_30_split(
        patients
    )

    # --------------------------------------------------------
    # 5-fold on 70
    # --------------------------------------------------------

    print()
    print("=" * 88)
    print("STEP 3 — STRATIFIED 5-FOLD CV INSIDE TRAIN 70")
    print("=" * 88)

    folds = make_five_folds(
        train70
    )

    validate_splits(
        train70,
        test30,
        folds
    )

    summary_rows = print_split_summary(
        train70,
        test30,
        folds
    )

    # --------------------------------------------------------
    # Save patient membership files first
    # --------------------------------------------------------

    print()
    print("=" * 88)
    print("STEP 4 — SAVE PERMANENT PATIENT SPLITS")
    print("=" * 88)

    master_rows = []

    train_ids = ids(train70)
    test_ids = ids(test30)

    for row in sorted(
        patients,
        key=lambda r: r["patient_id"]
    ):
        r = strip_files([row])[0]

        if row["patient_id"] in test_ids:
            r["outer_split"] = "test"
        elif row["patient_id"] in train_ids:
            r["outer_split"] = "development"
        else:
            raise RuntimeError(
                f"Unassigned patient: {row['patient_id']}"
            )

        master_rows.append(r)

    master_fields = PATIENT_FIELDS + [
        "outer_split"
    ]

    write_csv(
        MASTER_SPLIT,
        master_rows,
        master_fields
    )

    write_csv(
        TRAIN70_CSV,
        strip_files(train70),
        PATIENT_FIELDS
    )

    write_csv(
        TEST30_CSV,
        strip_files(test30),
        PATIENT_FIELDS
    )

    for fold_no, fold in enumerate(
        folds,
        start=1
    ):
        fold_dir = SPLIT_ROOT / f"fold_{fold_no}"

        write_csv(
            fold_dir / "train_patients.csv",
            strip_files(fold["train"]),
            PATIENT_FIELDS
        )

        write_csv(
            fold_dir / "val_patients.csv",
            strip_files(fold["val"]),
            PATIENT_FIELDS
        )

    write_csv(
        SUMMARY_CSV,
        summary_rows,
        [
            "group",
            "patient_count",
            "low",
            "medium",
            "high"
        ]
    )

    config = {
        "random_seed": RANDOM_SEED,
        "split_unit": "patient",
        "outer_split": {
            "development": TRAIN_COUNT,
            "final_test": TEST_COUNT
        },
        "cv": {
            "folds": N_FOLDS,
            "train_per_fold": 56,
            "val_per_fold": 14
        },
        "stratification": {
            "source": "abnormal WMH physical lesion volume",
            "groups": [
                "low",
                "medium",
                "high"
            ],
            "method": "rank-based tertiles"
        },
        "slice_mode": "2d_axial",
        "channels": list(CHANNEL_ORDER),
        "classes": {
            "0": "background",
            "1": "normal_wmh",
            "2": "abnormal_wmh"
        },
        "ventricle_policy": "folded_into_background",
        "exclude_empty_slices": EXCLUDE_EMPTY_SLICES,
        "distance_conditioning": False,
        "important": (
            "Patient split happens before slicing. "
            "Never perform random train/val/test splitting on slices."
        )
    }

    atomic_write_text(
        CONFIG_JSON,
        json.dumps(
            config,
            indent=2
        )
    )

    # --------------------------------------------------------
    # Slice generation
    # --------------------------------------------------------

    print()
    print("=" * 88)
    print("STEP 5 — AXIAL SLICING INTO TRAIN / VAL / TEST FOLDERS")
    print("=" * 88)

    total_saved = materialize_slices(
        patients=patients,
        test30=test30,
        folds=folds
    )

    print()
    print("=" * 88)
    print("FINAL SUMMARY")
    print("=" * 88)

    print("Patients total     :", EXPECTED_PATIENTS)
    print("Development train  :", TRAIN_COUNT)
    print("Final test         :", TEST_COUNT)
    print("CV folds           :", N_FOLDS)
    print("Per fold train     : 56 patients")
    print("Per fold val       : 14 patients")
    print()
    print("Channel order      : T1, T2, FLAIR")
    print("Classes            : 0 BG, 1 nWMH, 2 abWMH")
    print("Ventricle          : background")
    print("Distance channel   : NONE")
    print()
    print("Patient leakage    : PASS")
    print("Slice-level split  : NEVER USED")
    print()
    print("Total saved image/label partition pairs:")
    print(total_saved)
    print()
    print("Permanent split folder:")
    print(SPLIT_ROOT)
    print()
    print("Single source of truth:")
    print(MASTER_SPLIT)
    print()
    print("IMPORTANT:")
    print(
        "Do not regenerate this split after baseline training starts. "
        "All future baseline training must read these saved files."
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        log_file.flush()
        log_file.close()
