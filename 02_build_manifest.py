from __future__ import annotations

from pathlib import Path
import csv
import re
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(r"C:\Users\Preethi\Documents\LESION")

OUTPUT_DIR = PROJECT_DIR / "output"

MODEL_INPUT_DIR = OUTPUT_DIR / "MS_100_model_input"
MASKS_DIR = OUTPUT_DIR / "MS_100_patient_masks"
PREPROCESSED_DIR = OUTPUT_DIR / "MS_100_patient_preprocessed"
REGISTERED_DIR = OUTPUT_DIR / "MS_100_patient_registered"
FULL_DIR = OUTPUT_DIR / "MS_100_patient_full"
NIFTI_DIR = OUTPUT_DIR / "MS_100_patient_nifti"

MANIFEST_PATH = PROJECT_DIR / "manifest.csv"
AUDIT_REPORT_PATH = PROJECT_DIR / "dataset_audit.txt"


# ============================================================
# HELPERS
# ============================================================

def is_nifti(path: Path) -> bool:
    name = path.name.lower()

    return (
        name.endswith(".nii")
        or name.endswith(".nii.gz")
    )


def normalize_patient_id(path: Path, scan_root: Path) -> str | None:
    """
    Find patient ID without accidentally reading the '100'
    from folder names such as MS_100_patient_masks.

    Priority:
        1. filename
        2. relative parent folders
    """

    # --------------------------------------------------------
    # 1. Search filename FIRST
    # --------------------------------------------------------

    filename = path.name

    # Typical:
    # 001_FLAIR.nii.gz
    # 001_T1.nii.gz
    # patient_001_mask.nii.gz
    match = re.search(
        r"(?<!\d)(\d{3})(?!\d)",
        filename,
    )

    if match:
        patient_id = match.group(1)

        # Dataset patients expected 001-100
        value = int(patient_id)

        if 1 <= value <= 100:
            return f"{value:03d}"

    # --------------------------------------------------------
    # 2. Search relative path parts
    # --------------------------------------------------------

    try:
        relative = path.relative_to(scan_root)
    except ValueError:
        relative = path

    # Do not search scan_root itself because it contains MS_100
    for part in reversed(relative.parts[:-1]):

        # Exact folder e.g. 001
        if re.fullmatch(r"\d{1,3}", part):

            value = int(part)

            if 1 <= value <= 100:
                return f"{value:03d}"

        # Embedded ID e.g. patient_001
        match = re.search(
            r"(?<!\d)(\d{3})(?!\d)",
            part,
        )

        if match:

            value = int(match.group(1))

            if 1 <= value <= 100:
                return f"{value:03d}"

    return None


def classify_file(path: Path) -> str:

    name = path.name.lower()

    # ========================================================
    # IMAGE MODALITIES
    # ========================================================

    if "flair" in name:
        return "flair"

    if re.search(
        r"(^|[_\-.])t1([_\-.]|$)",
        name,
    ):
        return "t1"

    if re.search(
        r"(^|[_\-.])t2([_\-.]|$)",
        name,
    ):
        return "t2"

    # ========================================================
    # MASKS
    # ========================================================

    if "vent" in name:
        return "ventricle_mask"

    if (
        "abnormal" in name
        or "abwmh" in name
        or "lesion" in name
    ):
        return "lesion_mask"

    if (
        "normal" in name
        or "nwmh" in name
    ):
        return "normal_wmh_mask"

    if (
        "mask" in name
        or "label" in name
        or "seg" in name
        or "annotation" in name
    ):
        return "other_mask"

    return "other"


# ============================================================
# DIRECTORY SCANNER
# ============================================================

def scan_directory(
    root: Path,
    source_name: str,
    records: dict,
) -> None:

    if not root.exists():

        print(
            f"WARNING: Missing folder: {root}"
        )

        return

    print()
    print(f"Scanning: {root}")

    nifti_count = 0
    identified_count = 0
    unidentified_files = []

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        if not is_nifti(path):
            continue

        nifti_count += 1

        patient_id = normalize_patient_id(
            path,
            root,
        )

        if patient_id is None:

            unidentified_files.append(
                str(path)
            )

            continue

        identified_count += 1

        file_type = classify_file(path)

        records[patient_id]["all_files"].append(
            str(path)
        )

        records[patient_id]["sources"].add(
            source_name
        )

        # Keep first detected path of each class
        if file_type not in records[patient_id]:

            records[patient_id][file_type] = str(
                path
            )

    print(
        f"  NIfTI files found: {nifti_count}"
    )

    print(
        f"  Files assigned to patients: {identified_count}"
    )

    if unidentified_files:

        print(
            f"  WARNING: {len(unidentified_files)} "
            "file(s) had no detectable patient ID"
        )

        print("  First few:")

        for file in unidentified_files[:5]:

            print(
                f"    {file}"
            )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    records = defaultdict(
        lambda: {
            "sources": set(),
            "all_files": [],
        }
    )

    # ========================================================
    # SCAN
    # ========================================================

    scan_directory(
        MODEL_INPUT_DIR,
        "model_input",
        records,
    )

    scan_directory(
        MASKS_DIR,
        "masks",
        records,
    )

    scan_directory(
        PREPROCESSED_DIR,
        "preprocessed",
        records,
    )

    scan_directory(
        REGISTERED_DIR,
        "registered",
        records,
    )

    scan_directory(
        FULL_DIR,
        "full",
        records,
    )

    scan_directory(
        NIFTI_DIR,
        "nifti",
        records,
    )

    # ========================================================
    # PATIENT IDs
    # ========================================================

    patient_ids = sorted(
        records.keys(),
        key=lambda x: int(x),
    )

    print()
    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    print(
        f"Patients detected: {len(patient_ids)}"
    )

    if patient_ids:

        print(
            f"First patient: {patient_ids[0]}"
        )

        print(
            f"Last patient:  {patient_ids[-1]}"
        )

    # ========================================================
    # MANIFEST
    # ========================================================

    fieldnames = [
        "patient_id",
        "flair",
        "t1",
        "t2",
        "ventricle_mask",
        "normal_wmh_mask",
        "lesion_mask",
        "other_mask",
        "sources",
        "n_files",
        "audit_ok",
        "preprocessing_done",
        "label_engineering_done",
        "split",
    ]

    rows = []

    complete_count = 0
    incomplete_count = 0

    for patient_id in patient_ids:

        r = records[patient_id]

        flair = r.get(
            "flair",
            "",
        )

        t1 = r.get(
            "t1",
            "",
        )

        t2 = r.get(
            "t2",
            "",
        )

        vent = r.get(
            "ventricle_mask",
            "",
        )

        nwmh = r.get(
            "normal_wmh_mask",
            "",
        )

        lesion = r.get(
            "lesion_mask",
            "",
        )

        other_mask = r.get(
            "other_mask",
            "",
        )

        has_image = bool(
            flair or t1 or t2
        )

        has_annotation = bool(
            vent
            or nwmh
            or lesion
            or other_mask
        )

        audit_ok = (
            has_image
            and has_annotation
        )

        if audit_ok:

            complete_count += 1

        else:

            incomplete_count += 1

        rows.append(
            {
                "patient_id": patient_id,

                "flair": flair,

                "t1": t1,

                "t2": t2,

                "ventricle_mask": vent,

                "normal_wmh_mask": nwmh,

                "lesion_mask": lesion,

                "other_mask": other_mask,

                "sources": ";".join(
                    sorted(r["sources"])
                ),

                "n_files": len(
                    r["all_files"]
                ),

                "audit_ok": int(
                    audit_ok
                ),

                "preprocessing_done": 0,

                "label_engineering_done": 0,

                "split": "",
            }
        )

    # ========================================================
    # SAVE MANIFEST
    # ========================================================

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # AUDIT REPORT
    # ========================================================

    report_lines = []

    report_lines.append(
        "=" * 70
    )

    report_lines.append(
        "MS3SEG DATASET AUDIT"
    )

    report_lines.append(
        "=" * 70
    )

    report_lines.append("")

    report_lines.append(
        f"Patients detected: {len(patient_ids)}"
    )

    report_lines.append(
        f"Patients passing basic audit: {complete_count}"
    )

    report_lines.append(
        f"Patients needing review: {incomplete_count}"
    )

    report_lines.append("")

    report_lines.append(
        "Detected patient IDs:"
    )

    report_lines.append(
        ", ".join(patient_ids)
    )

    report_lines.append("")

    # ========================================================
    # MISSING DATA
    # ========================================================

    for row in rows:

        missing = []

        if not row["flair"]:

            missing.append(
                "FLAIR"
            )

        if not (
            row["ventricle_mask"]
            or row["normal_wmh_mask"]
            or row["lesion_mask"]
            or row["other_mask"]
        ):

            missing.append(
                "annotation"
            )

        if missing:

            report_lines.append(
                f"Patient {row['patient_id']}: "
                f"missing {', '.join(missing)}"
            )

    AUDIT_REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print()
    print(
        f"Passing basic audit: {complete_count}"
    )

    print(
        f"Need review:         {incomplete_count}"
    )

    print()

    print(
        f"Manifest saved to:\n"
        f"  {MANIFEST_PATH}"
    )

    print()

    print(
        f"Audit report saved to:\n"
        f"  {AUDIT_REPORT_PATH}"
    )

    print()
    print("=" * 70)
    print("PATIENT COUNT CHECK")
    print("=" * 70)

    if len(patient_ids) == 100:

        print(
            "GOOD: Exactly 100 patients detected."
        )

    else:

        print(
            f"WARNING: Expected 100 patients but "
            f"detected {len(patient_ids)}."
        )

        print(
            "Do not preprocess yet."
        )


if __name__ == "__main__":
    main()