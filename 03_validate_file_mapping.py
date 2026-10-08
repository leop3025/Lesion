from pathlib import Path
import csv


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(r"C:\Users\Preethi\Documents\LESION")

MANIFEST_PATH = PROJECT_DIR / "manifest.csv"

PATIENTS_TO_CHECK = [
    "001",
    "002",
    "003",
    "050",
    "100",
]


# ============================================================
# MAIN
# ============================================================

def main():

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        rows = {
            row["patient_id"]: row
            for row in reader
        }

    print("=" * 80)
    print("MS3SEG FILE MAPPING CHECK")
    print("=" * 80)

    for patient_id in PATIENTS_TO_CHECK:

        print()
        print("=" * 80)
        print(f"PATIENT {patient_id}")
        print("=" * 80)

        row = rows.get(patient_id)

        if row is None:
            print("Patient not found in manifest.")
            continue

        fields = [
            "flair",
            "t1",
            "t2",
            "ventricle_mask",
            "normal_wmh_mask",
            "lesion_mask",
            "other_mask",
            "sources",
            "n_files",
        ]

        for field in fields:

            value = row.get(field, "")

            print(
                f"{field:22s}: {value}"
            )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()