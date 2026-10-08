from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Preethi\Documents\LESION"
)

AUDIT_DIR = PROJECT_ROOT / "data" / "audit"

SUMMARY_CSV = AUDIT_DIR / "summary.csv"

VISUAL_REVIEW_CSV = (
    AUDIT_DIR /
    "visual_qc_review.csv"
)

PLAN_CSV = (
    AUDIT_DIR /
    "preprocess_plan.csv"
)


# ============================================================
# LOAD STAGE-1 SUMMARY
# ============================================================

if not SUMMARY_CSV.exists():
    raise FileNotFoundError(
        f"Stage-1 summary not found:\n{SUMMARY_CSV}"
    )


summary = pd.read_csv(
    SUMMARY_CSV,
    dtype={
        "patient_id": str
    },
    keep_default_na=False
)

summary["patient_id"] = (
    summary["patient_id"]
    .astype(str)
    .str.zfill(3)
)


# ============================================================
# OPTIONAL VISUAL REVIEW
# ============================================================

review_lookup = {}

if VISUAL_REVIEW_CSV.exists():

    review = pd.read_csv(
        VISUAL_REVIEW_CSV,
        dtype={
            "patient_id": str
        },
        keep_default_na=False
    )

    review["patient_id"] = (
        review["patient_id"]
        .astype(str)
        .str.zfill(3)
    )

    review_lookup = (
        review
        .set_index("patient_id")
        .to_dict(orient="index")
    )


# ============================================================
# CREATE PER-PATIENT PLAN
# ============================================================

rows = []

for _, patient in summary.iterrows():

    pid = patient["patient_id"]

    review = review_lookup.get(
        pid,
        {}
    )

    # --------------------------------------------------------
    # Based on completed Stage-1 review:
    #
    # Skull stripping and N4 are skipped by default.
    #
    # These are intentionally per-patient columns so any
    # patient can later be changed to YES without modifying
    # the preprocessing script.
    # --------------------------------------------------------

    skull_strip = "NO"
    apply_n4 = "NO"

    visual_skull = str(
        review.get(
            "visual_skull_stripped",
            ""
        )
    ).strip().upper()

    visual_n4 = str(
        review.get(
            "visual_n4_needed",
            ""
        )
    ).strip().upper()

    if visual_skull == "NO":
        skull_strip = "YES"

    elif visual_skull == "YES":
        skull_strip = "NO"

    if visual_n4 == "YES":
        apply_n4 = "YES"

    elif visual_n4 == "NO":
        apply_n4 = "NO"

    rows.append(
        {
            "patient_id": pid,

            # Stage-1 decision
            "apply_skull_strip": skull_strip,
            "apply_n4": apply_n4,

            # T1/T2 already have registered versions and
            # Stage-1 geometry was validated.
            "register_t1_to_flair": "NO",
            "register_t2_to_flair": "NO",

            # Masks already geometrically match axial FLAIR.
            "register_masks_to_flair": "NO",

            # We will still put everything on the exact final
            # reference grid.
            "resample_to_reference": "YES",

            "normalize_t1": "YES",
            "normalize_t2": "YES",
            "normalize_flair": "YES",

            "plan_status": "APPROVED",

            "notes": (
                "Stage-1 QC reviewed; skull stripping/N4 "
                "skipped unless individually overridden."
            )
        }
    )


plan = pd.DataFrame(rows)

plan.to_csv(
    PLAN_CSV,
    index=False
)


print("=" * 72)
print("PREPROCESSING PLAN CREATED")
print("=" * 72)

print(f"Patients: {len(plan)}")

print()
print("Skull stripping YES:",
      (plan["apply_skull_strip"] == "YES").sum())

print("N4 correction YES:",
      (plan["apply_n4"] == "YES").sum())

print("Registration T1 YES:",
      (plan["register_t1_to_flair"] == "YES").sum())

print("Registration T2 YES:",
      (plan["register_t2_to_flair"] == "YES").sum())

print()
print("Plan saved:")
print(PLAN_CSV)

print()
print(
    "You may edit individual YES/NO values in this CSV "
    "before running Stage 2."
)