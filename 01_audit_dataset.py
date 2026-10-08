"""
00_extract_archives.py

Extracts the MS3SEG dataset archives found in the Leision folder into
data/raw/<archive_name>/, skipping any archive that's already been
extracted (checkpointed via a marker file, not the full patient manifest,
since extraction happens once per archive, not once per patient).

Usage:
    python 00_extract_archives.py
    python 00_extract_archives.py --force        # re-extract everything
    python 00_extract_archives.py --only MS_100_patient_masks
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ============================================================
# CONFIG — EDIT THESE PATHS FOR YOUR MACHINE
# ============================================================

# Folder containing the raw archives (your screenshot: Documents/Leision)
ARCHIVE_DIR = Path(r"C:\Users\Preethi\Documents\DATASET-LESION")

# Where extracted data should go
OUTPUT_DIR = Path(r"C:\Users\Preethi\Documents\LESION\output")

# Path to 7z.exe. If 7-Zip is on your PATH, "7z" alone works.
SEVEN_ZIP_EXE = r"C:\Program Files\7-Zip\7z.exe"

LOG_PATH = OUTPUT_DIR / "_extraction_log.txt"

# ============================================================
# ARCHIVE GROUPS
# ============================================================
# Each entry: output subfolder name -> "first part" filename to hand to 7z.
# 7z automatically pulls in .part2, .part3, etc. as long as they're in the
# same directory as part1 and follow 7-Zip's/WinRAR's naming convention.
#
# IMPORTANT: fill in the REAL filenames (with extension) once you can see
# them. Right-click a file -> Properties -> General tab shows the true
# extension even if Explorer's Name column is truncated.

ARCHIVE_GROUPS: dict[str, str] = {
    "MS_100_model_input":            "MS_100_model_input",            # single archive, no parts
    "MS_100_patient_masks":          "MS_100_patient_masks",          # single archive, no parts
    "MS_100_patient_preprocessed":   "MS_100_patient_preprocessed",   # single archive, no parts
    "MS_100_patient_registered":     "MS_100_patient_registered",     # single archive, no parts
    "MS_100_patient_full":           "MS_100_patient_full.part1",     # multi-part -> point at part1
    "MS_100_patient_nifti":          "MS_100_patient_nifti.part1",    # multi-part -> point at part1
    "MS_100_patient_gifs":           "MS_100_patient_gifs.part1",     # multi-part -> point at part1
}


# ============================================================
# HELPERS
# ============================================================

def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(message)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def find_archive_file(archive_dir: Path, stem_hint: str) -> Path | None:
    """
    Find the actual archive file on disk matching the stem hint,
    regardless of the true extension (.zip / .rar / .7z / .001 / etc).
    """
    if not stem_hint:
        return None

    # Exact match first (in case you already filled in the full filename+ext)
    candidate = archive_dir / stem_hint
    if candidate.exists():
        return candidate

    # Otherwise search for anything starting with the stem hint
    matches = sorted(archive_dir.glob(f"{stem_hint}*"))
    matches = [m for m in matches if m.is_file()]

    if not matches:
        return None
    if len(matches) > 1:
        log(f"  WARNING: multiple files match '{stem_hint}*' -> {matches}. "
            f"Using first: {matches[0].name}")
    return matches[0]


def marker_path(output_folder: Path) -> Path:
    return output_folder / "_EXTRACTION_COMPLETE"


def is_already_extracted(output_folder: Path) -> bool:
    return marker_path(output_folder).exists()


def mark_extracted(output_folder: Path) -> None:
    marker_path(output_folder).write_text("ok", encoding="utf-8")


def check_7zip_available() -> None:
    if shutil.which(SEVEN_ZIP_EXE) is None and not Path(SEVEN_ZIP_EXE).exists():
        log(f"ERROR: 7-Zip not found at '{SEVEN_ZIP_EXE}' and not on PATH.")
        log("Install 7-Zip from https://www.7-zip.org/ or fix SEVEN_ZIP_EXE.")
        sys.exit(1)


def extract_archive(archive_file: Path, output_folder: Path) -> bool:
    """
    Run 7z x (extract with full paths) into output_folder.
    Returns True on success.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    cmd = [
        SEVEN_ZIP_EXE,
        "x",                          # extract with full paths
        str(archive_file),
        f"-o{output_folder}",
        "-y",                         # assume yes to prompts
    ]

    log(f"  Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        log(f"  FAILED (exit code {result.returncode})")
        log(f"  stdout: {result.stdout[-2000:]}")
        log(f"  stderr: {result.stderr[-2000:]}")
        return False

    return True


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                         help="Re-extract even if marker file exists")
    parser.add_argument("--only", type=str, default=None,
                         help="Only process this archive group name")
    args = parser.parse_args()

    check_7zip_available()

    groups = ARCHIVE_GROUPS
    if args.only:
        if args.only not in groups:
            log(f"ERROR: '{args.only}' not in ARCHIVE_GROUPS. "
                f"Options: {list(groups.keys())}")
            sys.exit(1)
        groups = {args.only: groups[args.only]}

    log(f"\n=== Extraction run started ===")
    log(f"Archive dir: {ARCHIVE_DIR}")
    log(f"Output dir:  {OUTPUT_DIR}")

    results = {}

    for group_name, stem_hint in groups.items():
        output_folder = OUTPUT_DIR / group_name
        log(f"\n[{group_name}]")

        if is_already_extracted(output_folder) and not args.force:
            log(f"  Already extracted -> skipping (marker found at "
                f"{marker_path(output_folder)}). Use --force to redo.")
            results[group_name] = "skipped"
            continue

        archive_file = find_archive_file(ARCHIVE_DIR, stem_hint)

        if archive_file is None:
            log(f"  ERROR: could not find archive file for hint "
                f"'{stem_hint}' in {ARCHIVE_DIR}")
            results[group_name] = "not_found"
            continue

        log(f"  Found archive: {archive_file.name}")

        success = extract_archive(archive_file, output_folder)

        if success:
            mark_extracted(output_folder)
            log(f"  Done -> {output_folder}")
            results[group_name] = "extracted"
        else:
            results[group_name] = "failed"

    log("\n=== Summary ===")
    for group_name, status in results.items():
        log(f"  {group_name}: {status}")


if __name__ == "__main__":
    main()