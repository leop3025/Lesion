from pathlib import Path
import numpy as np
import nibabel as nib

root = Path(
    r"C:\Users\Preethi\Documents\LESION"
    r"\output\MS_100_patient_preprocessed"
    r"\MS_100_patient_preprocessed"
)

passed = 0
failed = 0

print("=" * 70)
print("MS3SEG PREPROCESSED NIFTI GEOMETRY VALIDATION")
print("=" * 70)

for i in range(1, 101):

    sid = f"{i:03d}"
    folder = root / sid

    flair_path = folder / f"{sid}_FLAIR.nii.gz"
    t1_path = folder / f"{sid}_T1WI_reg.nii.gz"
    t2_path = folder / f"{sid}_T2WI_reg.nii.gz"

    try:
        flair = nib.load(flair_path)
        t1 = nib.load(t1_path)
        t2 = nib.load(t2_path)

        t1_ok = (
            flair.shape[:3] == t1.shape[:3]
            and np.allclose(
                flair.header.get_zooms()[:3],
                t1.header.get_zooms()[:3]
            )
            and np.allclose(
                flair.affine,
                t1.affine,
                atol=1e-4
            )
        )

        t2_ok = (
            flair.shape[:3] == t2.shape[:3]
            and np.allclose(
                flair.header.get_zooms()[:3],
                t2.header.get_zooms()[:3]
            )
            and np.allclose(
                flair.affine,
                t2.affine,
                atol=1e-4
            )
        )

        if t1_ok and t2_ok:
            print(f"{sid}: PASS")
            passed += 1
        else:
            print(f"{sid}: FAIL")

            if not t1_ok:
                print("   T1 geometry mismatch")

            if not t2_ok:
                print("   T2 geometry mismatch")

            failed += 1

    except Exception as e:
        print(f"{sid}: ERROR")
        print("  ", e)
        failed += 1


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("Total: ", passed + failed)
print("Passed:", passed)
print("Failed:", failed)