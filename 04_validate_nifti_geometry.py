import nibabel as nib
import numpy as np

base = r"C:\Users\Preethi\Documents\LESION\output\MS_100_patient_preprocessed\MS_100_patient_preprocessed\001"

flair = nib.load(base + r"\001_FLAIR.nii.gz")
t1 = nib.load(base + r"\001_T1WI_reg.nii.gz")
t2 = nib.load(base + r"\001_T2WI_reg.nii.gz")

print("=" * 60)
print("SUBJECT 001 GEOMETRY CHECK")
print("=" * 60)

for name, img in [
    ("FLAIR", flair),
    ("T1WI_reg", t1),
    ("T2WI_reg", t2)
]:
    print("\n", name)
    print("Shape:", img.shape)
    print("Voxel size:", img.header.get_zooms()[:3])
    print("Orientation:", nib.aff2axcodes(img.affine))
    print("Affine:")
    print(img.affine)


print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print("\nFLAIR vs T1:")
print("Shape:", flair.shape[:3] == t1.shape[:3])
print(
    "Voxel size:",
    np.allclose(
        flair.header.get_zooms()[:3],
        t1.header.get_zooms()[:3]
    )
)
print(
    "Affine:",
    np.allclose(flair.affine, t1.affine, atol=1e-4)
)

print("\nFLAIR vs T2:")
print("Shape:", flair.shape[:3] == t2.shape[:3])
print(
    "Voxel size:",
    np.allclose(
        flair.header.get_zooms()[:3],
        t2.header.get_zooms()[:3]
    )
)
print(
    "Affine:",
    np.allclose(flair.affine, t2.affine, atol=1e-4)
)