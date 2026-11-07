import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import math

path = r"C:\Users\Preethi\Documents\LESION\six_class_masks\001\001_SixClass_Mask.nii.gz"

img = nib.load(path)
data = np.asarray(img.dataobj).astype(np.uint8)

names = [
    "Background",
    "Ventricle",
    "Normal WMH",
    "PV Abnormal WMH",
    "Deep Abnormal WMH"
]

colors = ["black", "cyan", "yellow", "red", "magenta"]

print("Patient: 001")
print("Mask shape:", data.shape)
print("\nFIVE CLASS LABEL CHECK")
print("-" * 45)

for i in range(5):
    count = np.count_nonzero(data == i)
    print(f"Label {i}: {names[i]} = {count} voxels")

unique_labels = np.unique(data)

print("\nUnique labels:", unique_labels)

if np.array_equal(unique_labels, np.arange(5)):
    print("PASS: All 5 classes (0-4) are present.")
elif np.all(np.isin(unique_labels, np.arange(5))):
    print("PASS: Only valid labels (0-4) found.")
    print("Some classes are absent in this patient.")
else:
    print("WARNING: Unexpected labels found!")

cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)

n_slices = data.shape[2]
cols = 5
rows = math.ceil(n_slices / cols)

fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3))
axes = np.array(axes).flatten()

for z in range(n_slices):
    slice_data = np.rot90(data[:, :, z])

    axes[z].imshow(slice_data, cmap=cmap, norm=norm)
    axes[z].set_title(f"Slice {z}")
    axes[z].axis("off")

for z in range(n_slices, len(axes)):
    axes[z].axis("off")

legend = [
    Patch(facecolor=colors[i], label=f"{i}: {names[i]}")
    for i in range(5)
]

fig.legend(
    handles=legend,
    loc="lower center",
    ncol=3,
    fontsize=10
)

plt.suptitle("Patient 001 - Five Class Ground Truth", fontsize=16)
plt.tight_layout(rect=[0, 0.10, 1, 0.96])
plt.show()