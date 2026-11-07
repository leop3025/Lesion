
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load the predicted segmentation
file_path = r"C:\Users\Preethi\Documents\LESION\logs\distance_map_unet_smoke\fold_1\predictions_nifti\012_pred.nii.gz"

nii = nib.load(file_path)
volume = nii.get_fdata()

print("Shape:", volume.shape)
print("Data type:", volume.dtype)
print("Unique labels:", np.unique(volume))
print("Voxel spacing (mm):", nii.header.get_zooms()[:3])

# Display the middle slice
slice_idx = volume.shape[2] // 2

plt.figure(figsize=(8, 8))
plt.imshow(volume[:, :, slice_idx].T, cmap="viridis", origin="lower")
plt.title(f"Prediction - Slice {slice_idx}")
plt.colorbar(label="Label / Predicted value")
plt.axis("off")
plt.tight_layout()
plt.show()
