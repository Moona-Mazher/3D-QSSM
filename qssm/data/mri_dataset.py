from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    """
    Generic dataset for loading preprocessed 3D MRI volumes.

    Expected input:
        - NIfTI files (.nii or .nii.gz)
        - Each volume should already be spatially prepared
          for the model input size (default: 160 x 160 x 160)

    Returns:
        Tensor with shape [1, H, W, D]
    """

    def __init__(
        self,
        image_paths,
        expected_size=(160, 160, 160),
        normalize=True,
    ):
        self.image_paths = [
            Path(p) for p in image_paths
        ]

        self.expected_size = expected_size
        self.normalize = normalize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = nib.load(str(image_path))
        image = image.get_fdata(dtype=np.float32)

        if image.shape != self.expected_size:
            raise ValueError(
                f"{image_path} has shape {image.shape}, "
                f"but expected {self.expected_size}. "
                "Please preprocess the MRI before training."
            )

        if self.normalize:
            mean = image.mean()
            std = image.std()

            image = (
                image - mean
            ) / (std + 1e-8)

        image = torch.from_numpy(
            image
        ).float()

        # [H, W, D] -> [C, H, W, D]
        image = image.unsqueeze(0)

        return image


def find_nifti_files(data_dir):
    """
    Find all NIfTI images inside a directory.
    """

    data_dir = Path(data_dir)

    files = list(
        data_dir.rglob("*.nii")
    )

    files += list(
        data_dir.rglob("*.nii.gz")
    )

    return sorted(files)
