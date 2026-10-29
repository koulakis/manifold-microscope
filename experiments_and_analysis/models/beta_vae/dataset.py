"""Dataset utilities for the adapted beta-VAE implementation.

Copied and adapted from https://github.com/1Konny/Beta-VAE. The upstream project is MIT licensed; see this package's
`LICENSE` and the repository's `docs/source/third_party/beta_vae.rst` for copyright and attribution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset, DataLoader

if TYPE_CHECKING:
    import numpy as np


def is_power_of_2(num: int) -> bool:
    """Check whether a nonzero integer is a power of two.

    Args:
        num: Integer to check.

    Returns:
        True if num is a positive power of two; otherwise False.
    """
    return ((num & (num - 1)) == 0) and num != 0


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor) -> None:
        """Wrap an image tensor as a dataset.

        Args:
            data_tensor: Images of shape (n c h w).

        Returns:
            None.
        """
        self.data_tensor = data_tensor

    def __getitem__(self, index: int | slice) -> torch.Tensor:
        """Return an image or selected image batch.

        Args:
            index: Image index or slice along the sample axis.

        Returns:
            Image tensor of shape (c h w) for an integer index, or an image batch of shape (n_selected c h w) for a
            slice.
        """
        return self.data_tensor[index]

    def __len__(self) -> int:
        """Return the number of images.

        Returns:
            Number of images in the dataset.
        """
        return self.data_tensor.size(0)


def training_loader(
    data: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
    device: str = "cpu"
) -> DataLoader:
    """Build a shuffled loader for grayscale 64-by-64 images.

    Args:
        data: Grayscale images of shape (n 64 64).
        batch_size: Number of points or images per batch.
        num_workers: Number of data-loader workers.
        device: Device used for tensor computations.

    Returns:
        DataLoader yielding shuffled image tensors of shape (b c h w), including any smaller final batch.
    """
    if data.shape[1:] != (64, 64):
        raise ValueError("beta-VAE requires grayscale images of shape (64, 64).")
    images = torch.from_numpy(data).unsqueeze(1).float()
    return DataLoader(
        CustomTensorDataset(images),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda")
    )
