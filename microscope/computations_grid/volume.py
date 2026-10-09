import numpy as np
import torch

from microscope.computations_grid.basic import riemannian_metric
from microscope.patches import extract_patches, stack_patches


def volume_element_batch(
        features_on_grid: torch.Tensor,
        difference_intervals: list[float]
) -> torch.Tensor:
    """Estimate sqrt(det(g)) from central finite differences.

    Args:
        features_on_grid: Tensor of shape (s1, ..., sk, features).
        difference_intervals: One coordinate spacing per grid axis.

    Returns:
        A tensor of shape (s1 - 2, ..., sk - 2) on the input device.
        All grid axes are cropped; pad periodic axes first."""
    metric = riemannian_metric(features_on_grid, difference_intervals)

    return torch.sqrt(torch.linalg.det(metric))


def volume_element(
    features_on_grid: np.ndarray,
    difference_intervals: list[float],
    cyclic_dimensions: list[int],
    patch_sizes: list[int],
    device: str | torch.device | None = None
) -> np.ndarray:
    """Estimate the volume element of a NumPy grid in patches.

    Args:
        features_on_grid: Floating array of shape (s1, ..., sk, features).
        difference_intervals: One coordinate spacing per grid axis.
        cyclic_dimensions: Zero-based periodic grid-axis indices.
        patch_sizes: One patch size per grid axis, each greater than two.
        device: Torch device; None selects CUDA when available, otherwise CPU.

    Returns:
        A NumPy array of shape (s1_out, ..., sk_out), where s_i_out = s_i
        for periodic axes and s_i - 2 otherwise: one volume element per point.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dims = len(patch_sizes)
    overlaps = dims * [2]

    patches = extract_patches(
        features_on_grid,
        patch_sizes,
        overlaps,
        cyclic_dimensions=cyclic_dimensions
    )

    volume_patches = []
    for patch in patches.reshape(-1, *patches.shape[dims:]):
        points_pt_patch = torch.from_numpy(patch).to(device)

        volume_pred = volume_element_batch(
            points_pt_patch,
            difference_intervals=difference_intervals
        ).cpu().detach().numpy()
        volume_patches.append(volume_pred)

    volume_patches = np.stack(volume_patches, axis=0).reshape(
        *patches.shape[:dims],
        *list(np.array(patches.shape[dims:2*dims]) - np.array(overlaps)),
        *patches.shape[2*dims:-1],
    )

    volume_element_array = stack_patches(volume_patches, n_features=0)

    # Truncate any padded zeros introduced in the patch extraction.
    expected_array_shape = tuple(
        slice(
            0,
            s if i in cyclic_dimensions else s - 2
        )
        for i, s in enumerate(features_on_grid.shape[:-1])
    )

    return volume_element_array[expected_array_shape]


def compute_total_volume(element: np.ndarray, range_sizes: list[float]) -> float:
    grid_volume = np.prod(range_sizes)

    # Note that the total area = sum(element) * (grid_vol / N^2) = sum(element) / N^2 * grid_vol
    # = mean(element) * grid_vol.
    return element.mean() * grid_volume
