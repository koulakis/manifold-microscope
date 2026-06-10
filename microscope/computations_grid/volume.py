import numpy as np
import torch

from microscope.computations_grid.basic import riemannian_metric
from microscope.patches import extract_patches, stack_patches


def volume_element_batch(
        features_on_grid: torch.Tensor,
        difference_intervals: list[float]
) -> torch.Tensor:
    """Given a grid with features on the space of the manifold, it computes an estimate of the volume element per point
    of the grid, excluding border points, using a finite element approximation to compute the Riemannian metric and
    then the volume element.

    Args:
        features_on_grid: A tensor of shape (s1 ... sk f), where s_i is the number of points of the i-th dimension
            of its grid and f the number of features.
        difference_intervals: The h value for each dimension of the grid.

    Returns:
        A tensor of shape (s1_ ... sk_) which has a single volume element value per point. The new dimensions s_i_
        equal to s_i if the i-th dimension is cyclic, else to s_i - 2 as the computations cannot be performed on
        the borders.
    """
    metric = riemannian_metric(features_on_grid, difference_intervals)

    return torch.sqrt(torch.linalg.det(metric))


def volume_element(
    features_on_grid: np.ndarray,
    difference_intervals: list[float],
    cyclic_dimensions: list[int],
    patch_sizes: list[int],
    device: str = "cuda:0"
) -> np.ndarray:
    """Given a grid with features on the space of the manifold, it computes an estimate of the volume element per point
    of the grid, excluding border points, using a finite element approximation to compute the Riemannian metric and
    then the volume element. It performs the computation on batches of given size to reduce visual memory requirements.

    Args:
        features_on_grid: A tensor of shape (s1 ... sk f), where s_i is the number of points of the i-th dimension
            of its grid and f the number of features.
        difference_intervals: The h value for each dimension of the grid.
        cyclic_dimensions: An optional set of dimensions where the grid is cyclic.
        patch_sizes: The size of the patch to use. One value per dimension.
        device: The torch device.

    Returns:
        A tensor of shape (s1_ ... sk_) which has a single volume element value per point. The new dimensions s_i_
        equal to s_i if the i-th dimension is cyclic, else to s_i - 2 as the computations cannot be performed on
        the borders.
    """
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
