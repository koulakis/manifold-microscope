from typing import Optional, Union

import numpy as np
import torch
from einops import einops
from tqdm import tqdm

from microscope.computations_grid.basic import crop_dim_borders, partial_derivatives_across_all_dims_batched
from microscope.cyclic_dimensions import get_difference_intervals


def _distances_to_tangent_space(tangent_vectors: torch.Tensor, points: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    tangent_vectors_T = einops.rearrange(tangent_vectors, "... m n -> ... n m")

    projection_matrix = (
        tangent_vectors
        @ torch.linalg.inv(tangent_vectors_T @ tangent_vectors)
        @ tangent_vectors_T
    )

    point_diffs = points[None] - x[:, None]
    points_tangent_projections = point_diffs @ projection_matrix
    return torch.linalg.norm(point_diffs - points_tangent_projections, dim=-1)


def _subsample_dims(tensor: torch.Tensor, take_every_k: int, dim_idxs: set[int]) -> torch.Tensor:
    for dim in dim_idxs:
        tensor = torch.index_select(
            tensor,
            dim=dim,
            index=torch.arange(0, tensor.shape[dim], step=take_every_k, device=tensor.device)
        )

    return tensor


def reach_per_point(
    features_on_grid: np.ndarray,
    range_sizes: list[float],
    patch_sizes: list[int],
    cyclic_dimensions: Optional[list[int]] = None,
    subsample_points: Optional[int] = None,
    batch_size: int = 10,
    return_witnesses: bool = False,
    device: str = "cuda:0"
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Estimate the reach based on https://arxiv.org/pdf/1705.04565. The formula is:

    reach = min_{x != y in M} ||x - y||^2 / (2 d(y - x, T_xM))

    and finite differences are used to approximate the tangent space T_xM.

    Args:
        features_on_grid: A tensor of shape (s1 ... sk f), where s_i is the number of points of the i-th dimension
            of its grid and f the number of features.
        range_sizes: The sizes of the value range of each dimension of the grid.
        patch_sizes: The sizes of the patches used along each dimension of the grid.
        cyclic_dimensions: A set of dimensions where the grid is cyclic.
        batch_size: The size of batches on which the local reach is computed.
        subsample_points: If set to some integer, then the local reach will be computed only on every
            n-th point. The tangent spaces will be approximated though with all points.
        return_witnesses: If true, return the pair point producing the smallest local reach.
        device: The torch device.

    Returns:
        The reach estimate per point in a tensor of shape (s1_ ... sk_). The new dimensions s_i_ equal to
        s_i if the i-th dimension is cyclic, else to s_i - 2 as the computations cannot be performed on the borders.
    """
    if cyclic_dimensions is None:
        cyclic_dimensions = []
    if len(patch_sizes) != len(range_sizes):
        raise ValueError(f"Patch ({len(patch_sizes)}) and range ({len(range_sizes)}) should have same length.")
    dims_shape = features_on_grid.shape[:-1]
    n_dims = len(dims_shape)

    intrinsic_dim = len(dims_shape)
    dim_idxs = set(np.arange(intrinsic_dim))

    features_on_grid_pt = torch.from_numpy(features_on_grid)
    features_on_grid_cropped = crop_dim_borders(features_on_grid_pt, dim_idxs.difference(cyclic_dimensions))

    if subsample_points is not None:
        features_on_grid_cropped = _subsample_dims(features_on_grid_cropped, subsample_points, dim_idxs)

    features_on_grid_cropped_flat = features_on_grid_cropped.reshape(-1, features_on_grid_cropped.shape[-1]).to(device)

    difference_intervals = get_difference_intervals(
        n_samples_per_dim=list(dims_shape),
        range_sizes=range_sizes,
        cyclic_dimensions=cyclic_dimensions
    )
    tangent_vectors = torch.from_numpy(partial_derivatives_across_all_dims_batched(
        features_on_grid,
        cyclic_dimensions=cyclic_dimensions,
        difference_intervals=difference_intervals,
        patch_sizes=patch_sizes
    ))

    if subsample_points is not None:
        tangent_vectors = _subsample_dims(tangent_vectors, subsample_points, dim_idxs)

    tangent_vectors_flat = tangent_vectors.reshape(-1, *tangent_vectors.shape[-2:])

    reach_estimates = []
    estimate_witnesses = []
    num_ranges = len(features_on_grid_cropped_flat) // batch_size
    iteration_batch_ranges = [
        (i*batch_size, (i + 1)*batch_size)
        for i in range(num_ranges)
    ]

    if len(features_on_grid_cropped_flat) % batch_size != 0:
        iteration_batch_ranges += [(batch_size*num_ranges, len(features_on_grid_cropped_flat))]

    for s, e in tqdm(iteration_batch_ranges):
        x = features_on_grid_cropped_flat[s:e].to(device)
        T_x = tangent_vectors_flat[s:e].to(device)
        d_y_tang_x = _distances_to_tangent_space(T_x, features_on_grid_cropped_flat, x)

        x_estimate = torch.linalg.norm(x[:, None] - features_on_grid_cropped_flat[None], dim=-1)**2 / (2*d_y_tang_x)
        # Here x is included on the points to compare with. To make sure this pair is not picked,
        # we assign to it the largest possible value.
        max_value = float(10 * (torch.nan_to_num(x_estimate) + 1).max())
        x_estimate = torch.nan_to_num(x_estimate, nan=max_value)

        estimate_witnesses.append(torch.argmin(x_estimate, dim=-1).cpu().numpy())
        reach_estimates.append(x_estimate.min(dim=-1)[0].cpu().detach())

    reach_estimates = np.concatenate(reach_estimates, axis=0)
    estimate_witnesses = np.concatenate(estimate_witnesses, axis=0)

    grid_reach_estimates = reach_estimates.reshape(features_on_grid_cropped.shape[:-1])

    # If the reach has been computed on subsample points, repeat the values to get the original shape.
    if subsample_points is not None:
        dim_variables = [f'i{i}' for i in range(n_dims)]
        repeat_variables = [f'r{i}' for i in range(n_dims)]
        repeat_pattern = [f'({d} {r})' for d, r in zip(dim_variables, repeat_variables)]
        grid_reach_estimates = einops.repeat(
            grid_reach_estimates,
            f"{' '.join(dim_variables)} -> {' '.join(repeat_pattern)}",
            **{r: subsample_points for r in repeat_variables}
            )

    if return_witnesses:
        return grid_reach_estimates, estimate_witnesses
    else:
        return grid_reach_estimates
