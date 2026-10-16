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
    device: str | torch.device | None = None
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Estimate local reach from sampled pairs and finite-difference tangents.

    Uses min_{y != x} ||y - x||^2 / (2 d(y - x, T_x M)); see
    https://arxiv.org/abs/1705.04565. The minimum over x estimates global reach.

    Args:
        features_on_grid: Floating array of shape (s1, ..., sk, features).
        range_sizes: Full parameter-range length along each grid axis.
        patch_sizes: Derivative patch sizes, one per axis, each greater than two.
        cyclic_dimensions: Zero-based periodic grid-axis indices; None means none.
        subsample_points: Optional positive stride along every axis for both
            queries and candidates. Tangents are computed from the full grid.
        batch_size: Number of query points per pair-search batch.
        return_witnesses: Also return minimizing candidate indices.
        device: Torch device; None selects CUDA when available, otherwise CPU.

    Returns:
        A NumPy array of local estimates of shape (s1_out, ..., sk_out), where
        s_i_out = s_i for periodic axes and s_i - 2 otherwise. Subsampled
        estimates are repeated and truncated back to this shape.

        If return_witnesses is true, return (estimates, witnesses). The integer
        witnesses array has shape (N,), with N = prod(ceil(s_i_out / stride))
        and stride = subsample_points or 1. Each entry is a flat index into the
        cropped, subsampled point array. Witnesses are not repeated or reshaped.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
    cropped_grid_shape = features_on_grid_cropped.shape[:-1]

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
        patch_sizes=patch_sizes,
        device=device
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
        # Repetition can overshoot dimensions not divisible by the subsampling step.
        grid_reach_estimates = grid_reach_estimates[tuple(slice(0, s) for s in cropped_grid_shape)]

    if return_witnesses:
        return grid_reach_estimates, estimate_witnesses
    else:
        return grid_reach_estimates
