from typing import Optional

import numpy as np
import torch
from einops import einops

from microscope.computations_grid.basic import partial_derivatives_across_all_dims, crop_dim_borders, riemannian_metric
from microscope.patches import extract_patches, stack_patches


def christoffel_symbols(
    metric: torch.Tensor,
    metric_inv: torch.Tensor,
    difference_intervals: list[float]
):
    """Estimate Christoffel symbols of both kinds by finite differences.

    Args:
        metric: Metric tensor of shape (s1, ..., sk, k, k).
        metric_inv: Its precomputed inverse, with the same shape.
        difference_intervals: One coordinate spacing per grid axis.

    Returns:
        (gamma_first_kind, gamma_second_kind), each a tensor of shape
        (s1 - 2, ..., sk - 2, k, k, k). All grid axes are cropped."""
    n_dims = len(metric.shape) - 2
    metric_derivatives = partial_derivatives_across_all_dims(
        metric,
        manifold_dim=n_dims,
        difference_intervals=difference_intervals
    )
    gamma_first_kind = 0.5*(
        einops.rearrange(metric_derivatives, "... j k i-> ... i j k")
        + einops.rearrange(metric_derivatives, "... k i j-> ... i j k")
        - metric_derivatives
    )

    dim_idxs = set(np.arange(n_dims))
    metric_inv_cropped = crop_dim_borders(metric_inv, dim_idxs)
    gamma_second_kind = einops.einsum(metric_inv_cropped, gamma_first_kind, "... i r, ... j k r -> ... i j k")

    return gamma_first_kind, gamma_second_kind


def riemannian_curvature_tensor(
    gamma_first_kind: torch.Tensor,
    gamma_second_kind: torch.Tensor,
    difference_intervals: list[float],
    kinds: list[str] = ("first", "second")
) -> dict[str, Optional[torch.Tensor]]:
    """Estimate Riemann tensors by differentiating Christoffel symbols.

    Args:
        gamma_first_kind: First-kind symbols of shape (s1, ..., sk, k, k, k).
        gamma_second_kind: Second-kind symbols with the same shape.
        difference_intervals: One coordinate spacing per grid axis.
        kinds: Requested kinds: "first", "second", or both.

    Returns:
        A dictionary with keys "first" and "second"; unrequested entries are
        None. Requested tensors have shape (s1 - 2, ..., sk - 2, k, k, k, k).
        Each kind differentiates its corresponding Christoffel symbols."""
    n_dims = len(gamma_first_kind.shape) - 3
    gamma_derivatives_first_kind = partial_derivatives_across_all_dims(
        gamma_first_kind,
        manifold_dim=n_dims,
        difference_intervals=difference_intervals
    )
    gamma_derivatives_second_kind = partial_derivatives_across_all_dims(
        gamma_second_kind,
        manifold_dim=n_dims,
        difference_intervals=difference_intervals
    )

    dim_idxs = set(np.arange(n_dims))
    gamma_first_kind = crop_dim_borders(gamma_first_kind, dim_idxs)
    gamma_second_kind = crop_dim_borders(gamma_second_kind, dim_idxs)

    result = {"first": None, "second": None}

    if "first" in kinds:
        riemannian_tensor_first_kind = (
            einops.rearrange(gamma_derivatives_first_kind, "... j l i k -> ... i j k l")
            - einops.rearrange(gamma_derivatives_first_kind, "... j k i l -> ... i j k l")
            + einops.einsum(gamma_first_kind, gamma_second_kind, "... i l r, ... r j k -> ... i j k l")
            - einops.einsum(gamma_first_kind, gamma_second_kind, "... i k r, ... r j l -> ... i j k l")
        )
        result["first"] = riemannian_tensor_first_kind

    if "second" in kinds:
        riemannian_tensor_second_kind = (
            einops.rearrange(gamma_derivatives_second_kind, "... i j l k -> ... i j k l")
            - gamma_derivatives_second_kind
            + einops.einsum(gamma_second_kind, gamma_second_kind, "... r j l, ... i r k -> ... i j k l")
            - einops.einsum(gamma_second_kind, gamma_second_kind, "... r j k, ... i r l -> ... i j k l")
        )
        result["second"] = riemannian_tensor_second_kind

    return result


def scalar_curvature_batch(
    tensor: torch.Tensor,
    difference_intervals: list[float],
    normalize: bool = False
) -> torch.Tensor:
    """Estimate scalar curvature from a tensor grid using finite differences.

    Args:
        tensor: Tensor of shape (s1, ..., sd, features).
        difference_intervals: One coordinate spacing per grid axis.
        normalize: Divide scalar curvature by d * (d - 1); requires d >= 2.
            This does not rescale the manifold to unit volume.

    Returns:
        A tensor of shape (s1 - 6, ..., sd - 6) on the input device.
        All grid axes lose three samples at each end; pad periodic axes first.

    Raises:
        ValueError: If normalize is requested for d < 2."""
    intrinsic_dim = len(tensor.shape[:-1])
    if normalize and intrinsic_dim < 2:
        raise ValueError("Normalized scalar curvature requires intrinsic dimension at least 2.")

    metric = riemannian_metric(tensor, difference_intervals)
    metric_inv = torch.linalg.inv(metric)

    gamma_first_kind, gamma_second_kind = christoffel_symbols(
        metric,
        metric_inv,
        difference_intervals=difference_intervals
    )

    riemannian_tensor_first_kind = riemannian_curvature_tensor(
        gamma_first_kind,
        gamma_second_kind,
        difference_intervals=difference_intervals,
        kinds=["first"]
    )["first"]

    n_dims = set(np.arange(intrinsic_dim))
    metric_inv_cropped = crop_dim_borders(metric_inv, n_dims, crop=2)
    ricci_first_kind = einops.einsum(
        metric_inv_cropped,
        riemannian_tensor_first_kind, "... a b, ... a i b j -> ... i j"
    )
    if normalize:
        ricci_first_kind = 1 / (intrinsic_dim - 1) * ricci_first_kind

    scalar = einops.einsum(metric_inv_cropped, ricci_first_kind, "... i j, ... j i -> ...")
    if normalize:
        scalar = 1 / intrinsic_dim * scalar

    return scalar


def scalar_curvature(
    features_on_grid: np.ndarray,
    difference_intervals: list[float],
    cyclic_dimensions: list[int],
    patch_sizes: list[int],
    normalize: bool = False,
    device: str | torch.device | None = None
) -> np.ndarray:
    """Estimate scalar curvature of a NumPy grid in patches.

    Args:
        features_on_grid: Floating array of shape (s1, ..., sd, features).
        difference_intervals: One coordinate spacing per grid axis.
        cyclic_dimensions: Zero-based periodic grid-axis indices.
        patch_sizes: One patch size per grid axis, each greater than six.
        normalize: Divide scalar curvature by d * (d - 1); requires d >= 2.
            This does not rescale the manifold to unit volume.
        device: Torch device; None selects CUDA when available, otherwise CPU.

    Returns:
        A NumPy array of shape (s1_out, ..., sd_out), where s_i_out = s_i
        for periodic axes and s_i - 6 otherwise: one scalar curvature per point.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    dims = len(patch_sizes)
    overlaps = dims * [6]

    patches = extract_patches(
        features_on_grid,
        patch_sizes,
        overlaps,
        cyclic_dimensions=cyclic_dimensions
    )

    volume_patches = []
    for patch in patches.reshape(-1, *patches.shape[dims:]):
        points_pt_patch = torch.from_numpy(patch).to(device)

        volume_pred = scalar_curvature_batch(
            points_pt_patch,
            difference_intervals=difference_intervals,
            normalize=normalize
        ).cpu().detach().numpy()
        volume_patches.append(volume_pred)

    volume_patches = np.stack(volume_patches, axis=0).reshape(
        *patches.shape[:dims],
        *list(np.array(patches.shape[dims:2 * dims]) - np.array(overlaps)),
        *patches.shape[2 * dims:-1],
    )

    curvature_array = stack_patches(volume_patches, n_features=0)

    # Truncate any padded zeros introduced in the patch extraction.
    expected_array_shape = tuple(
        slice(
            0,
            s if i in cyclic_dimensions else s - 6
        )
        for i, s in enumerate(features_on_grid.shape[:-1])
    )

    return curvature_array[expected_array_shape]


def compute_total_curvature(element: np.ndarray, curvature: np.ndarray, range_sizes: list[float]) -> float:
    grid_volume = np.prod(range_sizes)

    # Note that the total curvature = curv * element * (grid_vol / N^2)
    # = (curv * element) / N^2 * grid_vol = mean(curv * element) * grid_vol.
    return (curvature * element).mean() * grid_volume
