"""Finite-difference second fundamental form and Gauss scalar curvature."""

import numpy as np
import torch

from microscope.computations_grid.basic import (
    crop_dim_borders,
    partial_derivatives_across_all_dims,
)
from microscope.computations_grid.curvature import _scalar_curvature_in_patches


def _second_fundamental_form_and_metric(
    tensor: torch.Tensor,
    difference_intervals: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsic_dim = tensor.ndim - 1
    grid_axes = set(range(intrinsic_dim))

    # J[..., a, i] is the ambient component a of the i-th tangent vector.
    tangent = partial_derivatives_across_all_dims(
        tensor, intrinsic_dim, difference_intervals
    )
    second_derivatives = partial_derivatives_across_all_dims(
        tangent, intrinsic_dim, difference_intervals
    )
    tangent = crop_dim_borders(tangent, grid_axes)

    metric = tangent.transpose(-1, -2) @ tangent
    tangent_components = torch.einsum(
        "...ar,...aij->...rij", tangent, second_derivatives
    )
    coefficients = torch.linalg.solve(
        metric, tangent_components.flatten(start_dim=-2)
    ).unflatten(-1, (intrinsic_dim, intrinsic_dim))
    form = second_derivatives - torch.einsum(
        "...ar,...rij->...aij", tangent, coefficients
    )
    return form, metric


def second_fundamental_form(
    tensor: torch.Tensor,
    difference_intervals: list[float],
) -> torch.Tensor:
    """Estimate the normal-valued second fundamental form of a grid embedding.

    Args:
        tensor: Embedded points of shape (s1, ..., sd, ambient_features).
        difference_intervals: One coordinate spacing per grid axis.

    Returns:
        A tensor of shape (s1 - 4, ..., sd - 4, ambient_features, d, d)
        on the input device. ``form[..., a, i, j]`` is the ambient component
        of B(∂i, ∂j). All grid axes are cropped by two at each end; pad
        periodic axes first. Full tangent rank is required.

    The tangent projection uses only a d-by-d solve. No ambient-sized normal
    basis or ambient-by-ambient projector is constructed.
    """
    form, _ = _second_fundamental_form_and_metric(tensor, difference_intervals)
    return form


def scalar_curvature_batch(
    tensor: torch.Tensor,
    difference_intervals: list[float],
    normalize: bool = False,
) -> torch.Tensor:
    """Estimate extrinsic scalar curvature via the Euclidean Gauss identity.

    The sign convention matches intrinsic ``curvature.scalar_curvature_batch``:
    a radius-r sphere has positive scalar curvature d(d-1)/r². ``normalize``
    divides by d(d-1), and requires d >= 2. The output has shape
    (s1 - 6, ..., sd - 6), matching the intrinsic batch function.

    For the separable sin/cos and sinh/cosh parameterizations used by the
    analytic manifold examples, matched central-difference factors cancel
    in the Gauss contraction. Their errors can therefore be dominated by
    roundoff even on coarse grids and grow as spacing decreases. Generic
    smooth embeddings retain finite-difference truncation error.
    """
    intrinsic_dim = tensor.ndim - 1
    if normalize and intrinsic_dim < 2:
        raise ValueError("Normalized scalar curvature requires intrinsic dimension at least 2.")

    form, metric = _second_fundamental_form_and_metric(tensor, difference_intervals)
    metric_inv = torch.linalg.inv(metric)

    # Sum over the ambient axis once. The remaining Gram tensor has only d^4
    # entries per point, even when the ambient embedding is very large.
    gram = torch.einsum("...aij,...akl->...ijkl", form, form)
    scalar = (
        torch.einsum("...ij,...kl,...ijkl->...", metric_inv, metric_inv, gram)
        - torch.einsum("...ik,...jl,...ijkl->...", metric_inv, metric_inv, gram)
    )
    if normalize:
        scalar = scalar / (intrinsic_dim * (intrinsic_dim - 1))

    # B needs two finite differences; the public curvature API keeps the
    # intrinsic method's three-point border and patch halo.
    return crop_dim_borders(scalar, set(range(intrinsic_dim)))


def scalar_curvature(
    features_on_grid: np.ndarray,
    difference_intervals: list[float],
    cyclic_dimensions: list[int],
    patch_sizes: list[int],
    normalize: bool = False,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Estimate extrinsic scalar curvature with the intrinsic NumPy grid API.

    For periodic axes, the output retains the input length; other axes lose
    three points at each end. ``device=None`` selects CUDA if available, else
    CPU. Patches bound the memory used by the ambient-vector-valued form.
    """
    return _scalar_curvature_in_patches(
        features_on_grid, difference_intervals, cyclic_dimensions, patch_sizes,
        normalize, device, scalar_curvature_batch
    )
