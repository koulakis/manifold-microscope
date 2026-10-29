from typing import Optional, cast

import numpy as np
import torch
from einops import einops

from microscope.computations_grid.basic import partial_derivatives_across_all_dims, crop_dim_borders
from microscope.patches import _iter_patches_with_halo
from microscope._grid_validation import validate_cyclic_axes, validate_integer_list, validate_spacings
from microscope.computations_grid._curvature_geometry import metric_and_inverse, validate_tensor


def christoffel_symbols(
    metric: torch.Tensor,
    metric_inv: torch.Tensor,
    difference_intervals: list[float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate Christoffel symbols of both kinds by finite differences.

    Args:
        metric: Metric tensor of shape (s1, ..., sk, k, k).
        metric_inv: Its precomputed inverse, with the same shape.
        difference_intervals: One coordinate spacing per grid axis.

    Returns:
        (gamma_first_kind, gamma_second_kind), each a tensor of shape (s1 - 2, ..., sk - 2, k, k, k). All grid axes are
        cropped.
    """
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
    kinds: list[str] | tuple[str, ...] = ("first", "second")
) -> dict[str, Optional[torch.Tensor]]:
    """Estimate Riemann tensors by differentiating Christoffel symbols.

    Args:
        gamma_first_kind: First-kind symbols of shape (s1, ..., sk, k, k, k).
        gamma_second_kind: Second-kind symbols with the same shape.
        difference_intervals: One coordinate spacing per grid axis.
        kinds: Requested kinds: "first", "second", or both.

    Returns:
        A dictionary with keys "first" and "second"; unrequested entries are None. Requested tensors have shape
        (s1 - 2, ..., sk - 2, k, k, k, k). Each kind differentiates its corresponding Christoffel symbols.
    """
    if not set(kinds) <= {"first", "second"}:
        raise ValueError("Riemann kinds must be 'first' or 'second'.")
    n_dims = len(gamma_first_kind.shape) - 3
    gamma_derivatives_first_kind = (
        partial_derivatives_across_all_dims(gamma_first_kind, n_dims, difference_intervals)
        if "first" in kinds else None
    )
    gamma_derivatives_second_kind = (
        partial_derivatives_across_all_dims(gamma_second_kind, n_dims, difference_intervals)
        if "second" in kinds else None
    )

    dim_idxs = set(np.arange(n_dims))
    gamma_first_kind = crop_dim_borders(gamma_first_kind, dim_idxs)
    gamma_second_kind = crop_dim_borders(gamma_second_kind, dim_idxs)

    result: dict[str, Optional[torch.Tensor]] = {"first": None, "second": None}

    if gamma_derivatives_first_kind is not None:
        riemannian_tensor_first_kind = (
            einops.rearrange(gamma_derivatives_first_kind, "... j l i k -> ... i j k l")
            - einops.rearrange(gamma_derivatives_first_kind, "... j k i l -> ... i j k l")
            + einops.einsum(gamma_first_kind, gamma_second_kind, "... i l r, ... r j k -> ... i j k l")
            - einops.einsum(gamma_first_kind, gamma_second_kind, "... i k r, ... r j l -> ... i j k l")
        )
        result["first"] = riemannian_tensor_first_kind

    if gamma_derivatives_second_kind is not None:
        riemannian_tensor_second_kind = (
            einops.rearrange(gamma_derivatives_second_kind, "... i j l k -> ... i j k l")
            - gamma_derivatives_second_kind
            + einops.einsum(gamma_second_kind, gamma_second_kind, "... r j l, ... i r k -> ... i j k l")
            - einops.einsum(gamma_second_kind, gamma_second_kind, "... r j k, ... i r l -> ... i j k l")
        )
        result["second"] = riemannian_tensor_second_kind

    return result


def _scalar_curvature_from_metric(
    metric: torch.Tensor,
    metric_inv: torch.Tensor,
    difference_intervals: list[float],
    normalize: bool
) -> torch.Tensor:
    """Apply the intrinsic formula to metric fields, cropping two grid borders.

    Args:
        metric: Metric field, shape (s1 ... sd d d).
        metric_inv: Inverse metric field with the same shape.
        difference_intervals: Coordinate spacing along each of the d grid axes.
        normalize: Divide scalar curvature by d * (d - 1).
    """
    intrinsic_dim = metric.ndim - 2
    gamma_first_kind, gamma_second_kind = christoffel_symbols(
        metric,
        metric_inv,
        difference_intervals=difference_intervals
    )

    riemannian_tensors = riemannian_curvature_tensor(
        gamma_first_kind,
        gamma_second_kind,
        difference_intervals=difference_intervals,
        kinds=["first"]
    )
    # The first-kind entry is populated because it was explicitly requested.
    riemannian_tensor_first_kind = cast(torch.Tensor, riemannian_tensors["first"])

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


def _validate_curvature_grid(
    features_on_grid: np.ndarray,
    difference_intervals: list[float],
    cyclic_dimensions: list[int],
    patch_sizes: list[int]
) -> None:
    """Shared NumPy-grid validation for the intrinsic and extrinsic wrappers.

    Require float32/float64 arrays with ambient dimension f >= d >= 1, one finite nonzero real spacing and one integer
    patch size >= 7 per axis. Periodic axes must be unique valid indices. Each nonperiodic axis needs at least seven
    samples; periodic axes need at least three. Finiteness of embedding values is checked by validate_tensor when
    processing tiles.

    Args:
        features_on_grid: Embedding array, shape (s1 ... sd f).
        difference_intervals: Coordinate spacing along each of the d grid axes.
        cyclic_dimensions: Zero-based periodic grid axes.
        patch_sizes: Patch size along each of the d grid axes.
    """
    if features_on_grid.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise TypeError("Curvature requires float32 or float64 inputs.")
    dims = features_on_grid.ndim - 1
    if dims < 1 or features_on_grid.shape[-1] < dims:
        raise ValueError("A grid needs ambient dimension >= intrinsic dimension >= 1.")
    validate_spacings(difference_intervals, dims)
    validate_integer_list(patch_sizes, "patch sizes", minimum=7)
    if len(patch_sizes) != dims:
        raise ValueError("One patch size is required per grid axis.")
    validate_cyclic_axes(cyclic_dimensions, dims)
    if any(n < (3 if axis in cyclic_dimensions else 7)
           for axis, n in enumerate(features_on_grid.shape[:-1])):
        raise ValueError("Grid axes need at least seven noncyclic or three cyclic samples.")


def scalar_curvature_batch(
    tensor: torch.Tensor,
    difference_intervals: list[float],
    normalize: bool = False
) -> torch.Tensor:
    """Estimate scalar curvature from a tensor grid using finite differences.

    Args:
        tensor: Tensor of shape (s1, ..., sd, features).
        difference_intervals: One coordinate spacing per grid axis.
        normalize: Divide scalar curvature by d * (d - 1); requires d >= 2. This does not rescale the manifold to unit
            volume.

    Returns:
        A tensor of shape (s1 - 6, ..., sd - 6) on the input device. All grid axes lose three samples at each end; pad
        periodic axes first.

    Raises:
        ValueError: For invalid grids/spacings or normalize with d < 2.
        torch.linalg.LinAlgError: For numerically rank-deficient tangents,
            with the first affected batch-grid index in the message.

    Well-conditioned Gram metrics are inverted directly. Points failing the dtype-aware spectral check use QR tangent
    factors instead. This cannot remove all sensitivity of metric differentiation in poorly conditioned coordinate
    charts; use float64 and appropriate scales.
    """
    validate_tensor(tensor, difference_intervals, border=3)
    intrinsic_dim = len(tensor.shape[:-1])
    if normalize and intrinsic_dim < 2:
        raise ValueError("Normalized scalar curvature requires intrinsic dimension at least 2.")

    tangent = partial_derivatives_across_all_dims(tensor, intrinsic_dim, difference_intervals)
    metric, metric_inv = metric_and_inverse(tangent)
    del tangent

    return _scalar_curvature_from_metric(metric, metric_inv, difference_intervals, normalize)


def _scalar_curvature_in_patches(
    features_on_grid: np.ndarray,
    difference_intervals: list[float],
    cyclic_dimensions: list[int],
    patch_sizes: list[int],
    normalize: bool,
    device: str | torch.device | None
) -> np.ndarray:
    """Estimate intrinsic curvature using embedding tiles with a three-point halo.

    Args:
        features_on_grid: Embedding array, shape (s1 ... sd f).
        difference_intervals: Coordinate spacing along each of the d grid axes.
        cyclic_dimensions: Zero-based periodic grid axes.
        patch_sizes: Patch size along each of the d grid axes.
        normalize: Divide scalar curvature by d * (d - 1).
        device: Torch device; None selects CUDA when available, otherwise CPU.
    """
    _validate_curvature_grid(
        features_on_grid,
        difference_intervals,
        cyclic_dimensions,
        patch_sizes
    )
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_shape = tuple(
        s if i in cyclic_dimensions else s - 6
        for i, s in enumerate(features_on_grid.shape[:-1])
    )
    curvature_array = np.empty(output_shape, dtype=features_on_grid.dtype)
    for destination, patch in _iter_patches_with_halo(
        features_on_grid,
        patch_sizes,
        3,
        cyclic_dimensions
    ):
        if any(stride < 0 for stride in patch.strides):
            patch = np.ascontiguousarray(patch)
        points_pt_patch = torch.from_numpy(patch).to(device)
        curvature_array[destination] = scalar_curvature_batch(
            points_pt_patch,
            difference_intervals=difference_intervals,
            normalize=normalize
        ).cpu().detach().numpy()
    return curvature_array


def scalar_curvature(
    features_on_grid: np.ndarray,
    difference_intervals: list[float],
    cyclic_dimensions: list[int],
    patch_sizes: list[int],
    normalize: bool = False,
    device: str | torch.device | None = None
) -> np.ndarray:
    """Estimate scalar curvature of a NumPy grid using metric-first tiles.

    Args:
        features_on_grid: Floating array of shape (s1, ..., sd, features).
        difference_intervals: One coordinate spacing per grid axis.
        cyclic_dimensions: Zero-based periodic grid-axis indices.
        patch_sizes: Image-tile size along each grid axis.
        normalize: Divide scalar curvature by d * (d - 1); requires d >= 2. This does not rescale the manifold to unit
            volume.
        device: Torch device; None selects CUDA when available, otherwise CPU.

    Returns:
        A NumPy array of shape (s1_out, ..., sd_out), where s_i_out = s_i for periodic axes and s_i - 6 otherwise: one
        scalar curvature per point.

    Requires a finite regular floating grid; see _validate_curvature_grid and validate_tensor for detailed requirements.
    Each metric and inverse is computed once using a one-point image halo, then curvature is computed from those small
    fields with a two-point halo. Two host arrays of shape (metric_grid_shape, d, d) are stored; this storage is
    independent of the number of embedding features. Image tiles alone move to the selected device; periodic boundaries
    are wrapped locally without padding the full embedding. The dense tensor API remains differentiable.
    """
    _validate_curvature_grid(
        features_on_grid,
        difference_intervals,
        cyclic_dimensions,
        patch_sizes
    )
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dims = features_on_grid.ndim - 1
    if normalize and dims < 2:
        raise ValueError("Normalized scalar curvature requires intrinsic dimension at least 2.")
    metric_shape = tuple(n if a in cyclic_dimensions else n-2
                         for a, n in enumerate(features_on_grid.shape[:-1]))
    metric_array = np.empty((*metric_shape, dims, dims), dtype=features_on_grid.dtype)
    inverse_array = np.empty_like(metric_array)

    with torch.no_grad():
        for destination, patch in _iter_patches_with_halo(
            features_on_grid,
            patch_sizes,
            1,
            cyclic_dimensions
        ):
            if any(stride < 0 for stride in patch.strides):
                patch = np.ascontiguousarray(patch)
            tensor = torch.from_numpy(patch).to(device)
            validate_tensor(tensor, difference_intervals, border=1)
            tangent = partial_derivatives_across_all_dims(tensor, dims, difference_intervals)
            metric, inverse = metric_and_inverse(tangent)
            metric_array[destination] = metric.cpu().numpy()
            inverse_array[destination] = inverse.cpu().numpy()
            del tensor, tangent, metric, inverse

        output_shape = tuple(n if a in cyclic_dimensions else n-6
                             for a, n in enumerate(features_on_grid.shape[:-1]))
        curvature_array = np.empty(output_shape, dtype=features_on_grid.dtype)
        metric_patch_sizes = [p-2 for p in patch_sizes]
        metric_tiles = _iter_patches_with_halo(metric_array, metric_patch_sizes, 2, cyclic_dimensions)
        inverse_tiles = _iter_patches_with_halo(inverse_array, metric_patch_sizes, 2, cyclic_dimensions)
        for (destination, metric_patch), (_, inverse_patch) in zip(metric_tiles, inverse_tiles):
            metric = torch.from_numpy(metric_patch).to(device)
            inverse = torch.from_numpy(inverse_patch).to(device)
            prediction = _scalar_curvature_from_metric(metric, inverse, difference_intervals, normalize)
            curvature_array[destination] = prediction.cpu().numpy()
            del metric, inverse, prediction
    return curvature_array


def compute_total_curvature(
    element: np.ndarray,
    curvature: np.ndarray,
    range_sizes: list[float]
) -> float:
    """Integrate scalar curvature against the volume element.

    Args:
        element: Volume-element grid, shape (s1 ... sd).
        curvature: Scalar-curvature grid, shape (s1 ... sd), aligned with element.
        range_sizes: Parameter-range length along each of the d grid axes.
    """
    grid_volume = np.prod(range_sizes)

    # Note that the total curvature = curv * element * (grid_vol / N^2)
    # = (curv * element) / N^2 * grid_vol = mean(curv * element) * grid_vol.
    return (curvature * element).mean() * grid_volume
