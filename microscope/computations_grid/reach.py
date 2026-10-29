from typing import Optional, Union

import numpy as np
import torch
from einops import einops
from tqdm import tqdm

from microscope.computations_grid.basic import crop_dim_borders, partial_derivatives_across_all_dims_batched
from microscope.computations_grid._curvature_geometry import tangent_geometry
from microscope._grid_validation import validate_cyclic_axes, validate_integer_list, validate_spacings
from microscope.cyclic_dimensions import get_difference_intervals


def _chord_geometry(points: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stable chord lengths and unit chords for pair-search methods.

    Args:
        points: Candidate points, shape (c f).
        x: Query points, shape (b f).

    Returns:
        (lengths, unit_chords), tensors of shapes (b c) and (b c f). Coincident pairs have zero length and zero unit
        chord.
    """
    differences = points[None] - x[:, None]
    scale = differences.abs().amax(dim=-1)
    differences = differences / torch.where(scale > 0, scale, 1.)[..., None]
    length = torch.linalg.vector_norm(differences, dim=-1)
    unit = differences / torch.where(length > 0, length, 1.)[..., None]
    return scale*length, unit


def _pair_geometry(
    frame: torch.Tensor,
    points: torch.Tensor,
    x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return chord lengths and normal fractions without an ambient projector.

    Normalize chords before projection, avoiding squared-distance overflow and norm underflow. Residuals at the QR
    projection's roundoff scale are zero. Nonfinite intermediate values are checked by the caller, never filled in.

    Args:
        frame: Orthonormal query tangent frames, shape (b f d).
        points: Candidate points, shape (c f).
        x: Query points, shape (b f).

    Returns:
        (distances, normal_fractions), tensors of shape (b c) containing chord lengths and the fractions orthogonal to
        each query tangent space.
    """
    distance, unit = _chord_geometry(points, x)
    normal = unit - (unit @ frame) @ frame.transpose(-1, -2)
    fraction = torch.linalg.vector_norm(normal, dim=-1)
    tolerance = 8*torch.finfo(frame.dtype).eps*frame.shape[-1]
    fraction = torch.where(fraction <= tolerance, 0., fraction)
    return distance, fraction


def _distances_to_tangent_space(
    tangent_vectors: torch.Tensor,
    points: torch.Tensor,
    x: torch.Tensor
) -> torch.Tensor:
    """Compute candidate distances to each query tangent space.

    Args:
        tangent_vectors: Query tangent matrices, shape (b f d).
        points: Candidate points, shape (c f).
        x: Query points, shape (b f).

    Returns:
        Normal-distance tensor of shape (b c), one value per query/candidate pair.
    """
    _, frame, _ = tangent_geometry(tangent_vectors)
    distance, normal_fraction = _pair_geometry(frame, points, x)
    return distance*normal_fraction


def _subsample_dims(
    tensor: torch.Tensor,
    take_every_k: int,
    dim_idxs: set[int]
) -> torch.Tensor:
    """Select every kth value along the chosen tensor axes.

    Args:
        tensor: Input tensor, shape (s1 ... sn).
        take_every_k: Sample stride along the selected axes.
        dim_idxs: Zero-based axes to subsample.

    Returns:
        Tensor of shape (t1 ... tn), where ti = ceil(si / take_every_k) on selected axes and si on all other axes.
    """
    for dim in dim_idxs:
        tensor = torch.index_select(
            tensor,
            dim=dim,
            index=torch.arange(
                0,
                tensor.shape[dim],
                step=take_every_k,
                device=tensor.device
            )
        )

    return tensor


def _validate_reach_grid(
    features_on_grid: np.ndarray,
    range_sizes: list[float],
    patch_sizes: list[int],
    cyclic_dimensions: list[int]
) -> None:
    """Validate grid geometry.

    Require a NumPy float32/float64 array with ambient dimension f >= d >= 1. Provide one integer patch size >= 3 and
    one finite nonzero real range per grid axis. Periodic axes must be unique valid indices.

    Args:
        features_on_grid: Embedding array, shape (s1 ... sd f).
        range_sizes: Parameter-range length along each of the d grid axes.
        patch_sizes: Patch size along each of the d grid axes.
        cyclic_dimensions: Zero-based periodic grid axes.

    Returns:
        None.
    """
    if features_on_grid.dtype not in (np.float32, np.float64):
        raise TypeError("Reach requires float32 or float64 inputs.")
    n_dims = features_on_grid.ndim - 1
    if n_dims < 1 or features_on_grid.shape[-1] < n_dims:
        raise ValueError("A grid needs ambient dimension >= intrinsic dimension >= 1.")
    validate_integer_list(patch_sizes, "patch sizes", minimum=3)
    validate_spacings(range_sizes, n_dims)
    if len(patch_sizes) != len(range_sizes):
        raise ValueError(f"Patch ({len(patch_sizes)}) and range ({len(range_sizes)}) should have same length.")
    validate_cyclic_axes(cyclic_dimensions, n_dims)


def _validate_reach_options(
    batch_size: int,
    candidate_batch_size: int,
    subsample_points: int | None,
    min_pair_distance: float | None
) -> None:
    """Validate reach search controls.

    Batch sizes and an explicit subsampling stride must be positive integers. An explicit pair cutoff must be a finite
    nonnegative scalar.

    Args:
        batch_size: Number of query points processed together.
        candidate_batch_size: Number of candidate points processed together.
        subsample_points: Optional stride along each grid axis; None uses every point.
        min_pair_distance: Pair-separation cutoff; None diagnoses unresolved near pairs.

    Returns:
        None.
    """
    validate_integer_list([batch_size], "batch size", minimum=1)
    validate_integer_list([candidate_batch_size], "candidate batch size", minimum=1)
    if subsample_points is not None:
        validate_integer_list([subsample_points], "subsample stride", minimum=1)
    if min_pair_distance is not None and (not np.isfinite(min_pair_distance) or min_pair_distance < 0):
        raise ValueError("Minimum pair distance must be a finite nonnegative real scalar.")


def _validate_reach_samples(features_on_grid: np.ndarray) -> None:
    """Validate reach sample extents and finiteness.

    Each grid axis needs at least three samples and all coordinates must be finite. Pair-dependent separation and
    numerical checks stay in the caller.

    Args:
        features_on_grid: Embedding array, shape (s1 ... sd f).

    Returns:
        None.
    """
    dims_shape = features_on_grid.shape[:-1]
    if any(n < 3 for n in dims_shape):
        raise ValueError("Each grid axis needs at least three samples.")
    if not np.isfinite(features_on_grid).all():
        raise ValueError("Reach inputs must contain only finite values.")


def reach_per_point(
    features_on_grid: np.ndarray,
    range_sizes: list[float],
    patch_sizes: list[int],
    cyclic_dimensions: Optional[list[int]] = None,
    subsample_points: Optional[int] = None,
    batch_size: int = 10,
    return_witnesses: bool = False,
    device: str | torch.device | None = None,
    candidate_batch_size: int = 1024,
    min_pair_distance: Optional[float] = None
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Estimate local reach from sampled pairs and finite-difference tangents.

    Uses min_{y != x} ||y - x||^2 / (2 d(y - x, T_x M)); see https://arxiv.org/abs/1705.04565. The minimum over x
    estimates global reach.

    Requires a regular floating grid and valid search controls; see the local _validate_reach_grid,
    _validate_reach_options and _validate_reach_samples helpers for detailed input requirements.

    Args:
        features_on_grid: Floating array of shape (s1, ..., sk, features).
        range_sizes: Full parameter-range length along each grid axis.
        patch_sizes: Derivative patch size along each grid axis.
        cyclic_dimensions: Zero-based periodic grid-axis indices; None means none.
        subsample_points: Optional query/candidate grid stride; tangents use the full grid.
        batch_size: Number of query points per pair-search batch.
        return_witnesses: Also return minimizing candidate indices.
        device: Torch device; None selects CUDA when available, otherwise CPU.
        candidate_batch_size: Maximum candidates per pair-search chunk. Only query/candidate chunks move to the selected
            device.
        min_pair_distance: Separation cutoff; None diagnoses unresolved near pairs.

    Returns:
        A NumPy array of local estimates of shape (s1_out, ..., sk_out), where s_i_out = s_i for periodic axes and s_i -
        2 otherwise. Subsampled estimates are repeated and truncated back to this shape.

        If return_witnesses is true, return (estimates, witnesses). The integer witnesses array has shape (N,), with N =
        prod(ceil(s_i_out / stride)) and stride = subsample_points or 1. Each entry is a flat index into the cropped,
        subsampled point array. Witnesses are not repeated or reshaped. Infinite estimates keep a distinct eligible
        witness.

    Raises:
        ValueError: For invalid arguments/nonfinite inputs, no distinct eligible candidate, or unresolved
            near-coincident pairs with the default distance policy. Exact duplicate coordinates are always excluded.
        torch.linalg.LinAlgError: For numerical tangent rank loss, with the
            original input-grid location of the query.
        FloatingPointError: For nonfinite geometry or finite-ratio overflow; rescale the embedding or use float64.

    The formula estimates an underlying smooth embedded manifold without boundary. It does not compute boundary
    normal-cone contributions for a bounded parameter image (for example, the inner boundary of an annulus).
    Finite-difference tangents and input roundoff can cause underestimation; the result is not an unconditional upper
    bound on true reach. Normal fractions <= 8 * dtype epsilon * intrinsic dimension are treated as zero, giving
    infinity for numerically flat chords. Generic parameterizations need a resolution/convergence check; exact duplicate
    removal alone does not resolve near-duplicate instability. Candidate search is exhaustive over eligible sampled
    pairs; no nearest-neighbor approximation is used.

    With the default cutoff, positive sampled separations <= 8 * dtype epsilon * norm(max(abs(sampled coordinates),
    axis=0)) raise. Exact duplicates are excluded. A positive cutoff can omit a real bottleneck.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if cyclic_dimensions is None:
        cyclic_dimensions = []
    _validate_reach_grid(
        features_on_grid,
        range_sizes,
        patch_sizes,
        cyclic_dimensions
    )
    n_dims = features_on_grid.ndim - 1
    _validate_reach_options(
        batch_size,
        candidate_batch_size,
        subsample_points,
        min_pair_distance
    )
    dims_shape = features_on_grid.shape[:-1]
    _validate_reach_samples(features_on_grid)
    difference_intervals = get_difference_intervals(list(dims_shape), range_sizes, cyclic_dimensions)
    if any(stride < 0 for stride in features_on_grid.strides):
        features_on_grid = np.ascontiguousarray(features_on_grid)
    dim_idxs = set(range(n_dims))

    features_on_grid_pt = torch.from_numpy(features_on_grid)
    features_on_grid_cropped = crop_dim_borders(features_on_grid_pt, dim_idxs.difference(cyclic_dimensions))
    cropped_grid_shape = features_on_grid_cropped.shape[:-1]

    if subsample_points is not None:
        features_on_grid_cropped = _subsample_dims(features_on_grid_cropped, subsample_points, dim_idxs)

    features_on_grid_cropped_flat = features_on_grid_cropped.reshape(-1, features_on_grid_cropped.shape[-1])
    count = len(features_on_grid_cropped_flat)
    if count < 2:
        raise ValueError("Reach needs at least two distinct sampled candidate coordinates.")
    sampled_shape = features_on_grid_cropped.shape[:-1]
    sampled_points = features_on_grid_cropped_flat.numpy()
    max_coordinates = np.maximum(np.abs(sampled_points.min(axis=0)),
                                 np.abs(sampled_points.max(axis=0)))
    coordinate_scale = np.hypot.reduce(max_coordinates.astype(np.float64))
    if not np.isfinite(coordinate_scale):
        raise FloatingPointError("Coordinate norm overflow; rescale the embedding.")
    automatic_cutoff = 8*np.finfo(features_on_grid.dtype).eps*coordinate_scale
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

    reach_estimates = np.empty(count, dtype=features_on_grid.dtype)
    estimate_witnesses = np.empty(count, dtype=np.int64)
    for s in tqdm(range(0, count, batch_size), desc=f'Reach ({device})', unit='batch'):
        e = min(s + batch_size, count)
        x = features_on_grid_cropped_flat[s:e].to(device)
        T_x = tangent_vectors_flat[s:e].to(device)
        grid_indices = np.column_stack(np.unravel_index(np.arange(s, e), sampled_shape))
        grid_indices *= subsample_points or 1
        grid_indices += [0 if a in cyclic_dimensions else 1 for a in range(n_dims)]
        with torch.no_grad():
            _, frame, _ = tangent_geometry(T_x, grid_indices=torch.from_numpy(grid_indices))
            best = torch.full(
                (e-s,),
                torch.inf,
                dtype=x.dtype,
                device=device
            )
            witnesses = torch.full(
                (e-s,),
                -1,
                dtype=torch.long,
                device=device
            )
            geometry_ok = torch.ones((), dtype=torch.bool, device=device)
            near_separation = torch.full_like(best, torch.inf)
            for c in range(0, count, candidate_batch_size):
                points = features_on_grid_cropped_flat[c:c+candidate_batch_size].to(device)
                distance, fraction = _pair_geometry(frame, points, x)
                geometry_ok &= torch.isfinite(distance).all() & torch.isfinite(fraction).all()
                if min_pair_distance is None:
                    near = (distance > 0) & (distance <= automatic_cutoff)
                    near_separation = torch.minimum(near_separation,
                        torch.where(near, distance, torch.inf).min(dim=-1).values)
                eligible = distance > (min_pair_distance or 0.)
                ratios = distance / (2*fraction)
                geometry_ok &= (~eligible | (fraction == 0) | torch.isfinite(ratios)).all()
                ratios = torch.where(eligible, ratios, torch.inf)
                values, indices = ratios.min(dim=-1)
                has_candidate = eligible.any(dim=-1)
                # All-flat chunks have only infinite ratios; choose an eligible
                # witness explicitly so argmin cannot accidentally select self.
                indices = torch.where(torch.isinf(values), eligible.long().argmax(dim=-1), indices)
                improve = has_candidate & ((witnesses < 0) | (values < best))
                best = torch.where(improve, values, best)
                witnesses = torch.where(improve, c+indices, witnesses)
            if not geometry_ok:
                raise FloatingPointError(
                    "Reach geometry overflow or nonfinite ratios; rescale the embedding or use float64."
                )
            unresolved = torch.isfinite(near_separation)
            if unresolved.any():
                index = int(torch.nonzero(unresolved, as_tuple=False)[0, 0])
                raise ValueError(
                    f"Unresolved near-coincident pair at grid index {tuple(grid_indices[index])}: "
                    f"separation {near_separation[index].item():.3g} <= {automatic_cutoff:.3g}. "
                    "Set min_pair_distance explicitly: a positive cutoff filters pairs; "
                    "zero includes all distinct coordinates."
                )
            if (witnesses < 0).any():
                index = int(torch.nonzero(witnesses < 0, as_tuple=False)[0, 0])
                raise ValueError(f"No distinct eligible candidate at grid index {tuple(grid_indices[index])}.")
            reach_estimates[s:e] = best.cpu().numpy()
            estimate_witnesses[s:e] = witnesses.cpu().numpy()

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
