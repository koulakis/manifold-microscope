"""Experimental reach estimate from normal curvature and bottleneck chords."""
from typing import NamedTuple

import numpy as np
import torch

from microscope._grid_validation import cyclic_axes, integer_list, spacings
from microscope.computations_grid.curvature_extrinsic import _normal_geometry
from microscope.computations_grid.reach import _chord_geometry
from microscope.cyclic_dimensions import get_difference_intervals
from microscope.patches import _iter_patches_with_halo


class ReachComponents(NamedTuple):
    """Global estimate, its two components and flat sampled bottleneck indices."""
    reach: float
    curvature_radius: float
    bottleneck_radius: float
    bottleneck_pair: tuple[int, int] | None


def _max_normal_curvature(form, n_directions=32, iterations=4):
    """Approximate max_{|v|=1} |B(v,v)| from an orthonormal normal-valued form.

    Axis/pair directions plus fixed-seed random starts are refined by alternating
    the normal direction B(v,v) with an extreme eigenvector of its shape operator.
    Each exact-arithmetic refinement is nondecreasing. The remaining multi-start
    optimization is not a certified global maximum in higher codimension.
    Computation after the Gram matrix uses only intrinsic-size tensors.
    """
    dims = form.shape[-1]
    form = .5*(form + form.transpose(-1, -2))
    scale = form.abs().amax(dim=(-3, -2, -1))
    scaled = form / torch.where(scale > 0, scale, 1.)[..., None, None, None]
    flattened = scaled.flatten(-2)
    gram = flattened.transpose(-1, -2) @ flattened
    if not torch.isfinite(gram).all():
        raise FloatingPointError("Nonfinite second fundamental form; rescale the embedding.")
    gram = gram.reshape(-1, dims, dims, dims, dims)
    identity = np.eye(dims)
    starts = [*identity]
    for i in range(dims):
        for j in range(i+1, dims):
            starts.extend([(identity[i]+identity[j])/np.sqrt(2),
                           (identity[i]-identity[j])/np.sqrt(2)])
    if n_directions:
        random = np.random.default_rng(0).standard_normal((n_directions, dims))
        starts.extend(random/np.linalg.norm(random, axis=-1, keepdims=True))
    vectors = torch.as_tensor(np.asarray(starts), dtype=form.dtype, device=form.device)
    vectors = vectors.expand(gram.shape[0], -1, -1)
    best = torch.zeros(gram.shape[0], dtype=form.dtype, device=form.device)
    for step in range(iterations+1):
        operators = torch.einsum("nijkl,nsk,nsl->nsij", gram, vectors, vectors)
        values = torch.einsum("nsi,nsij,nsj->ns", vectors, operators, vectors)
        best = torch.maximum(best, values.amax(dim=-1))
        if step < iterations:
            eigenvalues, eigenvectors = torch.linalg.eigh(.5*(operators+operators.transpose(-1, -2)))
            indices = eigenvalues.abs().argmax(dim=-1)
            vectors = eigenvectors.gather(-1, indices[..., None, None].expand(*indices.shape, dims, 1)).squeeze(-1)
    result = scale.reshape(-1)*best.clamp_min(0).sqrt()
    if not torch.isfinite(result).all():
        raise FloatingPointError("Normal curvature overflow; rescale the embedding or use float64.")
    return result.reshape(scale.shape)


def _bottleneck_radius(points, frames, batch_size, candidate_batch_size,
                       tolerance, min_pair_distance, device, *, return_per_point=False):
    """Search all sampled pairs for chords approximately normal at both ends."""
    coordinates = points.numpy()
    maxima = np.maximum(np.abs(coordinates.min(axis=0)), np.abs(coordinates.max(axis=0)))
    automatic_cutoff = 8*np.finfo(coordinates.dtype).eps*np.hypot.reduce(maxima.astype(np.float64))
    if not np.isfinite(automatic_cutoff):
        raise FloatingPointError("Coordinate scale overflow; rescale the embedding.")
    estimates = np.full(len(points), np.inf, dtype=coordinates.dtype)
    witnesses = np.full(len(points), -1, dtype=np.int64)
    has_distinct = torch.zeros((), dtype=torch.bool, device=device)
    for start in range(0, len(points), batch_size):
        x = points[start:start+batch_size].to(device)
        qx = frames[start:start+batch_size].to(device)
        best = torch.full((len(x),), torch.inf, dtype=points.dtype, device=device)
        partners = torch.full((len(x),), -1, dtype=torch.long, device=device)
        valid_geometry = torch.ones((), dtype=torch.bool, device=device)
        unresolved = torch.zeros((), dtype=torch.bool, device=device)
        for candidate in range(0, len(points), candidate_batch_size):
            y = points[candidate:candidate+candidate_batch_size].to(device)
            qy = frames[candidate:candidate+candidate_batch_size].to(device)
            distance, unit = _chord_geometry(y, x)
            tx2 = (unit @ qx).square().sum(dim=-1)
            ty2 = torch.einsum("bca,cad->bcd", unit, qy).square().sum(dim=-1)
            valid_geometry &= (torch.isfinite(distance).all() & torch.isfinite(tx2).all()
                               & torch.isfinite(ty2).all())
            if min_pair_distance is None:
                unresolved |= ((distance > 0) & (distance <= automatic_cutoff)).any()
            separated = distance > (min_pair_distance or 0.)
            has_distinct |= separated.any()
            eligible = (separated
                        & (tx2 <= tolerance*tolerance) & (ty2 <= tolerance*tolerance))
            # At a true double normal this is half the chord. The symmetric
            # normal-distance correction removes the fixed angular-tolerance
            # bias for circles and cannot undershoot either endpoint ratio
            # when tangent frames are exact.
            normal = (1-torch.maximum(tx2, ty2)).clamp_min(0).sqrt()
            ratios = distance/(2*normal)
            valid_geometry &= (~eligible | torch.isfinite(ratios)).all()
            ratios = torch.where(eligible, ratios, torch.inf)
            values, indices = ratios.min(dim=-1)
            improve = values < best
            best = torch.where(improve, values, best)
            partners = torch.where(improve, candidate+indices, partners)
        if not valid_geometry:
            raise FloatingPointError("Bottleneck geometry overflow; rescale the embedding or use float64.")
        if unresolved:
            raise ValueError("Unresolved near-coincident sampled separation; set min_pair_distance explicitly.")
        estimates[start:start+len(x)] = best.cpu().numpy()
        witnesses[start:start+len(x)] = partners.cpu().numpy()
    if not has_distinct:
        raise ValueError("No distinct eligible sampled coordinates remain.")
    if return_per_point:
        return estimates, witnesses
    query = int(estimates.argmin())
    pair = (query, int(witnesses[query])) if witnesses[query] >= 0 else None
    return float(estimates[query]), pair


def _repeat_to_grid(values, shape, stride):
    """Repeat sampled spatial axes; leave any trailing component axis intact."""
    if stride > 1:
        for axis in range(len(shape)):
            values = np.repeat(values, stride, axis=axis)
    return values[tuple(slice(0, n) for n in shape)]


def reach_per_point(
    features_on_grid: np.ndarray,
    range_sizes: list[float],
    patch_sizes: list[int],
    cyclic_dimensions: list[int] | None = None,
    subsample_points: int | None = None,
    batch_size: int = 10,
    return_witnesses: bool = False,
    device: str | torch.device | None = None,
    *,
    candidate_batch_size: int = 1024,
    bottleneck_tolerance: float = .1,
    n_directions: int = 32,
    curvature_iterations: int = 4,
    min_pair_distance: float | None = None,
    return_components: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate pointwise reach from curvature radii and bottleneck chords.

    This implements a practical curvature/bottleneck decomposition inspired by
    Aamari et al., arXiv:1705.04565, Theorem 3.4, and arXiv:2207.06074,
    Theorem 3.5. It is not the latter paper's geodesic/SDR statistical estimator.
    Smooth embedded compact manifolds without boundary are the intended setting.

    Inputs and periodic coordinates follow ``reach.reach_per_point``. Patch
    sizes must be >=5. Second derivatives exclude two samples at each
    nonperiodic end, leaving shape (n_i-4), while periodic axes keep n_i.
    ``subsample_points`` selects every kth center on this grid; derivatives
    still use the original adjacent samples. Sampled values are repeated and
    truncated back to the cropped center-grid shape, as in the original API.
    At least two sampled centers with a distinct eligible pair are required.
    No full ambient projector or
    all-pairs array is stored.

    Curvature is max_v ||II(v,v)|| for metric-unit tangent vectors. The shared
    second fundamental form and QR factors supply orthonormal coordinates.
    ``n_directions`` additional fixed-seed directions supplement axis/pair
    starts; ``curvature_iterations`` spectral refinements improve each start.
    Curves are exact for the estimated form; hypersurfaces are exact after
    one refinement. Higher codimension needs direction/iteration convergence.
    Scalar curvature and the flattened/Frobenius form norm are not substitutes.

    Bottlenecks require unit-chord tangential projections at BOTH endpoints
    <= ``bottleneck_tolerance`` (a sine in [0,1)). Candidates use the larger
    endpoint tangent-ratio value, equal to half chord length for exact double
    normals. The finite angular filter can miss bottlenecks; increase sampling
    and vary tolerance. No accepted partner gives infinite bottleneck radius
    and witness -1, which is not proof of absence. Search remains quadratic.

    ``min_pair_distance`` follows the existing reach policy: None diagnoses
    roundoff-scale positive separations, zero admits all distinct coordinates,
    a positive cutoff explicitly filters pairs and can omit true bottlenecks.
    Exact duplicates are excluded. Flat curvature gives infinite radius.
    Nonfinite input/geometry and rank loss raise, without regularization.

    Returns a NumPy grid of the pointwise minimum of both radii, with the input
    dtype. Its minimum estimates global reach. ``return_witnesses=True`` returns
    ``(reach_grid, witnesses)``. With ``return_components=True`` always returns
    ``(reach_grid, witnesses, components)``; this flag implies witness output.
    ``components.shape == reach_grid.shape + (2,)``: curvature radius first,
    bottleneck radius second, both in distance units. These are pointwise
    decomposition diagnostics, not generally the original tangent-ratio field.

    Witnesses are int64, flat and NOT repeated, one per sampled center. Each
    indexes the best bottleneck partner in the flattened cropped/subsampled
    grid, even when curvature determines the combined estimate. No accepted
    partner gives bottleneck infinity and witness -1. Flat curvature gives
    infinite curvature radius. Finite positive radii that overflow the input
    dtype raise rather than silently becoming infinity.

    The angular test is symmetric in exact arithmetic: a pair failing at one
    endpoint is rejected in both directions. Staggered samples or tangent error
    can reject a resolved bottleneck. Near the threshold floating-point effects
    may differ between directions. Density alone does not guarantee detection;
    vary tolerance as well as grid resolution and subsampling.
    Finite differences, directional optimization, sampling and angular filtering
    introduce separate errors; neither component is a certified true-reach bound.
    """
    if not isinstance(features_on_grid, np.ndarray) or features_on_grid.dtype not in (np.float32, np.float64):
        raise TypeError("Two-step reach requires a NumPy float32 or float64 array.")
    dims = features_on_grid.ndim-1
    if dims < 1 or features_on_grid.shape[-1] < dims:
        raise ValueError("A grid needs ambient dimension >= intrinsic dimension >=1.")
    range_sizes = spacings(range_sizes, dims)
    cyclic = cyclic_axes(cyclic_dimensions if cyclic_dimensions is not None else [], dims)
    patches = integer_list(patch_sizes, "patch sizes", minimum=5)
    if len(patches) != dims:
        raise ValueError("One patch size is required per grid axis.")
    if any(n < (3 if a in cyclic else 5) for a, n in enumerate(features_on_grid.shape[:-1])):
        raise ValueError("Grid axes need five noncyclic or three cyclic samples.")
    stride = integer_list([subsample_points if subsample_points is not None else 1], "subsample stride", minimum=1)[0]
    batch_size = integer_list([batch_size], "batch size", minimum=1)[0]
    candidate_batch_size = integer_list([candidate_batch_size], "candidate batch size", minimum=1)[0]
    n_directions = integer_list([n_directions], "direction count")[0]
    curvature_iterations = integer_list([curvature_iterations], "curvature iterations")[0]
    tolerance = np.asarray(bottleneck_tolerance)
    if (tolerance.ndim or tolerance.dtype.kind not in "iuf" or not np.isfinite(tolerance)
            or tolerance < 0 or tolerance >= 1):
        raise ValueError("Bottleneck tolerance must be a finite sine in [0,1).")
    if min_pair_distance is not None:
        cutoff = np.asarray(min_pair_distance)
        if cutoff.ndim or cutoff.dtype.kind not in "iuf" or not np.isfinite(cutoff) or cutoff < 0:
            raise ValueError("Minimum pair distance must be finite and nonnegative.")
        min_pair_distance = float(cutoff)
    if not np.isfinite(features_on_grid).all():
        raise ValueError("Two-step reach inputs must be finite.")
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    intervals = get_difference_intervals(list(features_on_grid.shape[:-1]), range_sizes, cyclic)
    shape = tuple(n if a in cyclic else n-4 for a, n in enumerate(features_on_grid.shape[:-1]))
    sampled_shape = tuple((n+stride-1)//stride for n in shape)
    if np.prod(sampled_shape) < 2:
        raise ValueError("Two-step reach needs at least two sampled centers.")
    selection = tuple(slice(None, None, stride) if a in cyclic else slice(2, -2, stride) for a in range(dims))
    points = np.ascontiguousarray(features_on_grid[selection]).reshape(-1, features_on_grid.shape[-1])
    frames = np.empty((*sampled_shape, features_on_grid.shape[-1], dims), dtype=features_on_grid.dtype)
    curvatures = np.empty(sampled_shape, dtype=features_on_grid.dtype)
    with torch.no_grad():
        for destination, patch in _iter_patches_with_halo(features_on_grid, patches, 2, cyclic):
            offsets = [(-s.start) % stride for s in destination]
            if any(first >= s.stop-s.start for first, s in zip(offsets, destination)):
                continue
            local = tuple(slice(first, None, stride) for first in offsets)
            target = tuple(slice((s.start+first)//stride, (s.stop+stride-1)//stride)
                           for first, s in zip(offsets, destination))
            tensor = torch.from_numpy(np.ascontiguousarray(patch)).to(device)
            form, _, inverse, frame = _normal_geometry(tensor, intervals, return_frame=True)
            form, inverse, frame = form[local], inverse[local], frame[local]
            orthonormal = torch.einsum("...pi,...apq,...qj->...aij", inverse, form, inverse)
            flat = orthonormal.reshape(-1, *orthonormal.shape[-3:])
            values = torch.cat([_max_normal_curvature(flat[s:s+batch_size], n_directions, curvature_iterations)
                                for s in range(0, len(flat), batch_size)])
            curvatures[target] = values.reshape(frame.shape[:-2]).cpu().numpy()
            frames[target] = frame.cpu().numpy()
        bottleneck, witnesses = _bottleneck_radius(torch.from_numpy(points), torch.from_numpy(frames.reshape(-1, *frames.shape[-2:])),
            batch_size, candidate_batch_size, float(tolerance), min_pair_distance, device, return_per_point=True)
    radii = np.full_like(curvatures, np.inf)
    with np.errstate(over="ignore", divide="ignore"):
        np.divide(1., curvatures, out=radii, where=curvatures > 0)
    if not np.isfinite(radii[curvatures > 0]).all():
        raise FloatingPointError("Curvature radius overflow; rescale the embedding or use float64.")
    bottleneck = bottleneck.reshape(sampled_shape)
    estimates = _repeat_to_grid(np.minimum(radii, bottleneck), shape, stride)
    if return_components:
        components = _repeat_to_grid(np.stack((radii, bottleneck), axis=-1), shape, stride)
        return estimates, witnesses, components
    if return_witnesses:
        return estimates, witnesses
    return estimates


def reach_curvature_bottleneck(
    features_on_grid: np.ndarray,
    range_sizes: list[float],
    patch_sizes: list[int],
    cyclic_dimensions: list[int] | None = None,
    subsample_points: int | None = None,
    batch_size: int = 10,
    device: str | torch.device | None = None,
    *,
    candidate_batch_size: int = 1024,
    bottleneck_tolerance: float = .1,
    n_directions: int = 32,
    curvature_iterations: int = 4,
    min_pair_distance: float | None = None,
    return_components: bool = False,
) -> float | ReachComponents:
    """Compatibility wrapper returning global minima of ``reach_per_point``.

    ``return_components=True`` retains the original scalar ReachComponents
    result; the new ``reach_per_point`` returns arrays instead. Both use the
    same geometry and filtering. Prefer the pointwise API for analysis.
    """
    estimates, witnesses, components = reach_per_point(
        features_on_grid, range_sizes, patch_sizes, cyclic_dimensions,
        subsample_points, batch_size, device=device, return_components=True,
        candidate_batch_size=candidate_batch_size, bottleneck_tolerance=bottleneck_tolerance,
        n_directions=n_directions, curvature_iterations=curvature_iterations,
        min_pair_distance=min_pair_distance,
    )
    stride = subsample_points or 1
    sampled = components[tuple(slice(None, None, stride) for _ in estimates.shape)]
    query = int(sampled[..., 1].argmin())
    pair = (query, int(witnesses[query])) if witnesses[query] >= 0 else None
    result = ReachComponents(float(estimates.min()), float(components[..., 0].min()),
                             float(components[..., 1].min()), pair)
    return result if return_components else result.reach
