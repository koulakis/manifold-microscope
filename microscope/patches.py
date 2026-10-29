from collections.abc import Iterator
from typing import Optional

import numpy as np
from einops import einops

from microscope.cyclic_dimensions import pad_cyclic_dimensions
from microscope._grid_validation import validate_cyclic_axes, validate_integer_list


def _iter_patches_with_halo(
    array: np.ndarray,
    patch_sizes: list[int],
    halo: int,
    cyclic_dimensions: list[int]
) -> Iterator[tuple[tuple[slice, ...], np.ndarray]]:
    """Yield (output slices, input tile) without padding the whole array.

    Leading grid axes use the given patch sizes. Each output tile is disjoint; its input includes ``halo`` samples on
    each side. Nonperiodic outputs lose that border, while periodic outputs retain every sample. Interior tiles are
    views; only tiles crossing periodic boundaries need wrapped copies. Trailing feature axes are untouched. Callers
    validate sizes and axes.

    Args:
        array: Input array, shape (s1 ... sd f1 ... fk).
        patch_sizes: Patch size along each of the d grid axes.
        halo: Number of extra samples on each side of an output tile.
        cyclic_dimensions: Zero-based periodic grid axes.

    Yields:
        (destination, patch): output-grid slices and an array of shape (p1 ... pd f1 ... fk). Each pi includes both halo
        margins and is at most patch_sizes[i]; trailing feature axes match array.
    """
    dims = len(patch_sizes)
    shape = array.shape[:dims]
    output_shape = [n if a in cyclic_dimensions else n-2*halo
                    for a, n in enumerate(shape)]
    steps = [p-2*halo for p in patch_sizes]
    counts = [(n+step-1)//step for n, step in zip(output_shape, steps)]
    for index in np.ndindex(*counts):
        starts = [i*step for i, step in zip(index, steps)]
        lengths = [min(step, n-start) for step, n, start
                   in zip(steps, output_shape, starts)]
        source, wrapped = [], []
        for axis, (start, length, n) in enumerate(zip(starts, lengths, shape)):
            first = start-halo if axis in cyclic_dimensions else start
            last = first+length+2*halo
            if first >= 0 and last <= n:
                source.append(slice(first, last))
            else:
                source.append(slice(None))
                wrapped.append((axis, np.arange(first, last) % n))
        if wrapped:
            # Gather all axes together: sequential np.take calls would retain
            # entire other wrapped axes in intermediate copies at corners.
            coordinates = dict(wrapped)
            indices = [coordinates[a] if a in coordinates else np.arange(s.start, s.stop)
                       for a, s in enumerate(source)]
            patch = array[np.ix_(*indices)]
        else:
            patch = array[tuple(source)]
        destination = tuple(slice(start, start+length)
                            for start, length in zip(starts, lengths))
        yield destination, patch


def _validate_patch_inputs(
    array: np.ndarray,
    patch_sizes: list[int],
    overlaps: list[int],
    cyclic_dimensions: list[int]
) -> None:
    """Validate patch geometry.

    Patch sizes must cover between one and all leading array axes. Sizes and overlaps must be integer sequences of equal
    length, with sizes > overlaps >= 0. Periodic axes must be unique valid indices. Every grid axis must be nonempty,
    and nonperiodic axes must exceed their overlap.

    Args:
        array: Input array, shape (s1 ... sd f1 ... fk).
        patch_sizes: Patch size along each of the d grid axes.
        overlaps: Overlap along each patched grid axis.
        cyclic_dimensions: Zero-based periodic grid axes.

    Returns:
        None.
    """
    validate_integer_list(patch_sizes, "patch sizes", minimum=1)
    validate_integer_list(overlaps, "patch overlaps")
    n_dim = len(patch_sizes)
    if not 1 <= n_dim <= array.ndim:
        raise ValueError("patch sizes must cover between one and all array axes.")
    validate_cyclic_axes(cyclic_dimensions, n_dim)
    if len(overlaps) != n_dim:
        raise ValueError(
            f"The shape dimensions of the patch sizes ({n_dim}) differ from the "
            f"shape dimensions of the overlaps ({len(overlaps)})."
        )
    if any(p <= overlap for p, overlap in zip(patch_sizes, overlaps)):
        raise ValueError("Each patch size must exceed its overlap.")
    if any(n == 0 or (axis not in cyclic_dimensions and n <= overlaps[axis])
           for axis, n in enumerate(array.shape[:n_dim])):
        raise ValueError("Grid axes must have a nonempty patch interior.")


def extract_patches(
    array: np.ndarray,
    patch_sizes: list[int],
    overlaps: list[int],
    cyclic_dimensions: Optional[list[int]] = None,
    verbose: bool = False
) -> np.ndarray:
    """Generalized function to extract overlapping patches from the first n - d dimensions of a multidimensional array,
    leaving the last d dimensions untouched. If size of the array is not divisible by the resulting step size along all
    spatial dimensions, it will be wrap-padded to a compatible size. This means that the caller is responsible to
    truncate outputs on a later stage to compensate for those artificial padded values.

    Requires compatible patch geometry; see _validate_patch_inputs.

    Args:
        array: Input array, shape (s1 ... sd f1 ... fk).
        patch_sizes: The size of each patch along the first n - d dimensions.
        overlaps: The overlap along each of the first n - d dimensions.
        cyclic_dimensions: A set of dimensions where the grid is cyclic.
        verbose: If true it prints info and warning messages.

    Returns:
        Patch-grid view of shape (g1 ... gd p1 ... pd f1 ... fk), where pi = patch_sizes[i] and gi counts patches along
        grid axis i. Feature axes are unchanged; padding may allocate a backing array.
    """
    if cyclic_dimensions is None:
        cyclic_dimensions = []
    _validate_patch_inputs(
        array,
        patch_sizes,
        overlaps,
        cyclic_dimensions
    )
    n_dim = len(patch_sizes)
    n_features = len(array.shape) - n_dim

    # Handle cyclic dimensions by padding the array
    array = pad_cyclic_dimensions(
        array,
        cyclic_dimensions=cyclic_dimensions,
        pad_sizes=[overlaps[dim] // 2 for dim in cyclic_dimensions]
    )

    patch_sizes_array = np.array(patch_sizes)
    overlaps_array = np.array(overlaps)
    step_sizes = patch_sizes_array - overlaps_array

    array_shape = np.array(array.shape[:n_dim])
    feature_shape = array.shape[n_dim:]

    residue = array_shape - overlaps_array - (step_sizes * ((array_shape - overlaps_array) // step_sizes))
    # noinspection PyUnresolvedReferences
    if (residue != 0).any():
        array_old_shape = array_shape
        # noinspection PyTypeChecker
        array = np.pad(
            array,
            pad_width=
            tuple((0, r) for r in ((step_sizes - residue) % step_sizes))
            + tuple(((0, 0) for _ in range(n_features))),
            mode="wrap"
        )
        array_shape = np.array(array.shape[:n_dim])
        feature_shape = array.shape[n_dim:]

        if verbose:
            print(
                f"WARNING: Array shape minus patch size {array_shape - patch_sizes_array} "
                f"is not divisible by the step size "
                f"{step_sizes} (patch_size - overlap) along all dimensions. Zero padding the array of size "
                f"({array_old_shape}) to ({array_shape}) to compensate for that."
            )

    # Compute new shape and strides for patches
    patch_shape = tuple((array_shape - overlaps_array) // step_sizes) + tuple(patch_sizes_array) + feature_shape
    strides = array.strides
    patch_strides = (
        tuple(
            stride * step
            for stride, step
            in zip(strides[:n_dim], step_sizes))
        + strides[:n_dim]
        + strides[n_dim:]
    )
    # Extract overlapping patches using as_strided
    patches = np.lib.stride_tricks.as_strided(array, shape=patch_shape, strides=patch_strides)

    return patches


def stack_patches(patches: np.ndarray, n_features: int) -> np.ndarray:
    """Reconstructs an array from non-overlapping patches using einops.

    Args:
        patches: Patch grid, shape (g1 ... gd p1 ... pd f1 ... fk).
        n_features: The number of features which are not patched.

    Returns:
        Reassembled array of shape (g1 * p1, ..., gd * pd, f1, ..., fk), with n_features trailing feature axes.
    """
    # Number of spatial dimensions.
    n_dim = (len(patches.shape) - n_features) // 2

    # Dynamically construct einops pattern
    grid_dims = [f"grid{i}" for i in range(n_dim)]
    patch_dims = [f"patch{i}" for i in range(n_dim)]
    zipped_dims = [f"({g} {p})" for g, p in zip(grid_dims, patch_dims)]
    einops_pattern = f"{' '.join(grid_dims)} {' '.join(patch_dims)} ... -> {' '.join(zipped_dims)} ..."

    # Rearrange patches to reconstruct the array
    array = einops.rearrange(patches, einops_pattern)

    return array
