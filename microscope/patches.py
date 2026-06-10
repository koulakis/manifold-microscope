from typing import Optional

import numpy as np
from einops import einops

from microscope.cyclic_dimensions import pad_cyclic_dimensions


def extract_patches(
        array: np.ndarray,
        patch_sizes: list[int],
        overlaps: list[int],
        cyclic_dimensions: Optional[list[int]] = None,
        verbose: bool = False
) -> np.ndarray:
    """
    Generalized function to extract overlapping patches from the first n - d dimensions of a multidimensional array,
    leaving the last d dimensions untouched. If size of the array is not divisible by the resulting step size
    along all spacial dimensions, it will be 0-padded to the next size that is divisible. This means that the caller
    is responsible to truncate outputs on a later stage to compensate for those artificial padded values.

    Args:
        array: The input array to split.
        patch_sizes: The size of each patch along the first n - d dimensions.
        overlaps: The overlap along each of the first n - d dimensions.
        cyclic_dimensions: A set of dimensions where the grid is cyclic.
        verbose: If true it prints info and warning messages.


    Returns:
        A view of the overlapping patches, preserving their structure.
    """
    if cyclic_dimensions is None:
        cyclic_dimensions = []

    n_dim = len(patch_sizes)
    n_features = len(array.shape) - n_dim

    if len(overlaps) != n_dim:
        raise ValueError(
            f"The shape dimensions of the patch sizes ({n_dim}) differ from the "
            f"shape dimensions of the overlaps ({len(overlaps)})."
        )
    # Handle cyclic dimensions by padding the array
    array = pad_cyclic_dimensions(
        array,
        cyclic_dimensions=cyclic_dimensions,
        pad_sizes=[overlaps[dim] // 2 for dim in cyclic_dimensions]
    )

    patch_sizes = np.array(patch_sizes)
    overlaps = np.array(overlaps)
    step_sizes = patch_sizes - overlaps

    array_shape = np.array(array.shape[:n_dim])
    feature_shape = array.shape[n_dim:]

    residue = array_shape - overlaps - (step_sizes * ((array_shape - overlaps) // step_sizes))
    # noinspection PyUnresolvedReferences
    if (residue != 0).any():
        array_old_shape = array_shape
        # noinspection PyTypeChecker
        array = np.pad(
            array,
            pad_width=
            tuple((0, r) for r in (step_sizes - residue))
            + tuple(((0, 0) for _ in range(n_features))),
            mode="wrap"
        )
        array_shape = np.array(array.shape[:n_dim])
        feature_shape = array.shape[n_dim:]

        if verbose:
            print(
                f"WARNING: Array shape minus patch size {array_shape - patch_sizes} is not divisible by the step size "
                f"{step_sizes} (patch_size - overlap) along all dimensions. Zero padding the array of size "
                f"({array_old_shape}) to ({array_shape}) to compensate for that."
            )

    # Compute new shape and strides for patches
    patch_shape = tuple((array_shape - overlaps) // step_sizes) + tuple(patch_sizes) + feature_shape
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
    """
    Reconstructs an array from non-overlapping patches using einops.

    Args:
        patches: The patches arranged in a grid, with shape (g1, g2, ... p1, p2 ... n_features).
        n_features: The number of features which are not patched.

    Returns:
        An array with all the patches stacked along their corresponding dimensions.
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
