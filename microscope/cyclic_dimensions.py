import numpy as np

from microscope._grid_validation import cyclic_axes, integer_list, spacings


def pad_cyclic_dimensions(
        array: np.ndarray,
        cyclic_dimensions: list[int],
        pad_sizes: list[int]
) -> np.ndarray:
    """Pad an array on both sides along cyclic dimensions in a circular way.

    Args:
        array: The array to pad.
        cyclic_dimensions: The dimensions along which to pad.
        pad_sizes: The size of the padding per cyclic dimension. It will be applied on both sides,
            i.e. the array will grow by 2*pad_size.

    Returns:
        The padded array.
    """
    cyclic_dimensions = cyclic_axes(cyclic_dimensions, array.ndim)
    pad_sizes = integer_list(pad_sizes, "pad sizes")
    if len(cyclic_dimensions) != len(pad_sizes):
        raise ValueError(
            f"Expected the number of cyclic dimensions ({len(cyclic_dimensions)}) to equal the number"
            f"of pad sizes ({len(pad_sizes)})."
        )
    padding = [(0, 0)]*array.ndim
    for dim, pad_size in zip(cyclic_dimensions, pad_sizes):
        if array.shape[dim] == 0:
            raise ValueError("Cannot wrap an empty cyclic axis.")
        padding[dim] = (pad_size, pad_size)
    return np.pad(array, padding, mode="wrap") if any(pad_sizes) else array


def get_difference_intervals(
        n_samples_per_dim: list[int],
        range_sizes: list[float],
        cyclic_dimensions: list[int]
) -> list[float]:
    """Computes the finite difference corresponding to the different dimensions. The difference is constant for all
    intervals and computed by dividing the range size with the number of samples if the dimension is cyclic else the
    number of samples minus 1.

    Args:
        n_samples_per_dim: The number of samples per dimension of the data.
        range_sizes: The value ranges along each dimension.
        cyclic_dimensions: Zero-based indices of the periodic grid axes.

    Returns:
        A list with the difference interval per dimension.
    """
    n_samples_per_dim = integer_list(n_samples_per_dim, "sample counts", minimum=1)
    cyclic_dimensions = cyclic_axes(cyclic_dimensions, len(n_samples_per_dim))
    range_sizes = spacings(range_sizes, len(n_samples_per_dim))
    if any(n < 2 for axis, n in enumerate(n_samples_per_dim) if axis not in cyclic_dimensions):
        raise ValueError("Noncyclic axes need at least two samples to define a spacing.")
    return [
        size / n_samples
        if dim in cyclic_dimensions
        else size / (n_samples - 1)

        for dim, [size, n_samples]
        in enumerate(zip(range_sizes, n_samples_per_dim))
    ]
