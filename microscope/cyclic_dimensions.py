import numpy as np

from microscope._grid_validation import validate_cyclic_axes, validate_integer_list, validate_spacings


def _validate_padding_inputs(
    array: np.ndarray,
    cyclic_dimensions: list[int],
    pad_sizes: list[int]
) -> None:
    """Validate circular padding sizes and axes.

    Require unique valid periodic axes and one nonnegative integer pad size per axis. Empty periodic axes cannot be
    wrapped.

    Args:
        array: Input array, shape (s1 ... sn).
        cyclic_dimensions: Zero-based periodic grid axes.
        pad_sizes: Number of samples added at each end of each periodic axis.

    Returns:
        None.
    """
    validate_cyclic_axes(cyclic_dimensions, array.ndim)
    validate_integer_list(pad_sizes, "pad sizes")
    if len(cyclic_dimensions) != len(pad_sizes):
        raise ValueError(
            f"Expected the number of cyclic dimensions ({len(cyclic_dimensions)}) to equal the number"
            f"of pad sizes ({len(pad_sizes)})."
        )
    for dim in cyclic_dimensions:
        if array.shape[dim] == 0:
            raise ValueError("Cannot wrap an empty cyclic axis.")


def pad_cyclic_dimensions(
    array: np.ndarray,
    cyclic_dimensions: list[int],
    pad_sizes: list[int]
) -> np.ndarray:
    """Pad an array on both sides along cyclic dimensions in a circular way.

    Requires compatible axes and padding; see _validate_padding_inputs.

    Args:
        array: Input array, shape (s1 ... sn).
        cyclic_dimensions: The dimensions along which to pad.
        pad_sizes: Number of samples added at each end of each periodic axis.

    Returns:
        Array with each cyclic axis length increased by twice its corresponding pad size; all other axis lengths are
        unchanged.
    """
    _validate_padding_inputs(array, cyclic_dimensions, pad_sizes)
    padding = [(0, 0)]*array.ndim
    for dim, pad_size in zip(cyclic_dimensions, pad_sizes):
        padding[dim] = (pad_size, pad_size)
    return np.pad(array, padding, mode="wrap") if any(pad_sizes) else array


def _validate_difference_interval_inputs(
    n_samples_per_dim: list[int],
    range_sizes: list[float],
    cyclic_dimensions: list[int]
) -> None:
    """Validate sample counts, ranges and periodic axes.

    Require positive integer sample counts, with at least two samples on nonperiodic axes. Ranges must be finite nonzero
    reals, one per axis; periodic axes must be unique valid indices.

    Args:
        n_samples_per_dim: Number of coordinate samples along each grid axis.
        range_sizes: Parameter-range length along each of the d grid axes.
        cyclic_dimensions: Zero-based periodic grid axes.

    Returns:
        None.
    """
    validate_integer_list(n_samples_per_dim, "sample counts", minimum=1)
    validate_cyclic_axes(cyclic_dimensions, len(n_samples_per_dim))
    validate_spacings(range_sizes, len(n_samples_per_dim))
    if any(n < 2 for axis, n in enumerate(n_samples_per_dim) if axis not in cyclic_dimensions):
        raise ValueError("Noncyclic axes need at least two samples to define a spacing.")


def get_difference_intervals(
    n_samples_per_dim: list[int],
    range_sizes: list[float],
    cyclic_dimensions: list[int]
) -> list[float]:
    """Computes the finite difference corresponding to the different dimensions. The difference is constant for all
    intervals and computed by dividing the range size with the number of samples if the dimension is cyclic else the
    number of samples minus 1.

    Requires compatible counts and ranges; see _validate_difference_interval_inputs.

    Args:
        n_samples_per_dim: The number of samples per dimension of the data.
        range_sizes: The value ranges along each dimension.
        cyclic_dimensions: Zero-based indices of the periodic grid axes.

    Returns:
        A list with the difference interval per dimension.
    """
    _validate_difference_interval_inputs(
        n_samples_per_dim,
        range_sizes,
        cyclic_dimensions
    )
    return [
        size / n_samples
        if dim in cyclic_dimensions
        else size / (n_samples - 1)

        for dim, [size, n_samples]
        in enumerate(zip(range_sizes, n_samples_per_dim))
    ]
