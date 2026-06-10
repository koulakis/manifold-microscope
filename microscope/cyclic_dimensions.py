import numpy as np


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
    if len(cyclic_dimensions) != len(pad_sizes):
        raise ValueError(
            f"Expected the number of cyclic dimensions ({len(cyclic_dimensions)}) to equal the number"
            f"of pad sizes ({len(pad_sizes)})."
        )
    for dim, pad_size in zip(cyclic_dimensions, pad_sizes):
        pad_front = array.take(indices=range(-pad_size, 0), axis=dim)
        pad_back = array.take(indices=range(pad_size), axis=dim)
        array = np.concatenate([pad_front, array, pad_back], axis=dim)

    return array


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
        cyclic_dimensions: An array of booleans indicating which dimensions are cyclic.

    Returns:
        A list with the difference interval per dimension.
    """
    return [
        size / n_samples
        if dim in cyclic_dimensions
        else size / (n_samples - 1)

        for dim, [size, n_samples]
        in enumerate(zip(range_sizes, n_samples_per_dim))
    ]
