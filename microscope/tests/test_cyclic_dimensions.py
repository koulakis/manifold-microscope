import numpy as np
import pytest

from microscope.cyclic_dimensions import get_difference_intervals, pad_cyclic_dimensions


def test_padding_wraps_multiple_grid_axes():
    array = np.arange(12).reshape(3, 4)

    padded = pad_cyclic_dimensions(array, cyclic_dimensions=[0, 1], pad_sizes=[1, 2])

    expected = np.pad(array, ((1, 1), (2, 2)), mode="wrap")
    np.testing.assert_array_equal(padded, expected)


def test_padding_requires_one_size_per_cyclic_axis():
    with pytest.raises(ValueError, match="number of cyclic dimensions"):
        pad_cyclic_dimensions(np.zeros((2, 3)), cyclic_dimensions=[0, 1], pad_sizes=[1])


def test_difference_intervals_distinguish_periodic_and_bounded_axes():
    intervals = get_difference_intervals(
        n_samples_per_dim=[4, 5], range_sizes=[2.0, 8.0], cyclic_dimensions=[0]
    )

    assert intervals == [0.5, 2.0]
