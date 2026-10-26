"""Spacing validation preserves caller precision and computation dtype."""
import numpy as np
import pytest
import torch

from microscope._grid_validation import spacings
from microscope.computations_grid.basic import partial_derivative_approximation


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32, np.int64])
def test_spacings_preserve_numpy_scalar_types(dtype):
    values = np.array([1, 2], dtype=dtype)
    result = spacings(values, 2)
    assert isinstance(result, list)
    assert all(type(value) is dtype for value in result)
    np.testing.assert_array_equal(result, values)


def test_spacings_preserve_mixed_python_and_numpy_scalars():
    values = [np.float32(.1), np.float64(.2), .3, 1]
    result = spacings(values, len(values))
    assert [type(value) for value in result] == [type(value) for value in values]
    assert result == values


@pytest.mark.parametrize("values", [[0.], [np.nan], [np.inf], [1j], [".1"], [True], [None],
                                    [np.float32(.1), True]])
def test_spacings_reject_invalid_real_values(values):
    with pytest.raises(ValueError, match="spacing|interval"):
        spacings(values, len(values))


@pytest.mark.parametrize("values", [[.1], [[.1, .2]], .1])
def test_spacings_reject_wrong_shape_or_count(values):
    with pytest.raises(ValueError, match="spacing|interval"):
        spacings(values, 2)


@pytest.mark.parametrize("tensor_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("spacing_dtype", [np.float32, np.float64])
def test_derivative_follows_tensor_dtype(tensor_dtype, spacing_dtype):
    h = spacing_dtype(.125)
    coordinates = torch.arange(9, dtype=tensor_dtype)*float(h)
    derivative = partial_derivative_approximation(coordinates.square(), 0, h)
    assert derivative.dtype == tensor_dtype
    # Central differences of x² are exact on this dyadic grid.
    torch.testing.assert_close(derivative, 2*coordinates[1:-1], rtol=0, atol=0)
