"""Spacing validation preserves caller precision and computation dtype."""
import numpy as np
import pytest
import torch

from microscope._grid_validation import validate_cyclic_axes, validate_integer_list, validate_spacings
from microscope.computations_grid.basic import partial_derivative_approximation


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32, np.int64])
def test_spacings_preserve_numpy_scalar_types(dtype):
    values = list(np.array([1, 2], dtype=dtype))
    original = values.copy()
    assert validate_spacings(values, 2) is None
    assert all(value is before for value, before in zip(values, original))


def test_spacings_preserve_mixed_python_and_numpy_scalars():
    values = [np.float32(.1), np.float64(.2), .3, 1]
    original = values.copy()
    assert validate_spacings(values, len(values)) is None
    assert all(value is before for value, before in zip(values, original))


@pytest.mark.parametrize("values", [[0.], [-0.], [np.nan], [np.inf], [-np.inf]])
def test_spacings_reject_invalid_real_values(values):
    with pytest.raises(ValueError, match="spacing|interval"):
        validate_spacings(values, len(values))


@pytest.mark.parametrize("values", [[], [.1], [.1, .2, .3]])
def test_spacings_reject_wrong_count(values):
    with pytest.raises(ValueError, match="spacing|interval"):
        validate_spacings(values, 2)


def test_spacings_accept_negative_and_small_nonzero_values():
    validate_spacings([-1., np.nextafter(0., 1.)], 2)


@pytest.mark.parametrize("values,minimum", [([0, 2], 0), ([1, 3], 1), ([], 1)])
def test_integer_validation_leaves_values_unchanged(values, minimum):
    original = values.copy()
    assert validate_integer_list(values, "sizes", minimum) is None
    assert values == original


@pytest.mark.parametrize("values,minimum", [([-1], 0), ([0, 3], 1)])
def test_integer_validation_rejects_values_below_minimum(values, minimum):
    with pytest.raises(ValueError, match="sizes"):
        validate_integer_list(values, "sizes", minimum)


def test_cyclic_axis_validation_preserves_order():
    axes = [2, 0]
    assert validate_cyclic_axes(axes, 3) is None
    assert axes == [2, 0]


@pytest.mark.parametrize("axes", [[-1], [2], [0, 0]])
def test_cyclic_axis_validation_rejects_invalid_indices(axes):
    with pytest.raises(ValueError, match="cyclic"):
        validate_cyclic_axes(axes, 2)


@pytest.mark.parametrize("tensor_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("spacing_dtype", [np.float32, np.float64])
def test_derivative_follows_tensor_dtype(tensor_dtype, spacing_dtype):
    h = spacing_dtype(.125)
    coordinates = torch.arange(9, dtype=tensor_dtype)*float(h)
    derivative = partial_derivative_approximation(coordinates.square(), 0, h)
    assert derivative.dtype == tensor_dtype
    # Central differences of x² are exact on this dyadic grid.
    torch.testing.assert_close(derivative, 2*coordinates[1:-1], rtol=0, atol=0)
