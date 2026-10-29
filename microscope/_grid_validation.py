"""Validation shared by regular-grid sampling and computations."""
from collections.abc import Iterable

import numpy as np


def integer_list(
    values: Iterable[int],
    name: str,
    minimum: int = 0
) -> list[int]:
    """Validate an integer sequence and return Python integer values.

    Accept integer iterables with values >= minimum; booleans are rejected. Return a list of Python ints, raising
    ValueError for invalid sequences.

    Args:
        values: Integer sequence of shape (d).
        name: Parameter name used in errors.
        minimum: Smallest accepted integer.
    """
    try:
        values = list(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of integers.") from error
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           or v < minimum for v in values):
        raise ValueError(f"{name} must contain integers >= {minimum}.")
    return [int(v) for v in values]


def cyclic_axes(values: Iterable[int], dims: int) -> list[int]:
    """Validate and normalize periodic-axis indices.

    Indices must be unique integers in [0, dims); booleans are rejected.

    Args:
        values: Periodic-axis indices, shape (k).
        dims: Number of available array axes.
    """
    axes = integer_list(values, "cyclic dimensions")
    if len(set(axes)) != len(axes) or any(axis >= dims for axis in axes):
        raise ValueError(f"cyclic dimensions must be unique axes in [0, {dims}).")
    return axes


def spacings(values: Iterable[float], dims: int) -> list[float]:
    """Validate real finite nonzero spacings, preserving each scalar's type.

    Require exactly dims finite, nonzero real scalars in one dimension. Reject booleans and complex values; preserve
    each accepted scalar type.

    Args:
        values: Coordinate spacings, shape (d).
        dims: Expected number of spacings.
    """
    try:
        values = list(values)
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError("difference intervals must be finite nonzero spacings.") from error
    if array.ndim != 1 or len(array) != dims:
        raise ValueError(f"Expected {dims} difference intervals, one spacing per grid axis.")
    if (array.dtype.kind not in "iuf" or any(isinstance(v, (bool, np.bool_)) for v in values)
            or not np.isfinite(array).all() or np.any(array == 0)):
        raise ValueError("difference intervals must be finite nonzero spacings.")
    return values
