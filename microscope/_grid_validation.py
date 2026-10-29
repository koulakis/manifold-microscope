"""Validation shared by regular-grid sampling and computations."""
from math import isfinite


def validate_integer_list(
    values: list[int],
    name: str,
    minimum: int = 0
) -> None:
    """Validate the minimum value in a list of integers.

    Require values >= minimum.

    Args:
        values: Integer values, one per grid axis or search control.
        name: Parameter name used in errors.
        minimum: Smallest accepted integer.
    """
    if any(v < minimum for v in values):
        raise ValueError(f"{name} must contain integers >= {minimum}.")


def validate_cyclic_axes(values: list[int], dims: int) -> None:
    """Validate periodic-axis indices.

    Indices must be unique and in [0, dims).

    Args:
        values: Periodic-axis indices, shape (k).
        dims: Number of available array axes.
    """
    validate_integer_list(values, "cyclic dimensions")
    if len(set(values)) != len(values) or any(axis >= dims for axis in values):
        raise ValueError(f"cyclic dimensions must be unique axes in [0, {dims}).")


def validate_spacings(values: list[float], dims: int) -> None:
    """Validate finite nonzero coordinate spacings.

    Require exactly dims finite, nonzero values.

    Args:
        values: Coordinate spacings, shape (d).
        dims: Expected number of spacings.
    """
    if len(values) != dims:
        raise ValueError(f"Expected {dims} difference intervals, one spacing per grid axis.")
    if any(not isfinite(value) or value == 0 for value in values):
        raise ValueError("difference intervals must be finite nonzero spacings.")
