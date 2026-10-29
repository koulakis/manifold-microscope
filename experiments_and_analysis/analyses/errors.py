"""Corresponding-point reconstruction/projection errors used in the paper."""

from __future__ import annotations

import numpy as np


def pointwise_distances(original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """Compute Euclidean reconstruction errors for corresponding points.

    Args:
        original: Original samples of shape (n ...).
        reconstructed: Reconstructed samples of shape (n ...), with matching flattened features.

    Returns:
        Euclidean reconstruction-error array of shape (n), one value per input sample.
    """
    if len(original) != len(reconstructed):
        raise ValueError('Original and reconstruction lengths differ.')
    difference = original.reshape(len(original), -1) - reconstructed.reshape(len(reconstructed), -1)
    return np.linalg.norm(difference, axis=1)


def summarize(distances: np.ndarray, number_of_train_points: int) -> dict:
    """Summarize finite, nonempty pointwise reconstruction errors.

    Args:
        distances: Pointwise distances of shape (n).
        number_of_train_points: Number of training samples used to fit the model.

    Returns:
        Dictionary containing sample counts, maximum and mean reconstruction error, and the metric description.
    """
    if not len(distances) or not np.isfinite(distances).all():
        raise ValueError('Errors must be nonempty and finite.')
    return dict(
        number_of_train_points=int(number_of_train_points),
        number_of_test_points=len(distances),
        hausdorff_distance=float(distances.max()),
        average_distance=float(distances.mean()),
        pointwise_metric='max ||x - reconstruction(x)|| (Hausdorff proxy)'
    )
