"""Corresponding-point reconstruction/projection errors used in the paper."""
import numpy as np


def pointwise_distances(original, reconstructed):
    if len(original) != len(reconstructed):
        raise ValueError('Original and reconstruction lengths differ.')
    difference = original.reshape(len(original), -1) - reconstructed.reshape(len(reconstructed), -1)
    return np.linalg.norm(difference, axis=1)


def summarize(distances, number_of_train_points):
    if not len(distances) or not np.isfinite(distances).all():
        raise ValueError('Errors must be nonempty and finite.')
    return dict(number_of_train_points=int(number_of_train_points),
                number_of_test_points=len(distances), hausdorff_distance=float(distances.max()),
                average_distance=float(distances.mean()),
                pointwise_metric='max ||x - reconstruction(x)|| (Hausdorff proxy)')
