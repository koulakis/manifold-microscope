"""Projection using the unchanged MMLS neighbor/SVD implementation."""
import numpy as np

from .projector import ANNMMLSProjector


def predict(projector: ANNMMLSProjector, data: np.ndarray) -> np.ndarray:
    """Project samples and restore their original feature shape.

    Args:
        projector: Fitted MMLS projector.
        data: Input samples of shape (n ...), with sample axis first.

    Returns:
        Projected NumPy array with the same shape (n ...) as data.
    """
    return projector.project(data.reshape(len(data), -1)).reshape(data.shape)
