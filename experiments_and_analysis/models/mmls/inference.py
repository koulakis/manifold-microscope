"""Projection using the unchanged MMLS neighbor/SVD implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from .projector import ANNMMLSProjector


def predict(projector: ANNMMLSProjector, data: np.ndarray) -> np.ndarray:
    """Project samples and restore their original feature shape.

    Args:
        projector: Fitted MMLS projector.
        data: Input samples of shape (n ...), with sample axis first.
    """
    return projector.project(data.reshape(len(data), -1)).reshape(data.shape)
