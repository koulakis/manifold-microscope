"""Explicit MMLS settings reach the unchanged projector."""
import numpy as np

from experiments_and_analysis.models.mmls import fitting as mmls
from experiments_and_analysis.models.config import MMLSConfig


def test_projector_receives_configured_sigma_and_threshold(monkeypatch):
    received = {}

    class Projector:
        def __init__(self, points, **kwargs):
            received.update(kwargs)

    monkeypatch.setattr(mmls, 'ANNMMLSProjector', Projector)
    config = MMLSConfig(number_of_dims=1, number_of_neighbors=2, sigma=.3,
                        exact_nn_threshold=17, verbose=False, projection_batch_size=3)
    mmls.fit(np.arange(16, dtype=np.float32).reshape(4, 2, 2), config, 'cpu')
    assert received['sigma'] == .3
    assert received['exact_nn_threshold'] == 17
    assert received['device'] == 'cpu'
    assert received['projection_method'] == 'thin_svd'
    assert received['projection_batch_size'] == 3
