"""Explicit MMLS settings must reach the projector, rather than only its log."""
import numpy as np

from experiment_scripts.manifold_fitting import mmls
from experiment_scripts.model_configs import MMLSConfig


def test_projector_receives_configured_sigma_and_threshold(tmp_path, monkeypatch):
    train = np.arange(16, dtype=np.float32).reshape(4, 2, 2)
    test = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    monkeypatch.setattr(mmls, "load_dataset_fixed_test_split", lambda **kwargs: (train, test, None, None))
    received = {}

    class Projector:
        def __init__(self, points, **kwargs):
            received.update(kwargs)

        def project(self, points):
            return points

    monkeypatch.setattr(mmls, "ANNMMLSProjector", Projector)
    config = MMLSConfig(output_dir=str(tmp_path), exported_datasets_dir=tmp_path,
                        dataset="custom_dsprites_balanced", number_of_dims=1,
                        training_ratio=1., ratio_per_dim=False, noise_sigma=0.,
                        number_of_neighbors=2, sigma=0.3, exact_nn_threshold=17,
                        device="cpu", verbose=False, projection_batch_size=3)
    mmls.fit_mmls(config)
    assert received["sigma"] == 0.3
    assert received["exact_nn_threshold"] == 17
    assert received["device"] == "cpu"
    assert received["projection_method"] == "thin_svd"
    assert received["projection_batch_size"] == 3
