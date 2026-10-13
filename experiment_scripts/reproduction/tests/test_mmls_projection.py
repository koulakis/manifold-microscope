"""Thin SVD must preserve the weighted tangent projection on regular inputs."""
import numpy as np
import pytest

from experiment_scripts.toy_manifolds_experiment.manifold_fitting_no_noise import ANNMMLSProjector


@pytest.mark.parametrize("batch_size", [1, 3, 32])
def test_default_thin_svd_matches_original(batch_size):
    rng = np.random.default_rng(52)
    train = rng.normal(size=(20, 9))
    queries = rng.normal(size=(5, 9))
    options = dict(d=3, k=8, sigma=1.3, device="cpu")
    expected = ANNMMLSProjector(train, **options, projection_method="original").project(queries)
    actual = ANNMMLSProjector(train, **options,
                              projection_batch_size=batch_size).project(queries)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("method", ["original", "thin_svd"])
def test_projection_onto_embedded_affine_plane(method):
    rng = np.random.default_rng(7)
    basis, _ = np.linalg.qr(rng.normal(size=(6, 2)))
    offset = rng.normal(size=6)
    train = offset + rng.normal(size=(20, 2)) @ basis.T
    queries = rng.normal(size=(4, 6))
    expected = offset + (queries - offset) @ basis @ basis.T
    actual = ANNMMLSProjector(train, d=2, k=7, device="cpu",
                              projection_method=method).project(queries)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_fast_constructor_does_not_build_unused_ann_index(monkeypatch):
    calls = []
    monkeypatch.setattr(ANNMMLSProjector, "build_index", lambda self: calls.append(self))
    train = np.arange(18, dtype=np.float64).reshape(6, 3)
    original = ANNMMLSProjector(train, d=1, k=2, device="cpu", exact_nn_threshold=4,
                                projection_method="original")
    fast = ANNMMLSProjector(train, d=1, k=2, device="cpu", exact_nn_threshold=4)
    assert calls == [original]
    assert fast.ann_index is None
    fast.project(train[:1])  # Same exact-neighbor branch; no index is needed.
    assert calls == [original]
