import numpy as np
import pytest

from microscope.datasets.custom_dsprites import dsprites
from microscope.datasets.dataset_split import interpolation_splitting, weighted_uniform_splitting
from microscope.datasets.noise_adding import null_space


def test_interpolation_split_selects_a_regular_grid():
    data = np.arange(12).reshape(3, 4, 1, 1)
    target = np.arange(12).reshape(3, 4)

    train, train_target, test, test_target, mask = interpolation_splitting(
        data, target, stride=2, subsample_dims=[0, 1]
    )

    np.testing.assert_array_equal(train_target, [0, 2, 8, 10])
    np.testing.assert_array_equal(test_target, [1, 3, 4, 5, 6, 7, 9, 11])
    assert train.shape == (4, 1, 1)
    assert test.shape == (8, 1, 1)
    np.testing.assert_array_equal(train[:, 0, 0], train_target)
    np.testing.assert_array_equal(test[:, 0, 0], test_target)
    np.testing.assert_array_equal(np.flatnonzero(mask), [0, 2, 8, 10])


def test_weighted_split_forwards_probabilities_and_preserves_source_order(monkeypatch):
    data = np.arange(24).reshape(6, 2, 2)
    target = np.arange(6)
    weights = np.arange(1, 7, dtype=float)
    weights /= weights.sum()
    received = {}

    def choose(population, *, size, replace, p):
        received.update(population=population, n_samples=size, replace=replace, probabilities=p)
        return np.array([4, 1, 3])

    monkeypatch.setattr(np.random, "choice", choose)
    train, train_target, test, test_target, mask = weighted_uniform_splitting(
        data, target, weights, train_sample_ratio=0.5
    )

    assert received["population"] == 6
    assert received["n_samples"] == 3
    assert received["replace"] is False
    np.testing.assert_array_equal(received["probabilities"], weights)
    np.testing.assert_array_equal(train_target, [1, 3, 4])
    np.testing.assert_array_equal(test_target, [0, 2, 5])
    np.testing.assert_array_equal(train, data[mask])
    np.testing.assert_array_equal(test, data[~mask])


@pytest.mark.parametrize("ratio", [-0.01, 1.01])
def test_weighted_split_rejects_invalid_ratios(ratio):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        weighted_uniform_splitting(
            np.zeros((2, 1, 1)), np.zeros(2), np.full(2, 0.5), ratio
        )


def test_batched_null_space_is_orthonormal_and_orthogonal():
    matrices = np.array([
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
    ])

    bases = null_space(matrices)

    assert bases.shape == (2, 1, 3)
    np.testing.assert_allclose(np.linalg.norm(bases, axis=-1), 1.0)
    np.testing.assert_allclose(matrices @ bases.swapaxes(-1, -2), 0.0, atol=1e-14)


def test_small_dsprites_grid_has_expected_factor_axes_and_labels():
    data, target = dsprites(
        image_size=8,
        sprite_initial_size=4,
        deformation_transforms_initial_size_factor=1,
        sprites=("heart", "square"),
        n_sizes=2,
        n_angles=4,
        max_translation=2,
        translation_stride=2,
        blur_sigma=None,
        processes=(None, None, None, None),
        verbose=False,
    )

    assert data.shape == (2, 2, 4, 2, 2, 8, 8)
    assert target.shape == data.shape[:-2]
    np.testing.assert_array_equal(target[:, 0, 0, 0, 0], [2, 0])
    assert data.dtype == np.float32
    assert data.min() == -1
    assert data.max() == 1
