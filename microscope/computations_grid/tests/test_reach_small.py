"""Pointwise reach contracts without large pairwise-distance workloads."""

import numpy as np
import pytest

from microscope.computations_grid.reach import reach_per_point


def circle(n, periodic=True):
    angles = np.linspace(0, 2*np.pi if periodic else np.pi, n, endpoint=not periodic)
    return angles, 2*np.stack([np.cos(angles), np.sin(angles)], axis=-1)


@pytest.mark.parametrize("batch_size", [1, 7, 40])
def test_circle_pointwise_reach_and_witnesses(batch_size):
    angles, points = circle(24)
    reach, witnesses = reach_per_point(points, [2*np.pi], [10], [0],
                                      batch_size=batch_size, return_witnesses=True, device="cpu")
    np.testing.assert_allclose(reach, np.full(24, 2.), rtol=0, atol=1e-11)
    assert witnesses.shape == (24,)
    assert np.issubdtype(witnesses.dtype, np.integer)
    assert ((witnesses >= 0) & (witnesses < 24)).all()
    assert (witnesses != np.arange(24)).all()
    # All distinct circle pairs tie. Check the witnessed value, not tie-breaking.
    diffs = points[witnesses] - points
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    witnessed_reach = np.sum(diffs**2, axis=-1) / (2*np.abs(np.sum(diffs*normals, axis=-1)))
    np.testing.assert_allclose(witnessed_reach, reach, rtol=0, atol=1e-11)


@pytest.mark.parametrize("periodic,n", [(True, 24), (True, 25), (False, 26), (False, 27)])
def test_subsampled_reach_restores_cropped_shape(periodic, n):
    _, points = circle(n, periodic)
    reach, witnesses = reach_per_point(points, [2*np.pi if periodic else np.pi], [10],
                                      [0] if periodic else [], subsample_points=2,
                                      batch_size=7, return_witnesses=True, device="cpu")
    size = n if periodic else n-2
    assert reach.shape == (size,)
    np.testing.assert_allclose(reach, np.full(size, 2.), rtol=0, atol=1e-11)
    # Witness indices address the flattened sampled candidate set, before repetition.
    sampled_size = (size+1)//2
    assert witnesses.shape == (sampled_size,)
    assert ((witnesses >= 0) & (witnesses < sampled_size)).all()
    assert (witnesses != np.arange(sampled_size)).all()


def test_reach_rejects_mismatched_patch_and_range_dimensions():
    _, points = circle(24)
    with pytest.raises(ValueError, match="same length"):
        reach_per_point(points, [2*np.pi], [10, 10], device="cpu")
