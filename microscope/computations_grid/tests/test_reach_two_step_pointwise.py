"""Pointwise component references, output compatibility and filter false negatives."""
import numpy as np
import pytest
import torch

from microscope.computations_grid.reach_two_step import (
    _bottleneck_radius, reach_curvature_bottleneck, reach_per_point,
)


@pytest.fixture(params=["cuda:0", "cpu"])
def device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def circle(n=24, dtype=np.float64):
    t = np.arange(n)*2*np.pi/n
    return (2*np.column_stack([np.cos(t), np.sin(t)])).astype(dtype)


@pytest.mark.parametrize("return_witnesses", [False, True])
@pytest.mark.parametrize("return_components", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_return_conventions_and_component_axis(device, dtype, return_witnesses, return_components):
    output = reach_per_point(circle(dtype=dtype), [2*np.pi], [8], [0], 3, 5,
        return_witnesses, device, candidate_batch_size=7, return_components=return_components)
    if return_components:
        assert isinstance(output, tuple) and len(output) == 3
        values, witnesses, components = output
        assert components.shape == (24, 2) and components.dtype == dtype
        np.testing.assert_allclose(components, 2., atol=3e-5)
        np.testing.assert_array_equal(values, components.min(axis=-1))
    elif return_witnesses:
        assert isinstance(output, tuple) and len(output) == 2
        values, witnesses = output
    else:
        assert isinstance(output, np.ndarray)
        values = output
    assert values.shape == (24,) and values.dtype == dtype
    np.testing.assert_allclose(values, 2., atol=3e-5)
    if return_components or return_witnesses:
        assert witnesses.shape == (8,) and witnesses.dtype == np.int64
        # Candidate indices refer to sampled coordinates, not repeated values.
        assert np.all((witnesses >= 0) & (witnesses < 8))
        sampled = circle(dtype=dtype)[::3]
        np.testing.assert_allclose(np.linalg.norm(sampled-sampled[witnesses], axis=-1)/2, 2., atol=3e-5)


@pytest.mark.parametrize("stride", [1, 4])
def test_full_parabola_curvature_field_and_partial_repetition(device, stride):
    t = np.linspace(-1, 1, 17)
    points = np.column_stack([t, t*t])
    values, witnesses, components = reach_per_point(points, [2.], [7],
        subsample_points=stride, device=device, return_components=True)
    assert values.shape == (13,) and components.shape == (13, 2)
    sampled_t = t[2:-2:stride]
    # Independently: curvature of (t,t²) is 2/(1+4t²)^(3/2).
    reference = (1+4*sampled_t**2)**1.5/2
    reference = reference[np.arange(13)//stride]
    np.testing.assert_allclose(components[:, 0], reference, rtol=1e-11)
    assert np.isposinf(components[:, 1]).all()
    assert np.all(witnesses == -1)
    np.testing.assert_array_equal(values, components[:, 0])


def test_mixed_periodic_grid_shapes_and_spatial_only_repetition(device):
    t, theta = np.meshgrid(np.linspace(-1, 1, 9), np.arange(14)*2*np.pi/14, indexing="ij")
    points = np.stack([t, 2*np.cos(theta), 2*np.sin(theta)], axis=-1)
    values, witnesses, components = reach_per_point(points, [2., 2*np.pi], [7, 8], [1],
        subsample_points=3, device=device, return_components=True)
    assert values.shape == (5, 14) and components.shape == (5, 14, 2)
    assert witnesses.shape == (10,)
    np.testing.assert_allclose(components[..., 0], 2., rtol=1e-11)
    np.testing.assert_allclose(values, 2., rtol=1e-11)
    np.testing.assert_array_equal(values, components.min(axis=-1))


@pytest.mark.parametrize("query_batch,candidate_batch", [(1, 7), (11, 13)])
def test_fold_has_local_bottleneck_at_both_endpoints_and_global_compatibility(device, query_batch, candidate_batch):
    t = np.arange(96)*2*np.pi/96
    points = np.column_stack([np.sin(t), np.sin(2*t), .01*np.cos(t)])
    values, witnesses, components = reach_per_point(points, [2*np.pi], [9], [0],
        batch_size=query_batch, candidate_batch_size=candidate_batch, device=device, return_components=True)
    np.testing.assert_allclose(components[[0, 48], 1], .01, atol=1e-12)
    assert witnesses[0] == 48 and witnesses[48] == 0
    assert np.all(components[[0, 48], 0] > .05)
    np.testing.assert_array_equal(values, components.min(axis=-1))
    summary = reach_curvature_bottleneck(points, [2*np.pi], [9], [0],
        batch_size=query_batch, candidate_batch_size=candidate_batch, device=device, return_components=True)
    assert summary.reach == pytest.approx(values.min(), abs=1e-12)
    assert summary.curvature_radius == pytest.approx(components[..., 0].min(), abs=1e-12)
    assert summary.bottleneck_radius == pytest.approx(components[..., 1].min(), abs=1e-12)
    assert summary.bottleneck_pair in ((0, 48), (48, 0))


def test_ellipse_keeps_bottleneck_witness_when_curvature_wins(device):
    points = circle(64)*np.array([1.5, .5])
    values, witnesses, components = reach_per_point(points, [2*np.pi], [9], [0],
        device=device, return_components=True)
    assert values[0] == pytest.approx(1/3, abs=1e-11)
    assert components[0, 1] > components[0, 0]
    assert witnesses[0] >= 0  # Witness describes the bottleneck component.


def test_flat_field_and_no_accepted_partner(device):
    t = np.arange(11.)
    points = np.column_stack([t, 2*t, -.5*t])
    values, witnesses, components = reach_per_point(points, [10.], [7],
        device=device, return_components=True)
    assert np.isposinf(values).all() and np.isposinf(components).all()
    assert np.all(witnesses == -1)


def test_staggered_sampled_branches_can_be_rejected_at_both_ends(device):
    # Exact tangents to two parallel lines, separated by gap=.1. Staggering
    # by half a .05 grid step makes the nearest unit chord's tangential
    # projection .025/sqrt(.025²+.1²) = .2425356 > .1.
    spacing, gap = .05, .1
    bottom = np.column_stack([np.arange(5)*spacing, np.zeros(5)])
    top = np.column_stack([(np.arange(5)+.5)*spacing, np.full(5, gap)])
    points = torch.from_numpy(np.concatenate([bottom, top]))
    frames = torch.zeros((10, 2, 1), dtype=torch.float64)
    frames[:, 0, 0] = 1
    strict, strict_witness = _bottleneck_radius(points, frames, 3, 4, .1, None,
        device, return_per_point=True)
    assert np.isposinf(strict).all() and np.all(strict_witness == -1)
    relaxed, witnesses = _bottleneck_radius(points, frames, 3, 4, .3, None,
        device, return_per_point=True)
    # Exact endpoint tangent ratio; exhaustive search sees this near-bottleneck
    # even with the stricter normality filter failing. True parallel-line reach
    # is .05, whereas the sampled ratio is .053125 (6.25% high).
    reference = (gap**2+(spacing/2)**2)/(2*gap)
    np.testing.assert_allclose(relaxed, reference, rtol=1e-12)
    assert np.all(witnesses[:5] >= 5) and np.all(witnesses[5:] < 5)


def test_bad_tangent_at_one_end_rejects_pair_in_both_directions(device):
    points = torch.tensor([[0., 0.], [1., 0.]], dtype=torch.float64)
    frames = torch.tensor([[[0.], [1.]], [[1.], [0.]]], dtype=torch.float64)
    values, witnesses = _bottleneck_radius(points, frames, 1, 1, .1, None,
        device, return_per_point=True)
    assert np.isposinf(values).all() and np.all(witnesses == -1)
