"""Small independent references; dense accuracy tests remain in their original files."""

import numpy as np
import pytest
import torch

from microscope.computations_grid.basic import (
    partial_derivatives_across_all_dims,
    partial_derivatives_across_all_dims_batched,
    riemannian_metric,
)
from microscope.computations_grid.curvature import (
    christoffel_symbols,
    riemannian_curvature_tensor,
    scalar_curvature,
    scalar_curvature_batch,
)
from microscope.computations_grid.volume import volume_element


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_affine_grid_preserves_derivative_axes_and_metric(dtype):
    # Unequal spacings and a nonorthogonal Jacobian detect axis swaps.
    u, v = np.meshgrid(np.arange(11) * .25, np.arange(13) * .5, indexing="ij")
    points = np.stack([2*u + v, u + 3*v, 4*u - 2*v], axis=-1).astype(dtype)
    tensor = torch.from_numpy(points)
    jacobian = np.array([[2., 1.], [1., 3.], [4., -2.]], dtype=dtype)
    expected_derivatives = np.broadcast_to(jacobian, (9, 11, 3, 2))
    expected_metric = np.broadcast_to(np.array([[21., -3.], [-3., 14.]]), (9, 11, 2, 2))
    direct = partial_derivatives_across_all_dims(tensor, 2, [.25, .5])
    np.testing.assert_allclose(direct.numpy(), expected_derivatives, atol=1e-6)
    np.testing.assert_allclose(riemannian_metric(tensor, [.25, .5]).numpy(), expected_metric)
    kwargs = dict(patch_sizes=[9, 10], cyclic_dimensions=[], device="cpu")
    batched = partial_derivatives_across_all_dims_batched(points, [.25, .5], **kwargs)
    assert batched.dtype == dtype
    np.testing.assert_allclose(batched, expected_derivatives, atol=1e-6)
    np.testing.assert_allclose(volume_element(points, [.25, .5], **kwargs),
                               np.full((9, 11), np.sqrt(285)), rtol=1e-6)
    np.testing.assert_allclose(scalar_curvature(points, [.25, .5], **kwargs),
                               np.zeros((5, 7)), atol=1e-5)


@pytest.fixture
def polar_metric():
    # Euclidean plane in polar coordinates: g = diag(1, r**2).
    r = torch.linspace(1., 2., 51, dtype=torch.float64)
    metric = torch.zeros((51, 7, 2, 2), dtype=torch.float64)
    metric[..., 0, 0] = 1
    metric[..., 1, 1] = r[:, None]**2
    first, second = christoffel_symbols(metric, torch.linalg.inv(metric), [.02, .1])
    return r, first, second


def test_christoffel_symbols_polar_reference(polar_metric):
    r, first, second = polar_metric
    radii = r[1:-1, None]
    expected_first = torch.zeros_like(first)
    expected_first[..., 0, 1, 1] = radii
    expected_first[..., 1, 0, 1] = radii
    expected_first[..., 1, 1, 0] = -radii
    expected_second = torch.zeros_like(second)
    expected_second[..., 0, 1, 1] = -radii
    expected_second[..., 1, 0, 1] = 1/radii
    expected_second[..., 1, 1, 0] = 1/radii
    # Central differences of r**2 are exact, apart from rounding.
    torch.testing.assert_close(first, expected_first, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(second, expected_second, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("kind", ["first", "second"])
def test_flat_polar_metric_has_zero_riemann_tensor(polar_metric, kind):
    _, first, second = polar_metric
    result = riemannian_curvature_tensor(first, second, [.02, .1], kinds=[kind])
    assert result["second" if kind == "first" else "first"] is None
    assert result[kind].shape == (47, 3, 2, 2, 2, 2)
    # Differentiating 1/r introduces O(h**2) error; h=.02, r>=1.
    torch.testing.assert_close(result[kind], torch.zeros_like(result[kind]), rtol=0, atol=1e-3)


def test_sphere_curvature_normalization_and_patch_seams():
    latitude = np.linspace(-.7, .7, 41)
    longitude = np.linspace(0, 2*np.pi, 48, endpoint=False)
    u, v = np.meshgrid(latitude, longitude, indexing="ij")
    points = 2*np.stack([np.cos(u)*np.cos(v), np.cos(u)*np.sin(v), np.sin(u)], axis=-1)
    intervals = [latitude[1] - latitude[0], 2*np.pi/48]
    kwargs = dict(cyclic_dimensions=[1], device="cpu")
    raw = scalar_curvature(points, intervals, patch_sizes=[13, 17], **kwargs)
    normalized = scalar_curvature(points, intervals, patch_sizes=[13, 17], normalize=True, **kwargs)
    padded = np.pad(points, ((0, 0), (3, 3), (0, 0)), mode="wrap")
    direct = scalar_curvature_batch(torch.from_numpy(padded), intervals).numpy()
    np.testing.assert_allclose(raw, direct, rtol=1e-10, atol=1e-10)
    # Sphere radius 2: scalar curvature 2/R**2, normalized curvature 1/R**2.
    np.testing.assert_allclose(raw, np.full((35, 48), .5), rtol=0, atol=.01)
    np.testing.assert_allclose(normalized, raw/2, rtol=1e-12, atol=1e-12)


def test_rank_deficient_grid_cannot_compute_curvature():
    u, _ = np.meshgrid(np.arange(9.), np.arange(9.), indexing="ij")
    points = torch.from_numpy(np.stack([u, np.zeros_like(u)], axis=-1))
    with pytest.raises(torch.linalg.LinAlgError):
        scalar_curvature_batch(points, [1., 1.])


def test_normalized_curvature_rejects_intrinsic_dimension_one():
    # Division by d*(d-1) is undefined for curves; request a clear rejection.
    points = torch.arange(9., dtype=torch.float64)[:, None]
    with pytest.raises(ValueError, match="dimension|normaliz"):
        scalar_curvature_batch(points, [1.], normalize=True)


@pytest.mark.parametrize("patch_sizes", [[6, 9], [5, 9]])
def test_curvature_rejects_patch_sizes_without_an_interior(patch_sizes):
    u, v = np.meshgrid(np.arange(9.), np.arange(9.), indexing="ij")
    points = np.stack([u, v], axis=-1)
    with pytest.raises(ValueError):
        scalar_curvature(points, [1., 1.], [], patch_sizes, device="cpu")
