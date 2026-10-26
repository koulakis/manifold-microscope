"""Independent local geometry and grid-contract checks for extrinsic curvature."""

import numpy as np
import pytest
import torch

from microscope.computations_grid.basic import partial_derivatives_across_all_dims
from microscope.computations_grid.curvature import scalar_curvature as intrinsic_curvature
from microscope.computations_grid.curvature_extrinsic import (
    scalar_curvature,
    scalar_curvature_batch,
    second_fundamental_form,
)
from microscope.cyclic_dimensions import get_difference_intervals
from microscope.manifold_examples.sampling_grid import sample_ellipsoid_on_grid


def test_saddle_form_is_normal_symmetric_and_has_negative_curvature():
    coordinates = np.linspace(-.5, .5, 21)
    u, v = np.meshgrid(coordinates, coordinates, indexing="ij")
    points = np.stack([u, v, .5*(u*u - v*v), np.zeros_like(u)], axis=-1)
    tensor = torch.from_numpy(points)
    intervals = [.05, .05]

    form = second_fundamental_form(tensor, intervals)
    tangent = partial_derivatives_across_all_dims(tensor, 2, intervals)[1:-1, 1:-1]
    assert form.shape == (17, 17, 4, 2, 2)
    torch.testing.assert_close(form, form.transpose(-1, -2), atol=1e-12, rtol=0)
    torch.testing.assert_close(
        torch.einsum("...ar,...aij->...rij", tangent, form),
        torch.zeros((17, 17, 2, 2, 2), dtype=tensor.dtype),
        atol=1e-12, rtol=0,
    )
    torch.testing.assert_close(
        form[8, 8, 2], torch.diag(torch.tensor([1., -1.], dtype=tensor.dtype)),
        atol=1e-12, rtol=0,
    )
    assert abs(scalar_curvature_batch(tensor, intervals)[7, 7].item() + 2) < 1e-12


def test_sphere_codimension_normalization_and_periodic_patch_seams():
    latitude = np.linspace(-.7, .7, 41)
    longitude = np.linspace(0, 2*np.pi, 48, endpoint=False)
    u, v = np.meshgrid(latitude, longitude, indexing="ij")
    sphere = 2*np.stack([np.cos(u)*np.cos(v), np.cos(u)*np.sin(v), np.sin(u)], axis=-1)
    points = np.concatenate([sphere, np.zeros((*sphere.shape[:-1], 2))], axis=-1)
    intervals = [latitude[1] - latitude[0], 2*np.pi/48]
    kwargs = dict(cyclic_dimensions=[1], device="cpu")

    raw = scalar_curvature(points, intervals, patch_sizes=[13, 17], **kwargs)
    normalized = scalar_curvature(points, intervals, patch_sizes=[13, 17], normalize=True, **kwargs)
    direct = scalar_curvature_batch(
        torch.from_numpy(np.pad(points, ((0, 0), (3, 3), (0, 0)), mode="wrap")), intervals
    ).numpy()
    intrinsic = intrinsic_curvature(points, intervals, patch_sizes=[13, 17], **kwargs)
    assert raw.shape == (35, 48)
    np.testing.assert_allclose(raw, direct, atol=1e-12, rtol=0)
    np.testing.assert_allclose(raw, .5, atol=1e-12, rtol=0)
    np.testing.assert_allclose(normalized, raw/2, atol=1e-12, rtol=0)
    np.testing.assert_allclose(raw, intrinsic, atol=.01, rtol=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_affine_grid_has_zero_form_and_curvature(dtype):
    u, v = np.meshgrid(np.arange(11)*.25, np.arange(13)*.5, indexing="ij")
    points = np.stack([2*u + v, u + 3*v, 4*u - 2*v], axis=-1).astype(dtype)
    form = second_fundamental_form(torch.from_numpy(points), [.25, .5])
    assert form.shape == (7, 9, 3, 2, 2)
    assert form.dtype == torch.from_numpy(points).dtype
    np.testing.assert_allclose(form.numpy(), 0, atol=1e-6 if dtype == np.float32 else 1e-13)
    result = scalar_curvature(points, [.25, .5], [], [9, 10], device="cpu")
    assert result.shape == (5, 7)
    assert result.dtype == dtype
    np.testing.assert_allclose(result, 0, atol=1e-6 if dtype == np.float32 else 1e-12)


def test_three_sphere_has_dimension_three_scalar_curvature():
    sizes = [16, 16, 24]
    points, _, _, _ = sample_ellipsoid_on_grid(
        semi_axes=[2, 2, 2, 2], n_samples_per_dim=sizes,
        ambient_space_dim=5, fraction_of_angles=.8,
        apply_random_isometry=False,
    )
    intervals = get_difference_intervals(
        sizes, [.6*np.pi, .6*np.pi, 2*np.pi], [2]
    )
    result = scalar_curvature_batch(torch.from_numpy(points), intervals)
    assert result.shape == (10, 10, 18)
    # d(d-1)/r² = 3*2/4, including codimension two in ambient R^5.
    torch.testing.assert_close(result, torch.full_like(result, 1.5), atol=1e-10, rtol=0)


def test_flat_product_of_circles_has_two_nonzero_normal_curvatures():
    u, v = np.meshgrid(
        np.linspace(0, 2*np.pi, 24, endpoint=False),
        np.linspace(0, 2*np.pi, 28, endpoint=False), indexing="ij"
    )
    points = np.stack([2*np.cos(u), 2*np.sin(u), 3*np.cos(v), 3*np.sin(v)], axis=-1)
    form = second_fundamental_form(torch.from_numpy(points), [2*np.pi/24, 2*np.pi/28])
    assert np.linalg.norm(form[0, 0, :, 0, 0]) > .5
    assert np.linalg.norm(form[0, 0, :, 1, 1]) > .3
    result = scalar_curvature(points, [2*np.pi/24, 2*np.pi/28], [0, 1], [12, 13], device="cpu")
    assert result.shape == (24, 28)
    np.testing.assert_allclose(result, 0, atol=1e-11)


def test_rank_failure_and_normalization_dimension_match_intrinsic():
    u, _ = np.meshgrid(np.arange(9.), np.arange(9.), indexing="ij")
    singular = torch.from_numpy(np.stack([u, np.zeros_like(u)], axis=-1))
    with pytest.raises(torch.linalg.LinAlgError):
        scalar_curvature_batch(singular, [1., 1.])
    curve = torch.arange(9., dtype=torch.float64)[:, None]
    with pytest.raises(ValueError, match="dimension|normaliz"):
        scalar_curvature_batch(curve, [1.], normalize=True)


def test_coarse_ellipsoid_cancels_stencil_bias_against_implicit_reference():
    # For x.T A x = 1 in R^3, scalar curvature = 2 det(A) / ||Ax||^4.
    # This reference avoids the sampler's symbolic coordinate formula.
    radii = np.array([1., 1.5, 3.])
    points, _, _, _ = sample_ellipsoid_on_grid(
        semi_axes=radii, n_samples_per_dim=[16, 19], ambient_space_dim=3,
        fraction_of_angles=.8, apply_random_isometry=False,
    )
    normal = points[3:-3, 3:-3] / radii**2
    truth = 2 / np.prod(radii**2) / np.sum(normal**2, axis=-1)**2
    result = scalar_curvature_batch(torch.from_numpy(points), [.6*np.pi/15, 2*np.pi/19])
    np.testing.assert_allclose(result.numpy(), truth, atol=2e-12, rtol=0)


def test_quartic_graph_extrinsic_error_converges_at_second_order():
    # Generic graph, without the matched trigonometric derivative factors.
    # Compare at a fixed physical point to exclude changes of cropped domain.
    u0, v0 = .6, .7
    fu, fv = .8*u0**3 + .1*v0, 1.2*v0**3 + .1*u0
    truth = 2*(2.4*u0**2 * 3.6*v0**2 - .1**2)/(1 + fu**2 + fv**2)**2
    errors = []
    for h in [.04, .02, .01]:
        u, v = np.meshgrid(u0 + np.arange(-3, 4)*h,
                           v0 + np.arange(-3, 4)*h, indexing="ij")
        points = np.stack([u, v, .2*u**4 + .3*v**4 + .1*u*v], axis=-1)
        result = scalar_curvature_batch(torch.from_numpy(points), [h, h]).item()
        errors.append(abs(result - truth))
    # Halving h reduces the leading O(h^2) error by approximately four.
    np.testing.assert_allclose(np.array(errors[:-1])/errors[1:], 4, rtol=.02)
