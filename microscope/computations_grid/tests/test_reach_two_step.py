"""Independent normal-curvature and double-normal bottleneck references."""
import numpy as np
import pytest
import torch

from microscope.computations_grid.reach_two_step import (
    ReachComponents, _bottleneck_radius, _max_normal_curvature,
    reach_curvature_bottleneck,
)
from microscope.manifold_examples.sampling_grid import sample_ellipsoid_on_grid


@pytest.fixture(params=["cuda:0", "cpu"])
def device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def circle(n=64, radius=2., dtype=np.float64, covers=1):
    t = np.linspace(0, covers*2*np.pi, n, endpoint=False)
    return (radius*np.column_stack([np.cos(t), np.sin(t)])).astype(dtype)


def estimate(points, device, **kwargs):
    return reach_curvature_bottleneck(points, [2*np.pi], [9], [0],
                                      device=device, return_components=True, **kwargs)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", [25, 64])
def test_circle_has_dimension_independent_radius_and_no_angle_bias(device, dtype, n):
    result = estimate(circle(n=n, dtype=dtype), device)
    assert isinstance(result, ReachComponents)
    tol = 3e-5 if dtype == np.float32 else 1e-11
    assert result.reach == pytest.approx(2., abs=tol)
    assert result.curvature_radius == pytest.approx(2., abs=tol)
    # Raw half-chord with a fixed angular tolerance would be biased below2.
    if np.isfinite(result.bottleneck_radius):
        assert result.bottleneck_radius == pytest.approx(2., abs=tol)


@pytest.mark.parametrize("ambient", [4, 7])
def test_three_sphere_uses_operator_norm_not_scalar_or_frobenius(device, ambient):
    sizes, fraction = [7, 7, 12], .8
    points, _, _, _ = sample_ellipsoid_on_grid([2]*4, sizes, ambient,
        apply_random_isometry=False, fraction_of_angles=fraction)
    result = reach_curvature_bottleneck(points, [(2*fraction-1)*np.pi]*2+[2*np.pi],
        [7]*3, [2], subsample_points=2, device=device, return_components=True)
    assert result.curvature_radius == pytest.approx(2., abs=1e-10)
    assert result.reach == pytest.approx(2., abs=1e-10)


def test_ellipse_is_curvature_limited(device):
    points = circle(n=64, radius=1.)*np.array([3., 1.])
    result = estimate(points, device)
    assert result.curvature_radius == pytest.approx(1/3, abs=1e-11)
    assert result.reach == pytest.approx(1/3, abs=1e-11)
    assert result.bottleneck_radius > result.curvature_radius


def test_higher_codimension_form_norm_is_not_flattened_spectral_norm(device):
    form = torch.zeros((1, 4, 2, 2), dtype=torch.float64, device=device)
    form[0, 2] = torch.diag(torch.tensor([1., -1.], device=device))
    form[0, 3] = torch.tensor([[0., 1.], [1., 0.]], device=device)
    # ||B(v,v)||²=cos²(2theta)+sin²(2theta)=1 for every unit v.
    assert _max_normal_curvature(form).item() == pytest.approx(1., abs=1e-12)
    assert torch.linalg.matrix_norm(form.flatten(-2), ord=2).item() == pytest.approx(np.sqrt(2))


def test_off_axis_hypersurface_shape_operator_is_exact_after_refinement(device):
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    operator = rotation @ np.diag([3., -2.]) @ rotation.T
    form = torch.zeros((1, 3, 2, 2), dtype=torch.float64, device=device)
    form[0, 2] = torch.as_tensor(operator, device=device)
    value = _max_normal_curvature(form, n_directions=0, iterations=1)
    assert value.item() == pytest.approx(3., abs=1e-12)


def test_general_codimension_refines_toward_independent_analytic_maximum(device):
    form = torch.zeros((1, 4, 2, 2), dtype=torch.float64, device=device)
    form[0, 2] = torch.tensor([[2., 0.], [0., 1.]], device=device)
    form[0, 3] = torch.tensor([[0., 2.], [2., 0.]], device=device)
    # With x=cos²(theta), ||B(v,v)||²=1+18x-15x², max6.4 at x=.6.
    sampled = _max_normal_curvature(form, n_directions=0, iterations=0).item()
    refined = _max_normal_curvature(form, n_directions=32, iterations=16).item()
    assert sampled <= refined <= np.sqrt(6.4)*(1+1e-12)
    assert refined == pytest.approx(np.sqrt(6.4), rel=1e-6)


def test_parallel_components_have_independent_bottleneck_scale(device):
    theta = np.linspace(0, 2*np.pi, 24, endpoint=False)
    gap = .01
    base = np.column_stack([2*np.cos(theta), 2*np.sin(theta)])
    points = np.concatenate([np.column_stack([base, np.full(24, -gap)]),
                             np.column_stack([base, np.full(24, gap)])])
    tangent = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros(24)])
    frames = np.tile(tangent, (2, 1))[:, :, None]
    radius, pair = _bottleneck_radius(torch.from_numpy(points), torch.from_numpy(frames),
        7, 11, .1, None, device)
    assert radius == pytest.approx(gap, abs=1e-12)
    assert (pair[0] < 24) != (pair[1] < 24)


def test_bottleneck_filter_checks_both_endpoint_tangents(device):
    points = torch.tensor([[0., 0.], [1., 0.]], dtype=torch.float64)
    frames = torch.tensor([[[0.], [1.]], [[1.], [0.]]], dtype=torch.float64)
    radius, pair = _bottleneck_radius(points, frames, 1, 1, .1, None, device)
    assert np.isposinf(radius) and pair is None


def test_connected_fold_is_bottleneck_limited(device):
    t = np.linspace(0, 2*np.pi, 96, endpoint=False)
    gap = .01
    points = np.column_stack([np.sin(t), np.sin(2*t), gap*np.cos(t)])
    result = estimate(points, device, candidate_batch_size=13)
    assert result.reach == pytest.approx(gap, abs=1e-10)
    assert result.bottleneck_radius == pytest.approx(gap, abs=1e-10)
    assert result.curvature_radius > 5*gap
    a, b = result.bottleneck_pair
    assert np.linalg.norm(points[a]-points[b])/2 == pytest.approx(gap, abs=1e-10)


def test_curvature_radius_converges_for_nonlinear_parameterization(device):
    errors = []
    for n in [32, 64, 128]:
        t = np.linspace(0, 2*np.pi, n, endpoint=False)
        theta = t+.6*np.sin(t)
        points = np.column_stack([np.cos(theta), np.sin(theta)])
        result = estimate(points, device)
        errors.append(abs(result.curvature_radius-1.))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < .02


@pytest.mark.parametrize("patches,stride", [([5, 5], 1), ([8, 7], 2), ([11, 12], 3)])
def test_partial_tiles_subsampled_centers_and_product_normal_spaces(device, patches, stride):
    u, v = np.meshgrid(2*np.pi*np.arange(13)/13, 2*np.pi*np.arange(17)/17, indexing="ij")
    points = np.stack([2*np.cos(u), 2*np.sin(u), .5*np.cos(v), .5*np.sin(v)], axis=-1)
    result = reach_curvature_bottleneck(points, [2*np.pi]*2, patches, [0, 1],
        subsample_points=stride, device=device, return_components=True)
    assert result.curvature_radius == pytest.approx(.5, abs=1e-11)
    assert result.reach == pytest.approx(.5, abs=1e-11)


def test_open_grid_curvature_matches_parabola_reference_and_handles_reverse(device):
    t = np.linspace(-1, 1, 17)
    points = np.column_stack([t, t*t])
    for array in [points, points[::-1]]:
        result = reach_curvature_bottleneck(array, [2.], [7], subsample_points=2,
            device=device, return_components=True)
        # Curvature2 at t=0 gives radius1/2; the sampled center set includes0.
        assert result.curvature_radius == pytest.approx(.5, abs=1e-12)


def test_flat_affine_grid_has_infinite_components(device):
    t = np.arange(11.)
    points = np.column_stack([t, 2*t, -.5*t])
    result = reach_curvature_bottleneck(points, [10.], [7], device=device, return_components=True)
    assert np.isposinf(result.reach) and np.isposinf(result.curvature_radius)
    assert np.isposinf(result.bottleneck_radius) and result.bottleneck_pair is None


def test_ambient_isometry_and_scale(device):
    points = circle(n=32)*np.array([3., 1.])
    rng = np.random.default_rng(7)
    rotation, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    reference = estimate(points, device).reach
    transformed = points @ rotation.T*4 + np.arange(5)
    result = estimate(transformed, device)
    assert result.reach == pytest.approx(4*reference, abs=1e-10)


def test_near_pair_policy_and_filter(device):
    points = circle(n=48, covers=2)
    with pytest.raises(ValueError, match="separation|coincident"):
        estimate(points, device)
    result = estimate(points, device, min_pair_distance=1e-8)
    assert result.reach == pytest.approx(2., abs=1e-11)
    with pytest.raises(ValueError, match="distinct"):
        estimate(points, device, min_pair_distance=5.)


@pytest.mark.parametrize("option,value", [
    ("bottleneck_tolerance", -1), ("bottleneck_tolerance", 1),
    ("bottleneck_tolerance", np.nan), ("bottleneck_tolerance", True),
    ("batch_size", 0), ("candidate_batch_size", 0), ("subsample_points", False),
    ("n_directions", -1), ("curvature_iterations", 1.5), ("min_pair_distance", -1),
])
def test_invalid_options_are_rejected(device, option, value):
    with pytest.raises(ValueError):
        estimate(circle(), device, **{option: value})


@pytest.mark.parametrize("dtype", [np.float16, np.int64, np.complex64])
def test_unsupported_dtype(device, dtype):
    with pytest.raises(TypeError, match="float32|float64"):
        estimate(circle().astype(dtype), device)


def test_nonfinite_rank_and_undersampling_errors(device):
    points = circle()
    points[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        estimate(points, device)
    with pytest.raises(ValueError, match="two sampled"):
        estimate(circle(), device, subsample_points=64)
    with pytest.raises(torch.linalg.LinAlgError, match="rank"):
        estimate(np.ones((12, 2)), device)
