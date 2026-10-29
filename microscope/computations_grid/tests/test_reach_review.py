"""Reach edge cases and independent geometric references, on CUDA and CPU."""
import numpy as np
import pytest
import torch

from microscope.computations_grid.reach import reach_per_point


@pytest.fixture(params=["cuda:0", "cpu"])
def device(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def circle(n=24, radius=2., dtype=np.float64, covers=1, warp=0.):
    t = np.linspace(0, covers*2*np.pi, n, endpoint=False)
    theta = t + warp*np.sin(t)
    return (radius*np.column_stack([np.cos(theta), np.sin(theta)])).astype(dtype)


def reach(points, device, **kwargs):
    return reach_per_point(points, [2*np.pi], [10], [0], device=device, **kwargs)


@pytest.mark.parametrize("dtype,eps", [(np.float32, 1e-5), (np.float64, 1e-8),
                                      (np.float32, 5e-4)])
def test_full_rank_skew_plane_is_flat(device, dtype, eps):
    u, v = np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9), indexing="ij")
    points = np.stack([u+v, eps*v, np.zeros_like(u)], axis=-1).astype(dtype)
    result, witnesses = reach_per_point(points, [2., 2.], [5, 5], device=device,
                                       return_witnesses=True)
    assert np.isposinf(result).all()
    assert (witnesses != np.arange(49)).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rotated_affine_line_keeps_infinity_and_distinct_witness(device, dtype):
    t = np.linspace(-1, 1, 13)
    points = (t[:, None]*np.array([1., 2., -3., .5, 4.]) + 7).astype(dtype)
    result, witnesses = reach_per_point(points, [2.], [7], device=device,
                                       return_witnesses=True)
    assert np.isposinf(result).all()
    assert (witnesses != np.arange(11)).all()


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_inputs_are_rejected(device, invalid):
    points = circle()
    points[5, 0] = invalid
    with pytest.raises(ValueError, match="finite"):
        reach(points, device)


def test_overflow_is_explicit(device):
    with pytest.raises(FloatingPointError, match="overflow|finite|rescale"):
        reach(circle(radius=1e20, dtype=np.float32), device)


@pytest.mark.parametrize("mode", ["subsample", "interior", "duplicate_candidates"])
def test_no_distinct_candidate_is_rejected(device, mode):
    if mode == "interior":
        points = np.array([[-1., 1.], [0., 0.], [1., 1.]])
        range_sizes = [2.]
        patch_sizes = [5]
        cyclic_dimensions = []
        subsample_points = None
    elif mode == "duplicate_candidates":
        points = np.tile(circle(12), (2, 1))
        range_sizes = [4*np.pi]
        patch_sizes = [10]
        cyclic_dimensions = [0]
        subsample_points = 12
    else:
        points = circle()
        range_sizes = [2*np.pi]
        patch_sizes = [10]
        cyclic_dimensions = [0]
        subsample_points = 24
    with pytest.raises(ValueError, match="distinct|candidate|sample"):
        reach_per_point(
            features_on_grid=points,
            range_sizes=range_sizes,
            patch_sizes=patch_sizes,
            cyclic_dimensions=cyclic_dimensions,
            subsample_points=subsample_points,
            device=device
        )


def test_exact_duplicate_coordinates_are_excluded(device):
    result, witnesses = reach(np.tile(circle(12), (2, 1)), device, return_witnesses=True)
    np.testing.assert_allclose(result, 2., atol=1e-11, rtol=0)
    points = np.tile(circle(12), (2, 1))
    assert (np.linalg.norm(points[witnesses]-points, axis=-1) > .1).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_near_duplicate_diagnostic_and_explicit_distance_filter(device, dtype):
    points = circle(covers=2, dtype=dtype)
    with pytest.raises(ValueError, match="separation|coincident|close"):
        reach(points, device)
    result, witnesses = reach(points, device, min_pair_distance=1e-8,
                              return_witnesses=True, candidate_batch_size=7)
    np.testing.assert_allclose(result, 2., rtol=0, atol=3e-6 if dtype == np.float32 else 1e-11)
    assert (np.linalg.norm(points[witnesses]-points, axis=-1) > 1e-8).all()
    with pytest.raises(ValueError, match="candidate|distinct"):
        reach(points, device, min_pair_distance=5.)
    # Zero deliberately opts out of the automatic unresolved-pair diagnostic.
    unguarded = reach(points, device, min_pair_distance=0.)
    if dtype == np.float64:
        assert unguarded.min() < 1e-12


@pytest.mark.parametrize("radius", [1e-4, 2., 1e4])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reach_and_separation_diagnostic_scale_with_embedding(device, radius, dtype):
    result = reach(circle(radius=radius, dtype=dtype), device)
    np.testing.assert_allclose(result, radius, rtol=2e-5 if dtype == np.float32 else 1e-11)
    with pytest.raises(ValueError, match="separation|coincident|close"):
        reach(circle(radius=radius, covers=2, dtype=dtype), device)


@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.parametrize("argument", ["batch_size", "subsample_points", "candidate_batch_size"])
def test_positive_integer_options(device, argument, value):
    with pytest.raises(ValueError, match="integer"):
        reach(circle(), device, **{argument: value})


@pytest.mark.parametrize("value", [-1., np.nan, np.inf])
def test_distance_filter_validation(device, value):
    with pytest.raises(ValueError, match="distance"):
        reach(circle(), device, min_pair_distance=value)


@pytest.mark.parametrize("dtype", [np.int64, np.bool_, np.complex64, np.float16])
def test_unsupported_dtype_is_explicit(device, dtype):
    with pytest.raises(TypeError, match="float32|float64"):
        reach(circle().astype(dtype), device)


def test_reversed_numpy_view_preserves_reach(device):
    result = reach(circle()[::-1], device)
    np.testing.assert_allclose(result, 2., rtol=0, atol=1e-11)


def test_true_rank_loss_reports_grid_location(device):
    u, v = np.meshgrid(np.arange(9.), np.arange(9.), indexing="ij")
    points = np.stack([u+v, np.zeros_like(u), np.zeros_like(u)], axis=-1)
    with pytest.raises(torch.linalg.LinAlgError, match="rank.*grid|grid.*rank"):
        reach_per_point(points, [8., 8.], [5, 5], device=device)


def test_rank_diagnostic_maps_later_batch_to_input_grid(device):
    t = np.arange(9.)-4
    points = np.column_stack([t*t, np.zeros_like(t)])
    with pytest.raises(torch.linalg.LinAlgError, match=r"grid index \(4,\)"):
        reach_per_point(points, [8.], [5], batch_size=2, device=device)


@pytest.mark.parametrize("batch_size,chunk_size", [(1, 1), (7, 4), (40, 100)])
def test_parabola_matches_independent_pointwise_reference(device, batch_size, chunk_size):
    t = np.linspace(-1, 1, 17)
    points = np.column_stack([t, t*t])
    result, witnesses = reach_per_point(points, [2.], [7], device=device,
        batch_size=batch_size, candidate_batch_size=chunk_size, return_witnesses=True)
    sampled, s = points[1:-1], t[1:-1]
    normals = np.column_stack([-2*s, np.ones_like(s)])
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    differences = sampled[None]-sampled[:, None]
    denominator = 2*np.abs(np.sum(differences*normals[:, None], axis=-1))
    np.fill_diagonal(denominator, 1.)
    ratios = np.sum(differences**2, axis=-1)/denominator
    np.fill_diagonal(ratios, np.inf)
    np.testing.assert_allclose(result, ratios.min(axis=-1), rtol=1e-11)
    np.testing.assert_allclose(ratios[np.arange(len(s)), witnesses], result, rtol=1e-11)


@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
def test_multiaxis_subsampling_and_witnesses_across_chunks(device, chunk_size):
    u, v = np.meshgrid(2*np.pi*np.arange(15)/15, 2*np.pi*np.arange(17)/17, indexing="ij")
    points = np.stack([2*np.cos(u), 2*np.sin(u), .5*np.cos(v), .5*np.sin(v)], axis=-1)
    result, witnesses = reach_per_point(points, [2*np.pi]*2, [7, 8], [0, 1],
        subsample_points=3, batch_size=7, candidate_batch_size=chunk_size,
        return_witnesses=True, device=device)
    np.testing.assert_allclose(result, .5, rtol=0, atol=1e-11)
    assert result.shape == (15, 17)
    sampled = points[::3, ::3].reshape(-1, 4)
    assert witnesses.shape == (30,)
    assert (witnesses != np.arange(30)).all()
    # Independent two-dimensional normal space of the product of two circles.
    differences = sampled[witnesses]-sampled
    normal1 = np.sum(differences[:, :2]*sampled[:, :2]/2, axis=-1)
    normal2 = np.sum(differences[:, 2:]*sampled[:, 2:]/.5, axis=-1)
    ratios = np.sum(differences**2, axis=-1)/(2*np.hypot(normal1, normal2))
    np.testing.assert_allclose(ratios, .5, rtol=0, atol=1e-11)


def test_global_bottleneck_is_not_lost_to_candidate_chunking(device):
    # Torus of revolution: reach min(r, R-r); the inner-hole bottleneck wins.
    u, v = np.meshgrid(2*np.pi*np.arange(24)/24, 2*np.pi*np.arange(16)/16, indexing="ij")
    major, minor = .75, .5
    points = np.stack([(major+minor*np.cos(v))*np.cos(u),
                       (major+minor*np.cos(v))*np.sin(u), minor*np.sin(v)], axis=-1)
    result = reach_per_point(points, [2*np.pi]*2, [8]*2, [0, 1], device=device,
                             batch_size=32, candidate_batch_size=37)
    assert result.min() == pytest.approx(.25, abs=1e-11)


def test_automatic_diagnostic_preserves_resolved_narrow_bottleneck(device):
    u, v = np.meshgrid(2*np.pi*np.arange(16)/16, 2*np.pi*np.arange(12)/12, indexing="ij")
    gap, minor = 1e-6, .5
    major = minor+gap
    points = np.stack([(major+minor*np.cos(v))*np.cos(u),
                       (major+minor*np.cos(v))*np.sin(u), minor*np.sin(v)], axis=-1)
    result = reach_per_point(points, [2*np.pi]*2, [8]*2, [0, 1], device=device,
                             batch_size=32, candidate_batch_size=37)
    assert result.min() == pytest.approx(gap, rel=1e-9)


def test_nonlinear_parameterization_converges_against_exact_tangents(device):
    errors = []
    for n in [16, 64, 128]:
        points = circle(n=n, radius=1., warp=.8)
        differences = points[None]-points[:, None]
        normal = np.abs(np.sum(differences*points[:, None], axis=-1))
        np.fill_diagonal(normal, 1.)
        ratios = np.sum(differences**2, axis=-1)/(2*normal)
        np.fill_diagonal(ratios, np.inf)
        assert ratios.min() == pytest.approx(1., abs=1e-10)
        errors.append(abs(float(reach(points, device).min())-1.))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < .07
