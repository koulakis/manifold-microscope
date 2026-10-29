"""Independent regressions for the seven curvature-review findings."""
import numpy as np
import pytest
import torch

from microscope.computations_grid import curvature, curvature_extrinsic
from microscope.cyclic_dimensions import get_difference_intervals, pad_cyclic_dimensions
from microscope.patches import extract_patches

METHODS = [curvature.scalar_curvature, curvature_extrinsic.scalar_curvature]
BATCHES = [curvature.scalar_curvature_batch, curvature_extrinsic.scalar_curvature_batch]


def paraboloid(dtype=np.float64):
    u, v = np.meshgrid(np.arange(9.), np.arange(9.), indexing="ij")
    return np.stack([u, v, u*u + v*v], axis=-1).astype(dtype)


@pytest.mark.parametrize("method", METHODS)
def test_partial_patch_does_not_invert_artificial_wrap_metrics(method):
    angles = 2*np.pi*np.arange(8)/8
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    circle = np.concatenate([circle, circle[:2]])
    z = np.arange(10)[:, None] + np.arange(11)[None, :]*.125
    points = np.concatenate([np.broadcast_to(circle[:, None], (10, 11, 2)), z[..., None]], axis=-1).copy()
    # Injective helical strip on a cylinder: scalar curvature is exactly zero.
    # Wrapping the nonperiodic tail creates collinear tangents outside the domain.
    for patches in ([10, 11], [9, 9], [20, 20]):
        result = method(points, [1., .125], [], patches, device="cpu")
        assert result.shape == (4, 5)
        np.testing.assert_allclose(result, 0., atol=1e-11)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("intervals", [[1.], [1., 1., 1.], [0., 1.], [np.nan, 1.], [np.inf, 1.]])
def test_spacing_errors_are_rejected_before_computation(method, intervals):
    with pytest.raises(ValueError, match="spacing|interval"):
        method(paraboloid(), intervals, [], [9, 9], device="cpu")


@pytest.mark.parametrize("batch", BATCHES)
def test_batch_and_form_validate_spacing_count(batch):
    with pytest.raises(ValueError, match="spacing|interval"):
        batch(torch.from_numpy(paraboloid()), [1.])
    with pytest.raises(ValueError, match="spacing|interval"):
        curvature_extrinsic.second_fundamental_form(torch.from_numpy(paraboloid()), [1.])


@pytest.mark.parametrize("dtype,k", [(np.float32, 1e4), (np.float64, 1e8)])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_gauss_contraction_preserves_small_sectional_curvature(dtype, k, sign):
    h = .001
    u, v = np.meshgrid(np.arange(-3, 4)*h, np.arange(-3, 4)*h, indexing="ij")
    points = np.stack([u, v, .5*(k*u*u + sign/k*v*v)], axis=-1).astype(dtype)
    # At the origin g=I, principal curvatures k and sign/k, so R=2*sign.
    result = curvature_extrinsic.scalar_curvature_batch(torch.from_numpy(points), [h, h])
    np.testing.assert_allclose(result.numpy(), 2*sign, atol=3e-6 if dtype == np.float32 else 1e-10)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", [np.int64, np.bool_, np.complex64, np.float16])
def test_unsupported_input_dtype_is_explicit(method, dtype):
    with pytest.raises(TypeError, match="float32|float64"):
        method(paraboloid(dtype), [1., 1.], [], [9, 9], device="cpu")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("axes", [[1, 1], [-1], [2]])
def test_invalid_periodic_axes_are_rejected(method, axes):
    with pytest.raises(ValueError, match="cyclic|periodic"):
        method(paraboloid(), [1., 1.], axes, [9, 9], device="cpu")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("patches", [[9], [6, 9], [5, 9]])
def test_invalid_patch_sizes_are_rejected(method, patches):
    with pytest.raises(ValueError, match="patch"):
        method(paraboloid(), [1., 1.], [], patches, device="cpu")


@pytest.mark.parametrize("method", METHODS)
def test_negative_stride_input_is_independent_of_patch_padding(method):
    points = paraboloid()
    expected = method(points, [1., 1.], [], [9, 9], device="cpu")[::-1]
    for patches in ([9, 9], [8, 8]):
        np.testing.assert_allclose(method(points[::-1], [1., 1.], [], patches, device="cpu"), expected,
                                   atol=1e-12)


@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("dtype,eps", [(np.float32, 1e-5), (np.float64, 1e-8)])
def test_full_rank_plane_survives_gram_matrix_precision_loss(batch, dtype, eps):
    u, v = np.meshgrid(np.arange(9.), np.arange(9.), indexing="ij")
    points = np.stack([u+v, eps*v, np.zeros_like(u)], axis=-1).astype(dtype)
    np.testing.assert_allclose(batch(torch.from_numpy(points), [1., 1.]).numpy(), 0., atol=1e-10)


@pytest.mark.parametrize("batch", BATCHES)
def test_true_rank_loss_has_grid_diagnostic(batch):
    u, _ = np.meshgrid(np.arange(9.), np.arange(9.), indexing="ij")
    points = torch.from_numpy(np.stack([u, np.zeros_like(u)], axis=-1))
    with pytest.raises(torch.linalg.LinAlgError, match="rank.*grid|grid.*rank"):
        batch(points, [1., 1.])


def test_divisible_axis_does_not_receive_a_whole_extra_padding_step():
    patches = extract_patches(np.zeros((10, 11, 3)), [8, 8], [6, 6], [])
    assert patches.shape[:2] == (2, 3)


@pytest.mark.parametrize("kind", ["first", "second"])
def test_riemann_only_differentiates_requested_kind(monkeypatch, kind):
    original = curvature.partial_derivatives_across_all_dims
    calls = []
    def record(tensor, *args, **kwargs):
        calls.append(tensor)
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(curvature, "partial_derivatives_across_all_dims", record)
    first = torch.zeros((7, 7, 2, 2, 2), dtype=torch.float64)
    second = torch.ones_like(first)
    result = curvature.riemannian_curvature_tensor(first, second, [1., 1.], kinds=[kind])
    assert len(calls) == 1
    assert calls[0] is (first if kind == "first" else second)
    assert result[kind] is not None


def test_spacing_generation_does_not_silently_zip_mismatched_axes():
    with pytest.raises(ValueError, match="range|interval|spacing"):
        get_difference_intervals([9, 9], [1.], [])


def test_periodic_padding_can_wrap_more_than_one_period():
    array = np.arange(2.)
    np.testing.assert_array_equal(pad_cyclic_dimensions(array, [0], [3]),
                                  np.pad(array, (3, 3), mode="wrap"))


def test_sectional_gauss_formula_in_codimension_two_and_sheared_coordinates():
    h = .02
    a, b = np.meshgrid(np.arange(-3, 4)*h, np.arange(-3, 4)*h, indexing="ij")
    u, v = a + 2*b, b
    points = np.stack([u, v, .6*u*u + .4*u*v + .4*v*v,
                       -.35*u*u - .2*u*v + .15*v*v], axis=-1)
    # In unsheared coordinates at the origin g=I and the two normal Hessians
    # give R=2*(1.2*.8 + (-.7)*.3 - .4**2 - (-.2)**2)=1.1.
    result = curvature_extrinsic.scalar_curvature_batch(torch.from_numpy(points), [h, h])
    np.testing.assert_allclose(result.numpy(), 1.1, atol=1e-11)


@pytest.mark.parametrize("batch", BATCHES)
def test_qr_curvature_preserves_tensor_autograd(batch):
    generator = torch.Generator().manual_seed(42)
    tensor = torch.randn((7, 7, 4), generator=generator, dtype=torch.float64, requires_grad=True)
    batch(tensor, [1., 1.]).sum().backward()
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
