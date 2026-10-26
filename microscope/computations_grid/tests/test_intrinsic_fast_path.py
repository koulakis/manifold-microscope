"""Independent inverse references and conditioning-based QR selection."""
import pytest
import torch

from microscope.computations_grid import _curvature_geometry as geometry
from microscope.computations_grid import curvature


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_well_conditioned_intrinsic_never_calls_qr(monkeypatch, dtype):
    def forbidden(*args, **kwargs):
        raise AssertionError("A well-conditioned metric must not require QR.")
    monkeypatch.setattr(geometry, "tangent_geometry", forbidden)
    # g=[[4,2],[2,10]], inverse=[[10,-2],[-2,4]]/36.
    tangent = torch.tensor([[2., 1.], [0., 3.], [0., 0.]], dtype=dtype).expand(5, 3, 2)
    metric, inverse = geometry.metric_and_inverse(tangent)
    expected = torch.tensor([[10., -2.], [-2., 4.]], dtype=dtype)/36
    torch.testing.assert_close(inverse, expected.expand_as(inverse))
    torch.testing.assert_close(metric @ inverse, torch.eye(2, dtype=dtype).expand_as(metric))
    u, v = torch.meshgrid(torch.arange(9, dtype=dtype), torch.arange(9, dtype=dtype), indexing="ij")
    plane = torch.stack([2*u+v, 3*v, torch.zeros_like(u)], dim=-1)
    torch.testing.assert_close(curvature.scalar_curvature_batch(plane, [1., 1.]),
                               torch.zeros((3, 3), dtype=dtype))


@pytest.mark.parametrize("dtype,eps", [(torch.float32, 1e-5), (torch.float64, 1e-8)])
def test_mixed_batch_only_sends_bad_point_to_qr(monkeypatch, dtype, eps):
    tangent = torch.tensor([[[1., 0.], [0., 1.], [0., 0.]],
                            [[1., 1.], [0., eps], [0., 0.]],
                            [[2., 0.], [0., 3.], [0., 0.]]], dtype=dtype)
    original = geometry.tangent_geometry
    selected = []
    def record(values, **kwargs):
        selected.append(values.detach().clone())
        return original(values, **kwargs)
    monkeypatch.setattr(geometry, "tangent_geometry", record)
    metric, inverse = geometry.metric_and_inverse(tangent)
    assert len(selected) == 1
    torch.testing.assert_close(selected[0], tangent[1:2])
    expected = torch.tensor([[1 + 1/eps**2, -1/eps**2], [-1/eps**2, 1/eps**2]], dtype=dtype)
    torch.testing.assert_close(inverse[1], expected, rtol=2e-6, atol=0)
    torch.testing.assert_close(inverse[0], torch.eye(2, dtype=dtype))
    torch.testing.assert_close(inverse[2], torch.diag(torch.tensor([.25, 1/9], dtype=dtype)))
    torch.testing.assert_close(metric, tangent.transpose(-1, -2) @ tangent)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_positive_metric_still_falls_back_when_poorly_conditioned(monkeypatch, dtype):
    separation = torch.finfo(dtype).eps**.375
    tangent = torch.diag(torch.tensor([1., separation], dtype=dtype)).unsqueeze(0)
    assert torch.linalg.eigvalsh(tangent.transpose(-1, -2) @ tangent).min() > 0
    original = geometry.tangent_geometry
    selected = []
    def record(values, **kwargs):
        selected.append(values.shape)
        return original(values, **kwargs)
    monkeypatch.setattr(geometry, "tangent_geometry", record)
    _, inverse = geometry.metric_and_inverse(tangent)
    assert selected
    expected = torch.diag(torch.tensor([1., 1/separation**2], dtype=dtype)).unsqueeze(0)
    torch.testing.assert_close(inverse, expected)


def test_fallback_rank_error_uses_original_grid_coordinates():
    identity = torch.eye(2, dtype=torch.float64)
    tangent = identity.expand(2, 2, 2, 2).clone()
    tangent[1, 0] = torch.tensor([[1., 1.], [0., 0.]])
    with pytest.raises(torch.linalg.LinAlgError, match=r"grid index \(1, 0\)"):
        geometry.metric_and_inverse(tangent)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_tangents_are_rejected_before_spectral_check(monkeypatch, invalid):
    def forbidden(*args, **kwargs):
        raise AssertionError("Nonfinite metrics must not reach an eigensolver.")
    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)
    tangent = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    tangent[0, 0, 0] = invalid
    with pytest.raises(FloatingPointError, match="Metric"):
        geometry.metric_and_inverse(tangent)


@pytest.mark.parametrize("mode", ["fast", "qr", "mixed"])
def test_adaptive_inverse_values_and_gradients_match_qr(mode):
    values = [torch.tensor([[1.2, .1], [.2, 1.5], [.1, -.2]], dtype=torch.float64),
              torch.tensor([[1., 1.], [0., 1e-5], [0., 0.]], dtype=torch.float64)]
    indices = {"fast": [0], "qr": [1], "mixed": [0, 1]}[mode]
    tangent = torch.stack([values[i] for i in indices]).requires_grad_()
    reference = tangent.detach().clone().requires_grad_()
    _, actual = geometry.metric_and_inverse(tangent)
    _, _, inverse_factor = geometry.tangent_geometry(reference)
    expected = inverse_factor @ inverse_factor.transpose(-1, -2)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-10)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.isfinite(tangent.grad).all()
    torch.testing.assert_close(tangent.grad, reference.grad, rtol=1e-6, atol=1e-8)
