"""Small analytic regressions for CPU selection and nested device forwarding."""

import numpy as np
import pytest
import torch

from microscope.computations_grid.basic import partial_derivatives_across_all_dims_batched
from microscope.computations_grid.curvature import scalar_curvature
from microscope.computations_grid.reach import reach_per_point
from microscope.computations_grid.volume import volume_element
from microscope.computations_grid.tests._device import get_test_device


@pytest.mark.parametrize("available", [False, True])
def test_dense_grid_default_requires_cuda(monkeypatch, available):
    monkeypatch.delenv("MICROSCOPE_TEST_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    if available:
        assert get_test_device() == "cuda:0"
    else:
        with pytest.raises(RuntimeError, match="MICROSCOPE_TEST_DEVICE=cpu"):
            get_test_device()


@pytest.mark.parametrize("available", [False, True])
def test_dense_grid_cpu_override_is_explicit(monkeypatch, available):
    monkeypatch.setenv("MICROSCOPE_TEST_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    assert get_test_device() == "cpu"


@pytest.mark.parametrize("device", ["mps", "meta"])
def test_dense_grid_rejects_unsupported_device(monkeypatch, device):
    monkeypatch.setenv("MICROSCOPE_TEST_DEVICE", device)
    with pytest.raises(ValueError, match="MICROSCOPE_TEST_DEVICE"):
        get_test_device()


@pytest.mark.parametrize("index", [1, 2])
def test_dense_grid_checks_requested_cuda_index(monkeypatch, index):
    monkeypatch.setenv("MICROSCOPE_TEST_DEVICE", f"cuda:{index}")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    if index == 1:
        assert get_test_device() == "cuda:1"
    else:
        with pytest.raises(ValueError, match="unavailable CUDA device"):
            get_test_device()


@pytest.mark.parametrize("device", [None, "cpu", torch.device("cpu")])
def test_plane_on_cpu(device, monkeypatch):
    # Explicit CPU must work even on a host where CUDA is available.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device is not None)
    coordinates = np.linspace(-1, 1, 11)
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    points = np.stack([x, y, np.zeros_like(x)], axis=-1)
    intervals = [0.2, 0.2]
    derivatives = partial_derivatives_across_all_dims_batched(
        features_on_grid=points,
        difference_intervals=intervals,
        patch_sizes=[9, 9],
        cyclic_dimensions=[],
        device=device
    )
    expected = np.broadcast_to(np.array([[1., 0.], [0., 1.], [0., 0.]]), (9, 9, 3, 2))
    np.testing.assert_allclose(derivatives, expected, atol=1e-12)
    volume = volume_element(
        features_on_grid=points,
        difference_intervals=intervals,
        patch_sizes=[9, 9],
        cyclic_dimensions=[],
        device=device
    )
    curvature = scalar_curvature(
        features_on_grid=points,
        difference_intervals=intervals,
        cyclic_dimensions=[],
        patch_sizes=[9, 9],
        device=device
    )
    np.testing.assert_allclose(volume, np.ones((9, 9)), atol=1e-12)
    np.testing.assert_allclose(curvature, np.zeros((5, 5)), atol=1e-12)


@pytest.mark.parametrize("device", [None, "cpu", torch.device("cpu")])
def test_circle_reach_on_cpu(device, monkeypatch):
    # If reach fails to forward explicit CPU, derivatives try to use CUDA.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device is not None)
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    points = 2 * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    reach = reach_per_point(
        features_on_grid=points,
        range_sizes=[2 * np.pi],
        patch_sizes=[10],
        cyclic_dimensions=[0],
        batch_size=4,
        device=device
    )
    np.testing.assert_allclose(reach, np.full(24, 2.), atol=1e-12)
