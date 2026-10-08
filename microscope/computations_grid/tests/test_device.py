"""Small analytic regressions for CPU selection and nested device forwarding."""

import numpy as np
import pytest
import torch

from microscope.computations_grid.basic import partial_derivatives_across_all_dims_batched
from microscope.computations_grid.curvature import scalar_curvature
from microscope.computations_grid.reach import reach_per_point
from microscope.computations_grid.volume import volume_element


@pytest.mark.parametrize("device", [None, "cpu", torch.device("cpu")])
def test_plane_on_cpu(device, monkeypatch):
    # Explicit CPU must work even on a host where CUDA is available.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device is not None)
    coordinates = np.linspace(-1, 1, 11)
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    points = np.stack([x, y, np.zeros_like(x)], axis=-1)
    intervals = [0.2, 0.2]
    kwargs = dict(patch_sizes=[9, 9], cyclic_dimensions=[])
    if device is not None:
        kwargs["device"] = device

    derivatives = partial_derivatives_across_all_dims_batched(points, intervals, **kwargs)
    expected = np.broadcast_to(np.array([[1., 0.], [0., 1.], [0., 0.]]), (9, 9, 3, 2))
    np.testing.assert_allclose(derivatives, expected, atol=1e-12)
    np.testing.assert_allclose(volume_element(points, intervals, **kwargs), np.ones((9, 9)), atol=1e-12)
    np.testing.assert_allclose(scalar_curvature(points, intervals, **kwargs), np.zeros((5, 5)), atol=1e-12)


@pytest.mark.parametrize("device", [None, "cpu", torch.device("cpu")])
def test_circle_reach_on_cpu(device, monkeypatch):
    # If reach fails to forward explicit CPU, derivatives try to use CUDA.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device is not None)
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    points = 2 * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    kwargs = dict(patch_sizes=[10], cyclic_dimensions=[0], batch_size=4)
    if device is not None:
        kwargs["device"] = device
    reach = reach_per_point(points, [2 * np.pi], **kwargs)
    np.testing.assert_allclose(reach, np.full(24, 2.), atol=1e-12)
