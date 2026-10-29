"""Metric-first stitching, conditioning, and bounded embedding work."""
import numpy as np
import pytest
import torch

from microscope.computations_grid import curvature
from microscope.computations_grid import _curvature_geometry as geometry
from microscope.patches import _iter_patches_with_halo


def immersion(shape, cyclic, dtype):
    coordinates = [np.arange(n)*(2*np.pi/n if a in cyclic else .2)
                   for a, n in enumerate(shape)]
    grid = np.meshgrid(*coordinates, indexing="ij")
    features = []
    for a, coordinate in enumerate(grid):
        features.extend([np.cos(coordinate), np.sin(coordinate)]
                        if a in cyclic else [coordinate])
    features.append(.1*sum(np.sin(x)**2 for x in grid))
    return np.stack(features, axis=-1).astype(dtype), [x[1]-x[0] for x in coordinates]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape,cyclic,patches", [
    ((11,), [], [8]),
    ((3,), [0], [10]),
    ((11, 12), [], [8, 9]),
    ((10, 11), [1], [9, 8]),
    ((3, 4), [1, 0], [10, 11]),
    ((9, 8, 10), [0, 2], [8, 7, 9]),
    ((8, 7, 8, 9), [3], [7, 8, 9, 10]),
])
def test_metric_first_matches_embedding_tiles(dtype, shape, cyclic, patches):
    points, intervals = immersion(shape, cyclic, dtype)
    for normalize in ([False, True] if len(shape) > 1 else [False]):
        previous = curvature._scalar_curvature_in_patches(
            features_on_grid=points,
            difference_intervals=intervals,
            cyclic_dimensions=cyclic,
            patch_sizes=patches,
            normalize=normalize,
            device="cpu"
        )
        result = curvature.scalar_curvature(
            features_on_grid=points,
            difference_intervals=intervals,
            cyclic_dimensions=cyclic,
            patch_sizes=patches,
            normalize=normalize,
            device="cpu"
        )
        assert result.dtype == dtype
        assert result.shape == tuple(n if a in cyclic else n-6 for a, n in enumerate(shape))
        np.testing.assert_allclose(result, previous, atol=3e-5 if dtype == np.float32 else 1e-12,
                                   rtol=3e-5 if dtype == np.float32 else 1e-11)


def test_each_metric_point_is_computed_once_without_global_image_padding(monkeypatch):
    shape, cyclic, patches = (11, 12, 9), [0, 2], [8, 9, 7]
    points, intervals = immersion(shape, cyclic, np.float64)
    # More features than any metric tensor axis, allowing derivative calls to
    # distinguish image tiles from the subsequent metric/Christoffel fields.
    points = np.pad(points, [(0, 0)]*3+[(0, 29)], mode="constant")
    original = curvature.partial_derivatives_across_all_dims
    work = []
    def record(tensor, *args, **kwargs):
        if tensor.ndim == 4:
            work.append(tensor.shape[:-1])
        return original(tensor, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("Embedding boundaries must be wrapped within individual tiles.")
    monkeypatch.setattr(curvature, "partial_derivatives_across_all_dims", record)
    monkeypatch.setattr(np, "pad", forbidden)
    result = curvature.scalar_curvature(points, intervals, cyclic, patches, device="cpu")
    assert result.shape == (11, 6, 9)
    assert sum(np.prod(np.asarray(s)-2) for s in work) == 11*10*9
    assert all(all(n <= p for n, p in zip(s, patches)) for s in work)


@pytest.mark.parametrize("dtype,eps", [(np.float32, 1e-5), (np.float64, 1e-8)])
def test_tiled_metric_retains_qr_inverse(dtype, eps, monkeypatch):
    u, v = np.meshgrid(np.arange(11.), np.arange(10.), indexing="ij")
    points = np.stack([u+v, eps*v, np.zeros_like(u)], axis=-1).astype(dtype)
    original = geometry.tangent_geometry
    selected = []
    def record(tangent, **kwargs):
        selected.append(tangent.shape)
        return original(tangent, **kwargs)
    monkeypatch.setattr(geometry, "tangent_geometry", record)
    result = curvature.scalar_curvature(points, [1., 1.], [], [8, 9], device="cpu")
    assert selected
    np.testing.assert_allclose(result, 0., atol=1e-10)


@pytest.mark.parametrize("reverse", [False, True])
def test_metric_first_strided_views_match_contiguous(reverse):
    points, intervals = immersion((19, 21), [1], np.float64)
    view = points[::2, ::-1 if reverse else 1]
    result = curvature.scalar_curvature(
        features_on_grid=view,
        difference_intervals=[2*intervals[0], intervals[1]],
        cyclic_dimensions=[1],
        patch_sizes=[8, 9],
        device="cpu"
    )
    contiguous_result = curvature.scalar_curvature(
        features_on_grid=np.ascontiguousarray(view),
        difference_intervals=[2*intervals[0], intervals[1]],
        cyclic_dimensions=[1],
        patch_sizes=[8, 9],
        device="cpu"
    )
    np.testing.assert_array_equal(result, contiguous_result)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_metric_first_rejects_nonfinite_original_border(bad):
    points, intervals = immersion((10, 11), [1], np.float64)
    points[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        curvature.scalar_curvature(points, intervals, [1], [8, 9], device="cpu")


def test_metric_first_rejects_rank_loss():
    points = np.zeros((9, 9, 3), dtype=np.float64)
    with pytest.raises(torch.linalg.LinAlgError, match="rank.*grid"):
        curvature.scalar_curvature(points, [1., 1.], [], [8, 8], device="cpu")


@pytest.mark.parametrize("shape,cyclic,halo,patches", [
    ((3, 4), [0, 1], 3, [10, 11]),
    ((19, 21), [0, 1], 2, [8, 9]),
    ((11, 12, 13), [0, 2], 1, [7, 8, 9]),
])
def test_halo_tiles_wrap_corners_and_cover_each_output_once(shape, cyclic, halo, patches, monkeypatch):
    array = np.arange(np.prod(shape)*6).reshape(*shape, 2, 3)
    # Sequential full-axis gathers can create huge intermediate image copies.
    def forbidden(*args, **kwargs):
        raise AssertionError("Periodic corners need a single tile-sized gather.")
    monkeypatch.setattr(np, "take", forbidden)
    output_shape = tuple(n if a in cyclic else n-2*halo for a, n in enumerate(shape))
    visits = np.zeros(output_shape, dtype=int)
    for destination, tile in _iter_patches_with_halo(array, patches, halo, cyclic):
        indices = [np.arange(s.start, s.stop+2*halo)-(halo if a in cyclic else 0)
                   for a, s in enumerate(destination)]
        reference = array[np.ix_(*[x % n for x, n in zip(indices, shape)])]
        np.testing.assert_array_equal(tile, reference)
        assert all(n <= p for n, p in zip(tile.shape, patches))
        visits[destination] += 1
    np.testing.assert_array_equal(visits, 1)
