"""Reference cases for FPS and retained fixed splits; not yet executed."""
import numpy as np
import pytest

from microscope.datasets.dataset_split import maximal_distance_selection_torch
from microscope.datasets import generic_dataset_loader as loader


def test_farthest_points_on_a_line():
    points = np.array([[0.], [1.], [4.], [10.]], dtype=np.float32)
    selected, indices = maximal_distance_selection_torch(points, 3, device="cpu")
    np.testing.assert_array_equal(indices, [0, 3, 2])
    np.testing.assert_array_equal(selected[:, 0], [0., 10., 4.])


def test_duplicate_points_still_select_distinct_indices():
    points = np.zeros((4, 2), dtype=np.float32)
    _, indices = maximal_distance_selection_torch(points, 4, device="cpu")
    assert set(indices) == set(range(4))


def test_empty_selection():
    selected, indices = maximal_distance_selection_torch(np.empty((0, 2), np.float32), 0, device="cpu")
    assert selected.shape == (0, 2)
    assert indices.shape == (0,)


@pytest.mark.parametrize("size", [-1, 5])
def test_invalid_selection_size(size):
    with pytest.raises(ValueError):
        maximal_distance_selection_torch(np.zeros((4, 2), np.float32), size, device="cpu")


def test_split_forwards_device_and_preserves_conflicting_cache(tmp_path, monkeypatch):
    data = np.arange(20, dtype=np.float32).reshape(1, 20, 1, 1)
    target = np.zeros((1, 20), dtype=np.int64)
    slices = (slice(None),) * 4
    monkeypatch.setattr(loader, "load_dataset_no_split", lambda *args, **kwargs:
                        (data, target, [8], [1.], slices, slices, [1.], []))
    seen = []

    def select(points, size, *, device, verbose):
        seen.append(device)
        return points[:size], np.arange(size)

    monkeypatch.setattr(loader, "maximal_distance_selection_torch", select)
    export = loader.export_fixed_grid_test_set_and_rest_for_train
    export(loader.DatasetName.dsprites, 1, tmp_path, max_test_size=3, device="cpu")
    assert seen == ["cpu"]
    path = tmp_path / "dsprites__1.npz"
    with np.load(path, allow_pickle=True) as split:
        assert len(split["data_train"]) == 17
        assert len(split["data_test"]) == 3
        assert not np.any(split["train_idx"] & split["test_idx"])
        np.testing.assert_allclose(split["weights_test"], [1 / 3] * 3)
    original = path.read_bytes()
    export(loader.DatasetName.dsprites, 1, tmp_path, max_test_size=3, device="cpu")
    assert seen == ["cpu"]  # The matching cache was reused.
    with pytest.raises(ValueError, match="new output directory"):
        export(loader.DatasetName.dsprites, 1, tmp_path, max_test_size=2, device="cpu")
    assert path.read_bytes() == original
