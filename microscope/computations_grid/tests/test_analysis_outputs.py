"""Small output smoke tests: no real datasets, inference runs or pixel snapshots."""

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from microscope.computations_grid.data_analysis.data_analysis import (
    AnalysisResults,
    Measures,
    compute_measure_aggregates_multiclass,
    plot_3d_pca_projections_multiclass,
)


@pytest.fixture
def results():
    rng = np.random.default_rng(17)
    measures = Measures(
        volume_elements={name: rng.uniform(1, 2, 16) for name in ["a", "b"]},
        total_volumes={"a": 2., "b": 3.},
        normalized_volume_elements={name: rng.uniform(1, 2, 16) for name in ["a", "b"]},
        normalized_curvatures={name: np.linspace(-1, 1, 16) for name in ["a", "b"]},
        normalized_reaches={name: np.arange(16, dtype=float) + 1 for name in ["a", "b"]},
        normalized_distances={("a", "b"): rng.uniform(2, 3, 16)},
    )
    return AnalysisResults(measures, compute_measure_aggregates_multiclass(measures, [2.]),
                           [2.], (slice(None), slice(3, -3)))


@pytest.mark.parametrize("export_plots", [False, True])
def test_run_single_output_saves_raw_results_and_returns_clipped_aggregates(
    tmp_path, monkeypatch, results, export_plots
):
    # The runner imports optional CLI dependencies; geometry itself is tested elsewhere.
    pytest.importorskip("typer")
    from microscope.computations_grid.data_analysis import run_data_analysis as runner

    data = np.arange(2*22*3, dtype=float).reshape(2, 22, 3)
    calls = []

    def compute(**kwargs):
        assert kwargs["data"] is data
        assert kwargs["class_names"] == ["a", "b"]
        assert kwargs["normalize_for_volume"] is True
        assert kwargs["reach_subsample"] == 2
        assert kwargs["range_sizes"] == [2.]
        assert kwargs["cyclic_dimensions"] == []
        assert kwargs["patch_sizes"] == [9]
        return results

    def pairs(trimmed, **kwargs):
        np.testing.assert_array_equal(trimmed, data[:, 3:-3])
        assert kwargs["analysis_results"].measure_aggregates.normalized_min_reaches == pytest.approx({"a": 1.15, "b": 1.15})
        calls.append("pairs")

    def projections(trimmed, clipped, **kwargs):
        np.testing.assert_array_equal(trimmed, data[:, 3:-3])
        assert clipped.trim_slices_data == results.trim_slices_data
        assert kwargs["show"] is False
        calls.append("projections")

    monkeypatch.setattr(runner, "compute_measures_multiclass", compute)
    monkeypatch.setattr(runner, "plot_pairs_multiclass", pairs)
    monkeypatch.setattr(runner, "plot_3d_pca_projections_multiclass", projections)
    aggregates = runner.run_single_output(
        "checkpoint__layer", data, [2.], [], [9], ["a", "b"],
        tmp_path, tmp_path, tmp_path, tmp_path, normalize_for_volume=True,
        export_plots=export_plots,
    )
    assert calls == (["pairs", "projections"] if export_plots else [])
    assert aggregates.normalized_min_reaches == pytest.approx({"a": 1.15, "b": 1.15})
    with (tmp_path / "checkpoint__layer.pkl").open("rb") as stream:
        saved = pickle.load(stream)
    np.testing.assert_array_equal(saved.measures.normalized_reaches["a"], np.arange(16.) + 1)
    assert saved.measure_aggregates.normalized_min_reaches == {"a": 1., "b": 1.}


def test_pca_projection_exports_images_and_aligns_sampled_values(tmp_path, monkeypatch, results):
    from microscope.computations_grid.data_analysis import data_analysis as analysis

    data = np.random.default_rng(23).normal(size=(2, 16, 3))
    captured = []
    original = analysis._plot_with_scalar_values

    def check_alignment(title, projection, scalar_values, **kwargs):
        assert projection.shape == (16, 3)
        # With all samples selected, the same permutation must apply to every measure.
        reach = scalar_values["Local reach"]
        indices = reach.astype(int) - 1
        # Three PCA components preserve distances in this three-feature fixture.
        class_data = data[0 if title == "a" else 1][indices]
        np.testing.assert_allclose(
            np.linalg.norm(projection[:, None] - projection[None], axis=-1),
            np.linalg.norm(class_data[:, None] - class_data[None], axis=-1),
            atol=1e-12,
        )
        np.testing.assert_array_equal(scalar_values["Volume element"],
                                      results.measures.normalized_volume_elements[title][indices])
        np.testing.assert_array_equal(scalar_values["Scalar curvature"],
                                      results.measures.normalized_curvatures[title][indices])
        np.testing.assert_array_equal(scalar_values["Absolute scalar curvature"],
                                      np.abs(results.measures.normalized_curvatures[title][indices]))
        captured.append(title)
        original(title, projection, scalar_values, **kwargs)

    monkeypatch.setattr(analysis, "_plot_with_scalar_values", check_alignment)
    try:
        plot_3d_pca_projections_multiclass(data, results, tmp_path, tmp_path, "case",
                                          show=False, dpi=20, n_samples_for_plots=100)
        assert captured == ["a", "b"]
        for filename in ["case_a.png", "case_b.png", "case.png"]:
            with Image.open(tmp_path / filename) as image:
                assert min(image.size) > 0
                image.verify()
    finally:
        plt.close("all")


def test_merge_layer_evolution_exports_both_scales(tmp_path):
    pytest.importorskip("typer")
    from microscope.computations_grid.data_analysis.merge_analysis_outputs import merge_layer_evolution

    for scale in ["linear", "symlog"]:
        directory = tmp_path / "layer_evolution" / scale
        directory.mkdir(parents=True)
        for step, value in [("random", 0), ("50k", 255)]:
            Image.fromarray(np.full((4, 6, 3), value, dtype=np.uint8)).save(directory / f"epoch_{step}.png")
    output = tmp_path / "merged"
    output.mkdir()
    try:
        merge_layer_evolution(tmp_path, ["random", "50k"], output, dpi=20, figsize=(2, 2))
        for scale in ["linear", "symlog"]:
            with Image.open(output / f"layer_evolution_{scale}.png") as image:
                image.verify()
    finally:
        plt.close("all")
