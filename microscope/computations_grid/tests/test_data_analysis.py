"""Hand-calculated aggregates and small CPU integration tests for grid analysis."""

import copy

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from microscope.computations_grid.data_analysis.data_analysis import (
    AnalysisResults,
    Measures,
    clip_measure_multiclass,
    clip_results_multiclass,
    compute_measure_aggregates_multiclass,
    compute_measures_multiclass,
)


@pytest.fixture
def measures():
    return Measures(
        volume_elements={"a": np.array([1., 3.]), "b": np.array([2., 4.])},
        total_volumes={"a": 99., "b": 101.},
        normalized_volume_elements={"a": np.array([1., 3.]), "b": np.array([2., 4.])},
        normalized_curvatures={"a": np.array([2., -1.]), "b": np.array([-2., 4.])},
        normalized_reaches={"a": np.array([5., 7.]), "b": np.array([11., 13.])},
        normalized_distances={("a", "b"): np.array([2., 6.])},
    )


def test_measure_aggregates_hand_calculated(measures):
    result = compute_measure_aggregates_multiclass(measures, [2., 3.])
    assert result.total_volumes == {"a": 99., "b": 101.}
    assert result.normalized_total_volumes == {"a": 12., "b": 18.}
    assert result.normalized_total_curvatures == {"a": -3., "b": 36.}
    assert result.normalized_min_reaches == {"a": 5., "b": 11.}
    assert result.normalized_total_reaches == {"a": 78., "b": 222.}
    assert result.normalized_total_distances == {("a", "b"): 72.}
    # This API reports the grid mean of distance*mean(volume densities).
    assert result.normalized_average_distances == {("a", "b"): 12.}


@pytest.mark.parametrize("curvatures,expected_parts", [
    ([2., -1., 0., 0.], [.25, -.375]),
    ([2., 1., 3., 4.], [2.5, 0.]),
    ([-2., -1., -3., -4.], [0., -2.5]),
], ids=["mixed-with-zero-region", "all-positive", "all-negative"])
def test_curvature_parts_on_unit_volume_manifold(curvatures, expected_parts):
    # Four equally spaced parameter-grid points with geometric volume weights
    # [1/8, 3/8, 1/8, 3/8]. The whole manifold has volume 1.
    density = np.array([[.5, 1.5], [.5, 1.5]])
    measures = Measures(
        volume_elements={"a": density.copy()},
        total_volumes={"a": 1.},
        normalized_volume_elements={"a": density.copy()},
        normalized_curvatures={"a": np.array(curvatures).reshape(2, 2)},
        normalized_reaches={"a": np.ones((2, 2))},
        normalized_distances={},
    )
    result = compute_measure_aggregates_multiclass(measures, [1., 1.])
    assert result.normalized_total_volumes["a"] == pytest.approx(1.)
    positive = result.normalized_total_curvatures_positive["a"]
    negative = result.normalized_total_curvatures_negative["a"]
    # Non-contributing regions retain their volume weight and contribute zero.
    np.testing.assert_allclose([positive, negative], expected_parts, rtol=0, atol=1e-12)
    assert positive + negative == pytest.approx(result.normalized_total_curvatures["a"])
    # The API stores the negative contribution with its sign: sum magnitudes.
    expected_absolute = expected_parts[0] - expected_parts[1]
    assert positive + abs(negative) == pytest.approx(expected_absolute)


@pytest.mark.parametrize("margin,expected", [
    (0., [[0., 1.], [2., 100.]]),
    (.25, [[.75, 1.], [2., 26.5]]),
])
def test_clip_measure_quantiles_shape_keys_and_no_mutation(margin, expected):
    values = np.array([[0., 1.], [2., 100.]])
    original = values.copy()
    measure = {"a": values, ("a", "b"): values.copy()}
    result = clip_measure_multiclass(measure, margin)
    assert result.keys() == measure.keys()
    for key in result:
        np.testing.assert_array_equal(result[key], expected)
        assert result[key].shape == values.shape
        np.testing.assert_array_equal(measure[key], original)


def test_clip_results_preserves_metadata_and_recomputes_aggregates(measures):
    # Two-point quantiles at .25 are the endpoints moved 25% toward each other.
    results = AnalysisResults(measures, compute_measure_aggregates_multiclass(measures, [2., 3.]),
                              [2., 3.], (slice(None), slice(3, -3)))
    original = copy.deepcopy(results)
    clipped = clip_results_multiclass(results, .25)
    assert clipped is not results
    assert clipped.range_sizes == results.range_sizes
    assert clipped.trim_slices_data == results.trim_slices_data
    assert clipped.measures.total_volumes == {"a": 99., "b": 101.}
    expected = {
        "volume_elements": {"a": [1.5, 2.5], "b": [2.5, 3.5]},
        "normalized_volume_elements": {"a": [1.5, 2.5], "b": [2.5, 3.5]},
        "normalized_curvatures": {"a": [1.25, -.25], "b": [-.5, 2.5]},
        "normalized_reaches": {"a": [5.5, 6.5], "b": [11.5, 12.5]},
        "normalized_distances": {("a", "b"): [3., 5.]},
    }
    for field, arrays in expected.items():
        for key, values in arrays.items():
            np.testing.assert_array_equal(getattr(clipped.measures, field)[key], values)
            np.testing.assert_array_equal(getattr(results.measures, field)[key],
                                          getattr(original.measures, field)[key])
    # Independent values distinguish recomputation from copying old aggregates.
    assert clipped.measure_aggregates.normalized_total_curvatures == {"a": 3.75, "b": 22.5}
    assert clipped.measure_aggregates.normalized_min_reaches == {"a": 5.5, "b": 11.5}
    assert clipped.measure_aggregates.normalized_total_reaches == {"a": 73.5, "b": 217.5}
    assert clipped.measure_aggregates.normalized_total_distances == {("a", "b"): 63.}
    assert clipped.measure_aggregates.normalized_average_distances == {("a", "b"): 10.5}


def translated_spheres(n_latitude):
    latitude = np.linspace(-.6, .6, n_latitude)
    longitude = np.linspace(0, 2*np.pi, 16, endpoint=False)
    u, v = np.meshgrid(latitude, longitude, indexing="ij")
    points = 2*np.stack([np.cos(u)*np.cos(v), np.cos(u)*np.sin(v), np.sin(u)], axis=-1)
    return latitude, np.stack([points, points + np.array([0., 0., 5.])])


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("n_latitude,subsample", [(13, None), (14, 2), (13, 2)])
def test_multiclass_normalization_alignment_and_scaling(normalize, n_latitude, subsample):
    latitude, data = translated_spheres(n_latitude)
    original = data.copy()
    ranges = [1.2, 2*np.pi]
    results = compute_measures_multiclass(data, ["a", "b"], ranges, [1], [9, 10],
                                        normalize_for_volume=normalize, reach_subsample=subsample,
                                        reach_batch_size=7, device="cpu")
    np.testing.assert_array_equal(data, original)
    # Exact central differences of the trig parametrization include sinc factors.
    h = latitude[1] - latitude[0]
    density = 4 * (np.sin(h)/h) * (np.sin(2*np.pi/16)/(2*np.pi/16)) * np.cos(latitude[1:-1])
    expected_volume = density.mean() * np.prod(ranges)
    scale = 1/np.sqrt(2*expected_volume) if normalize else 1.
    shape = (n_latitude-6, 16)
    m = results.measures
    for name in ["a", "b"]:
        assert m.total_volumes[name] == pytest.approx(expected_volume, rel=1e-12)
        np.testing.assert_allclose(m.volume_elements[name], np.broadcast_to(density[:, None], (n_latitude-2, 16)))
        for field in [m.normalized_volume_elements, m.normalized_curvatures, m.normalized_reaches]:
            assert field[name].shape == shape
            assert np.isfinite(field[name]).all()
        np.testing.assert_allclose(m.normalized_volume_elements[name],
                                   np.broadcast_to((density[2:-2]*scale**2)[:, None], shape), rtol=1e-11)
        np.testing.assert_allclose(m.normalized_reaches[name], np.full(shape, 2*scale), rtol=1e-10)
        # Coarse longitude spacing: allow finite-difference error, not shape errors.
        np.testing.assert_allclose(m.normalized_curvatures[name]*scale**2, np.full(shape, .5), atol=.04, rtol=0)
    trimmed = data[results.trim_slices_data]
    np.testing.assert_array_equal(trimmed, data[:, 3:-3])
    assert m.normalized_distances.keys() == {("a", "b")}
    np.testing.assert_allclose(m.normalized_distances[("a", "b")], np.full(shape, 5*scale), rtol=1e-12)
    # The retained interior has a different density average from the initial grid.
    expected_retained_volume = (density[2:-2]*scale**2).mean()*np.prod(ranges)
    assert results.measure_aggregates.normalized_total_volumes == pytest.approx(
        {"a": expected_retained_volume, "b": expected_retained_volume})


@pytest.mark.parametrize("class_names", [["a"], ["a", "a"]])
def test_multiclass_rejects_mismatched_or_duplicate_class_names(class_names):
    _, data = translated_spheres(13)
    with pytest.raises(ValueError, match="class|name"):
        compute_measures_multiclass(data, class_names, [1.2, 2*np.pi], [1], [9, 10],
                                   normalize_for_volume=False, device="cpu")


def test_multiclass_rejects_mismatched_dimension_metadata():
    _, data = translated_spheres(13)
    with pytest.raises(ValueError):
        compute_measures_multiclass(data, ["a", "b"], [1.2], [1], [9, 10],
                                   normalize_for_volume=False, device="cpu")
