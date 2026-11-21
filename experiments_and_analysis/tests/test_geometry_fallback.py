"""Geometry precision recovery preserves original samples, caches and model fits."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from experiments_and_analysis.analyses import geometry
from experiments_and_analysis.data.preparation import Dataset
from experiments_and_analysis.outputs import read_json
from experiments_and_analysis.scripts import run as pipeline
from experiments_and_analysis.tests.test_external_models import IdentityAdapter


@pytest.fixture
def thin_grid():
    u, v = np.meshgrid(np.linspace(-1, 1, 9, dtype=np.float32),
                       np.linspace(-1, 1, 9, dtype=np.float32), indexing='ij')
    grid = np.stack([u + v, u + v + 1e-4 * u, 1e-4 * (u*u + v*v)], -1)[None]
    return Dataset(grid, [2, 2], [], [9, 9], ['thin'], feature_ndim=1)


def test_actual_nonfinite_normalization_retries_original_grid_in_float64(thin_grid):
    original = thin_grid.flat.copy()
    with pytest.raises(ValueError, match='only finite values'):
        geometry.compute(original, thin_grid)
    result = geometry.compute(original, thin_grid, fallback_to_float64_on_nan_measures=True)
    np.testing.assert_array_equal(original, thin_grid.flat)
    assert original.dtype == np.float32
    assert result.measures.normalized_curvatures['thin'].dtype == np.float64
    assert np.isfinite(result.measures.normalized_reaches['thin']).all()
    # An independent SVD avoids forming the ill-conditioned Gram determinant.
    grid = original.reshape(9, 9, 3).astype(np.float64)
    tangent = np.stack([(grid[2:, 1:-1] - grid[:-2, 1:-1]) / .5,
                        (grid[1:-1, 2:] - grid[1:-1, :-2]) / .5], -1)
    expected = np.linalg.svd(tangent, compute_uv=False).prod(-1).mean() * 4
    np.testing.assert_allclose(result.measures.total_volumes['thin'], expected, rtol=2e-7)
    # A missing sign subset historically yields an aggregate NaN, not a bad measure grid.
    assert np.isnan(result.measure_aggregates.normalized_total_curvatures_negative['thin'])


@pytest.mark.parametrize('bad_value', [np.nan, np.inf])
@pytest.mark.parametrize('aggregate', [False, True])
def test_nonfinite_returned_measures_trigger_one_retry(thin_grid, monkeypatch, bad_value, aggregate):
    reference = geometry.compute(thin_grid.flat.astype(np.float64), thin_grid)
    calls = []

    def backend(values, *args, **kwargs):
        calls.append(values.copy())
        result = deepcopy(reference)
        if values.dtype == np.float32:
            if aggregate:
                result.measure_aggregates.normalized_total_curvatures['thin'] = bad_value
            else:
                result.measures.normalized_curvatures['thin'].flat[0] = bad_value
        return result

    monkeypatch.setattr(geometry, 'compute_measures_multiclass', backend)
    result = geometry.compute(thin_grid.flat, thin_grid, fallback_to_float64_on_nan_measures=True)
    assert [x.dtype for x in calls] == [np.float32, np.float64]
    np.testing.assert_array_equal(calls[1], calls[0].astype(np.float64))
    assert np.isfinite(result.measures.normalized_curvatures['thin']).all()


@pytest.mark.parametrize('dtype, enabled, count', [(np.float32, True, 2), (np.float32, False, 1),
                                                 (np.float64, True, 1)])
def test_persistent_nonfinite_failure_is_not_retried_indefinitely(thin_grid, monkeypatch, dtype, enabled, count):
    calls = []

    def backend(values, *args, **kwargs):
        calls.append(values.dtype)
        raise FloatingPointError('Nonfinite finite-difference tangent vectors.')

    monkeypatch.setattr(geometry, 'compute_measures_multiclass', backend)
    with pytest.raises(FloatingPointError, match='Nonfinite'):
        geometry.compute(thin_grid.flat.astype(dtype), thin_grid, fallback_to_float64_on_nan_measures=enabled)
    assert len(calls) == count


def test_unrelated_validation_error_is_not_retried(thin_grid, monkeypatch):
    calls = []

    def backend(values, *args, **kwargs):
        calls.append(values.dtype)
        raise ValueError('Invalid patch sizes.')

    monkeypatch.setattr(geometry, 'compute_measures_multiclass', backend)
    with pytest.raises(ValueError, match='Invalid patch sizes'):
        geometry.compute(thin_grid.flat, thin_grid, fallback_to_float64_on_nan_measures=True)
    assert calls == [np.float32]


def test_recovery_keeps_fit_and_complete_analysis_then_reuses_float64_result(tmp_path, thin_grid, monkeypatch):
    source = tmp_path / 'source.npz'
    np.savez(source, data=thin_grid.grid)
    config: dict = dict(
        dataset=dict(name='thin', path=str(source), dimension=2, range_sizes=[2, 2], patch_sizes=[9, 9],
                     class_names=['thin'], feature_ndim=1, normalize=False),
        split=dict(method='random', weighted=False, test_fraction=.25, seed=7),
        model=dict(name='external', adapter=IdentityAdapter.identifier),
        inference=dict(batch_size=16, persist=True), analyses=['errors'], device='cpu',
    )
    root = tmp_path / 'outputs'
    adapter = IdentityAdapter()
    errors = pipeline.run(config, root, model_adapter=adapter)
    error_path = Path(errors['analyses'][0]['directory']) / 'metrics.json'
    error_time = error_path.stat().st_mtime_ns
    config['analyses'].append(dict(name='geometry', representations=['code']))

    def no_training(*args, **kwargs):
        pytest.fail('Recovery must reuse the completed model.')

    monkeypatch.setattr(pipeline, 'fit_model', no_training)
    recovered = pipeline.run(config, root, model_adapter=adapter, fallback_to_float64_on_nan_measures=True)
    assert recovered['model_id'] == errors['model_id']
    assert error_path.stat().st_mtime_ns == error_time
    assert adapter.fit_calls == 1
    directory = Path(recovered['analyses'][1]['directory'])
    assert read_json(directory / 'precision.json') == dict(
        input_dtype='float32', computation_dtype='float64',
        fallback_to_float64_on_nan_measures=True, fallback_used=True,
    )
    timestamp = (directory / 'measures.pkl').stat().st_mtime_ns
    reused = pipeline.run(config, root, model_adapter=adapter)
    assert reused == recovered
    assert (directory / 'measures.pkl').stat().st_mtime_ns == timestamp


def test_successful_float32_geometry_stays_float32(tmp_path):
    from experiments_and_analysis.tests.test_external_models import fixture_config

    config: dict = fixture_config(tmp_path)
    config['analyses'] = [dict(name='geometry', representations=['input'], normalize_for_volume=False)]
    # A curved one-dimensional grid has finite reach and zero scalar curvature.
    x = np.linspace(-1, 1, 24, dtype=np.float32)
    np.savez(config['dataset']['path'], data=np.stack([x, x*x], -1)[None])
    result = pipeline.run(config, tmp_path / 'outputs', model_adapter=IdentityAdapter(),
                          fallback_to_float64_on_nan_measures=True)
    precision = read_json(Path(result['analyses'][0]['directory']) / 'precision.json')
    assert precision['computation_dtype'] == 'float32'
    assert precision['fallback_used'] is False
