"""Caller-owned models retain units, checkpoints and ordinary pipeline reuse."""

from pathlib import Path

import numpy as np
import pytest

from experiments_and_analysis.config import load_config, resolve
from experiments_and_analysis.models.interface import TrainingPaused
from experiments_and_analysis.outputs import identity
from experiments_and_analysis.scripts import run as pipeline
from experiments_and_analysis.scripts.sweep import expand


class IdentityAdapter:
    identifier = 'test/identity/v1'

    def __init__(self, pause=False):
        self.pause = pause
        self.fit_calls = 0
        self.seen = None

    def fit(self, data, output_dir, seed, device):
        self.fit_calls += 1
        self.seen = data.copy()
        np.save(output_dir / 'checkpoint.npy', data)
        if self.pause:
            raise TrainingPaused('Saved test checkpoint')
        return ['checkpoint.npy']

    def load(self, output_dir, device):
        self.seen = np.load(output_dir / 'checkpoint.npy')

    def representation_names(self):
        return ['input', 'code', 'output']

    def batches(self, data, representation, batch_size, seed):
        for start in range(0, len(data), batch_size):
            yield start, data[start:start + batch_size]


def fixture_config(tmp_path):
    values = np.arange(48, dtype=np.float32).reshape(1, 24, 2) + 10
    path = tmp_path / 'grid.npz'
    np.savez(path, data=values)
    return dict(
        dataset=dict(name='fixture', path=str(path), dimension=1, range_sizes=[1],
                     patch_sizes=[12], class_names=['fixture'], feature_ndim=1, normalize=False),
        split=dict(method='random', weighted=False, test_fraction=.25, seed=7),
        model=dict(name='external', adapter=IdentityAdapter.identifier),
        inference=dict(batch_size=2, persist=True),
        analyses=['errors'],
    )


def test_external_fit_preserves_units_and_reuses_complete_stages(tmp_path):
    config = fixture_config(tmp_path)
    adapter = IdentityAdapter()
    root = tmp_path / 'outputs'
    result = pipeline.run(config, root, model_adapter=adapter)
    assert adapter.seen is not None
    assert adapter.seen.dtype == np.float32
    assert adapter.seen.min() >= 10
    distances = np.load(Path(result['analyses'][0]['directory']) / 'pointwise_distances.npy')
    np.testing.assert_array_equal(distances, 0)
    second = pipeline.run(config, root, model_adapter=adapter)
    assert result == second
    assert adapter.fit_calls == 1


def test_external_pause_requires_explicit_resume_and_matching_identity(tmp_path):
    config = fixture_config(tmp_path)
    adapter = IdentityAdapter(pause=True)
    root = tmp_path / 'outputs'
    paused = pipeline.run(config, root, model_adapter=adapter)
    assert paused['status'] == 'paused'
    assert not list((root / 'trained-models').rglob('COMPLETED'))
    with pytest.raises((RuntimeError, FileExistsError), match='[Ii]ncomplete|[Pp]artial'):
        pipeline.run(config, root, model_adapter=adapter)
    adapter.pause = False
    result = pipeline.run(config, root, model_adapter=adapter, resume_training=True)
    assert result['model_id'] == paused['model_id']
    assert len(result['analyses']) == 1
    assert all(pipeline.is_complete(path) for path in (root / 'trained-models').iterdir())


def test_missing_adapter_is_rejected_before_output_creation(tmp_path):
    config = fixture_config(tmp_path)
    with pytest.raises(ValueError, match='model_adapter'):
        pipeline.run(config, tmp_path / 'outputs')
    assert not (tmp_path / 'outputs').exists()


@pytest.mark.parametrize('batches', [
    [(1, np.ones((3, 2), dtype=np.float32))],
    [(0, np.ones((2, 2), dtype=np.float32))],
    [(0, np.full((3, 2), np.nan, dtype=np.float32))],
    [(0, np.ones((2, 2), dtype=np.float32)), (2, np.ones((1, 1), dtype=np.float32))],
])
def test_external_inference_rejects_missing_reordered_or_nonfinite_samples(batches):
    class InvalidAdapter(IdentityAdapter):
        def batches(self, data, representation, batch_size, seed):
            yield from batches
    adapter = InvalidAdapter()
    config = resolve(dict(model=dict(name='external', adapter=adapter.identifier)))
    with pytest.raises(ValueError, match='External inference'):
        list(pipeline.predictions(config, adapter, np.ones((3, 2), dtype=np.float32), 'output'))


@pytest.mark.parametrize('figure, expected', [
    (3, '0debd744413f01585ca2'),
    (4, 'e82b6a824ace02490eb5'),
    (5, 'bd764ab626d89d45da29'),
    (6, '96736fa1881fd51856f4'),
])
def test_original_paper_configuration_expansion_is_unchanged(figure, expected):
    directory = Path(__file__).parents[1] / 'scripts/configs/paper'
    assert identity(expand(load_config(directory / f'figure{figure}.yaml'))) == expected


def test_normalization_switch_must_be_boolean():
    with pytest.raises(ValueError, match='normalize'):
        resolve(dict(dataset=dict(normalize='false')))


def test_single_class_layer_plot_keeps_undefined_sign_average_as_gap(tmp_path):
    from microscope.computations_grid.data_analysis.data_analysis import (
        AnalysisResults, Measures, compute_measure_aggregates_multiclass,
    )
    from experiments_and_analysis.plotting.layers import create_layer_plots
    field = {'fixture': np.ones((3, 3))}
    measures = Measures(field, {'fixture': 1.0}, field, field, field, {})
    with pytest.warns(RuntimeWarning):
        aggregates = compute_measure_aggregates_multiclass(measures, [1.0, 1.0])
    assert np.isnan(aggregates.normalized_total_curvatures_negative['fixture'])
    result = AnalysisResults(measures, aggregates, [1.0, 1.0], (slice(None),))
    with pytest.warns(RuntimeWarning):
        create_layer_plots({'input': result, 'latent': result},
                           dataset_type='fixture', output_dir=tmp_path, bottleneck='latent')
    assert (tmp_path / 'layers_fixture.png').is_file()
