"""Quick workload and resume in the new layout; old roots are not reused."""
from pathlib import Path
import numpy as np
import pytest

from experiments_and_analysis.config import load_config
from experiments_and_analysis.data.preparation import Dataset
from experiments_and_analysis.paths import experiment_paths
from experiments_and_analysis.scripts import run as single, sweep


def recipe():
    config = load_config(Path(__file__).parents[1] / 'scripts/configs/paper/figure4_quick.yaml')
    config['base']['split'].update(method='random', weighted=False)
    config['outputs'] = []
    return config


def fake_dataset():
    return Dataset(np.arange(4000, dtype=np.float32).reshape(1, 1000, 2, 2), [1], [], [10], ['a'])


def test_six_trainings_per_model_and_resume(tmp_path, monkeypatch):
    calls = []

    def fit(config, data, directory, split_id):
        calls.append(config)
        (directory / 'fit.txt').write_text('fit')
        return ['fit.txt']

    monkeypatch.setattr(single, 'load_dataset', lambda *args: fake_dataset())
    monkeypatch.setattr(single, 'fit_model', fit)
    monkeypatch.setattr(single, 'load_model', lambda *args: object())
    monkeypatch.setattr(single, 'analyze', lambda *args: [])
    sweep.run(recipe(), tmp_path)
    assert len(calls) == 12
    assert sum(c['model']['name'] == 'beta_vae' for c in calls) == 6
    assert all(c['training']['seed'] == 473959 for c in calls)
    before = {path: path.read_bytes() for path in (tmp_path / 'trained-models').rglob('*') if path.is_file()}
    sweep.run(recipe(), tmp_path)
    assert len(calls) == 12
    assert all(path.read_bytes() == content for path, content in before.items())


def test_incomplete_stage_preserved_and_blocks_export(tmp_path, monkeypatch):
    config = recipe()
    job = sweep.expand(config)[0]
    partial = experiment_paths(tmp_path, job, job['dataset'])['model']
    partial.mkdir(parents=True)
    artifact = partial / 'checkpoint'
    artifact.write_text('keep me')
    monkeypatch.setattr(single, 'load_dataset', lambda *args: fake_dataset())
    monkeypatch.setattr(single, 'fit_model', lambda *args: pytest.fail('Must not fit'))
    with pytest.raises(RuntimeError, match='Incomplete output preserved'):
        sweep.run(config, tmp_path)
    assert artifact.read_text() == 'keep me'
    assert not list((tmp_path / 'plots').glob('*'))
