"""Small real toy runs check bundled IO, equivalence, resume and file counts."""
from copy import deepcopy
import csv
from pathlib import Path
import pickle
import struct

import numpy as np
import pytest

from experiments_and_analysis import bundles
from experiments_and_analysis.config import load_config, resolve
from experiments_and_analysis.outputs import identity, read_json
from experiments_and_analysis.scripts import run as single, sweep


def toy_config():
    return dict(dataset=dict(name='Circle', dimension=1, sample_size=8, ground_truth_size=12),
                split=dict(method='provided', weighted=False),
                model=dict(name='mmls', parameters=dict(number_of_neighbors=4, verbose=False)),
                analyses=['errors'])


def small_sweep():
    base = toy_config()
    base['output_layout'] = 'bundled'
    return dict(base=base, grid={'dataset.sample_size': [8, 12, 16], 'repetition': [
        dict(index=0, seed=7), dict(index=1, seed=8)]})


def export(directory):
    with (directory / 'results.pkl').open('rb') as stream:
        return pickle.load(stream)


def stage(row, name):
    result, = row['stages'][name].values()
    return result


@pytest.mark.parametrize('persist', [False, True])
def test_real_bundled_run_matches_directory_artifacts(tmp_path, persist):
    config = toy_config()
    config['inference'] = dict(persist=persist, evaluation_size=7, buffer='temporary')
    config['model']['parameters']['export_points'] = True
    ordinary = single.run(config, tmp_path / 'ordinary')
    config['output_layout'] = 'bundled'
    compact = single.run(config, tmp_path / 'compact')
    assert compact['experiment_id'] == ordinary['experiment_id']
    assert compact['model_id'] == ordinary['model_id']
    assert compact['analyses'][0]['id'] == ordinary['analyses'][0]['id']
    directory, = (tmp_path / 'compact/experiment-bundles').iterdir()
    row, = export(directory)['experiments'].values()
    metrics = read_json(Path(ordinary['analyses'][0]['directory']) / 'metrics.json')
    assert stage(row, 'analysis-outputs')['metrics.json'] == metrics
    np.testing.assert_array_equal(stage(row, 'analysis-outputs')['pointwise_distances.npy'],
                                  np.load(Path(ordinary['analyses'][0]['directory']) / 'pointwise_distances.npy'))
    split, = (tmp_path / 'ordinary/splits').glob(f"*__{ordinary['split_id']}/indices.npz")
    with np.load(split) as arrays:
        for name in ('train', 'test'):
            np.testing.assert_array_equal(stage(row, 'splits')['indices.npz'][name], arrays[name])
    assert row['configuration'] == resolve(config)
    assert stage(row, 'trained-models')['projector.json']['split_id'] == compact['split_id']
    assert stage(row, 'trained-models')['training_points.npy'].shape == (8, 2)
    assert ('values.npy' in stage(row, 'inference-outputs')) == persist
    if persist:
        assert stage(row, 'inference-outputs')['values.npy'].shape == (7, 2)
    assert not list(directory.rglob('working.npy'))
    assert not any(p.is_dir() for p in directory.iterdir())


def test_sweep_has_one_bundle_per_trial_and_reuses_after_restart(tmp_path, monkeypatch):
    config = small_sweep()
    config['outputs'] = [dict(name='bounds', filename='tiny_bundle', panels=[dict(
        label='circle', dimension=1, where={'dataset.name': 'Circle'})])]
    first = sweep.run(config, tmp_path)
    assert len(first['results']) == 6
    directories = list((tmp_path / 'experiment-bundles').iterdir())
    assert len(directories) == 2
    for directory in directories:
        rows = export(directory)['experiments']
        assert len(rows) == 3
        assert {r['configuration']['dataset']['sample_size'] for r in rows.values()} == {8, 12, 16}
        assert {p.name for p in directory.iterdir()} == {
            'family.pkl', 'progress.pkl', 'results.pkl', 'summary.csv'}
        assert not any(p.is_dir() for p in directory.iterdir())
        with (directory / 'summary.csv').open() as stream:
            assert len(list(csv.DictReader(stream))) == 3
    assert len(list((tmp_path / 'splits').iterdir())) == 1  # Shared reference only.
    assert not (tmp_path / 'trained-models').exists()
    assert (Path(first['plots'][0]) / 'tiny_bundle.png').is_file()
    before = {p: p.read_bytes() for d in directories for p in d.iterdir()}
    bundles._bundles.clear()
    monkeypatch.setattr(single, 'load_dataset', lambda *_: pytest.fail('Completed jobs must not load data'))
    second = sweep.run(config, tmp_path)
    assert second['results'] == first['results']
    assert second['plots'] == first['plots']
    assert all(p.read_bytes() == data for p, data in before.items())


def test_interruption_rolls_back_only_current_experiment_and_exports_previous(tmp_path, monkeypatch):
    config = small_sweep()
    config['grid'].pop('repetition')
    original = single.predictions
    calls = []

    def fail(config, *args):
        n = config['dataset']['sample_size']
        calls.append(n)
        if n == 12:
            raise KeyboardInterrupt()
        yield from original(config, *args)

    with monkeypatch.context() as failing:
        failing.setattr(single, 'predictions', fail)
        with pytest.raises(KeyboardInterrupt):
            sweep.run(config, tmp_path)
    directory, = (tmp_path / 'experiment-bundles').iterdir()
    assert len(export(directory)['experiments']) == 1
    bundles._bundles.clear()
    result = sweep.run(config, tmp_path)
    assert len(result['results']) == 3
    assert calls == [8, 12]
    assert len(export(directory)['experiments']) == 3


def test_torn_journal_tail_and_missing_export_are_recovered(tmp_path):
    config = toy_config()
    config['output_layout'] = 'bundled'
    single.run(config, tmp_path)
    directory, = (tmp_path / 'experiment-bundles').iterdir()
    with (directory / 'progress.pkl').open('ab') as stream:
        stream.write(struct.pack('<Q', 1000) + b'x' * 32 + b'unfinished')
    (directory / 'results.pkl').unlink()
    bundles._bundles.clear()
    config['dataset']['sample_size'] = 12
    single.run(config, tmp_path)
    assert len(export(directory)['experiments']) == 2
    bundles._bundles.clear()
    assert single.completed_result(resolve(config), tmp_path) is not None


def test_committed_journal_corruption_is_reported(tmp_path):
    config = toy_config()
    config['output_layout'] = 'bundled'
    single.run(config, tmp_path)
    directory, = (tmp_path / 'experiment-bundles').iterdir()
    journal = directory / 'progress.pkl'
    data = bytearray(journal.read_bytes())
    data[45] ^= 1
    journal.write_bytes(data)
    bundles._bundles.clear()
    with pytest.raises(ValueError, match='Corrupt bundle journal'):
        single.completed_result(resolve(config), tmp_path)


def test_stale_export_after_hard_stop_is_refreshed_without_rerunning(tmp_path, monkeypatch):
    config = dict(toy_config(), output_layout='bundled')
    with bundles.bundle_session(tmp_path):
        single.run(config, tmp_path)
        directory, = (tmp_path / 'experiment-bundles').iterdir()
        bundles.export_bundle(directory)
        config['dataset']['sample_size'] = 12
        single.run(config, tmp_path)
        # Mimic process termination before the session's export finally block.
        bundles._bundles.pop(directory.resolve())
    assert len(export(directory)['experiments']) == 1
    monkeypatch.setattr(single, 'load_dataset', lambda *_: pytest.fail('Committed job must be reused'))
    single.run(config, tmp_path)
    assert len(export(directory)['experiments']) == 2


def test_skipped_job_is_saved_independently(tmp_path):
    config = small_sweep()
    config['grid'] = {'dataset.sample_size': [3, 8]}
    result = sweep.run(config, tmp_path)
    assert result['results'][0]['status'] == 'skipped'
    assert len(result['results'][1]['analyses']) == 1
    directory, = (tmp_path / 'experiment-bundles').iterdir()
    assert len(export(directory)['experiments']) == 2
    with (directory / 'summary.csv').open() as stream:
        assert [r['status'] for r in csv.DictReader(stream)] == ['skipped', 'completed']


def test_family_separates_seeds_parameters_and_reference_settings():
    first = resolve(dict(toy_config(), output_layout='bundled'))
    key = identity(bundles.bundle_family(first))
    changed = deepcopy(first)
    changed['dataset']['sample_size'] = 5000
    assert identity(bundles.bundle_family(changed)) == key
    for section, name, value in [('dataset', 'seed', 10), ('dataset', 'reference_seed', 10),
                                 ('training', 'trial', 9), ('model', 'parameters', {'number_of_neighbors': 8})]:
        changed = deepcopy(first)
        changed[section][name] = value
        assert identity(bundles.bundle_family(changed)) != key


@pytest.mark.parametrize('changes', [dict(model=dict(name='beta_vae')),
                                   dict(split=dict(method='random')), dict(analyses=[])])
def test_unsupported_bundled_workflows_fail_before_outputs(changes):
    config = dict(toy_config(), output_layout='bundled', **changes)
    with pytest.raises(ValueError, match='Bundled outputs require'):
        resolve(config)


def test_full_figure_five_has_forty_bundles_and_ordinary_image_jobs():
    path = Path(__file__).parents[1] / 'scripts/configs/paper/figure5.yaml'
    jobs = sweep.expand(load_config(path))
    compact = [c for c in jobs if c['output_layout'] == 'bundled']
    assert len(compact) == 40640 and len(jobs) - len(compact) == 65
    assert len({identity(bundles.bundle_family(c)) for c in compact}) == 40
