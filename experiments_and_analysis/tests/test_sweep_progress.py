"""Sweep display settings preserve execution, caching, and visible failures."""
import sys

import pytest
from tqdm import tqdm
from typer.testing import CliRunner

from experiments_and_analysis.outputs import read_json
from experiments_and_analysis.scripts import sweep


@pytest.fixture
def small_sweep(monkeypatch):
    config = dict(grid={'repetition': [dict(index=i, seed=i+1) for i in range(3)]},
                  outputs=[dict(name='bounds', filename='test_bounds')])
    calls = []

    def completed(experiment, root):
        print('cache lookup')
        if experiment['training']['seed'] == 1:
            return dict(analyses=[], status='cached')

    def load(*args):
        print('dataset loaded')
        return object()

    def run(experiment, root, **kwargs):
        seed = experiment['training']['seed']
        calls.append(seed)
        print('experiment stdout')
        print('experiment stderr', file=sys.stderr)
        for _ in tqdm(range(1), desc='Inner training'):
            pass
        return dict(analyses=[], status='skipped' if seed == 2 else 'finished')

    def render(*args):
        print('plot output')
        return 'plot'

    monkeypatch.setattr(sweep.single, 'completed_result', completed)
    monkeypatch.setattr(sweep.single, 'load_dataset', load)
    monkeypatch.setattr(sweep.single, 'run', run)
    monkeypatch.setattr(sweep, 'render', render)
    return config, calls


@pytest.mark.parametrize('configured,override,quiet', [
    (False, None, False), (True, None, True), (True, False, False), (False, True, True)])
def test_sweep_progress_counts_cached_skipped_and_executed_jobs(
        tmp_path, monkeypatch, capsys, small_sweep, configured, override, quiet):
    config, calls = small_sweep
    config['quiet_experiments'] = configured
    result = sweep.run(config, tmp_path, quiet_experiments=override)
    captured = capsys.readouterr()
    assert 'Sweep' in captured.err and '3/3' in captured.err
    for message in ('cache lookup', 'dataset loaded', 'experiment stdout', 'plot output'):
        assert (message in captured.out) == (not quiet)
    for message in ('experiment stderr', 'Inner training'):
        assert (message in captured.err) == (not quiet)
    assert calls == [2, 3]
    assert [r['status'] for r in result['results']] == ['cached', 'skipped', 'finished']


def test_quiet_setting_preserves_saved_jobs_and_sweep_identity(tmp_path, small_sweep):
    config, _ = small_sweep
    sweep.run(config, tmp_path, dry_run=True)
    directory, = tmp_path.glob('experiment-configurations/sweep__*')
    before = read_json(directory / 'jobs.json')
    config['quiet_experiments'] = True
    sweep.run(config, tmp_path, dry_run=True)
    assert list(tmp_path.glob('experiment-configurations/sweep__*')) == [directory]
    assert read_json(directory / 'jobs.json') == before


def test_quiet_failure_propagates_and_restores_streams(tmp_path, monkeypatch, capsys, small_sweep):
    config, _ = small_sweep
    stdout, stderr = sys.stdout, sys.stderr

    def fail(*args, **kwargs):
        print('hidden detail')
        raise RuntimeError('experiment failed')

    monkeypatch.setattr(sweep.single, 'run', fail)
    with pytest.raises(RuntimeError, match='experiment failed'):
        sweep.run(config, tmp_path, quiet_experiments=True)
    assert sys.stdout is stdout and sys.stderr is stderr
    print('visible after failure', file=sys.stderr)
    captured = capsys.readouterr()
    assert 'hidden detail' not in captured.out
    assert 'visible after failure' in captured.err
    assert '3/3' not in captured.err


@pytest.mark.parametrize('flag,value', [('--quiet-experiments', True),
                                       ('--no-quiet-experiments', False), (None, None)])
def test_sweep_cli_quiet_override(tmp_path, monkeypatch, flag, value):
    calls = []
    monkeypatch.setattr(sweep, 'run', lambda *args, **kwargs: calls.append(kwargs))
    path = tmp_path / 'config.json'
    path.write_text('{"quiet_experiments": true}')
    args = ['--config', str(path), '--output-dir', str(tmp_path)]
    if flag:
        args.append(flag)
    result = CliRunner().invoke(sweep.app, args)
    assert result.exit_code == 0, result.output
    assert calls[0]['quiet_experiments'] is value
