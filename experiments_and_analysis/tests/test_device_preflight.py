"""Unavailable devices fail before datasets are loaded or stage files are written."""
from pathlib import Path

import pytest
import torch

from experiments_and_analysis.config import load_config
from experiments_and_analysis.scripts import run as single
from experiments_and_analysis.scripts import sweep


@pytest.mark.parametrize('runner', ['single', 'sweep'])
def test_cuda_failure_precedes_dataset_loading_and_output_creation(tmp_path, monkeypatch, runner):
    def fail_cuda(*args, **kwargs):
        raise RuntimeError('CUDA unknown error')

    monkeypatch.setattr(torch, 'empty', fail_cuda)
    monkeypatch.setattr(single, 'load_dataset', lambda *_: pytest.fail('Must not load the dataset'))
    config = dict(device='cuda:0')
    output = tmp_path / 'outputs'
    with pytest.raises(RuntimeError, match='CUDA unknown error'):
        if runner == 'single':
            single.run(config, output)
        else:
            sweep.run(dict(base=config), output)
    assert not output.exists()


def test_sweep_dry_run_never_initializes_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, 'empty', lambda *_args, **_kwargs: pytest.fail('Dry run must not use CUDA'))
    result = sweep.run(dict(base=dict(device='cuda:0')), tmp_path / 'outputs', dry_run=True)
    assert len(result['configurations']) == 1


def test_cached_sweep_does_not_require_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, 'empty', lambda *_args, **_kwargs: pytest.fail('Cached run must not use CUDA'))
    monkeypatch.setattr(single, 'completed_result', lambda *_: dict(analyses=[]))
    result = sweep.run(dict(base=dict(device='cuda:0')), tmp_path / 'outputs')
    assert result['results'] == [dict(analyses=[])]


def test_cpu_single_experiment_still_runs(tmp_path):
    from .test_pipeline import circle_config
    result = single.run(circle_config(tmp_path), tmp_path / 'outputs')
    assert result['analyses'][0]['name'] == 'errors'


def test_all_paper_quick_configs_export_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, 'empty', lambda *_args, **_kwargs: pytest.fail('Dry run must not use CUDA'))
    presets = Path(__file__).parents[1] / 'scripts/configs/paper'
    for figure, expected in [(4, 6), (5, 12), (6, 2)]:
        result = sweep.run(load_config(presets / f'figure{figure}.yaml'), tmp_path / 'outputs',
                           quick_run=True, dry_run=True, device='cuda:0')
        assert len(result['configurations']) == expected
