"""Unavailable devices fail before datasets are loaded or stage files are written."""
from pathlib import Path
import importlib

import pytest
import torch
import numpy as np

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


@pytest.mark.parametrize('name', ['figure3', 'figure4', 'figure4_quick', 'figure5', 'figure6'])
def test_paper_presets_request_cuda_but_allow_explicit_cpu_override(name):
    presets = Path(__file__).parents[1] / 'scripts/configs/paper'
    config = load_config(presets / f'{name}.yaml')
    assert all(c['device'] == 'cuda:0' for c in sweep.expand(config, quick_run=True))
    assert all(c['device'] == 'cpu' for c in sweep.expand(config, quick_run=True, device='cpu'))


def test_geometry_forwards_explicit_gpu_through_all_measure_kernels(monkeypatch, capsys):
    from experiments_and_analysis.analyses.geometry import compute
    from experiments_and_analysis.data.preparation import Dataset
    module = importlib.import_module('microscope.computations_grid.data_analysis.data_analysis')
    calls = []

    def volume(*, device, **kwargs):
        calls.append(('volume', device))
        return np.ones(24)

    def curvature(*, device, **kwargs):
        calls.append(('curvature', device))
        return np.tile([1., -1.], 12)

    def reach(*, device, **kwargs):
        calls.append(('reach', device))
        return np.ones(24)

    monkeypatch.setattr(module, 'volume_element', volume)
    monkeypatch.setattr(module, 'scalar_curvature', curvature)
    monkeypatch.setattr(module, 'reach_per_point', reach)
    dataset = Dataset(np.zeros((1, 24, 2)), [2*np.pi], [0], [10], ['circle'], feature_ndim=1)
    compute(dataset.flat, dataset, normalize_for_volume=False, device='cuda:0')
    assert calls == [('volume', 'cuda:0'), ('curvature', 'cuda:0'),
                     ('volume', 'cuda:0'), ('reach', 'cuda:0')]
    assert '[geometry] computing volume, curvature and reach on cuda:0' in capsys.readouterr().out
