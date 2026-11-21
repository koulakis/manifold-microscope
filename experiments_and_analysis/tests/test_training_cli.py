"""The Typer boundary preserves the selected training configuration."""
import json

from typer.testing import CliRunner

from experiments_and_analysis.config import resolve
from experiments_and_analysis.scripts import run as runner


def test_selected_training_repetition(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, 'run', lambda config, output: calls.append(resolve(config)))
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(dict(model=dict(name='beta_vae', parameters=dict(max_iter=1, strict_budget=True)),
                                    repetition=dict(index=2, seed=473959), split=dict(train_fraction=.5))))
    result = CliRunner().invoke(runner.app, ['--config', str(path), '--output-dir', str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]['training']['trial'] == 2
    assert calls[0]['training']['seed'] == 473959
    assert calls[0]['model']['parameters']['max_iter'] == 1
    assert calls[0]['model']['parameters']['strict_budget']


def test_mmls_projection_option(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, 'run', lambda config, output: calls.append(resolve(config)))
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(dict(model=dict(name='mmls', parameters=dict(projection_method='original')))))
    result = CliRunner().invoke(runner.app, ['--config', str(path), '--output-dir', str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert calls[0]['model']['parameters']['projection_method'] == 'original'


def test_geometry_precision_flag_is_a_runtime_option(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, 'run', lambda config, output, **kwargs: calls.append((config, kwargs)))
    path = tmp_path / 'config.json'
    path.write_text('{}')
    result = CliRunner().invoke(runner.app, ['--config', str(path), '--output-dir', str(tmp_path),
                                           '--fallback-to-float64-on-nan-measures'])
    assert result.exit_code == 0, result.output
    assert calls == [({}, {'fallback_to_float64_on_nan_measures': True})]
