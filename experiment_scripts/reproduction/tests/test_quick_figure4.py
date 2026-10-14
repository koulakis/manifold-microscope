"""Reduced recipe selection and read-only reuse, without running experiments."""
import json
from pathlib import Path

import pytest

from experiment_scripts.reproduction import common, quick_figure4 as recipe


def completed(path):
    path.mkdir(parents=True)
    (path / 'COMPLETED').write_text('fixture')
    return path


def inputs(tmp_path):
    figure3, figure6 = tmp_path / 'figure3', tmp_path / 'figure6'
    dataset = completed(tmp_path / 'dataset')
    full = completed(figure6 / 'fits' / 'custom_dsprites_balanced__beta_vae__1.0__4__0__2')
    (full / 'config.json').write_text(json.dumps({
        'seed': 473959, 'model_config': {'exported_datasets_dir': str(dataset)}}))
    for fraction in (0.05, 0.2, 0.4, 0.6, 0.8, 1.0):
        completed(figure3 / 'fits' / f'custom_dsprites_balanced__MMLS__{fraction}__4__0__2')
    return figure3, figure6


def test_five_trainings_six_evaluations_and_resume(tmp_path, monkeypatch):
    figure3, figure6 = inputs(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    calls = []

    def run(command, check):
        assert check
        calls.append(command)
        completed(Path(command[command.index('--output') + 1]))

    monkeypatch.setattr(common.subprocess, 'run', run)
    output = tmp_path / 'output'
    recipe.reproduce(output, figure3, figure6, device='cuda:0')
    fits = [c for c in calls if c[3].endswith('.fitting')]
    evaluations = [c for c in calls if c[3].endswith('.evaluation')]
    assert len(fits) == 5
    assert [float(c[c.index('--fraction') + 1]) for c in fits] == [.05, .2, .4, .6, .8]
    assert all(c[c.index('--seed') + 1] == '473959' for c in fits)
    assert len(evaluations) == 6
    assert len(calls) == 13  # Plus bound export and rendering.
    assert len(list((output / 'fits').glob('*MMLS*'))) == 6
    assert (output / 'fits/custom_dsprites_balanced__beta_vae__1.0__4__0__2').is_symlink()
    recipe.reproduce(output, figure3, figure6, device='cuda:0')
    assert len(calls) == 13
    assert all(p.read_bytes() == content for p, content in before.items())


def test_incomplete_stage_preserved_and_blocks_export(tmp_path, monkeypatch, capsys):
    figure3, figure6 = inputs(tmp_path)
    output = tmp_path / 'output'
    partial = output / 'fits/custom_dsprites_balanced__beta_vae__0.05__4__0__2'
    partial.mkdir(parents=True)
    artifact = partial / 'checkpoint'
    artifact.write_text('keep me')
    monkeypatch.setattr(common.subprocess, 'run', lambda *a, **k: pytest.fail('Must not launch'))
    with pytest.raises(RuntimeError, match='partial directory'):
        recipe.reproduce(output, figure3, figure6)
    assert 'WARNING' in capsys.readouterr().out
    assert artifact.read_text() == 'keep me'
    assert not (output / 'bounds').exists()
