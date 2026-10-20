"""Quick budgets, selected representations, and small complete CPU workflows."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from experiments_and_analysis.config import load_config, resolve
from experiments_and_analysis.outputs import read_json, stage_ids
from experiments_and_analysis.quick import QUICK_RUN, apply_quick_run
from experiments_and_analysis.scripts import run as single
from experiments_and_analysis.scripts import sweep


PAPER = Path(__file__).parents[1] / 'scripts/configs/paper'


@pytest.mark.parametrize('figure,count', [(3, 12), (4, 6), (5, 12), (6, 2)])
def test_quick_paper_jobs_preserve_datasets_layers_and_input_config(figure, count):
    config = load_config(PAPER / f'figure{figure}.yaml')
    original = deepcopy(config)
    jobs = sweep.expand(config, quick_run=True)
    assert config == original
    assert len(jobs) == count
    for job in jobs:
        assert job['split']['max_test_size'] == 100
        assert job['inference']['evaluation_size'] == 100
        if job['model']['name'] == 'beta_vae':
            assert job['model']['parameters']['max_iter'] == 10_000
            assert job['model']['parameters']['strict_budget']
        if job['dataset']['name'] in ('Sphere', 'Torus'):
            assert job['dataset']['ground_truth_size'] == 100
            assert job['dataset']['sample_size'] in (25, 250, 1000)
        if figure == 6:
            assert job['analyses'] == [dict(name='geometry', domain='grid',
                                           representations='all', reach_subsample=4)]
    if figure == 3:
        assert {job['dataset']['dimension'] for job in jobs} == {1, 2, 3, 4}
    if figure == 5:
        assert {job['dataset']['name'] for job in jobs} == {
            'Sphere', 'Torus', 'custom_dsprites_balanced', 'extended_coil20'}


def test_quick_fit_identity_is_distinct_and_shared_between_figures():
    fourth = sweep.expand(load_config(PAPER / 'figure4.yaml'), quick_run=True)
    shared = next(job for job in fourth if job['model']['name'] == 'beta_vae'
                  and job['split']['train_fraction'] == 1)
    sixth = sweep.expand(load_config(PAPER / 'figure6.yaml'), quick_run=True)[0]
    assert stage_ids(shared, shared['dataset']) == stage_ids(sixth, sixth['dataset'])
    full_budget = deepcopy(shared)
    full_budget['model']['parameters'].update(max_iter=2_000_000, strict_budget=False)
    assert stage_ids(shared, shared['dataset'])[1] != stage_ids(full_budget, full_budget['dataset'])[1]


def test_profile_preserves_smaller_limits_and_explicit_layers():
    config = dict(model=dict(name='beta_vae', parameters=dict(max_iter=2)),
                  split=dict(max_test_size=20), inference=dict(representations=['mu'], evaluation_size=10),
                  analyses=[dict(name='geometry', representations=['mu', 'output'])])
    quick = apply_quick_run(config)
    assert quick['model']['parameters']['max_iter'] == 2
    assert quick['split']['max_test_size'] == 20
    assert quick['inference']['evaluation_size'] == 10
    assert quick['inference']['representations'] == ['mu']
    assert quick['analyses'][0]['representations'] == ['mu', 'output']
    assert apply_quick_run(quick) == quick
    assert resolve(config)['model']['parameters']['max_iter'] == 2


def test_single_cli_uses_quick_profile_and_device(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(single, 'run', lambda config, output: calls.append(config))
    path = tmp_path / 'config.json'
    path.write_text('{"model": {"name": "beta_vae"}}')
    result = CliRunner().invoke(single.app, ['--config', str(path), '--output-dir', str(tmp_path),
                                            '--quick-run', '--device', 'cuda:0'])
    assert result.exit_code == 0, result.output
    assert calls[0]['model']['parameters']['max_iter'] == QUICK_RUN['vae_max_iter']
    assert calls[0]['device'] == 'cuda:0'


def test_sweep_cli_dry_run_exports_effective_quick_jobs(tmp_path):
    result = CliRunner().invoke(sweep.app, ['--config', str(PAPER / 'figure5.yaml'),
                                          '--output-dir', str(tmp_path), '--quick-run',
                                          '--dry-run', '--device', 'cpu'])
    assert result.exit_code == 0, result.output
    assert 'Prepared 12 experiment configurations.' in result.output
    jobs = read_json(next(tmp_path.glob('experiment-configurations/sweep__*/jobs.json')))
    assert len(jobs) == 12
    assert not (tmp_path / 'trained-models').exists()


def test_quick_random_evaluation_caps_predictions_preserving_train_and_grid(tmp_path, capsys):
    from experiments_and_analysis.data.preparation import load_source
    angle = np.arange(240) * 2 * np.pi / 240
    data = np.stack([np.cos(angle), np.sin(angle), .2 * np.cos(2 * angle)], axis=-1)[None]
    path = tmp_path / 'circle.npz'
    np.savez(path, data=data)
    config = dict(dataset=dict(name='circle_grid', path=str(path), dimension=1,
                                range_sizes=[2 * np.pi], cyclic_dimensions=[0],
                                patch_sizes=[12], class_names=['circle'], feature_ndim=1),
                  split=dict(method='random', weighted=False, test_fraction=.75),
                  model=dict(name='mmls', parameters=dict(number_of_neighbors=4, verbose=False)),
                  inference=dict(persist=True), analyses=['errors'])
    root = tmp_path / 'outputs'
    result = single.run(config, root, quick_run=True)
    with np.load(root / 'splits' / result['split_id'] / 'indices.npz') as split:
        assert len(split['train']) == 60 and len(split['test']) == 180
    assert len(load_source(config['dataset']).flat) == 240
    assert np.load(next((root / 'inference-outputs').glob('*/values.npy'))).shape == (100, 3)
    metrics = read_json(Path(result['analyses'][0]['directory']) / 'metrics.json')
    assert metrics['number_of_test_points'] == 100
    manifests = list(root.glob('*/**/manifest.json'))
    before = {p: p.read_bytes() for p in manifests}
    assert all(read_json(p)['elapsed_seconds'] >= 0 for p in manifests)
    single.run(config, root, quick_run=True)
    assert all(p.read_bytes() == value for p, value in before.items())
    assert 'original elapsed' in capsys.readouterr().out


@pytest.mark.parametrize('model_type', ['B', 'H'])
def test_selected_layer_stream_matches_full_inference(model_type):
    from experiments_and_analysis.models.beta_vae.inference import batches, representation_names
    from experiments_and_analysis.models.beta_vae.model import BetaVAE_B, BetaVAE_H
    model = BetaVAE_B(z_dim=2, nc=1) if model_type == 'B' else BetaVAE_H(z_dim=2, nc=1)
    data = np.random.default_rng(3).random((3, 64, 64), dtype=np.float32)
    names = representation_names(model)
    full = list(batches(model, data, names, batch_size=2, seed=23))
    for name in names:
        selected = list(batches(model, data, [name], batch_size=2, seed=23))
        for (start, output), (full_start, expected) in zip(selected, full):
            assert start == full_start and list(output) == [name]
            np.testing.assert_array_equal(output[name], expected[name])


def test_latent_only_inference_never_runs_decoder_or_exports_reconstruction(monkeypatch):
    from experiments_and_analysis.models.beta_vae import inference
    from experiments_and_analysis.models.beta_vae.model import BetaVAE_B
    model = BetaVAE_B(z_dim=2)
    monkeypatch.setattr(model.decoder[0], 'forward', lambda *_: pytest.fail('Decoder should be skipped'))
    monkeypatch.setattr(inference.F, 'sigmoid', lambda *_: pytest.fail('Output should not be copied'))
    outputs = list(inference.batches(model, np.zeros((1, 64, 64), dtype=np.float32), ['mu']))
    assert list(outputs[0][1]) == ['mu']


def test_quick_sweep_renders_three_sizes_and_reuses_completed_stages(tmp_path, monkeypatch):
    from .test_pipeline import circle_config
    config = dict(base=circle_config(tmp_path), grid={'split.train_fraction': [.4, .7, 1.]},
                  outputs=[dict(name='bounds', filename='quick_bounds', panels=[dict(
                      label='mmls', dimension=1, where={'model': 'mmls'})])])
    root = tmp_path / 'outputs'
    first = sweep.run(config, root, quick_run=True)
    assert len(first['results']) == 3
    plot_dir = Path(first['plots'][0])
    assert (plot_dir / 'quick_bounds.png').is_file()
    assert (plot_dir / 'quick_bounds.pdf').read_bytes().startswith(b'%PDF')
    assert read_json(plot_dir / 'manifest.json')['elapsed_seconds'] > 0
    monkeypatch.setattr(single, 'load_dataset', lambda *_: pytest.fail('Completed runs must skip loading'))
    second = sweep.run(config, root, quick_run=True)
    assert second['plots'] == first['plots'] and second['results'] == first['results']
