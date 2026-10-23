"""Small real CPU workflows and invariants for stage identity and reuse."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from experiments_and_analysis.config import resolve
from experiments_and_analysis.data.preparation import split_indices
from experiments_and_analysis.outputs import identity, read_json, stage_ids
from experiments_and_analysis.scripts import run as single
from experiments_and_analysis.scripts.sweep import expand


def circle_config(tmp_path):
    angle = np.arange(24) * 2 * np.pi / 24
    data = np.stack([np.cos(angle), np.sin(angle), .2 * np.cos(2 * angle)], axis=-1)[None]
    path = tmp_path / 'circle.npz'
    np.savez(path, data=data)
    return dict(dataset=dict(name='circle_grid', path=str(path), dimension=1,
                              range_sizes=[2 * np.pi], cyclic_dimensions=[0],
                              patch_sizes=[12], class_names=['circle'], feature_ndim=1),
                split=dict(method='random', weighted=False, test_fraction=.25, seed=7),
                model=dict(name='mmls', parameters=dict(number_of_neighbors=4, verbose=False)),
                analyses=['errors'])


def test_seeded_split_is_disjoint_reproducible_and_preserves_rng():
    points = np.arange(40).reshape(20, 2)
    weights = np.arange(1, 21, dtype=float)
    weights /= weights.sum()
    spec = resolve({})['split']
    state = np.random.get_state()
    first = split_indices(points, weights, spec)
    second = split_indices(points, weights, spec)
    assert len(first[0]) == 18 and len(first[1]) == 2
    assert not set(first[0]).intersection(first[1])
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(np.random.get_state()[1], state[1])


def test_stage_identity_separates_trials_but_reuses_training_for_new_analyses():
    config = resolve(dict(model=dict(name='beta_vae')))
    before = stage_ids(config, config['dataset'])
    changed = deepcopy(config)
    changed['analyses'] = [dict(name='geometry', representations=['mu'])]
    changed['inference']['representations'] = ['mu']
    assert stage_ids(changed, changed['dataset']) == before
    changed['training']['enabled'] = False
    assert stage_ids(changed, changed['dataset']) == before
    changed['training']['trial'] += 1
    assert stage_ids(changed, changed['dataset'])[1] != before[1]
    changed = deepcopy(config)
    changed['split']['seed'] += 1
    assert stage_ids(changed, changed['dataset'])[0] != before[0]


def test_real_mmls_run_reuses_fit_for_new_analysis_and_resumes(tmp_path, monkeypatch):
    config = circle_config(tmp_path)
    root = tmp_path / 'outputs'
    first = single.run(config, root)
    model_dir = next((root / 'trained-models').glob(f"*__{first['model_id']}"))
    before = {p: p.read_bytes() for p in model_dir.rglob('*') if p.is_file()}
    assert not (model_dir / 'training_points.npy').exists()
    metrics = read_json(Path(first['analyses'][0]['directory']) / 'metrics.json')
    assert metrics['number_of_train_points'] == 18
    assert metrics['number_of_test_points'] == 6
    assert 0 < metrics['hausdorff_distance'] < 1
    assert not list((root / 'inference-outputs').rglob('*.npy'))
    config['training'] = dict(enabled=False)
    config['analyses'].append(dict(name='geometry', representations=['input'], normalize_for_volume=False))
    second = single.run(config, root)
    assert first['model_id'] == second['model_id']
    assert len(second['analyses']) == 2
    assert all(p.read_bytes() == contents for p, contents in before.items())
    monkeypatch.setattr(single, 'predictions', lambda *args: pytest.fail('Completed analysis must skip inference'))
    third = single.run(config, root)
    assert second == third


def test_persisted_predictions_equal_recorded_errors(tmp_path):
    config = circle_config(tmp_path)
    config['inference'] = dict(persist=True)
    root = tmp_path / 'outputs'
    result = single.run(config, root)
    path = next((root / 'inference-outputs').glob('*/values.npy'))
    predictions = np.load(path)
    assert predictions.shape == (6, 3)
    assert np.isfinite(predictions).all()
    directory = Path(result['analyses'][0]['directory'])
    distances = np.load(directory / 'pointwise_distances.npy')
    assert distances.max() == read_json(directory / 'metrics.json')['hausdorff_distance']


def test_dataset_only_geometry_skips_models_and_inference(tmp_path):
    config = circle_config(tmp_path)
    config.update(model=dict(name=None), split=dict(method='full'), inference=dict(enabled=False),
                  analyses=[dict(name='geometry', representations=['input'], normalize_for_volume=False)])
    result = single.run(config, tmp_path / 'outputs')
    assert result['model_id'] is None
    assert not list((tmp_path / 'outputs/trained-models').iterdir())
    assert (Path(result['analyses'][0]['directory']) / 'measures.pkl').is_file()


def test_geometry_on_a_split_is_rejected_before_output_creation(tmp_path):
    config = circle_config(tmp_path)
    config['analyses'] = [dict(name='geometry', domain='test')]
    with pytest.raises(ValueError, match='full structured grid'):
        single.run(config, tmp_path / 'outputs')
    assert not (tmp_path / 'outputs').exists()


def test_sweep_exclusions_and_explicit_additions():
    config = dict(base=dict(model=dict(name='mmls')),
                  grid={'dataset.dimension': [1, 2], 'split.train_fraction': [.1, .5]},
                  exclude=[{'dataset.dimension': 1, 'split.train_fraction': .1}],
                  include=[dict(dataset=dict(dimension=3), split=dict(train_fraction=.8))])
    jobs = expand(config)
    assert len(jobs) == 4
    assert (1, .1) not in {(job['dataset']['dimension'], job['split']['train_fraction']) for job in jobs}


@pytest.mark.parametrize('standalone', [False, True])
def test_mmls_projector_export_round_trip(tmp_path, standalone):
    from experiments_and_analysis.models.config import MMLSConfig
    from experiments_and_analysis.models.mmls.fitting import fit, load_projector, save_projector
    from experiments_and_analysis.models.mmls.inference import predict
    rng = np.random.default_rng(8)
    data = rng.normal(size=(12, 3))
    query = rng.normal(size=(3, 3))
    options = MMLSConfig(number_of_dims=1, number_of_neighbors=4, verbose=False)
    projector = fit(data, options)
    save_projector(projector, options, tmp_path, split_id='fixture', export_points=standalone)
    loaded = load_projector(tmp_path, training_points=None if standalone else data)
    np.testing.assert_allclose(predict(projector, query), predict(loaded, query), rtol=1e-12, atol=1e-12)


def test_one_step_vae_checkpoint_and_named_inference(tmp_path):
    import torch
    from experiments_and_analysis.models.beta_vae.inference import batches
    from experiments_and_analysis.models.beta_vae.inference import load_autoencoder_from_checkpoint
    data = np.random.default_rng(5).random((3, 64, 64), dtype=np.float32)
    np.savez(tmp_path / 'images.npz', data=data[None])
    config = dict(dataset=dict(name='tiny_images', dimension=1, path=str(tmp_path / 'images.npz'),
                              range_sizes=[1], patch_sizes=[10], class_names=['a']),
                  split=dict(method='full'), model=dict(name='beta_vae', parameters=dict(
                      z_dim=2, batch_size=2, max_iter=1, max_epochs=1, strict_budget=True)),
                  inference=dict(domain='train', representations=['mu', 'output'], persist=True, batch_size=2))
    result = single.run(config, tmp_path / 'outputs')
    model_dir = next((tmp_path / 'outputs/trained-models').glob(f"*__{result['model_id']}"))
    assert len(list((tmp_path / 'outputs/inference-outputs').glob('*/values.npy'))) == 2
    checkpoint = torch.load(model_dir / 'checkpoints/last.pth', weights_only=False, map_location='cpu')
    assert checkpoint['iter'] == 1
    assert all(torch.isfinite(value).all() for value in checkpoint['model_states']['net'].values())
    model = load_autoencoder_from_checkpoint('B', model_dir / 'checkpoints/last.pth', latent_dim=2, device='cpu')
    outputs = list(batches(model, data, ['mu', 'output'], batch_size=2, decoder_sampling=False))
    assert [start for start, _ in outputs] == [0, 2]
    assert outputs[0][1]['mu'].shape == (2, 2)
    assert outputs[1][1]['output'].shape == (1, 1, 64, 64)
    np.testing.assert_array_equal(outputs[0][1]['mu'],
                                  next(batches(model, data, ['mu'], batch_size=2,
                                               decoder_sampling=False))[1]['mu'])


def test_grid_projection_geometry_reuses_persisted_inference(tmp_path, monkeypatch):
    config = circle_config(tmp_path)
    config['inference'] = dict(persist=True, buffer='temporary')
    config['analyses'] = [dict(name='geometry', representations=['output'], normalize_for_volume=False)]
    root = tmp_path / 'outputs'
    first = single.run(config, root)
    assert not list((root / 'inference-outputs').rglob('working.npy'))
    assert len(list((root / 'inference-outputs').rglob('values.npy'))) == 1
    config['analyses'][0]['normalize_for_volume'] = True
    monkeypatch.setattr(single, 'predictions', lambda *args: pytest.fail('Persisted inference must be reused'))
    second = single.run(config, root)
    assert first['model_id'] == second['model_id']
    assert first['analyses'][0]['id'] != second['analyses'][0]['id']


def test_sweep_renders_bounds_from_explicit_results(tmp_path):
    from experiments_and_analysis.scripts.sweep import run
    base = circle_config(tmp_path)
    config = dict(base=base, grid={'split.train_fraction': [.5, 1.]}, outputs=[dict(
        name='bounds', filename='tiny_bounds', panels=[dict(label='mmls', dimension=1, where={'model': 'mmls'})])])
    result = run(config, tmp_path / 'outputs')
    assert len(result['results']) == 2
    directory = Path(result['plots'][0])
    assert (directory / 'tiny_bounds.png').is_file()
    assert (directory / 'tiny_bounds.pdf').read_bytes().startswith(b'%PDF')


def test_paper_figures_four_and_six_share_the_full_vae_fit():
    from experiments_and_analysis.config import load_config
    configs = Path(__file__).parents[1] / 'scripts/configs/paper'
    figure4 = expand(load_config(configs / 'figure4.yaml'))
    shared = next(job for job in figure4 if job['model']['name'] == 'beta_vae'
                  and job['split']['train_fraction'] == 1 and job['training']['trial'] == 2)
    figure6 = expand(load_config(configs / 'figure6.yaml'))[0]
    assert stage_ids(shared, shared['dataset'])[1] == stage_ids(figure6, figure6['dataset'])[1]


def test_too_few_mmls_points_skip_without_leaving_a_partial_fit(tmp_path):
    config = circle_config(tmp_path)
    config['split']['train_fraction'] = .1
    result = single.run(config, tmp_path / 'outputs')
    assert result['status'] == 'skipped'
    assert not list((tmp_path / 'outputs/trained-models').iterdir())
    assert single.run(config, tmp_path / 'outputs') == result


def test_toy_reference_is_shared_across_training_sizes(tmp_path):
    spec = dict(name='Circle', dimension=1, sample_size=8, seed=7, ground_truth_size=12)
    first = single.load_dataset(spec, tmp_path)
    spec.update(sample_size=10, seed=8)
    second = single.load_dataset(spec, tmp_path)
    np.testing.assert_array_equal(first.flat[8:], second.flat[10:])
    assert len(list((tmp_path / 'splits').glob('reference__*'))) == 1


@pytest.mark.parametrize('config', [
    {'split': {'train_fraciton': .5}},
    {'analyses': [{'name': 'geometry', 'reach_subsampl': 2}]},
])
def test_configuration_typos_are_rejected_before_training(config):
    with pytest.raises(ValueError, match='Unknown'):
        resolve(config)
