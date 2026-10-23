"""Readable paths preserve identities, shared stages, and partial outputs."""
from copy import deepcopy
from pathlib import Path

import pytest

from experiments_and_analysis import paths
from experiments_and_analysis.config import load_config, resolve
from experiments_and_analysis.data.preparation import source_identity
from experiments_and_analysis.outputs import begin, experiment_identity, read_json, stage_directory, stage_ids
from experiments_and_analysis.scripts import run as single, sweep


PAPER = Path(__file__).parents[1] / 'scripts/configs/paper'


def test_fit_and_split_names_show_the_experiment_without_changing_ids(tmp_path):
    config = resolve(dict(model=dict(name='mmls'), split=dict(method='fixed_test', train_fraction=.2),
                          repetition=dict(index=2, seed=473959), analyses=['errors']))
    source = source_identity(config['dataset'])
    split_id, model_id = stage_ids(config, source)
    directories = paths.experiment_paths(tmp_path, config, source)
    assert directories['model'].name == (
        'custom_dsprites_balanced__dim4__mmls__k32__thin_svd__fixed_test__train20pct'
        f'__weighted__testmax500__seed473959__trial2__{model_id}')
    assert directories['split'].name.endswith(f'__seed473959__{split_id}')
    assert 'mmls' not in directories['split'].name and 'trial' not in directories['split'].name
    experiment_id = experiment_identity(config, source)
    assert directories['configuration'].name.endswith(f'__errors__{experiment_id}')


def test_toy_counts_vae_budgets_and_representations_are_visible(tmp_path):
    toy = resolve(dict(dataset=dict(name='Sphere', dimension=2, sample_size=250),
                       model=dict(name='mmls'), split=dict(method='provided')))
    toy_name = paths.model_label(toy)
    assert 'Sphere__dim2__n250__dataseed42' in toy_name
    assert 'provided' in toy_name and 'pct' not in toy_name
    vae = sweep.expand(load_config(PAPER / 'figure6.yaml'), quick_run=True)[0]
    label = paths.representation_label(vae, 'grid', 'encoder_relu_0')
    assert 'beta_vae__B__z10__iter10000' in label
    assert 'grid__encoder_relu_0' in label
    assert paths.sweep_label(load_config(PAPER / 'figure5.yaml'), []) == 'sweep__figure_5'


def test_input_geometry_and_shared_fit_paths_ignore_unrelated_options(tmp_path):
    jobs4 = sweep.expand(load_config(PAPER / 'figure4.yaml'), quick_run=True)
    fourth = next(c for c in jobs4 if c['model']['name'] == 'beta_vae'
                  and c['split']['train_fraction'] == 1)
    sixth = sweep.expand(load_config(PAPER / 'figure6.yaml'), quick_run=True)[0]
    assert paths.experiment_paths(tmp_path, fourth, fourth['dataset'])['model'] == (
        paths.experiment_paths(tmp_path, sixth, sixth['dataset'])['model'])
    changed = deepcopy(fourth)
    changed['model']['name'] = 'mmls'
    changed['split'].update(seed=7, train_fraction=.1)
    changed['training'].update(seed=7, trial=10)
    assert paths.representation_label(changed, 'grid', 'input') == (
        paths.representation_label(fourth, 'grid', 'input'))
    assert 'trial' not in paths.representation_label(fourth, 'grid', 'input')


@pytest.mark.parametrize('label', ['../../a/b: unsafe% name', '漢字' * 300, 'x' * 1000])
def test_path_components_are_portable_bounded_and_keep_hash(tmp_path, label):
    key = 'a' * 20
    directory = stage_directory(tmp_path, key, label)
    assert directory.parent == tmp_path
    assert directory.name.isascii() and len(directory.name.encode()) <= 242
    assert directory.name.endswith(f'__{key}')


def test_partial_output_is_preserved(tmp_path):
    directory = stage_directory(tmp_path, 'b' * 20, 'dataset__mmls__train20pct')
    directory.mkdir()
    (directory / 'checkpoint').write_text('keep')
    with pytest.raises(RuntimeError, match='Incomplete output preserved'):
        begin(directory, {})
    assert (directory / 'checkpoint').read_text() == 'keep'


def test_shared_cache_names_include_dataset_but_no_training_fraction(tmp_path):
    from .test_pipeline import circle_config
    config = circle_config(tmp_path)
    config['split'].update(method='fixed_test', train_fraction=.5, max_test_size=6)
    result = single.run(config, tmp_path / 'outputs')
    split_root = tmp_path / 'outputs/splits'
    weight, = split_root.glob('weights__*')
    test, = split_root.glob('test__*')
    assert 'circle_grid' in weight.name and '__uniform__' in weight.name
    assert 'circle_grid' in test.name and '__max6__' in test.name
    assert all('train50pct' not in p.name and 'mmls' not in p.name for p in (weight, test))
    split, = split_root.glob(f"*__{result['split_id']}")
    assert 'train50pct' in split.name
    assert read_json(test / 'manifest.json')['config']['max_test_size'] == 6
