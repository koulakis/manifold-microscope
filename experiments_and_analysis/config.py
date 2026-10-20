"""Plain configuration dictionaries shared by Python and CLI callers."""
from copy import deepcopy
from pathlib import Path
from dataclasses import asdict
import json

import yaml


def load_config(path):
    path = Path(path)
    with path.open() as stream:
        value = json.load(stream) if path.suffix == '.json' else yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError('The configuration must be a dictionary.')
    return value


def merge(base, changes):
    result = deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve(config):
    """Fill defaults without changing the caller's dictionary."""
    repetition = config.get('repetition', {'index': 0, 'seed': 42})
    seed = repetition['seed']
    defaults = dict(
        dataset=dict(name='custom_dsprites_balanced', dimension=4),
        split=dict(method='random', test_fraction=0.1, train_fraction=1.0,
                   seed=seed, weighted=True, max_test_size=500),
        model=dict(name=None, parameters={}),
        training=dict(enabled=True, seed=seed, trial=repetition['index']),
        inference=dict(enabled=True, domain='test', representations=['output'],
                       batch_size=64, seed=seed, decoder_sampling=True,
                       persist=False, buffer='memory'),
        analyses=[], repetition=repetition, device='cpu', num_workers=0)
    result = merge(defaults, config)
    unknown = set(result) - set(defaults)
    if unknown:
        raise ValueError(f'Unknown experiment options: {sorted(unknown)}')
    for section in ('split', 'model', 'training', 'inference'):
        allowed = set(defaults[section])
        if section == 'inference':
            allowed.add('evaluation_size')
        unknown = set(result[section]) - allowed
        if unknown:
            raise ValueError(f'Unknown {section} options: {sorted(unknown)}')
    split = result['split']
    if split['method'] not in ('random', 'fixed_test', 'full', 'provided'):
        raise ValueError('split.method must be random, fixed_test, full or provided.')
    if not 0 < split['train_fraction'] <= 1 or not 0 < split['test_fraction'] < 1:
        raise ValueError('Invalid training/test fraction.')
    if split['method'] == 'full' and split['train_fraction'] != 1:
        raise ValueError('A full split must keep the full training domain.')
    if result['model']['name'] not in (None, 'beta_vae', 'mmls'):
        raise ValueError('model.name must be beta_vae, mmls or null.')
    if result['dataset']['name'] in ('Sphere', 'Torus', 'Circle', 'Moons'):
        result['dataset'].setdefault('seed', seed)
        result['dataset'].setdefault('reference_seed', 0)
        result['dataset'].setdefault('ground_truth_size', 500)
    from .models.config import BetaVAEConfig, MMLSConfig
    parameters = result['model']['parameters']
    if result['model']['name'] == 'beta_vae':
        if set(parameters).intersection({'ckpt_dir', 'output_dir', 'device', 'num_workers', 'ckpt_name'}):
            raise ValueError('Set paths, device and workers outside model.parameters.')
        defaults = dict(model='H', objective='H', lr=1e-4, loss_threshold=10) if result['dataset']['name'] == 'extended_coil20' else {}
        options = asdict(BetaVAEConfig(ckpt_dir=Path('.'), output_dir=Path('.'), **{**defaults, **parameters}))
        result['model']['parameters'] = {k: v for k, v in options.items()
                                          if k not in ('ckpt_dir', 'output_dir', 'device', 'num_workers', 'ckpt_name')}
    elif result['model']['name'] == 'mmls':
        if 'number_of_dims' in parameters:
            raise ValueError('Set the intrinsic dimension in dataset.dimension.')
        options = asdict(MMLSConfig(**{'number_of_dims': result['dataset']['dimension'],
                          'number_of_neighbors': 2 * 2 ** result['dataset']['dimension'],
                          **{k: v for k, v in parameters.items() if k != 'export_points'}}))
        options.pop('number_of_dims')
        result['model']['parameters'] = {**options, 'export_points': parameters.get('export_points', False)}
    if result['inference']['batch_size'] < 1:
        raise ValueError('inference.batch_size must be positive.')
    if result['inference'].get('evaluation_size', 1) < 1:
        raise ValueError('inference.evaluation_size must be positive.')
    if result['inference']['buffer'] not in ('memory', 'temporary'):
        raise ValueError('inference.buffer must be memory or temporary.')
    if result['inference']['domain'] not in ('train', 'test', 'grid'):
        raise ValueError('inference.domain must be train, test or grid.')
    if (not result['analyses'] and result['model']['name'] is not None and result['inference']['enabled']
            and result['inference']['domain'] == 'test' and split['method'] == 'full'):
        raise ValueError('A full split has no held-out test domain; use train/grid or disable inference.')
    analyses = []
    for requested in result['analyses']:
        analysis = {'name': requested} if isinstance(requested, str) else dict(requested)
        if analysis['name'] not in ('errors', 'geometry'):
            raise ValueError(f"Unknown analysis: {analysis['name']}")
        allowed = {'name', 'domain', 'representations'}
        if analysis['name'] == 'geometry':
            allowed.update(['normalize_for_volume', 'normalize_curvatures', 'reach_subsample', 'reach_batch_size'])
        if set(analysis) - allowed:
            raise ValueError(f'Unknown analysis options: {sorted(set(analysis) - allowed)}')
        analysis.setdefault('domain', 'grid' if analysis['name'] == 'geometry'
                            else result['inference']['domain'])
        analysis.setdefault('representations', ['input'] if result['model']['name'] is None
                            else result['inference']['representations'])
        if analysis['domain'] not in ('train', 'test', 'grid'):
            raise ValueError('Analysis domain must be train, test or grid.')
        if analysis['name'] == 'geometry' and analysis['domain'] != 'grid':
            raise ValueError('Geometry requires the full structured grid, not a split.')
        if analysis['name'] == 'geometry' and split['method'] == 'provided':
            raise ValueError('Sampled toy points do not form a full parameter grid.')
        if analysis['name'] == 'errors' and analysis['representations'] != ['output']:
            raise ValueError('Reconstruction errors require the output representation.')
        if analysis['domain'] == 'test' and split['method'] == 'full':
            raise ValueError('A full split has no held-out test domain.')
        if (result['model']['name'] is None or not result['inference']['enabled']) and analysis['representations'] != ['input']:
            raise ValueError('Model inference is required for non-input representations.')
        analyses.append(analysis)
    result['analyses'] = analyses
    return result
