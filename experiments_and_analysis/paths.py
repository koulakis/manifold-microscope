"""Human-readable labels use only inputs relevant to the corresponding stage."""
from decimal import Decimal
from pathlib import Path

from .outputs import artifact_path, experiment_identity, stage_directory, stage_ids


def number(value):
    value = format(Decimal(str(value)), 'f')
    return value.rstrip('0').rstrip('.') if '.' in value else value


def percentage(fraction):
    return number(Decimal(str(fraction)) * 100)


def dataset_label(spec):
    parts = [spec['name']]
    if 'path' in spec:
        parts.append(Path(spec['path']).stem)
    if 'dimension' in spec:
        parts.append(f"dim{spec['dimension']}")
    if 'sample_size' in spec:
        parts.extend([f"n{spec['sample_size']}", f"dataseed{spec['seed']}"])
    return '__'.join(parts)


def selection_label(split, *, include_seed=True):
    parts = [split['method']]
    if split['method'] not in ('full', 'provided'):
        parts.extend([f"train{percentage(split['train_fraction'])}pct",
                      'weighted' if split['weighted'] else 'uniform'])
        if split['method'] == 'random':
            parts.append(f"test{percentage(split['test_fraction'])}pct")
        else:
            parts.append(f"testmax{split['max_test_size']}")
        if include_seed:
            parts.append(f"seed{split['seed']}")
    return '__'.join(parts)


def split_label(config):
    return f"{dataset_label(config['dataset'])}__{selection_label(config['split'])}"


def model_label(config):
    model, split, training = config['model'], config['split'], config['training']
    parts = [dataset_label(config['dataset']), model['name'] or 'input']
    options = model['parameters']
    if model['name'] == 'mmls':
        parts.extend([f"k{options['number_of_neighbors']}", options['projection_method']])
    elif model['name'] == 'beta_vae':
        parts.extend([options['model'], f"z{options['z_dim']}", f"iter{number(options['max_iter'])}"])
    parts.append(selection_label(split, include_seed=False))
    if split['method'] not in ('full', 'provided') and split['seed'] != training['seed']:
        parts.append(f"splitseed{split['seed']}")
    parts.extend([f"seed{training['seed']}", f"trial{training['trial']}"])
    return '__'.join(parts)


def representation_label(config, domain, representation):
    if representation == 'input':
        # Input grid geometry is shared across models, fractions and trials.
        label = dataset_label(config['dataset']) if domain == 'grid' else split_label(config)
        return f'{label}__{domain}__input'
    parts = [model_label(config), domain, representation]
    if domain == 'test' and 'evaluation_size' in config['inference']:
        parts.append(f"eval{config['inference']['evaluation_size']}")
    if representation == 'output' and config['model']['name'] == 'beta_vae':
        parts.append('sampled' if config['inference']['decoder_sampling'] else 'mean')
    return '__'.join(parts)


def experiment_paths(root, config, source):
    """Build readable paths while keeping raw IDs in manifests."""
    root = artifact_path(root)
    split_id, model_id = stage_ids(config, source)
    experiment_id = experiment_identity(config, source)
    label = model_label(config) if config['model']['name'] else f"{split_label(config)}__input"
    analyses = '-'.join(sorted({a['name'] for a in config['analyses']}))
    return dict(
        configuration=stage_directory(root / 'experiment-configurations', experiment_id,
                                      f'{label}__{analyses}' if analyses else label),
        split=stage_directory(root / 'splits', split_id, split_label(config)),
        model=stage_directory(root / 'trained-models', model_id, model_label(config)))


def sweep_label(config, experiments):
    names = [request['filename'] for request in config.get('outputs', [])]
    if not names:
        names = sorted({c['dataset']['name'] for c in experiments})
        names.extend(sorted({c['model']['name'] or 'input' for c in experiments}))
    return 'sweep__' + '__'.join(names)
