"""Run one configured experiment; keep IO and stage selection here."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Optional

from pathlib import Path
import pickle
from time import perf_counter

import numpy as np
import typer

from ..config import load_config, resolve
from ..bundles import output_root
from ..data.preparation import (
    candidates,
    load_source,
    normalize,
    prepare_split,
    sampling_weights,
    source_identity,
    toy_reference,
)
from ..models.config import MMLSConfig
from ..outputs import (
    STAGES,
    artifact_path,
    begin,
    experiment_identity,
    finish,
    identity,
    is_complete,
    load_array,
    read_json,
    report_reuse,
    save_array,
    stage_directory,
    stage_ids,
    write_json,
)
from ..paths import dataset_label, experiment_paths, representation_label
from ..quick import apply_quick_run

if TYPE_CHECKING:
    from ..bundles import BundlePath
    from ..data.preparation import Dataset
    from ..models.beta_vae.model import BetaVAE_H
    from ..models.mmls.projector import ANNMMLSProjector


app = typer.Typer(pretty_exceptions_enable=False)


def check_device(device: str) -> None:
    """Initialize the requested accelerator before loading data or writing stages.

    Args:
        device: Device used for tensor computations.

    Returns:
        None.
    """
    import torch
    target = torch.device(device)
    if target.type == 'cpu':
        return
    try:
        torch.empty(1, device=target)
    except (RuntimeError, AssertionError) as error:
        raise RuntimeError(f'Device preflight failed for {device}: {error}') from error


def load_dataset(
    spec: dict,
    root: str | Path,
    device: str = 'cpu'
) -> Dataset:
    """Load a dataset, reusing fixed toy reference points when available.

    Args:
        spec: Dataset or split specification.
        root: Root directory for experiment artifacts.
        device: Device used for tensor computations.

    Returns:
        Loaded Dataset, reusing a cached toy reference where available.
    """
    started = perf_counter()
    if spec['name'] not in ('Sphere', 'Torus', 'Circle', 'Moons'):
        dataset = load_source(spec)
    else:
        reference_config = dict(name=spec['name'], ground_truth_size=spec.get('ground_truth_size', 500),
                                reference_seed=spec.get('reference_seed', 0))
        reference_id = identity(reference_config)
        directory = stage_directory(Path(root) / 'splits', reference_id,
                                    f"reference__{spec['name']}__n{reference_config['ground_truth_size']}"
                                    f"__seed{reference_config['reference_seed']}")
        if begin(directory, reference_config):
            np.save(directory / 'points.npy', toy_reference(spec, device))
            finish(directory, ['points.npy'])
        dataset = load_source(spec, reference=np.load(directory / 'points.npy'))
    print(f"[data] loaded {spec['name']} in {perf_counter() - started:.2f}s")
    return dataset


def prepare_indices(
    dataset: Dataset,
    config: dict,
    root: Path,
    source: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Cache small volume weights and fixed-test indices shared by many fits.

    Args:
        dataset: Loaded dataset and its grid/split metadata.
        config: Experiment or sweep configuration.
        root: Root directory for experiment artifacts.
        source: Dataset identity including source provenance.

    Returns:
        (train_indices, test_indices), flat dataset index arrays of shapes (n_train) and (n_test).
    """
    spec = config['split']
    if spec['method'] in ('full', 'provided'):
        return prepare_split(dataset, spec, config['device'])
    weight_config = dict(dataset=source, weighted=spec['weighted'])
    weight_id = identity(weight_config)
    weight_dir = stage_directory(root / 'splits', weight_id,
                                 f"weights__{dataset_label(source)}__{'volume' if spec['weighted'] else 'uniform'}")
    if begin(weight_dir, weight_config):
        weights = sampling_weights(dataset, spec['weighted'], config['device'])
        np.save(weight_dir / 'weights.npy', weights)
        finish(weight_dir, ['weights.npy'])
    weights = np.load(weight_dir / 'weights.npy')
    fixed_test = None
    if spec['method'] == 'fixed_test':
        from microscope.datasets.dataset_split import maximal_distance_selection_torch
        test_config = dict(dataset=source, max_test_size=spec['max_test_size'])
        test_id = identity(test_config)
        test_dir = stage_directory(root / 'splits', test_id,
                                   f"test__{dataset_label(source)}__max{spec['max_test_size']}")
        if begin(test_dir, test_config):
            available = candidates(dataset)
            points = dataset.flat[available].reshape(len(available), -1)
            count = min(spec['max_test_size'], int(.3 * len(points)))
            if count < 1:
                raise ValueError('Dataset is too small for a fixed test split.')
            _, fixed_test = maximal_distance_selection_torch(points, count, device=config['device'])
            np.save(test_dir / 'indices.npy', fixed_test)
            finish(test_dir, ['indices.npy'])
        fixed_test = np.load(test_dir / 'indices.npy')
    return prepare_split(
        dataset,
        spec,
        config['device'],
        weights=weights,
        fixed_test=fixed_test
    )


def fit_model(
    config: dict,
    data: np.ndarray,
    directory: Path | BundlePath,
    split_id: str | None
) -> list[str]:
    """Explicit model selection; training functions own checkpoint IO.

    Args:
        config: Experiment or sweep configuration.
        data: Input samples of shape (n ...), with sample axis first.
        directory: Directory or virtual bundle path for this stage.
        split_id: Identifier of the training split, when available.

    Returns:
        Relative filenames of the model artifacts written by the selected training method.
    """
    from ..models.beta_vae import fitting as vae
    from ..models.mmls import fitting as mmls
    parameters = dict(config['model']['parameters'])
    match config['model']['name']:
        case 'beta_vae':
            vae.fit(
                data,
                directory,
                parameters,
                seed=config['training']['seed'],
                device=config['device'],
                num_workers=config['num_workers']
            )
            return ['model.json', 'checkpoints/last.pth']
        case 'mmls':
            np.random.seed(config['training']['seed'])
            export_points = parameters.pop('export_points', False)
            options = MMLSConfig(number_of_dims=config['dataset']['dimension'],
                                 **{'number_of_neighbors': 2 * 2 ** config['dataset']['dimension'], **parameters})
            projector = mmls.fit(data, options, config['device'])
            mmls.save_projector(
                projector,
                options,
                directory,
                split_id=split_id,
                export_points=export_points
            )
            return (['projector.json'] + (['training_points.npy'] if export_points else [])
                    + (['neighbors.pkl'] if projector.ann_index is not None else []))
        case _:
            raise ValueError('Fitting requires a model.')


def domain_data(dataset: Dataset, indices: np.ndarray | None) -> np.ndarray:
    """Select the requested samples and apply historical per-domain normalization.

    Args:
        dataset: Loaded dataset and its grid/split metadata.
        indices: Selected sample indices of shape (n), or None for the full grid.

    Returns:
        Selected sample array of shape (n_selected, *dataset.flat.shape[1:]), normalized unless the dataset supplies its
        own split.
    """
    data = dataset.flat if indices is None else dataset.flat[indices]
    # Toy manifolds already supply their analytic, normalized coordinates.
    return data if dataset.provided_train_count is not None else normalize(data)


def load_model(
    config: dict,
    directory: Path | BundlePath,
    training_points: np.ndarray | None
) -> BetaVAE_H | ANNMMLSProjector | None:
    """Restore the selected model for inference.

    Args:
        config: Experiment or sweep configuration.
        directory: Directory or virtual bundle path for this stage.
        training_points: Training samples of shape (n f), if not exported with the fit.

    Returns:
        Restored beta-VAE or MMLS projector, or None when no model is configured.
    """
    from ..models.beta_vae.inference import load_autoencoder_from_checkpoint
    from ..models.mmls.fitting import load_projector
    match config['model']['name']:
        case 'beta_vae':
            parameters = read_json(directory / 'model.json')
            return load_autoencoder_from_checkpoint(
                parameters['model'],
                directory / 'checkpoints/last.pth',
                latent_dim=parameters['z_dim'],
                device=config['device']
            )
        case 'mmls':
            np.random.seed(config['training']['seed'])
            return load_projector(directory, training_points=training_points, device=config['device'])
        case _:
            return None


def predictions(
    config: dict,
    model: BetaVAE_H | ANNMMLSProjector | None,
    data: np.ndarray,
    representation: str
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield arrays to the analyses without saving intermediate outputs.

    Args:
        config: Experiment or sweep configuration.
        model: Autoencoder used for inference.
        data: Input samples of shape (n ...), with sample axis first.
        representation: Requested input, latent, layer or output representation.

    Yields:
        (start_index, values), with each NumPy array of shape (b, *representation_shape). Beta-VAE images have shape (b
        1 h w), latent means (b z), and MMLS output retains the input shape (n ...).
    """
    from ..models.beta_vae.inference import batches
    from ..models.mmls.inference import predict
    match config['model']['name']:
        case 'beta_vae':
            options = config['inference']
            for start, output in batches(
                model,
                data,
                [representation],
                batch_size=options['batch_size'],
                device=config['device'],
                num_workers=config['num_workers'],
                seed=options['seed'],
                decoder_sampling=options['decoder_sampling']
            ):
                yield start, output[representation]
        case 'mmls':
            if representation != 'output':
                raise ValueError('MMLS exposes input and output only.')
            # Preserve the original neighbor branch, which depends on query count.
            np.random.seed(config['inference']['seed'])
            yield 0, predict(model, data)
        case _:
            raise ValueError('Inference requires a model.')


def analyze(
    config: dict,
    dataset: Dataset,
    indices: dict,
    train_count: int,
    model: BetaVAE_H | ANNMMLSProjector | None,
    root: Path | BundlePath,
    model_id: str | None
) -> list[dict]:
    """Run configured analyses over selected data domains and representations.

    Args:
        config: Experiment or sweep configuration.
        dataset: Loaded dataset and its grid/split metadata.
        indices: Train/test/grid selections and the split identifier; index arrays have shape (n).
        train_count: Number of training samples.
        model: Autoencoder used for inference.
        root: Root directory for experiment artifacts.
        model_id: Model-stage identifier, or None for input-only analyses.

    Returns:
        List of result records for each configured analysis, domain and representation.
    """
    from ..analyses.errors import pointwise_distances, summarize
    from ..analyses.geometry import compute
    from ..models.beta_vae.inference import representation_names
    results = []
    requests = config['analyses']
    if not requests and config['model']['name'] and config['inference']['enabled']:
        requests = [dict(name='inference', domain=config['inference']['domain'],
                         representations=config['inference']['representations'])]
    for analysis in requests:
        names = analysis['representations']
        available = (representation_names(model) if config['model']['name'] == 'beta_vae'
                     else ['input', 'output'] if model is not None else ['input'])
        names = available if names == 'all' else names
        if not set(names).issubset(available):
            raise ValueError(f'Unknown representations: {set(names) - set(available)}')
        domain = analysis['domain']
        selection = indices[domain]
        limit = config['inference'].get('evaluation_size')
        if domain == 'test' and limit is not None and len(selection) > limit:
            selection = np.sort(np.random.default_rng(config['inference']['seed']).choice(
                selection, size=limit, replace=False))
        data = None
        for representation in names:
            inference_config = dict(
                model=model_id if representation != 'input' else None,
                source=source_identity(config['dataset']),
                domain=domain,
                split=indices['split_id'] if domain != 'grid' else None,
                representation=representation,
                options=({k: v for k, v in config['inference'].items()
                          if k not in ('representations', 'domain', 'enabled')}
                         if representation != 'input' else {})
            )
            inference_id = identity(inference_config)
            label = representation_label(config, domain, representation)
            inference_dir = stage_directory(root / 'inference-outputs', inference_id, label)
            analysis_config = dict(inference=inference_id, analysis={k: v for k, v in analysis.items()
                                                                   if k != 'representations'})
            analysis_id = identity(analysis_config)
            directory = stage_directory(root / 'analysis-outputs', analysis_id, f"{label}__{analysis['name']}")
            record = dict(
                id=analysis_id,
                name=analysis['name'],
                domain=domain,
                representation=representation,
                directory=str(directory),
                dataset=config['dataset'],
                model=config['model']['name'],
                dimension=config['dataset']['dimension'],
                model_id=model_id,
                split_id=indices['split_id'],
                trial=config['training']['trial']
            )
            if not begin(directory, analysis_config):
                results.append(record)
                continue
            analysis_started = perf_counter()
            if data is None:
                data = domain_data(dataset, selection)
            inference_new = representation != 'input' and begin(inference_dir, inference_config)
            saved = inference_dir / 'values.npy'
            persist = config['inference']['persist'] and representation != 'input'
            temporary = inference_dir / 'working.npy'
            values = None
            distances = []
            try:
                inference_started = perf_counter()
                reused_values = saved.is_file() or representation == 'input'
                if representation == 'input':
                    values = data
                    batches = [(0, data)]
                elif reused_values:
                    values = load_array(saved, mmap_mode='r')
                    batches = [(0, values)]
                else:
                    batches = predictions(config, model, data, representation)
                for start, batch in batches:
                    if analysis['name'] == 'errors':
                        distances.append(pointwise_distances(data[start:start + len(batch)], batch))
                    if (analysis['name'] != 'errors' or persist) and values is None:
                        shape = (len(data), *batch.shape[1:])
                        if (persist or config['inference']['buffer'] == 'temporary') and isinstance(saved, Path):
                            path = saved if persist else temporary
                            values = np.lib.format.open_memmap(
                                path,
                                mode='w+',
                                dtype=batch.dtype,
                                shape=shape
                            )
                        else:
                            values = np.empty(shape, dtype=batch.dtype)
                    if values is not None and not reused_values:
                        values[start:start + len(batch)] = batch
                if isinstance(values, np.memmap):
                    values.flush()
                if persist and not isinstance(saved, Path) and not reused_values:
                    save_array(saved, values)
                inference_seconds = perf_counter() - inference_started
                if inference_new:
                    write_json(inference_dir / 'summary.json', dict(
                        number_of_points=len(data),
                        representation=representation,
                        persisted=persist,
                        shape=[len(data), *batch.shape[1:]],
                        dtype=str(batch.dtype)
                    ))
                    finish(inference_dir, ['summary.json'] + (['values.npy'] if persist else []),
                           elapsed_seconds=inference_seconds)
                elif representation != 'input' and not reused_values:
                    # Metadata-only completion cannot supply arrays to a new analysis.
                    print(f'[inference-outputs] recomputed {inference_id} in {inference_seconds:.2f}s')
                match analysis['name']:
                    case 'errors':
                        distances = np.concatenate(distances)
                        write_json(directory / 'metrics.json', summarize(distances, train_count))
                        save_array(directory / 'pointwise_distances.npy', distances)
                        files = ['metrics.json', 'pointwise_distances.npy']
                    case 'geometry':
                        options = {k: v for k, v in analysis.items()
                                   if k not in ('name', 'domain', 'representations')}
                        measures = compute(
                            values,
                            dataset,
                            device=config['device'],
                            **options
                        )
                        with (directory / 'measures.pkl').open('wb') as stream:
                            pickle.dump(measures, stream, -1)
                        files = ['measures.pkl']
                    case 'inference':
                        write_json(directory / 'inference.json', dict(directory=str(inference_dir)))
                        files = ['inference.json']
                write_json(directory / 'result.json', record)
                finish(directory, files + ['result.json'],
                       elapsed_seconds=perf_counter() - analysis_started - inference_seconds)
            finally:
                del values
                if temporary.exists():
                    temporary.unlink()
            results.append(record)
    return results


def completed_result(
    config: dict,
    root: Path,
    source: dict | None = None
) -> dict | None:
    """Return a reusable experiment result when all required stages are complete.

    Args:
        config: Experiment or sweep configuration.
        root: Root directory for experiment artifacts.
        source: Known dataset identity, or None to inspect the configured source.

    Returns:
        Reusable completed-experiment result dictionary, or None if required stages are missing or incompatible.
    """
    source = source_identity(config['dataset']) if source is None else source
    paths = experiment_paths(output_root(config, root), config, source)
    path = paths['configuration'] / 'result.json'
    if not path.is_file():
        return None
    result = read_json(path)
    directories = [paths['split']]
    if result['model_id'] is not None and result.get('status') != 'skipped':
        directories.append(paths['model'])
    directories.extend(artifact_path(record['directory']) for record in result['analyses'])
    if not all(is_complete(directory) for directory in directories):
        return None
    for directory in directories:
        report_reuse(directory)
    return result


def _run(
    config: dict,
    root: Path,
    dataset: Dataset | None,
    artifacts: Path | BundlePath
) -> dict:
    """One execution path for both directory and bundled stage artifacts.

    Args:
        config: Experiment or sweep configuration.
        root: Root directory for experiment artifacts.
        dataset: Loaded dataset and its grid/split metadata.
        artifacts: Root for ordinary or bundled stage artifacts.

    Returns:
        Experiment result dictionary containing its stage identifiers, artifact paths and analysis records.
    """
    source = source_identity(config['dataset'])
    split_id, model_id = stage_ids(config, source)
    experiment_id = experiment_identity(config, source)
    paths = experiment_paths(artifacts, config, source)
    configuration_path = paths['configuration'] / 'config.json'
    completed = completed_result(config, root, source)
    if completed is not None:
        return completed
    check_device(config['device'])
    for stage in STAGES:
        (artifacts / stage).mkdir(parents=True, exist_ok=True)
    configuration_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(configuration_path, config)
    dataset = load_dataset(config['dataset'], root, config['device']) if dataset is None else dataset
    split_dir = paths['split']
    if begin(split_dir, dict(dataset=source, split=config['split'])):
        train, test = prepare_indices(dataset, config, root, source)
        with (split_dir / 'indices.npz').open('wb') as stream:
            np.savez(stream, train=train, test=test)
        finish(split_dir, ['indices.npz'])
    with (split_dir / 'indices.npz').open('rb') as stream, np.load(stream) as split:
        train, test = split['train'], split['test']
    if (config['model']['name'] == 'mmls'
            and len(train) < config['model']['parameters']['number_of_neighbors']):
        result = dict(
            experiment_id=experiment_id,
            split_id=split_id,
            model_id=model_id,
            configuration=str(configuration_path),
            analyses=[],
            status='skipped',
            reason='Fewer training points than MMLS neighbors.'
        )
        write_json(configuration_path.with_name('result.json'), result)
        print(f"Skipping {experiment_id}: {result['reason']}")
        return result
    model_dir = paths['model']
    if config['model']['name'] is not None:
        if not config['training']['enabled'] and not is_complete(model_dir):
            raise RuntimeError(f'Training disabled but required fit is missing: {model_dir}')
        model_config = dict(
            split=split_id,
            model=config['model'],
            seed=config['training']['seed'],
            trial=config['training']['trial']
        )
        if begin(model_dir, model_config, requires=[split_dir]):
            points = domain_data(dataset, train)
            files = fit_model(config, points, model_dir, split_id)
            finish(model_dir, files)
        points = domain_data(dataset, train) if config['model']['name'] == 'mmls' else None
        model = load_model(config, model_dir, points)
    else:
        model, model_id = None, None
    indices = dict(
        train=train,
        test=test,
        grid=None,
        split_id=split_id
    )
    records = analyze(config, dataset, indices, len(train), model, artifacts, model_id)
    result = dict(
        experiment_id=experiment_id,
        split_id=split_id,
        model_id=model_id,
        configuration=str(configuration_path),
        analyses=records
    )
    write_json(configuration_path.with_name('result.json'), result)
    return result


def run(
    config: dict,
    output_dir: str | Path,
    dataset: Dataset | None = None,
    quick_run: bool = False
) -> dict:
    """Run a dictionary configuration. An optional loaded Dataset avoids repeated IO.

    Args:
        config: Experiment or sweep configuration.
        output_dir: Directory for generated outputs.
        dataset: Loaded dataset to reuse, or None to load from the configuration.
        quick_run: Apply the shared reduced experiment budgets.

    Returns:
        Experiment result dictionary containing its stage identifiers, artifact paths and analysis records.
    """
    config = apply_quick_run(config) if quick_run else resolve(config)
    root = Path(output_dir).expanduser().resolve()
    artifacts = output_root(config, root)
    if isinstance(artifacts, Path):
        return _run(
            config,
            root,
            dataset=dataset,
            artifacts=artifacts
        )
    with artifacts.bundle.transaction(experiment_identity(config, source_identity(config['dataset']))):
        return _run(
            config,
            root,
            dataset=dataset,
            artifacts=artifacts
        )


@app.command()
def main(
    config: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(...),
    quick_run: bool = False,
    device: Optional[str] = typer.Option(None, help='Override the configured device, e.g. cuda:0.')
) -> None:
    """Run one YAML/JSON experiment configuration.

    Args:
        config: Path to a YAML or JSON experiment configuration.
        output_dir: Directory for generated outputs.
        quick_run: Apply the shared reduced experiment budgets.
        device: Optional device override, such as cpu or cuda:0.

    Returns:
        None.
    """
    configuration = load_config(config)
    if device is not None:
        configuration['device'] = device
    if quick_run:
        configuration = apply_quick_run(configuration)
    run(configuration, output_dir)


if __name__ == '__main__':
    app()
