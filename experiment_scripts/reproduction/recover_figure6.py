"""Recover Figure 6 after training, using a fresh process for each layer.

Completed inputs are reused read-only. Existing incomplete output directories
are warned about and skipped; choose a new output root for a failed retry.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from .common import is_complete, run_experiment, write_json


def layer_names(inference):
    with zipfile.ZipFile(Path(inference) / 'intermediate_outputs_last.pth.npz') as archive:
        return [Path(name).stem for name in archive.namelist() if name.endswith('.npy')]


def analyze_layer(inference, output, layer, device):
    inference = Path(inference).resolve()

    def action(directory):
        import numpy as np
        from microscope.datasets.generic_dataset_loader import DatasetName, load_dataset
        from microscope.datasets import custom_dsprites, coil20
        from microscope.computations_grid.data_analysis.run_data_analysis import run_single_output

        config = json.loads((inference / 'config.json').read_text())
        dataset = DatasetName(config['dataset'])
        data, target, ranges, cyclic, patches = load_dataset(
            dataset_name=dataset, number_of_dims=config['dimension'],
            ratio_per_dim=False, training_ratio=1.0, noise_sigma=0.0,
            save_train_idx=False, weight_subsampling_by_manifold_volume=False,
            return_full_datasets=True, return_dataset_unflat_with_metadata=True,
            full_datasets_unclipped=True, device=device)
        grid_shape = data.shape[:-2]
        del data, target  # Only grid metadata is needed for layer measures.
        labels = (custom_dsprites.LABEL_MAPPING if dataset == DatasetName.custom_dsprites_balanced
                  else coil20.LABEL_MAPPING)
        with np.load(inference / 'intermediate_outputs_last.pth.npz') as archive:
            features = archive[layer]  # Read this member once, not all layers.
        features = features.reshape(*grid_shape, -1)
        measures = directory / 'measures'
        measures.mkdir()
        name = f'intermediate_outputs_last.pth__{layer}'
        print(f'Analyzing {dataset.value}: {layer}, shape={features.shape}', flush=True)
        run_single_output(name, features, list(ranges), list(cyclic), list(patches),
                          list(labels.values()), directory, directory, directory, measures,
                          normalize_for_volume=True, export_plots=False, device=device)
        write_json(directory / 'config.json', dict(inference_directory=str(inference),
                   layer=layer, device=device, normalize_for_volume=True))
        return [measures / f'{name}.pkl']

    return run_experiment(output, action, requires=[inference])


def collect_layers(output, parts, names):
    parts = Path(parts).resolve()

    def action(directory):
        (directory / 'measures').mkdir()
        files = []
        for layer in names:
            filename = f'intermediate_outputs_last.pth__{layer}.pkl'
            source = parts / layer / 'measures' / filename
            target = directory / 'measures' / filename
            target.symlink_to(source)
            files.append(target)
        write_json(directory / 'config.json', dict(parts=str(parts), layers=names))
        return files

    return run_experiment(output, action, requires=[parts / layer for layer in names])


def recover(source_root, output_root, device):
    source, output = Path(source_root).resolve(), Path(output_root).resolve()
    if source == output:
        raise ValueError('Use a separate recovery output root to preserve the failed run.')
    jobs = [('dsprites', 'custom_dsprites_balanced__beta_vae__1.0__4__0__2'),
            ('coil', 'extended_coil20__beta_vae__1.0__3__0__0')]

    def call(module, *args):
        subprocess.run([sys.executable, '-u', '-m', module, *map(str, args)], check=True)

    for label, model_name in jobs:
        model = source / 'fits' / model_name
        if not is_complete(model):
            raise RuntimeError(f'Training is not complete: {model}')
        inference = source / 'inference' / label
        if not is_complete(inference):
            inference = output / 'inference' / label
            call('experiment_scripts.reproduction.layers', 'infer', '--input', model,
                 '--output', inference, '--device', device, '--num-workers', 0)
        if not is_complete(inference):
            raise RuntimeError(f'Inference is incomplete: {inference}')
        names = layer_names(inference)
        for layer in names:
            call(__spec__.name, 'layer', '--input', inference,
                 '--output', output / 'analysis_parts' / label / layer,
                 '--layer', layer, '--device', device)
        if not collect_layers(output / 'analysis' / label, output / 'analysis_parts' / label, names):
            raise RuntimeError(f'Analysis is incomplete: {label}')

    call('experiment_scripts.reproduction.plots', '--figure', 6,
         '--output', output / 'figures/figure_6',
         '--dsprites-analysis', output / 'analysis/dsprites',
         '--coil-analysis', output / 'analysis/coil')
    if not is_complete(output / 'figures/figure_6'):
        raise RuntimeError('Figure 6 rendering is incomplete.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['run', 'layer'])
    parser.add_argument('--input', type=Path, required=True,
                        help='Original run root for run; inference directory for layer.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layer')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.stage == 'run':
        recover(args.input, args.output, args.device)
    elif args.layer is None:
        parser.error('layer stage requires --layer')
    elif not analyze_layer(args.input, args.output, args.layer, args.device):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
