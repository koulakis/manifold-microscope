"""Dataset export and single-model fits, using the existing numerical code."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path

from .common import run_experiment, write_json


def export_dataset(output, dataset, dimension, *, max_test_size=500, device='cuda:0'):
    def action(directory):
        from microscope.datasets.generic_dataset_loader import export_fixed_grid_test_set_and_rest_for_train
        export_fixed_grid_test_set_and_rest_for_train(
            dataset_name=dataset, number_of_dims=dimension, output_dir=directory,
            max_test_size=max_test_size, device=device)
        write_json(directory / 'config.json', dict(dataset=dataset, dimension=dimension,
                                                   max_test_size=max_test_size, device=device))
        return [directory / f'{dataset}__{dimension}.npz']

    return run_experiment(output, action)


def fit_mmls(output, datasets_dir, dataset, dimension, fraction, *, device='cuda:0'):
    datasets_dir = Path(datasets_dir).expanduser().resolve()

    def action(directory):
        from experiment_scripts.model_configs import MMLSConfig
        from experiment_scripts.manifold_fitting.mmls import fit_mmls as fit
        config = MMLSConfig(
            output_dir=str(directory), exported_datasets_dir=Path(datasets_dir),
            dataset=dataset, number_of_dims=dimension, training_ratio=fraction,
            ratio_per_dim=False, noise_sigma=0, number_of_neighbors=2 * 2 ** dimension,
            device=device)
        write_json(directory / 'config.json', asdict(config))
        fit(config)
        return list(directory.glob('distance_results_*.pkl'))

    return run_experiment(output, action, requires=[datasets_dir])


def fit_vae(output, datasets_dir, dataset, dimension, fraction, seed, *,
            device='cuda:0', num_workers=2, max_iter=None, max_epochs=None):
    datasets_dir = Path(datasets_dir).expanduser().resolve()

    def action(directory):
        import numpy as np
        import torch
        from experiment_scripts.model_configs import BetaVAEConfig
        from representation_learning.beta_vae.solver import Solver
        config = BetaVAEConfig(
            dataset=dataset, number_of_dims=dimension, training_ratio=fraction,
            ratio_per_dim=False, noise_sigma=0, ckpt_dir=directory / 'checkpoints',
            output_dir=directory, exported_datasets_dir=Path(datasets_dir),
            device=device, num_workers=num_workers)
        if dataset == 'extended_coil20':
            config = replace(config, model='H', objective='H', lr=1e-4, loss_threshold=10)
        overrides = {name: value for name, value in
                     [('max_iter', max_iter), ('max_epochs', max_epochs)] if value is not None}
        config = replace(config, **overrides)
        write_json(directory / 'config.json', dict(seed=seed, model_config=asdict(config)))
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        Solver(config).train()
        return [config.ckpt_dir / 'last.pth']

    return run_experiment(output, action, requires=[datasets_dir])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=['dataset', 'mmls', 'vae'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices=['custom_dsprites_balanced', 'extended_coil20'], required=True)
    parser.add_argument('--dimension', type=int, required=True)
    parser.add_argument('--datasets-dir', type=Path)
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=439)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-test-size', type=int, default=500)
    parser.add_argument('--max-iter', type=int)
    parser.add_argument('--max-epochs', type=int)
    args = parser.parse_args()
    if args.method == 'dataset':
        export_dataset(args.output, args.dataset, args.dimension,
                       max_test_size=args.max_test_size, device=args.device)
        return
    if args.datasets_dir is None:
        parser.error('--datasets-dir is required for fitting')
    if args.method == 'mmls':
        fit_mmls(args.output, args.datasets_dir, args.dataset, args.dimension, args.fraction, device=args.device)
    else:
        fit_vae(args.output, args.datasets_dir, args.dataset, args.dimension, args.fraction,
                args.seed, device=args.device, num_workers=args.num_workers,
                max_iter=args.max_iter, max_epochs=args.max_epochs)


if __name__ == '__main__':
    main()
