"""Direct beta-VAE layer inference and normalized geometry analysis."""
import argparse
import json
from pathlib import Path

from .common import run_experiment, write_json


def infer(output, model_dir, *, device='cuda:0', num_workers=2, seed=None, decoder_sampling=True):
    model_dir = Path(model_dir).expanduser().resolve()

    def action(directory):
        from microscope.datasets.generic_dataset_loader import DatasetName
        from representation_learning.beta_vae.inference_intermediate_layers import main as inference
        config = json.loads((model_dir / 'config.json').read_text())['model_config']
        write_json(directory / 'config.json', dict(model_directory=str(model_dir),
                                                   dataset=config['dataset'], dimension=config['number_of_dims'],
                                                   device=device, num_workers=num_workers,
                                                   seed=seed, decoder_sampling=decoder_sampling))
        inference(dataset=DatasetName(config['dataset']), number_of_dims=config['number_of_dims'],
                  only_latent_and_output=False, checkpoint_path=model_dir / 'checkpoints/last.pth',
                  output_dir=directory, random_model=False, batch_size=config['batch_size'],
                  num_workers=num_workers, latent_dim=config['z_dim'], model_type=config['model'],
                  device=device, seed=seed, decoder_sampling=decoder_sampling)
        return [directory / 'intermediate_outputs_last.pth.npz']

    return run_experiment(output, action, requires=[model_dir])


def analyze(output, inference_dir, *, device='cuda:0'):
    inference_dir = Path(inference_dir).expanduser().resolve()

    def action(directory):
        from microscope.datasets.generic_dataset_loader import DatasetName
        from microscope.computations_grid.data_analysis.run_data_analysis import main as analysis
        config = json.loads((inference_dir / 'config.json').read_text())
        analysis(inference_path=inference_dir, output_path=directory,
                 dataset=DatasetName(config['dataset']), model_type='beta_vae',
                 number_of_dims=config['dimension'], normalize_for_volume=True, only_evolution=True,
                 device=device)
        write_json(directory / 'config.json', dict(inference_directory=str(inference_dir),
                                                   normalize_for_volume=True, device=device))
        # Check all requested layer files, not just the last one.
        import numpy as np
        with np.load(inference_dir / 'intermediate_outputs_last.pth.npz') as outputs:
            return [directory / 'measures' / f'intermediate_outputs_last.pth__{layer}.pkl'
                    for layer in outputs.files]

    return run_experiment(output, action, requires=[inference_dir])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['infer', 'analyze'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--decoder-sampling', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.stage == 'infer':
        infer(args.output, args.input, device=args.device, num_workers=args.num_workers,
              seed=args.seed, decoder_sampling=args.decoder_sampling)
    else:
        analyze(args.output, args.input, device=args.device)


if __name__ == '__main__':
    main()
