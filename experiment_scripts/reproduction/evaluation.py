"""Fixed-test beta-VAE pointwise errors, extracted from the paper notebook."""
import argparse
import json
from pathlib import Path

from .common import run_experiment, write_json


def evaluate(output, model_dir, *, device='cuda:0', num_workers=2):
    model_dir = Path(model_dir).expanduser().resolve()

    def action(directory):
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        from microscope.datasets.generic_dataset_loader import load_dataset_fixed_test_split
        from representation_learning.beta_vae.dataset import CustomTensorDataset
        from representation_learning.beta_vae.inference_intermediate_layers import load_autoencoder_from_checkpoint
        config = json.loads((model_dir / 'config.json').read_text())['model_config']
        train, test, _, _ = load_dataset_fixed_test_split(
            datasets_dir=Path(config['exported_datasets_dir']), dataset_name=config['dataset'],
            number_of_dims=config['number_of_dims'], ratio_per_dim=False,
            training_ratio=config['training_ratio'], noise_sigma=0,
            weight_subsampling_by_manifold_volume=True)
        test = (test - test.min()) / (test.max() - test.min())
        loader = DataLoader(CustomTensorDataset(torch.from_numpy(test).unsqueeze(1).float()),
                            batch_size=config['batch_size'], shuffle=False,
                            num_workers=num_workers, pin_memory=device.startswith('cuda'))
        model = load_autoencoder_from_checkpoint(
            model_type=config['model'], checkpoint_path=model_dir / 'checkpoints/last.pth',
            latent_dim=config['z_dim'], device=device)
        distances = []
        with torch.no_grad():
            for images in loader:
                images = images.to(device)
                reconstruction, _, _ = model(images)
                error = (torch.sigmoid(reconstruction) - images).flatten(start_dim=1)
                distances.append(torch.linalg.vector_norm(error, dim=1).cpu().numpy())
        distances = np.concatenate(distances)
        result = dict(number_of_train_points=len(train), number_of_test_points=len(test),
                      hausdorff_distance=float(distances.max()), average_distance=float(distances.mean()),
                      pointwise_metric='max ||x - reconstruction(x)|| (Hausdorff proxy)',
                      dataset=config['dataset'], dimension=config['number_of_dims'],
                      fraction=config['training_ratio'], model_directory=str(model_dir))
        write_json(directory / 'metrics.json', result)
        return [directory / 'metrics.json']

    return run_experiment(output, action, requires=[model_dir])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()
    evaluate(args.output, args.model_dir, device=args.device, num_workers=args.num_workers)


if __name__ == '__main__':
    main()
