"""Full run for Figure 4: all 13 training sizes and three repetitions."""
import argparse
from pathlib import Path

from .common import experiment_name, run_stage as stage
from .paper import DS, experiments


def reproduce(output_root, *, device='cuda:0', num_workers=2, projection_method='thin_svd'):
    root = Path(output_root).expanduser().resolve()
    dataset = root / 'datasets' / f'{DS}__4'
    stage('fitting', dataset, 'dataset', '--dataset', DS, '--dimension', 4,
          '--max-test-size', 500, '--device', device)
    for method, name, dimension, fraction, repetition, seed in experiments([4]):
        model_name = experiment_name(name, method, fraction, dimension, repetition)
        model = root / 'fits' / model_name
        args = ['--dataset', name, '--dimension', dimension, '--fraction', fraction,
                '--datasets-dir', dataset, '--device', device]
        if method == 'MMLS':
            stage('fitting', model, 'mmls', *args, '--projection-method', projection_method)
        else:
            stage('fitting', model, 'vae', *args, '--seed', seed, '--num-workers', num_workers)
            stage('evaluation', root / 'evaluation' / model_name, '--model-dir', model,
                  '--device', device, '--num-workers', num_workers)
    stage('bounds', root / 'bounds/figure_4', '--figure', 4, '--results-root', root)
    stage('plots', root / 'figures/figure_4', '--figure', 4,
          '--bounds-dir', root / 'bounds/figure_4')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--projection-method', choices=['original', 'thin_svd'], default='thin_svd')
    args = parser.parse_args()
    reproduce(args.output_root, device=args.device, num_workers=args.num_workers,
              projection_method=args.projection_method)


if __name__ == '__main__':
    main()
