"""Quick run for Figure 4: six sizes and one repetition, reusing Figure 3/6 fits."""
import argparse
import json
from pathlib import Path

from .common import experiment_name, is_complete, run_stage as stage

FRACTIONS = (0.05, 0.2, 0.4, 0.6, 0.8, 1.0)
DATASET = 'custom_dsprites_balanced'
REPETITION = 2
SEED = 473959


def reuse(source, target):
    """Link a completed input without modifying it or replacing existing paths."""
    source = Path(source).resolve()
    if not is_complete(source):
        raise RuntimeError(f'Required input is incomplete: {source}')
    if target.is_symlink() and target.resolve() == source:
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(f'Reuse target already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)


def reproduce(output_root, figure3_root, figure6_root, *, device='cuda:0', num_workers=2):
    root = Path(output_root).expanduser().resolve()
    figure3, figure6 = Path(figure3_root).resolve(), Path(figure6_root).resolve()
    if root in (figure3, figure6):
        raise ValueError('Use a separate output root to preserve the source experiments.')
    full_name = experiment_name(DATASET, 'beta_vae', 1.0, 4, REPETITION)
    full_model = figure6 / 'fits' / full_name
    if not is_complete(full_model):
        raise RuntimeError(f'Required training is incomplete: {full_model}')
    config = json.loads((full_model / 'config.json').read_text())
    if config['seed'] != SEED:
        raise ValueError(f'The reused repetition must have seed {SEED}.')
    dataset = Path(config['model_config']['exported_datasets_dir']).resolve()
    inputs = [(dataset, root / 'datasets' / f'{DATASET}__4'),
              (full_model, root / 'fits' / full_name)]
    for fraction in FRACTIONS:
        name = experiment_name(DATASET, 'MMLS', fraction, 4, REPETITION)
        inputs.append((figure3 / 'fits' / name, root / 'fits' / name))
    # Check all required reuse inputs before creating any experiment outputs.
    for source, _ in inputs:
        if not is_complete(source):
            raise RuntimeError(f'Required input is incomplete: {source}')
    for source, target in inputs:
        reuse(source, target)

    for fraction in FRACTIONS:
        name = experiment_name(DATASET, 'beta_vae', fraction, 4, REPETITION)
        model = root / 'fits' / name
        if fraction != 1.0:
            stage('fitting', model, 'vae', '--dataset', DATASET, '--dimension', 4,
                  '--datasets-dir', dataset, '--fraction', fraction, '--seed', SEED,
                  '--device', device, '--num-workers', num_workers)
        stage('evaluation', root / 'evaluation' / name, '--model-dir', model,
              '--device', device, '--num-workers', num_workers)
    stage('bounds', root / 'bounds/figure_4', '--figure', 4, '--results-root', root)
    stage('plots', root / 'figures/figure_4', '--figure', 4,
          '--bounds-dir', root / 'bounds/figure_4')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--figure3-root', type=Path, required=True)
    parser.add_argument('--figure6-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()
    reproduce(args.output_root, args.figure3_root, args.figure6_root,
              device=args.device, num_workers=args.num_workers)


if __name__ == '__main__':
    main()
