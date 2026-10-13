"""The recovered experiment recipe and final plots for paper Figures 3–6."""
import argparse
from pathlib import Path

from . import bounds, evaluation, fitting, layers, plots, toy
from .common import experiment_name

FRACTIONS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DS = 'custom_dsprites_balanced'
COIL = 'extended_coil20'
DS_SEEDS = (439, 437875, 473959)
STAGES = ('fit', 'evaluate', 'infer', 'analyze', 'export', 'plot')


def experiments(figures):
    """Return shared image jobs once, selected by target figure."""
    jobs = []
    ds_dims = range(1, 5) if 3 in figures else ([4] if any(f in figures for f in (4, 5)) else [])
    for dataset, dims, repetitions in [(DS, ds_dims, range(3)),
                                       (COIL, [3] if 5 in figures else [], range(2))]:
        for dimension in dims:
            for fraction in FRACTIONS:
                if dimension == 1 and fraction < 0.1:
                    continue
                for repetition in repetitions:
                    jobs.append(('MMLS', dataset, dimension, fraction, repetition, None))
    if 4 in figures or 6 in figures:
        fractions = FRACTIONS if 4 in figures else (1.0,)
        repetitions = range(3) if 4 in figures else (2,)
        for fraction in fractions:
            for repetition in repetitions:
                jobs.append(('beta_vae', DS, 4, fraction, repetition, DS_SEEDS[repetition]))
    if 6 in figures:
        jobs.append(('beta_vae', COIL, 3, 1.0, 0, 438))
    return jobs


def reproduce(output_root, *, figures=(3, 4, 5, 6), stages=STAGES,
              device='cuda:0', num_workers=2):
    root = Path(output_root).expanduser().resolve()
    if 6 in figures and 'analyze' in stages and device != 'cuda:0':
        raise ValueError('Figure 6 analysis currently requires cuda:0; existing device plumbing is deferred.')
    if 5 in figures and 'fit' in stages and device != 'cuda:0':
        raise ValueError('Figure 5 toy fitting currently uses cuda:0 internally; device plumbing is deferred.')
    jobs = experiments(figures)

    if 'fit' in stages:
        for dataset, dimension in sorted({(job[1], job[2]) for job in jobs}):
            fitting.export_dataset(root / 'datasets' / f'{dataset}__{dimension}',
                                   dataset, dimension, max_test_size=500, device=device)
        for method, dataset, dimension, fraction, repetition, seed in jobs:
            output = root / 'fits' / experiment_name(dataset, method, fraction, dimension, repetition)
            dataset_dir = root / 'datasets' / f'{dataset}__{dimension}'
            if method == 'MMLS':
                fitting.fit_mmls(output, dataset_dir, dataset, dimension, fraction, device=device)
            else:
                fitting.fit_vae(output, dataset_dir, dataset, dimension, fraction, seed,
                                device=device, num_workers=num_workers)
        if 5 in figures:
            toy.fit_toy(root / 'toy', sizes=(25, 5005, 5), repetitions=20,
                        ground_truth=500, workers=16)

    if 'evaluate' in stages and 4 in figures:
        for method, dataset, dimension, fraction, repetition, _ in jobs:
            if method == 'beta_vae' and dataset == DS:
                name = experiment_name(dataset, method, fraction, dimension, repetition)
                evaluation.evaluate(root / 'evaluation' / name, root / 'fits' / name,
                                    device=device, num_workers=num_workers)

    if 6 in figures:
        for label, dataset, dimension, repetition in [('dsprites', DS, 4, 2), ('coil', COIL, 3, 0)]:
            model = root / 'fits' / experiment_name(dataset, 'beta_vae', 1.0, dimension, repetition)
            inference = root / 'inference' / label
            if 'infer' in stages:
                layers.infer(inference, model, device=device, num_workers=num_workers)
            if 'analyze' in stages:
                layers.analyze(root / 'analysis' / label, inference, device=device)

    for figure in figures:
        if figure != 6:
            exported = root / 'bounds' / f'figure_{figure}'
            if 'export' in stages:
                bounds.export(exported, root, figure)
            if 'plot' in stages:
                plots.plot_bounds(root / 'figures' / f'figure_{figure}', exported, figure)
        elif 'plot' in stages:
            plots.plot_layers(root / 'figures/figure_6', root / 'analysis/dsprites', root / 'analysis/coil')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--figures', type=int, nargs='+', choices=[3, 4, 5, 6], default=[3, 4, 5, 6])
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()
    reproduce(args.output_root, figures=args.figures, stages=args.stages,
              device=args.device, num_workers=args.num_workers)


if __name__ == '__main__':
    main()
