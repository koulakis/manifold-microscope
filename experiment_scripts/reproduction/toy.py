"""The original toy MMLS size sweep."""
import argparse
from pathlib import Path

from .common import run_experiment, write_json


def fit_toy(output, *, sizes=(25, 5005, 5), repetitions=20, ground_truth=500, workers=16,
            device='cuda:0', projection_method='thin_svd'):
    def action(directory):
        from experiment_scripts.toy_manifolds_experiment.fit_and_get_measures import fit_and_get_measures
        write_json(directory / 'config.json', dict(sizes=sizes, repetitions=repetitions,
                                                   ground_truth=ground_truth, workers=workers, device=device,
                                                   projection_method=projection_method))
        fit_and_get_measures(output_path=directory, N_range=tuple(sizes),
                             n_examples_per_size=repetitions, N_ground_truth=ground_truth,
                             max_workers=workers, fitting_method='MMLS', normalized=True, device=device,
                             projection_method=projection_method)
        return [directory / 'distance_and_measure_results.pkl']

    return run_experiment(output, action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs=3, default=(25, 5005, 5))
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--ground-truth', type=int, default=500)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--projection-method', choices=['original', 'thin_svd'], default='thin_svd',
                        help='MMLS projector (default: thin_svd; original uses covariance SVD).')
    args = parser.parse_args()
    fit_toy(args.output, sizes=args.sizes, repetitions=args.repetitions,
            ground_truth=args.ground_truth, workers=args.workers, device=args.device,
            projection_method=args.projection_method)


if __name__ == '__main__':
    main()
