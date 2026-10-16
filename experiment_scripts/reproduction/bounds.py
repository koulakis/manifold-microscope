"""Notebook-derived curve aggregation and the original fitted bound shapes."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
import warnings

from .common import is_complete, run_experiment, write_json

DS = 'custom_dsprites_balanced'
COIL = 'extended_coil20'
FILENAMES = {3: '28012026_dsprites_dim_bounds', 4: '28012026_all_models_bounds',
             5: '28012026_all_datasets_bounds'}


def _rows(parent, dataset, method, dimension, filename):
    rows = []
    for directory in sorted(Path(parent).glob(f'{dataset}__{method}__*')):
        if int(directory.name.split('__')[3]) != dimension:
            continue
        if not is_complete(directory):
            warnings.warn(f'Ignoring unfinished result: {directory}', stacklevel=2)
            continue
        path = next(directory.glob(filename))
        if path.suffix == '.json':
            rows.append(json.loads(path.read_text()))
        else:
            with path.open('rb') as stream:
                rows.append(pickle.load(stream))
    return rows


def _curve(rows, dimension):
    import numpy as np
    from sklearn.linear_model import LinearRegression

    grouped = defaultdict(list)
    for row in rows:
        grouped[row['number_of_train_points']].append(row['hausdorff_distance'])
    x = np.array(sorted(grouped), dtype=float)
    if len(x) < 2:
        raise ValueError('At least two completed training sizes are required for each bound curve.')
    repetitions = [grouped[int(n)] for n in x]
    empirical = np.array([np.mean(values) for values in repetitions])
    if not np.isfinite(empirical).all() or np.any(empirical <= 0) or np.any(x <= 1):
        raise ValueError('Log-log bounds require finite positive errors and training counts > 1.')
    regression = LinearRegression().fit(np.log(x).reshape(-1, 1), np.log(empirical))
    fitted = np.exp(regression.intercept_) * x ** regression.coef_[0]
    lower = (1 / x) ** (2 / (2 + dimension))
    upper = (np.log(x) / x) ** (2 / (2 + dimension))
    fefferman = (np.log(x) / x) ** (1 / dimension)
    # These are the active notebook formulas; its unused reach loads are omitted.
    exported = dict(x=x, empirical=empirical, empirical_all_repetitions=repetitions,
                    lower_genovese=np.quantile(fitted / lower, 0.01) * lower,
                    upper_genovese=np.quantile(fitted / upper, 0.99) * upper,
                    upper_fefferman=np.quantile(fitted / fefferman, 0.99) * fefferman)
    summary = dict(dimension=dimension, training_counts=x.tolist(),
                   repetition_counts=list(map(len, repetitions)),
                   alpha=float(-regression.coef_[0]), coefficient=float(np.exp(regression.intercept_)))
    return exported, summary


def export(output, results_root, figure):
    """Export completed results for Figure 3, 4 or 5."""
    root = Path(results_root)

    def action(directory):
        import numpy as np

        def mmls(dataset, dimension):
            return _rows(root / 'fits', dataset, 'MMLS', dimension, 'distance_results_*.pkl')

        if figure == 3:
            panels = [(f'dim_{d}', d, mmls(DS, d)) for d in range(1, 5)]
        elif figure == 4:
            panels = [('mmls', 4, mmls(DS, 4)),
                      ('beta_vae', 4, _rows(root / 'evaluation', DS, 'beta_vae', 4, 'metrics.json'))]
        elif figure == 5:
            toy_dir = root / 'toy'
            if not is_complete(toy_dir):
                warnings.warn(f'Cannot export Figure 5: unfinished toy experiment {toy_dir}', stacklevel=2)
                return []
            with (toy_dir / 'distance_and_measure_results.pkl').open('rb') as stream:
                toy = pickle.load(stream)
            panels = [(name.lower(), 2, [dict(number_of_train_points=n, hausdorff_distance=error)
                                         for n, errors in toy[name]['hausdorff_distances'].items()
                                         for error in errors]) for name in ('Sphere', 'Torus')]
            panels.extend([(COIL, 3, mmls(COIL, 3)), (DS, 4, mmls(DS, 4))])
        else:
            raise ValueError('Bound figures are 3, 4 and 5.')
        curves, summaries = {}, {}
        for name, dimension, rows in panels:
            curves[name], summaries[name] = _curve(rows, dimension)
        path = directory / (FILENAMES[figure] + '.npz')
        np.savez(path, **curves)
        write_json(directory / 'counts_and_regression.json', summaries)
        return [path, directory / 'counts_and_regression.json']

    return run_experiment(output, action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', type=int, choices=[3, 4, 5], required=True)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.output, args.results_root, args.figure)


if __name__ == '__main__':
    main()
