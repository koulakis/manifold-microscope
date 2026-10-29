"""The shared --quick-run profile; edit its budgets here."""

from __future__ import annotations

from copy import deepcopy

from .config import resolve


QUICK_RUN = dict(
    vae_max_iter=10_000,
    train_fractions=[0.2, 0.5, 1.0],  # Three sizes for curve fitting.
    repetitions=1,
    evaluation_size=100,
    toy_sample_sizes=[25, 250, 1000],
    toy_repetitions=1,
    toy_reference_points=100,
    reach_subsample=4,
)


def quick_grid(grid: dict) -> dict:
    """Reduce size/repetition axes, retaining model/dataset/layer choices.

    Args:
        grid: Sweep axes mapped to choices or integer ranges.
    """
    grid = deepcopy(grid)
    for path, sizes in [('split.train_fraction', QUICK_RUN['train_fractions']),
                        ('dataset.sample_size', QUICK_RUN['toy_sample_sizes'])]:
        if path in grid and (isinstance(grid[path], dict) or len(grid[path]) > len(sizes)):
            grid[path] = list(sizes)
    if 'repetition' in grid:
        count = QUICK_RUN['toy_repetitions'] if 'dataset.sample_size' in grid else QUICK_RUN['repetitions']
        # The last paper dSprites trial is also used by the layer figure.
        grid['repetition'] = grid['repetition'][-count:]
    return grid


def apply_quick_run(config: dict) -> dict:
    """Return a resolved experiment with capped budgets and unchanged grids/layers.

    Args:
        config: Experiment or sweep configuration.
    """
    result = resolve(config)
    if result['model']['name'] == 'beta_vae':
        parameters = result['model']['parameters']
        parameters['max_iter'] = min(parameters['max_iter'], QUICK_RUN['vae_max_iter'])
        parameters['strict_budget'] = True
    result['split']['max_test_size'] = min(result['split']['max_test_size'], QUICK_RUN['evaluation_size'])
    result['inference']['evaluation_size'] = min(
        result['inference'].get('evaluation_size', QUICK_RUN['evaluation_size']), QUICK_RUN['evaluation_size'])
    if result['dataset']['name'] in ('Sphere', 'Torus', 'Circle', 'Moons'):
        result['dataset']['ground_truth_size'] = min(
            result['dataset']['ground_truth_size'], QUICK_RUN['toy_reference_points'])
    for analysis in result['analyses']:
        if analysis['name'] == 'geometry':
            analysis['reach_subsample'] = QUICK_RUN['reach_subsample']
    return result
