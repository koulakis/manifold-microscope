"""Expand parameter choices, run experiments sequentially, then aggregate/plot."""
from copy import deepcopy
from itertools import product
from pathlib import Path
from time import perf_counter

import typer

from ..config import load_config, merge, resolve
from ..data.preparation import source_identity
from ..outputs import identity, write_json
from ..plotting.render import matches, render
from ..quick import apply_quick_run, quick_grid
from . import run as single

app = typer.Typer(pretty_exceptions_enable=False)


def set_choice(config, path, value):
    keys = path.split('.')
    current = config
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = deepcopy(value)


def expand(config, *, quick_run=False, device=None):
    """Cartesian choices plus explicit exclusions and additional configurations."""
    experiments = []
    for group in config.get('groups', [dict()]):
        base = merge(config.get('base', {}), group.get('base', {}))
        grid = {**config.get('grid', {}), **group.get('grid', {})}
        if quick_run:
            grid = quick_grid(grid)
        grid = {path: ([n for start, stop, step in choices['ranges'] for n in range(start, stop, step)]
                       if isinstance(choices, dict) else choices) for path, choices in grid.items()}
        for values in product(*grid.values()):
            candidate = deepcopy(base)
            for path, value in zip(grid, values):
                set_choice(candidate, path, value)
            if device is not None:
                candidate['device'] = device
            excluded = config.get('exclude', []) + group.get('exclude', [])
            if any(matches(candidate, rule) for rule in excluded):
                continue
            # Match resolved defaults too, e.g. default model parameters.
            resolved = apply_quick_run(candidate) if quick_run else resolve(candidate)
            if not any(matches(resolved, rule) for rule in excluded):
                experiments.append(resolved)
    for item in config.get('include', []):
        candidate = merge(config.get('base', {}), item)
        if device is not None:
            candidate['device'] = device
        experiments.append(apply_quick_run(candidate) if quick_run else resolve(candidate))
    return list({identity(experiment): experiment for experiment in experiments}.values())


def run(config: dict, output_dir, *, dry_run=False, quick_run=False, device=None):
    started = perf_counter()
    experiments = expand(config, quick_run=quick_run, device=device)
    root = Path(output_dir).expanduser().resolve()
    completed_results = []
    if not dry_run:
        completed_results = [single.completed_result(experiment, root) for experiment in experiments]
        devices = {experiment['device'] for experiment, completed in zip(experiments, completed_results)
                   if completed is None}
        for requested_device in sorted(devices):
            single.check_device(requested_device)
    configurations = root / 'experiment-configurations'
    configurations.mkdir(parents=True, exist_ok=True)
    # Effective jobs include the quick profile and device in this sweep's identity.
    sweep_id = identity(dict(config=config, experiments=experiments))
    configurations = configurations / f'sweep__{sweep_id}'
    configurations.mkdir(exist_ok=True)
    write_json(configurations / 'config.json', config)
    write_json(configurations / 'jobs.json', experiments)
    if dry_run:
        return dict(configurations=experiments, results=[], plots=[])
    results = []
    dataset, previous_source = None, None
    for experiment, completed in zip(experiments, completed_results):
        if completed is not None:
            results.append(completed)
            continue
        source = identity(source_identity(experiment['dataset']))
        if source != previous_source:
            del dataset
            dataset = single.load_dataset(experiment['dataset'], root, experiment['device'])
            previous_source = source
        results.append(single.run(experiment, root, dataset=dataset))
    records = [record for result in results for record in result['analyses']]
    plots = [str(render(records, request, root)) for request in config.get('outputs', [])]
    result = dict(results=results, plots=plots, elapsed_seconds=perf_counter() - started)
    write_json(configurations / 'results.json', result)
    print(f"[sweep] completed {len(experiments)} experiments in {result['elapsed_seconds']:.2f}s")
    return result


@app.command()
def main(config: Path = typer.Option(..., exists=True), output_dir: Path = typer.Option(...),
         dry_run: bool = False, quick_run: bool = False,
         device: str = typer.Option(None, help='Override the configured device, e.g. cuda:0.')):
    """Expand/run a YAML or JSON sweep; --dry-run writes its job list only."""
    result = run(load_config(config), output_dir, dry_run=dry_run, quick_run=quick_run, device=device)
    if dry_run:
        typer.echo(f"Prepared {len(result['configurations'])} experiment configurations.")


if __name__ == '__main__':
    app()
