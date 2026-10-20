"""Expand parameter choices, run experiments sequentially, then aggregate/plot."""
from copy import deepcopy
from itertools import product
from pathlib import Path

import typer

from ..config import load_config, merge, resolve
from ..data.preparation import source_identity
from ..outputs import identity, write_json
from ..plotting.render import matches, render
from . import run as single

app = typer.Typer(pretty_exceptions_enable=False)


def set_choice(config, path, value):
    keys = path.split('.')
    current = config
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = deepcopy(value)


def expand(config):
    """Cartesian choices plus explicit exclusions and additional configurations."""
    experiments = []
    for group in config.get('groups', [dict()]):
        base = merge(config.get('base', {}), group.get('base', {}))
        grid = {**config.get('grid', {}), **group.get('grid', {})}
        grid = {path: ([n for start, stop, step in choices['ranges'] for n in range(start, stop, step)]
                       if isinstance(choices, dict) else choices) for path, choices in grid.items()}
        for values in product(*grid.values()):
            candidate = deepcopy(base)
            for path, value in zip(grid, values):
                set_choice(candidate, path, value)
            excluded = config.get('exclude', []) + group.get('exclude', [])
            if any(matches(candidate, rule) for rule in excluded):
                continue
            # Match resolved defaults too, e.g. default model parameters.
            resolved = resolve(candidate)
            if not any(matches(resolved, rule) for rule in excluded):
                experiments.append(resolved)
    experiments.extend(resolve(merge(config.get('base', {}), item)) for item in config.get('include', []))
    return list({identity(experiment): experiment for experiment in experiments}.values())


def run(config: dict, output_dir, *, dry_run=False):
    experiments = expand(config)
    root = Path(output_dir).expanduser().resolve()
    configurations = root / 'experiment-configurations'
    configurations.mkdir(parents=True, exist_ok=True)
    sweep_id = identity(config)
    configurations = configurations / f'sweep__{sweep_id}'
    configurations.mkdir(exist_ok=True)
    write_json(configurations / 'config.json', config)
    write_json(configurations / 'jobs.json', experiments)
    if dry_run:
        return dict(configurations=experiments, results=[], plots=[])
    results = []
    dataset, previous_source = None, None
    for experiment in experiments:
        completed = single.completed_result(experiment, root)
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
    result = dict(results=results, plots=plots)
    write_json(configurations / 'results.json', result)
    return result


@app.command()
def main(config: Path = typer.Option(..., exists=True), output_dir: Path = typer.Option(...),
         dry_run: bool = False):
    """Expand/run a YAML or JSON sweep; --dry-run writes its job list only."""
    result = run(load_config(config), output_dir, dry_run=dry_run)
    if dry_run:
        typer.echo(f"Prepared {len(result['configurations'])} experiment configurations.")


if __name__ == '__main__':
    app()
