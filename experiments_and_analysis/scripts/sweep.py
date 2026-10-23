"""Expand parameter choices, run experiments sequentially, then aggregate/plot."""
from copy import deepcopy
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from itertools import product
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Optional

import typer
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

from ..config import load_config, merge, resolve
from ..bundles import bundle_session
from ..data.preparation import source_identity
from ..outputs import identity, stage_directory, write_json
from ..paths import sweep_label
from ..plotting.render import matches, render
from ..quick import apply_quick_run, quick_grid
from . import run as single

app = typer.Typer(pretty_exceptions_enable=False)


@contextmanager
def experiment_output(quiet):
    """Keep the sweep bar separate from experiment prints and nested bars."""
    if quiet:
        with open(os.devnull, 'w') as sink, redirect_stdout(sink), redirect_stderr(sink):
            yield
    else:
        # Inner tqdm instances must write directly to stderr. Wrapping their
        # refreshes in tqdm.write clears/repaints every active bar each time.
        with redirect_stdout(DummyTqdmFile(sys.stdout)):
            yield


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


def run(config: dict, output_dir, *, dry_run=False, quick_run=False, device=None,
        quiet_experiments=None):
    started = perf_counter()
    quiet = config.get('quiet_experiments', False) if quiet_experiments is None else quiet_experiments
    experiments = expand(config, quick_run=quick_run, device=device)
    root = Path(output_dir).expanduser().resolve()
    completed_results = []
    if not dry_run:
        with experiment_output(quiet):
            completed_results = [single.completed_result(experiment, root) for experiment in experiments]
        devices = {experiment['device'] for experiment, completed in zip(experiments, completed_results)
                   if completed is None}
        for requested_device in sorted(devices):
            single.check_device(requested_device)
    configurations = root / 'experiment-configurations'
    configurations.mkdir(parents=True, exist_ok=True)
    # Effective jobs include the quick profile and device in this sweep's identity.
    scientific_config = {key: value for key, value in config.items() if key != 'quiet_experiments'}
    sweep_id = identity(dict(config=scientific_config, experiments=experiments))
    configurations = stage_directory(configurations, sweep_id, sweep_label(config, experiments))
    configurations.mkdir(exist_ok=True)
    write_json(configurations / 'config.json', config)
    write_json(configurations / 'jobs.json', experiments)
    if dry_run:
        return dict(configurations=experiments, results=[], plots=[])
    results = []
    dataset, previous_source = None, None
    with bundle_session(root), tqdm(total=len(experiments), desc='Sweep', unit='experiment', file=sys.stderr,
              position=0, dynamic_ncols=True) as progress:
        for experiment, completed in zip(experiments, completed_results):
            with experiment_output(quiet):
                if completed is not None:
                    results.append(completed)
                else:
                    source = identity(source_identity(experiment['dataset']))
                    if source != previous_source:
                        del dataset
                        dataset = single.load_dataset(experiment['dataset'], root, experiment['device'])
                        previous_source = source
                    results.append(single.run(experiment, root, dataset=dataset))
            progress.update(1)
    records = [record for result in results for record in result['analyses']]
    with experiment_output(quiet):
        plots = [str(render(records, request, root)) for request in config.get('outputs', [])]
    result = dict(results=results, plots=plots, elapsed_seconds=perf_counter() - started)
    write_json(configurations / 'results.json', result)
    if not quiet:
        print(f"[sweep] completed {len(experiments)} experiments in {result['elapsed_seconds']:.2f}s")
    return result


@app.command()
def main(config: Path = typer.Option(..., exists=True), output_dir: Path = typer.Option(...),
         dry_run: bool = False, quick_run: bool = False,
         quiet_experiments: Optional[bool] = typer.Option(
             None, '--quiet-experiments/--no-quiet-experiments',
             help='Show only sweep progress; overrides quiet_experiments in the sweep config.'),
         device: str = typer.Option(None, help='Override the configured device, e.g. cuda:0.')):
    """Expand/run a YAML or JSON sweep; --dry-run writes its job list only."""
    result = run(load_config(config), output_dir, dry_run=dry_run, quick_run=quick_run, device=device,
                 quiet_experiments=quiet_experiments)
    if dry_run:
        typer.echo(f"Prepared {len(result['configurations'])} experiment configurations.")


if __name__ == '__main__':
    app()
