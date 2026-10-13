"""Completion handling shared by the experiment entry points."""
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings

COMPLETED = 'COMPLETED'


def is_complete(directory):
    return (Path(directory) / COMPLETED).is_file()


def run_experiment(directory, action, *, requires=()):
    """Run a new experiment; never overwrite completed or partial directories.

    action(directory) returns the essential output paths. The completion file
    is written only after the action returns and those files exist.
    """
    directory = Path(directory).expanduser().resolve()
    if is_complete(directory):
        print(f'Skipping completed experiment: {directory}')
        return True
    if directory.exists():
        warnings.warn(f'Skipping incomplete experiment (no {COMPLETED}): {directory}', stacklevel=2)
        return False
    missing = [str(path) for path in requires if not is_complete(path)]
    if missing:
        warnings.warn(f'Skipping {directory}; unfinished inputs: {", ".join(missing)}', stacklevel=2)
        return False
    directory.mkdir(parents=True)
    expected = action(directory)
    if not expected or not all(Path(path).is_file() for path in expected):
        warnings.warn(f'Experiment has no complete output; leaving unmarked: {directory}', stacklevel=2)
        return False
    (directory / COMPLETED).write_text(datetime.now(timezone.utc).isoformat() + '\n')
    return True


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def experiment_name(dataset, method, fraction, dimension, repetition):
    return '__'.join([dataset, method, str(float(fraction)), str(dimension), '0', str(repetition)])
