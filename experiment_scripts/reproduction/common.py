"""Completion handling shared by the experiment entry points."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

COMPLETED = 'COMPLETED'


def is_complete(directory):
    return (Path(directory) / COMPLETED).is_file()


def run_stage(module, output, *args):
    """Run a reproduction CLI in a fresh process, stopping on partial results."""
    output = Path(output)
    if is_complete(output):
        print(f'Skipping completed experiment: {output}', flush=True)
        return
    if output.exists():
        print(f'WARNING: Skipping incomplete experiment (no COMPLETED): {output}', flush=True)
        raise RuntimeError('Preserve the partial directory under another name before retrying.')
    subprocess.run([sys.executable, '-u', '-m', f'experiment_scripts.reproduction.{module}',
                    *map(str, args), '--output', str(output)], check=True)
    if not is_complete(output):
        raise RuntimeError(f'Stage did not complete: {output}')


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
        print(f'WARNING: Skipping incomplete experiment (no {COMPLETED}): {directory}')
        return False
    missing = [str(path) for path in requires if not is_complete(path)]
    if missing:
        print(f'WARNING: Skipping {directory}; unfinished inputs: {", ".join(missing)}')
        return False
    directory.mkdir(parents=True)
    expected = action(directory)
    if not expected or not all(Path(path).is_file() for path in expected):
        print(f'WARNING: Experiment has no complete output; leaving unmarked: {directory}')
        return False
    (directory / COMPLETED).write_text(datetime.now(timezone.utc).isoformat() + '\n')
    return True


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def experiment_name(dataset, method, fraction, dimension, repetition):
    return '__'.join([dataset, method, str(float(fraction)), str(dimension), '0', str(repetition)])
