"""Stage identities and IO; numerical functions do not use these helpers."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import time

# Increment when stage implementations change numerical behavior or artifact formats.
PIPELINE_VERSION = 1
STAGES = ('experiment-configurations', 'splits', 'trained-models',
          'inference-outputs', 'analysis-outputs', 'plots')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def identity(value):
    return sha256(canonical([PIPELINE_VERSION, value]).encode()).hexdigest()[:20]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


def is_complete(directory):
    directory = Path(directory)
    if not (directory / 'COMPLETED').is_file():
        return False
    manifest = read_json(directory / 'manifest.json')
    return all((directory / name).is_file() for name in manifest['outputs'])


def report_reuse(directory):
    directory = Path(directory)
    elapsed = read_json(directory / 'manifest.json').get('elapsed_seconds')
    timing = f'; original elapsed {elapsed:.2f}s' if elapsed is not None else ''
    print(f'[{directory.parent.name}] reused {directory.name}{timing}')


def begin(directory, config, *, requires=()):
    """Return False for compatible completed work; stop on partial outputs."""
    directory = Path(directory)
    if is_complete(directory):
        if canonical(read_json(directory / 'manifest.json')['config']) != canonical(config):
            raise ValueError(f'Output configuration mismatch: {directory}')
        report_reuse(directory)
        return False
    if directory.exists():
        raise RuntimeError(f'Incomplete output preserved: {directory}. Move it before retrying.')
    missing = [str(path) for path in requires if not is_complete(path)]
    if missing:
        raise RuntimeError(f'Unfinished inputs: {missing}')
    directory.mkdir(parents=True)
    write_json(directory / 'manifest.json', dict(config=config, outputs=[], started_at=time()))
    print(f'[{directory.parent.name}] starting {directory.name}')
    return True


def finish(directory, files, *, elapsed_seconds=None):
    directory = Path(directory)
    if not files or not all((directory / name).is_file() for name in files):
        raise RuntimeError(f'Missing stage outputs: {directory}')
    manifest = read_json(directory / 'manifest.json')
    manifest['outputs'] = list(files)
    manifest['elapsed_seconds'] = time() - manifest['started_at'] if elapsed_seconds is None else elapsed_seconds
    write_json(directory / 'manifest.json', manifest)
    (directory / 'COMPLETED').write_text(datetime.now(timezone.utc).isoformat() + '\n')
    print(f"[{directory.parent.name}] completed {directory.name} in {manifest['elapsed_seconds']:.2f}s")


def stage_ids(config, dataset_identity):
    split = identity(dict(dataset=dataset_identity, split=config['split']))
    model = identity(dict(split=split, model=config['model'],
                          seed=config['training']['seed'], trial=config['training']['trial']))
    return split, model
