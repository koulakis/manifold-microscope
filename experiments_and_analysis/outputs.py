"""Stage identities and IO; numerical functions do not use these helpers."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

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


def begin(directory, config, *, requires=()):
    """Return False for compatible completed work; stop on partial outputs."""
    directory = Path(directory)
    if is_complete(directory):
        if canonical(read_json(directory / 'manifest.json')['config']) != canonical(config):
            raise ValueError(f'Output configuration mismatch: {directory}')
        return False
    if directory.exists():
        raise RuntimeError(f'Incomplete output preserved: {directory}. Move it before retrying.')
    missing = [str(path) for path in requires if not is_complete(path)]
    if missing:
        raise RuntimeError(f'Unfinished inputs: {missing}')
    directory.mkdir(parents=True)
    write_json(directory / 'manifest.json', dict(config=config, outputs=[]))
    return True


def finish(directory, files):
    directory = Path(directory)
    if not files or not all((directory / name).is_file() for name in files):
        raise RuntimeError(f'Missing stage outputs: {directory}')
    manifest = read_json(directory / 'manifest.json')
    manifest['outputs'] = list(files)
    write_json(directory / 'manifest.json', manifest)
    (directory / 'COMPLETED').write_text(datetime.now(timezone.utc).isoformat() + '\n')


def stage_ids(config, dataset_identity):
    split = identity(dict(dataset=dataset_identity, split=config['split']))
    model = identity(dict(split=split, model=config['model'],
                          seed=config['training']['seed'], trial=config['training']['trial']))
    return split, model
