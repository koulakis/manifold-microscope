"""Stage identities and IO; numerical functions do not use these helpers."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from time import time

# Increment when stage implementations change numerical behavior or artifact formats.
PIPELINE_VERSION = 1
STAGES = ('experiment-configurations', 'splits', 'trained-models',
          'inference-outputs', 'analysis-outputs', 'plots')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def identity(value):
    return sha256(canonical([PIPELINE_VERSION, value]).encode()).hexdigest()[:20]


def experiment_identity(config, source):
    return identity(dict(config={k: v for k, v in config.items() if k != 'output_layout'}, source=source))


def artifact_path(path):
    """Accept filesystem paths or virtual paths within a pickle bundle."""
    if isinstance(path, (str, Path)) and '::' in str(path):
        from .bundles import bundled_path
        return bundled_path(str(path))
    if hasattr(path, 'read_text'):
        return path
    return Path(path)


def save_array(path, value):
    import numpy as np
    with artifact_path(path).open('wb') as stream:
        np.save(stream, value, allow_pickle=False)


def load_array(path, *, mmap_mode=None):
    import numpy as np
    path = artifact_path(path)
    if isinstance(path, Path):
        return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    with path.open('rb') as stream:
        return np.load(stream, allow_pickle=False)


def stage_directory(parent, key, label):
    """Build a readable directory name with its content hash appended."""
    # ASCII-only and bounded to fit a filesystem component including the hash.
    label = re.sub(r'[^A-Za-z0-9_.-]+', '_', label).strip('._-')[:220].rstrip('._-')
    return artifact_path(parent) / f'{label or "output"}__{key}'


def write_json(path, value):
    artifact_path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def read_json(path):
    return json.loads(artifact_path(path).read_text())


def is_complete(directory):
    directory = artifact_path(directory)
    if not (directory / 'COMPLETED').is_file():
        return False
    manifest = read_json(directory / 'manifest.json')
    return all((directory / name).is_file() for name in manifest['outputs'])


def report_reuse(directory):
    directory = artifact_path(directory)
    elapsed = read_json(directory / 'manifest.json').get('elapsed_seconds')
    timing = f'; original elapsed {elapsed:.2f}s' if elapsed is not None else ''
    print(f'[{directory.parent.name}] reused {directory.name}{timing}')


def begin(directory, config, *, requires=()):
    """Return False for compatible completed work; stop on partial outputs."""
    directory = artifact_path(directory)
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
    directory = artifact_path(directory)
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
