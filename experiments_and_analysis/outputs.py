"""Stage identities and IO; numerical functions do not use these helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from time import time

if TYPE_CHECKING:
    import numpy as np
    from .bundles import BundlePath


# Increment when stage implementations change numerical behavior or artifact formats.
PIPELINE_VERSION = 1
STAGES = ('experiment-configurations', 'splits', 'trained-models',
          'inference-outputs', 'analysis-outputs', 'plots')


def canonical(value: object) -> str:
    """Serialize a value deterministically for configuration comparisons.

    Args:
        value: Value to serialize or assign.

    Returns:
        Deterministic JSON string representing value.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(',', ':'),
        default=str
    )


def identity(value: object) -> str:
    """Hash the pipeline version and a value into an artifact identifier.

    Args:
        value: Value to serialize or assign.

    Returns:
        Twenty-character hexadecimal content identifier including the pipeline version.
    """
    return sha256(canonical([PIPELINE_VERSION, value]).encode()).hexdigest()[:20]


def experiment_identity(config: dict, source: dict) -> str:
    """Identify an experiment independently of its output layout.

    Args:
        config: Experiment or sweep configuration.
        source: Dataset identity including source provenance.

    Returns:
        Experiment identifier based on configuration and source, excluding output layout.
    """
    return identity(dict(config={k: v for k, v in config.items() if k != 'output_layout'}, source=source))


def artifact_path(path: str | Path | BundlePath) -> Path | BundlePath:
    """Accept filesystem paths or virtual paths within a pickle bundle.

    Args:
        path: Filesystem or virtual bundle path.

    Returns:
        Filesystem Path or BundlePath corresponding to the supplied path or locator.
    """
    if isinstance(path, (str, Path)) and '::' in str(path):
        from .bundles import bundled_path
        return bundled_path(str(path))
    if hasattr(path, 'read_text'):
        return path
    return Path(path)


def save_array(path: str | Path | BundlePath, value: np.ndarray) -> None:
    """Save an array without object pickling.

    Args:
        path: Filesystem or virtual bundle path.
        value: Array of shape (...), saved with its dtype and dimensions.

    Returns:
        None.
    """
    import numpy as np
    with artifact_path(path).open('wb') as stream:
        np.save(stream, value, allow_pickle=False)


def load_array(path: str | Path | BundlePath, mmap_mode: str | None = None) -> np.ndarray:
    """Load an array from a filesystem or bundle path.

    Args:
        path: Filesystem or virtual bundle path.
        mmap_mode: NumPy memory-map mode for filesystem arrays; None loads into memory.

    Returns:
        NumPy array with its stored shape and dtype; filesystem arrays may be memory-mapped when requested.
    """
    import numpy as np
    path = artifact_path(path)
    if isinstance(path, Path):
        return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    with path.open('rb') as stream:
        return np.load(stream, allow_pickle=False)


def stage_directory(
    parent: str | Path | BundlePath,
    key: str,
    label: str
) -> Path | BundlePath:
    """Build a readable directory name with its content hash appended.

    Args:
        parent: Parent directory or virtual bundle path.
        key: Content identifier appended to the directory label.
        label: Readable directory label, sanitized and bounded to fit a filesystem component.

    Returns:
        Filesystem or virtual path with the readable label and content identifier appended.
    """
    # ASCII-only and bounded to fit a filesystem component including the hash.
    label = re.sub(r'[^A-Za-z0-9_.-]+', '_', label).strip('._-')[:220].rstrip('._-')
    return artifact_path(parent) / f'{label or "output"}__{key}'


def write_json(path: str | Path | BundlePath, value: object) -> None:
    """Write a JSON artifact.

    Args:
        path: Filesystem or virtual bundle path.
        value: Value to serialize or assign.

    Returns:
        None.
    """
    artifact_path(path).write_text(json.dumps(value, indent=2, default=str) + '\n')


def read_json(path: str | Path | BundlePath) -> dict:
    """Read a JSON artifact.

    Args:
        path: Filesystem or virtual bundle path.

    Returns:
        Dictionary decoded from the JSON artifact.
    """
    return json.loads(artifact_path(path).read_text())


def is_complete(directory: Path | BundlePath) -> bool:
    """Check a stage completion marker and its recorded output files.

    Args:
        directory: Directory or virtual bundle path for this stage.

    Returns:
        True if the completion marker and all recorded output files are present; otherwise False.
    """
    directory = artifact_path(directory)
    if not (directory / 'COMPLETED').is_file():
        return False
    manifest = read_json(directory / 'manifest.json')
    return all((directory / name).is_file() for name in manifest['outputs'])


def report_reuse(directory: Path | BundlePath) -> None:
    """Report reuse and the original elapsed stage time.

    Args:
        directory: Directory or virtual bundle path for this stage.

    Returns:
        None.
    """
    directory = artifact_path(directory)
    elapsed = read_json(directory / 'manifest.json').get('elapsed_seconds')
    timing = f'; original elapsed {elapsed:.2f}s' if elapsed is not None else ''
    print(f'[{directory.parent.name}] reused {directory.name}{timing}')


def begin(
    directory: Path | BundlePath,
    config: dict,
    requires: list[Path | BundlePath] | tuple[Path | BundlePath, ...] = ()
) -> bool:
    """Return False for compatible completed work; stop on partial outputs.

    Args:
        directory: Directory or virtual bundle path for this stage.
        config: Experiment or sweep configuration.
        requires: Input stages that must already be complete.

    Returns:
        False when compatible completed outputs can be reused; True when a new stage should run.
    """
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


def finish(
    directory: Path | BundlePath,
    files: list[str],
    elapsed_seconds: float | None = None
) -> None:
    """Record the stage outputs and mark the stage complete.

    Args:
        directory: Directory or virtual bundle path for this stage.
        files: Relative names of the files produced by this stage.
        elapsed_seconds: Measured duration, or None to use time since the stage started.

    Returns:
        None.
    """
    directory = artifact_path(directory)
    if not files or not all((directory / name).is_file() for name in files):
        raise RuntimeError(f'Missing stage outputs: {directory}')
    manifest = read_json(directory / 'manifest.json')
    manifest['outputs'] = list(files)
    manifest['elapsed_seconds'] = time() - manifest['started_at'] if elapsed_seconds is None else elapsed_seconds
    write_json(directory / 'manifest.json', manifest)
    (directory / 'COMPLETED').write_text(datetime.now(timezone.utc).isoformat() + '\n')
    print(f"[{directory.parent.name}] completed {directory.name} in {manifest['elapsed_seconds']:.2f}s")


def stage_ids(config: dict, dataset_identity: dict) -> tuple[str, str]:
    """Compute the split and model stage identifiers.

    Args:
        config: Experiment or sweep configuration.
        dataset_identity: Dataset identity including source provenance.

    Returns:
        (split_id, model_id), content identifiers for the split and trained-model stages.
    """
    split = identity(dict(dataset=dataset_identity, split=config['split']))
    model = identity(dict(
        split=split,
        model=config['model'],
        seed=config['training']['seed'],
        trial=config['training']['trial']
    ))
    return split, model
