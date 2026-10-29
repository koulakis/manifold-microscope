"""Incremental toy artifacts with ordinary pickle/NumPy exports.

Virtual paths let the single runner use its existing stages without creating per-experiment files. A journal commits one
entire experiment at a time; results.pkl is the normal, single-pickle export for downstream analysis.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import BinaryIO
from contextlib import contextmanager
from copy import deepcopy
import csv
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import pickle
import struct
from time import perf_counter

import numpy as np

from .outputs import canonical, identity, stage_directory
from .paths import model_label

FORMAT_VERSION = 1
_bundles = {}
_sessions = {}


def bundle_family(config: dict) -> dict:
    """Remove per-size choices from a bundled experiment configuration.

    Args:
        config: Experiment or sweep configuration.

    Returns:
        Copied configuration with output_layout and dataset sample_size removed for grouping compatible experiments.
    """
    family = deepcopy(config)
    family.pop('output_layout', None)
    family['dataset'].pop('sample_size', None)
    return family


def _atomic_pickle(path: Path, value: object) -> None:
    """Write a pickle through an atomic filesystem replacement.

    Args:
        path: Filesystem path.
        value: Value to serialize or assign.

    Returns:
        None.
    """
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _decode(name: str, data: bytes) -> object:
    """Decode an exported artifact according to its filename.

    Args:
        name: Name used for labels or output files.
        data: Serialized artifact bytes.

    Returns:
        Decoded artifact: text/bytes, JSON or pickle data, a NumPy array with its stored shape, or a dictionary of
        stored arrays for an NPZ file.
    """
    if name.endswith('.json'):
        return json.loads(data)
    if name.endswith('.npy'):
        return np.load(io.BytesIO(data), allow_pickle=False)
    if name.endswith('.npz'):
        with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
            return {key: arrays[key] for key in arrays.files}
    if name.endswith('.pkl'):
        return pickle.loads(data)
    return data.decode('utf-8') if name == 'COMPLETED' else data


class Bundle:
    @property
    def journal(self) -> Path:
        """Return the path to the incremental experiment journal.

        Returns:
            Filesystem path to progress.pkl in the bundle directory.
        """
        return self.directory / 'progress.pkl'

    def _read(self, stream: BinaryIO) -> dict | None:
        """Read one complete journal event, detecting torn writes and corruption.

        Args:
            stream: Open journal stream positioned at an event header.

        Returns:
            Decoded journal-event dictionary, or None at the end of committed data or an incomplete trailing event.
        """
        header = stream.read(40)
        if len(header) < 40:
            return None  # Empty or interrupted final write.
        length = struct.unpack('<Q', header[:8])[0]
        # Check the remaining size before allocating memory for a payload.
        if length > os.fstat(stream.fileno()).st_size - stream.tell():
            return None
        payload = stream.read(length)
        if sha256(payload).digest() != header[8:]:
            raise ValueError(f'Corrupt bundle journal: {self.journal}')
        return pickle.loads(payload)

    def event(self, offset: int) -> dict:
        """Read a committed journal event, reusing the most recent event cache.

        Args:
            offset: Byte offset of a committed event.

        Returns:
            Committed event dictionary at the requested byte offset.
        """
        if self.cached is None or self.cached[0] != offset:
            with self.journal.open('rb') as stream:
                stream.seek(offset)
                event = self._read(stream)
            if event is None:
                raise ValueError(f'Missing committed experiment: {self.journal}')
            self.cached = offset, event
        return self.cached[1]

    def _index(
        self,
        event: dict,
        offset: int
    ) -> None:
        """Index the files committed by a journal event.

        Args:
            event: Committed experiment metadata and file payloads.
            offset: Byte offset of the event in the journal.

        Returns:
            None.
        """
        self.experiments[event['experiment_id']] = offset
        for name in event['files']:
            self.index[name] = offset
            self.dirs.update(str(p) if str(p) != '.' else '' for p in PurePosixPath(name).parents)

    def _scan(self) -> None:
        """Index complete journal events and retain the last committed offset.

        Returns:
            None.
        """
        if not self.journal.is_file():
            return
        with self.journal.open('rb') as stream:
            while True:
                offset = stream.tell()
                event = self._read(stream)
                if event is None:
                    break
                self._index(event, offset)
                self.end = stream.tell()

    def __init__(
        self,
        directory: Path,
        family: dict | None = None
    ) -> None:
        """Load bundle metadata and scan its committed journal.

        Args:
            directory: Directory or virtual bundle path for this stage.
            family: Expected bundle configuration, or None to use saved metadata.

        Returns:
            None.
        """
        self.directory = directory
        metadata = directory / 'family.pkl'
        exported_bytes = 0
        if metadata.is_file():
            with metadata.open('rb') as stream:
                saved = pickle.load(stream)
            if saved['format_version'] != FORMAT_VERSION:
                raise ValueError(f'Unsupported bundle format: {directory}')
            self.family = saved['configuration']
            exported_bytes = saved.get('exported_bytes', 0)
        elif family is not None:
            self.family = family
        else:
            raise FileNotFoundError(metadata)
        self.index, self.experiments, self.dirs = {}, {}, {''}
        self.pending, self.pending_dirs = {}, set()
        self.end = 0
        self.cached = None
        self.dirty = False
        self._scan()
        self.dirty = self.end != exported_bytes

    def read(self, name: str) -> bytes:
        """Read a pending or committed virtual file.

        Args:
            name: Name used for labels or output files.

        Returns:
            Raw bytes of the requested pending or committed virtual file.
        """
        if name in self.pending:
            return self.pending[name]
        if name not in self.index:
            raise FileNotFoundError(name)
        return self.event(self.index[name])['files'][name]

    def export(self) -> None:
        """Export committed experiments as standard pickle and CSV files.

        Returns:
            None.
        """
        if not self.experiments or (not self.dirty and (self.directory / 'results.pkl').is_file()
                                    and (self.directory / 'summary.csv').is_file()):
            return
        experiments, rows = {}, []
        for key, offset in self.experiments.items():
            stages = {}
            config, result = None, None
            event = self.event(offset)
            for name, data in event['files'].items():
                stage, directory, filename = name.split('/', 2)
                value = _decode(filename, data)
                stages.setdefault(stage, {}).setdefault(directory, {})[filename] = value
                if stage == 'experiment-configurations':
                    if filename == 'config.json':
                        config = value
                    elif filename == 'result.json':
                        result = value
            experiments[key] = dict(
                configuration=config,
                result=result,
                stages=stages,
                elapsed_seconds=event['elapsed_seconds']
            )
            base = dict(
                experiment_id=key,
                sample_size=config['dataset']['sample_size'],
                seed=config['training']['seed'],
                trial=config['training']['trial'],
                status=result.get('status', 'completed'),
                elapsed_seconds=event['elapsed_seconds']
            )
            count = len(rows)
            for files in stages.get('analysis-outputs', {}).values():
                if 'metrics.json' in files:
                    rows.append(dict(base, domain=files['result.json']['domain'], **files['metrics.json']))
            if len(rows) == count:
                rows.append(base)
        _atomic_pickle(self.directory / 'results.pkl', dict(
            format_version=FORMAT_VERSION, configuration=self.family, experiments=experiments))
        temporary = self.directory / 'summary.csv.tmp'
        with temporary.open('w', newline='') as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
                writer.writeheader()
                writer.writerows(rows)
        temporary.replace(self.directory / 'summary.csv')
        _atomic_pickle(self.directory / 'family.pkl', dict(format_version=FORMAT_VERSION,
                       configuration=self.family, exported_bytes=self.end))
        self.dirty = False

    @contextmanager
    def transaction(self, experiment_id: str) -> Iterator[None]:
        """Commit pending files together as one experiment journal event.

        Args:
            experiment_id: Identifier attached to the committed experiment.

        Yields:
            None while pending files are collected for one atomic journal event.
        """
        if self.pending or self.pending_dirs:
            raise RuntimeError('Nested bundle transaction')
        started = perf_counter()
        try:
            yield
            if self.pending:
                event = dict(experiment_id=experiment_id, files=self.pending,
                             elapsed_seconds=perf_counter() - started)
                payload = pickle.dumps(event, protocol=pickle.HIGHEST_PROTOCOL)
                self.directory.mkdir(parents=True, exist_ok=True)
                if not (self.directory / 'family.pkl').exists():
                    _atomic_pickle(self.directory / 'family.pkl', dict(
                        format_version=FORMAT_VERSION, configuration=self.family))
                with self.journal.open('r+b' if self.journal.exists() else 'w+b') as stream:
                    # Discard only a torn final write, retaining all committed jobs.
                    stream.seek(self.end)
                    stream.truncate()
                    stream.write(struct.pack('<Q', len(payload)) + sha256(payload).digest() + payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                    end = stream.tell()
                self._index(event, self.end)
                self.cached = None
                self.end, self.dirty = end, True
        finally:
            self.pending, self.pending_dirs = {}, set()
            root = self.directory.parent.parent
            if root not in _sessions:
                self.export()


def get_bundle(directory: str | Path, family: dict | None = None) -> Bundle:
    """Open or reuse a compatible bundle.

    Args:
        directory: Directory or virtual bundle path for this stage.
        family: Expected bundle configuration, or None to use saved metadata.

    Returns:
        Cached or newly opened Bundle matching the requested directory and family.
    """
    directory = Path(directory).resolve()
    if directory not in _bundles:
        _bundles[directory] = Bundle(directory, family)
    bundle = _bundles[directory]
    if family is not None and canonical(bundle.family) != canonical(family):
        raise ValueError(f'Bundle configuration mismatch: {directory}')
    return bundle


class BundlePath:
    """The small path/file interface used by the existing stage IO functions."""
    def __init__(
        self,
        bundle: Bundle,
        relative: str = ''
    ) -> None:
        """Create a virtual path inside an experiment bundle.

        Args:
            bundle: Bundle containing this virtual path.
            relative: Relative POSIX path inside the bundle.

        Returns:
            None.
        """
        self.bundle, self.relative = bundle, relative

    @property
    def name(self) -> str:
        """Return the final virtual path component.

        Returns:
            Final component of the relative virtual path.
        """
        return PurePosixPath(self.relative).name

    def __truediv__(self, name: str) -> BundlePath:
        """Append a relative component to this virtual path.

        Args:
            name: Name used for labels or output files.

        Returns:
            New virtual path with the supplied component appended.
        """
        return BundlePath(self.bundle, f'{self.relative}/{name}'.lstrip('/'))

    def __str__(self) -> str:
        """Return the bundle locator for this virtual path.

        Returns:
            Locator string in the form bundle_directory::relative_path.
        """
        return f'{self.bundle.directory}::{self.relative}'

    @property
    def parent(self) -> BundlePath:
        """Return the parent virtual path.

        Returns:
            Virtual path to the parent directory.
        """
        relative = str(PurePosixPath(self.relative).parent)
        return BundlePath(self.bundle, '' if relative == '.' else relative)

    def with_name(self, name: str) -> BundlePath:
        """Replace the final virtual path component.

        Args:
            name: Name used for labels or output files.

        Returns:
            Sibling virtual path with the requested final component.
        """
        return self.parent / name

    def is_file(self) -> bool:
        """Check whether the virtual file is pending or committed.

        Returns:
            True if the virtual file is pending or committed; otherwise False.
        """
        return self.relative in self.bundle.pending or self.relative in self.bundle.index

    def exists(self) -> bool:
        """Check whether the virtual path names a file or directory.

        Returns:
            True if the virtual path identifies a known file or directory; otherwise False.
        """
        return self.is_file() or self.relative in self.bundle.dirs or self.relative in self.bundle.pending_dirs

    def mkdir(
        self,
        parents: bool = False,
        exist_ok: bool = False
    ) -> None:
        """Register a pending virtual directory.

        Args:
            parents: Accepted for compatibility with filesystem paths; parent directories are implicit.
            exist_ok: Allow a directory that already exists.

        Returns:
            None.
        """
        if self.exists() and not exist_ok:
            raise FileExistsError(str(self))
        self.bundle.pending_dirs.add(self.relative)

    def write_text(self, value: str) -> None:
        """Store UTF-8 text as a pending virtual file.

        Args:
            value: Text content to store.

        Returns:
            None.
        """
        self.bundle.pending[self.relative] = value.encode('utf-8')

    def read_text(self) -> str:
        """Read a virtual file as UTF-8 text.

        Returns:
            UTF-8-decoded contents of the virtual file.
        """
        return self.bundle.read(self.relative).decode('utf-8')

    def unlink(self) -> None:
        """Remove a pending virtual file.

        Returns:
            None.
        """
        del self.bundle.pending[self.relative]

    @contextmanager
    def open(self, mode: str = 'r') -> Iterator[io.BytesIO | io.StringIO]:
        """Open a virtual file as an in-memory text or binary stream.

        Args:
            mode: Read or write mode: r, rb, w or wb.

        Yields:
            In-memory binary or text stream matching mode; writable contents are stored as a pending virtual file on
            exit.
        """
        if mode not in ('r', 'rb', 'w', 'wb'):
            raise ValueError(f'Unsupported bundle file mode: {mode}')
        binary, writing = 'b' in mode, 'w' in mode
        initial = b'' if writing else self.bundle.read(self.relative)
        with (io.BytesIO(initial) if binary else io.StringIO(initial.decode('utf-8'))) as stream:
            yield stream
            if writing:
                value = stream.getvalue()
                self.bundle.pending[self.relative] = value if binary else value.encode('utf-8')


def output_root(config: dict, root: str | Path) -> Path | BundlePath:
    """Select ordinary directories or a virtual bundle root.

    Args:
        config: Experiment or sweep configuration.
        root: Root directory for experiment artifacts.

    Returns:
        Filesystem root or virtual bundle root selected by the output configuration.
    """
    if config['output_layout'] == 'directories':
        return Path(root)
    family = bundle_family(config)
    key = identity(dict(format=FORMAT_VERSION, family=family))
    directory = stage_directory(Path(root) / 'experiment-bundles', key, model_label(family))
    return BundlePath(get_bundle(directory, family))


def bundled_path(locator: str) -> BundlePath:
    """Resolve a virtual bundle locator.

    Args:
        locator: Bundle directory and relative path separated by ::.

    Returns:
        Virtual path resolved from a bundle_directory::relative_path locator.
    """
    directory, relative = locator.split('::', 1)
    return BundlePath(get_bundle(directory), relative)


def export_bundle(directory: str | Path) -> None:
    """Refresh standard exports from committed progress, e.g. after a hard kill.

    Args:
        directory: Directory or virtual bundle path for this stage.

    Returns:
        None.
    """
    get_bundle(directory).export()


@contextmanager
def bundle_session(root: str | Path) -> Iterator[None]:
    """Export once at sweep exit, including on interruption, rather than per job.

    Args:
        root: Root directory for experiment artifacts.

    Yields:
        None while bundled experiments execute; pending exports are refreshed on exit.
    """
    root = Path(root).resolve()
    _sessions[root] = _sessions.get(root, 0) + 1
    try:
        yield
    finally:
        _sessions[root] -= 1
        if not _sessions[root]:
            del _sessions[root]
            for directory, bundle in list(_bundles.items()):
                if directory.parent == root / 'experiment-bundles':
                    bundle.export()
