"""Incremental toy artifacts with ordinary pickle/NumPy exports.

Virtual paths let the single runner use its existing stages without creating
per-experiment files. A journal commits one entire experiment at a time;
results.pkl is the normal, single-pickle export for downstream analysis.
"""
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


def bundle_family(config):
    family = deepcopy(config)
    family.pop('output_layout', None)
    family['dataset'].pop('sample_size', None)
    return family


def get_bundle(directory, family=None):
    directory = Path(directory).resolve()
    if directory not in _bundles:
        _bundles[directory] = Bundle(directory, family)
    bundle = _bundles[directory]
    if family is not None and canonical(bundle.family) != canonical(family):
        raise ValueError(f'Bundle configuration mismatch: {directory}')
    return bundle


def output_root(config, root):
    if config['output_layout'] == 'directories':
        return Path(root)
    family = bundle_family(config)
    key = identity(dict(format=FORMAT_VERSION, family=family))
    directory = stage_directory(Path(root) / 'experiment-bundles', key, model_label(family))
    return BundlePath(get_bundle(directory, family))


def bundled_path(locator):
    directory, relative = locator.split('::', 1)
    return BundlePath(get_bundle(directory), relative)


def export_bundle(directory):
    """Refresh standard exports from committed progress, e.g. after a hard kill."""
    get_bundle(directory).export()


@contextmanager
def bundle_session(root):
    """Export once at sweep exit, including on interruption, rather than per job."""
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


def _atomic_pickle(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Bundle:
    def __init__(self, directory, family=None):
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

    @property
    def journal(self):
        return self.directory / 'progress.pkl'

    def _index(self, event, offset):
        self.experiments[event['experiment_id']] = offset
        for name in event['files']:
            self.index[name] = offset
            self.dirs.update(str(p) if str(p) != '.' else '' for p in PurePosixPath(name).parents)

    def _read(self, stream):
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

    def _scan(self):
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

    def event(self, offset):
        if self.cached is None or self.cached[0] != offset:
            with self.journal.open('rb') as stream:
                stream.seek(offset)
                event = self._read(stream)
            if event is None:
                raise ValueError(f'Missing committed experiment: {self.journal}')
            self.cached = offset, event
        return self.cached[1]

    def read(self, name):
        if name in self.pending:
            return self.pending[name]
        if name not in self.index:
            raise FileNotFoundError(name)
        return self.event(self.index[name])['files'][name]

    @contextmanager
    def transaction(self, experiment_id):
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

    def export(self):
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
            experiments[key] = dict(configuration=config, result=result, stages=stages,
                                    elapsed_seconds=event['elapsed_seconds'])
            base = dict(experiment_id=key, sample_size=config['dataset']['sample_size'],
                        seed=config['training']['seed'], trial=config['training']['trial'],
                        status=result.get('status', 'completed'), elapsed_seconds=event['elapsed_seconds'])
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


def _decode(name, data):
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


class BundlePath:
    """The small path/file interface used by the existing stage IO functions."""
    def __init__(self, bundle, relative=''):
        self.bundle, self.relative = bundle, relative

    def __truediv__(self, name):
        return BundlePath(self.bundle, f'{self.relative}/{name}'.lstrip('/'))

    def __str__(self):
        return f'{self.bundle.directory}::{self.relative}'

    @property
    def name(self):
        return PurePosixPath(self.relative).name

    @property
    def parent(self):
        relative = str(PurePosixPath(self.relative).parent)
        return BundlePath(self.bundle, '' if relative == '.' else relative)

    def with_name(self, name):
        return self.parent / name

    def is_file(self):
        return self.relative in self.bundle.pending or self.relative in self.bundle.index

    def exists(self):
        return self.is_file() or self.relative in self.bundle.dirs or self.relative in self.bundle.pending_dirs

    def mkdir(self, *, parents=False, exist_ok=False):
        if self.exists() and not exist_ok:
            raise FileExistsError(str(self))
        self.bundle.pending_dirs.add(self.relative)

    def write_text(self, value):
        self.bundle.pending[self.relative] = value.encode('utf-8')

    def read_text(self):
        return self.bundle.read(self.relative).decode('utf-8')

    def unlink(self):
        del self.bundle.pending[self.relative]

    @contextmanager
    def open(self, mode='r'):
        if mode not in ('r', 'rb', 'w', 'wb'):
            raise ValueError(f'Unsupported bundle file mode: {mode}')
        binary, writing = 'b' in mode, 'w' in mode
        initial = b'' if writing else self.bundle.read(self.relative)
        with (io.BytesIO(initial) if binary else io.StringIO(initial.decode('utf-8'))) as stream:
            yield stream
            if writing:
                value = stream.getvalue()
                self.bundle.pending[self.relative] = value if binary else value.encode('utf-8')
