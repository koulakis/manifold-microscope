"""Dataset metadata, index splits and historical per-domain normalization."""
from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256
import os

import numpy as np

from microscope.computations_grid.volume import volume_element
from microscope.datasets.dataset_split import weighted_uniform_splitting, maximal_distance_selection_torch
from microscope.datasets.generic_dataset_loader import load_dataset_no_split


@dataclass
class Dataset:
    grid: np.ndarray
    range_sizes: list
    cyclic_dimensions: list
    patch_sizes: list
    class_names: list
    feature_ndim: int = 2
    volume_slices: tuple = ()
    split_slices: tuple = ()
    difference_intervals: list | None = None
    provided_train_count: int | None = None

    @property
    def flat(self):
        return self.grid.reshape(-1, *self.grid.shape[-self.feature_ndim:])

    @property
    def grid_shape(self):
        return self.grid.shape[:-self.feature_ndim]


def source_identity(spec):
    result = dict(spec)
    if 'path' in spec:
        path = Path(spec['path']).expanduser().resolve()
        info = path.stat()
        result.update(path=str(path), size=info.st_size, mtime_ns=info.st_mtime_ns)
    elif spec['name'] == 'extended_coil20':
        path = Path(os.environ.get('COIL20_PATH', '~/datasets/curvature-landscapes/coil-20-proc')).expanduser().resolve()
        files = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(path.glob('*.png'))]
        result.update(data_root=str(path), input_files=sha256(repr(files).encode()).hexdigest())
    return result


def normalize(data):
    """Historical min/max normalization, independently for each selected domain."""
    lower, upper = data.min(), data.max()
    if upper == lower:
        return np.zeros_like(data, dtype=np.float32)
    if lower == 0 and upper == 1:
        return data.astype(np.float32, copy=False)
    return ((data - lower) / (upper - lower)).astype(np.float32)


def toy_manifold(name):
    from microscope.datasets import toy_manifolds
    match name:
        case 'Sphere':
            return toy_manifolds.Sphere()
        case 'Torus':
            return toy_manifolds.Torus()
        case 'Circle':
            return toy_manifolds.Circle()
        case 'Moons':
            return toy_manifolds.Moons()
        case _:
            raise ValueError(f'Unknown toy manifold: {name}')


def toy_reference(spec, device='cpu'):
    state = np.random.get_state()
    try:
        np.random.seed(spec.get('reference_seed', 0))
        return toy_manifold(spec['name']).tesselation(
            N=spec.get('ground_truth_size', 500), normalized=True, device=device)
    finally:
        np.random.set_state(state)


def load_source(spec, *, reference=None):
    """IO boundary: load a package dataset or an explicitly described NPZ grid."""
    if spec['name'] in ('Sphere', 'Torus', 'Circle', 'Moons'):
        manifold = toy_manifold(spec['name'])
        state = np.random.get_state()
        try:
            np.random.seed(spec['seed'])
            train = manifold.sample(spec['sample_size'], normalized=True)
        finally:
            np.random.set_state(state)
        test = toy_reference(spec) if reference is None else reference
        # A sampled point cloud has no parameter-grid geometry contract.
        return Dataset(np.concatenate([train, test])[None], [], [], [], [spec['name']],
                       feature_ndim=1, provided_train_count=len(train))
    if 'path' in spec:
        with np.load(Path(spec['path']).expanduser(), allow_pickle=False) as archive:
            grid = archive['data']
        return Dataset(normalize(grid), list(spec['range_sizes']),
                       list(spec.get('cyclic_dimensions', [])), list(spec['patch_sizes']),
                       list(spec['class_names']), spec.get('feature_ndim', 2))
    data, _, patches, intervals, volume_slices, split_slices, ranges, cyclic = load_dataset_no_split(
        spec['name'], spec['dimension'])
    from microscope.datasets import custom_dsprites, coil20
    labels = coil20.LABEL_MAPPING if spec['name'] == 'extended_coil20' else custom_dsprites.LABEL_MAPPING
    return Dataset(data, list(ranges), list(cyclic), list(patches),
                   list(labels.values()), volume_slices=volume_slices, split_slices=split_slices,
                   difference_intervals=list(intervals))


def candidates(dataset):
    indices = np.arange(len(dataset.flat)).reshape(dataset.grid_shape)
    return indices[dataset.split_slices].reshape(-1)


def sampling_weights(dataset, weighted, device='cpu'):
    indices = candidates(dataset)
    if not weighted:
        return np.full(len(indices), 1 / len(indices))
    from microscope.cyclic_dimensions import get_difference_intervals
    grid = dataset.grid[dataset.volume_slices]
    grid = grid.reshape(*grid.shape[:-dataset.feature_ndim], -1)
    intervals = dataset.difference_intervals or get_difference_intervals(
        list(grid.shape[1:-1]), dataset.range_sizes, dataset.cyclic_dimensions)
    volumes = [volume_element(d, intervals, dataset.cyclic_dimensions,
                              dataset.patch_sizes, device=device) for d in grid]
    weights = np.stack([v / v.sum() for v in volumes])
    if weights.size != len(indices):
        raise ValueError('Volume weights and the interior split grid do not align.')
    return weights.reshape(-1) / weights.sum()


def split_indices(points, weights, spec, device='cpu', fixed_test=None):
    """Return local train/test indices; restore the caller's NumPy RNG state."""
    state = np.random.get_state()
    try:
        np.random.seed(spec['seed'])
        n = len(points)
        match spec['method']:
            case 'full':
                return np.arange(n), np.empty(0, dtype=np.int64)
            case 'fixed_test':
                count = min(spec['max_test_size'], int(0.3 * n))
                if count < 1:
                    raise ValueError('Dataset is too small for a fixed test split.')
                if fixed_test is None:
                    _, test = maximal_distance_selection_torch(points.reshape(n, -1), count, device=device)
                else:
                    test = fixed_test
            case 'random':
                count = max(1, int(spec['test_fraction'] * n))
                test = np.random.choice(n, count, replace=False)
            case _:
                raise ValueError(f"Unknown split: {spec['method']}")
        pool = np.setdiff1d(np.arange(n), test)
        if not len(pool):
            raise ValueError('The split has no training points.')
        probabilities = weights[pool] / weights[pool].sum()
        # Use the existing sampler on index-shaped data, retaining its sample order.
        _, _, _, _, selected = weighted_uniform_splitting(
            pool.reshape(-1, 1, 1), pool, probabilities, spec['train_fraction'])
        train = pool[selected]
        if not len(train):
            raise ValueError('The selected fraction has no training points.')
        return train, np.sort(test)
    finally:
        np.random.set_state(state)


def prepare_split(dataset, spec, device='cpu', *, weights=None, fixed_test=None):
    available = candidates(dataset)
    if spec['method'] == 'full':
        return available, np.empty(0, dtype=np.int64)
    if spec['method'] == 'provided':
        if dataset.provided_train_count is None:
            raise ValueError('This dataset does not provide a train/test split.')
        count = dataset.provided_train_count
        return available[:count], available[count:]
    if weights is None:
        weights = sampling_weights(dataset, spec['weighted'], device)
    train, test = split_indices(dataset.flat[available], weights, spec, device, fixed_test)
    return available[train], available[test]
