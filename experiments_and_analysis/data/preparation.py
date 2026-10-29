"""Dataset metadata, index splits and historical per-domain normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256
import os

import numpy as np

from microscope.computations_grid.volume import volume_element
from microscope.datasets.dataset_split import weighted_uniform_splitting, maximal_distance_selection_torch
from microscope.datasets.generic_dataset_loader import load_dataset_no_split

if TYPE_CHECKING:
    from microscope.datasets.toy_manifolds import Manifold


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
    def flat(self) -> np.ndarray:
        """View the grid with all class and parameter axes flattened into a sample axis."""
        return self.grid.reshape(-1, *self.grid.shape[-self.feature_ndim:])

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Return the class and parameter-grid dimensions."""
        return self.grid.shape[:-self.feature_ndim]


def source_identity(spec: dict) -> dict:
    """Describe the dataset source and its file provenance.

    Args:
        spec: Dataset or split specification.
    """
    result = dict(spec)
    if 'path' in spec:
        path = Path(spec['path']).expanduser().resolve()
        info = path.stat()
        result.update(path=str(path), size=info.st_size, mtime_ns=info.st_mtime_ns)
    elif spec['name'] == 'extended_coil20':
        path = (
            Path(os.environ.get('COIL20_PATH', '~/datasets/curvature-landscapes/coil-20-proc'))
            .expanduser()
            .resolve()
        )
        files = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(path.glob('*.png'))]
        result.update(data_root=str(path), input_files=sha256(repr(files).encode()).hexdigest())
    return result


def normalize(data: np.ndarray) -> np.ndarray:
    """Historical min/max normalization, independently for each selected domain.

    Args:
        data: Input samples of shape (n ...), with sample axis first.
    """
    lower, upper = data.min(), data.max()
    if upper == lower:
        return np.zeros_like(data, dtype=np.float32)
    if lower == 0 and upper == 1:
        return data.astype(np.float32, copy=False)
    return ((data - lower) / (upper - lower)).astype(np.float32)


def toy_manifold(name: str) -> Manifold:
    """Construct a supported analytic toy manifold.

    Args:
        name: Name used for labels or output files.
    """
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


def toy_reference(spec: dict, device: str = 'cpu') -> np.ndarray:
    """Generate a fixed reference tessellation while preserving the caller's RNG state.

    Args:
        spec: Dataset or split specification.
        device: Device used for tensor computations.
    """
    state = np.random.get_state()
    try:
        np.random.seed(spec.get('reference_seed', 0))
        return toy_manifold(spec['name']).tesselation(
            N=spec.get('ground_truth_size', 500), normalized=True, device=device)
    finally:
        np.random.set_state(state)


def load_source(spec: dict, reference: np.ndarray | None = None) -> Dataset:
    """IO boundary: load a package dataset or an explicitly described NPZ grid.

    Args:
        spec: Dataset or split specification.
        reference: Optional toy reference points of shape (n f).
    """
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
        return Dataset(
            np.concatenate([train, test])[None],
            [],
            [],
            [],
            [spec['name']],
            feature_ndim=1,
            provided_train_count=len(train)
        )
    if 'path' in spec:
        with np.load(Path(spec['path']).expanduser(), allow_pickle=False) as archive:
            grid = archive['data']
        return Dataset(
            normalize(grid),
            list(spec['range_sizes']),
            list(spec.get('cyclic_dimensions', [])),
            list(spec['patch_sizes']),
            list(spec['class_names']),
            spec.get('feature_ndim', 2)
        )
    data, _, patches, intervals, volume_slices, split_slices, ranges, cyclic = load_dataset_no_split(
        spec['name'], spec['dimension'])
    from microscope.datasets import custom_dsprites, coil20
    labels = coil20.LABEL_MAPPING if spec['name'] == 'extended_coil20' else custom_dsprites.LABEL_MAPPING
    return Dataset(
        data,
        list(ranges),
        list(cyclic),
        list(patches),
        list(labels.values()),
        volume_slices=volume_slices,
        split_slices=split_slices,
        difference_intervals=list(intervals)
    )


def candidates(dataset: Dataset) -> np.ndarray:
    """Return flat indices of points in the interior split grid.

    Args:
        dataset: Loaded dataset and its grid/split metadata.
    """
    indices = np.arange(len(dataset.flat)).reshape(dataset.grid_shape)
    return indices[dataset.split_slices].reshape(-1)


def sampling_weights(
    dataset: Dataset,
    weighted: bool,
    device: str = 'cpu'
) -> np.ndarray:
    """Compute uniform or volume-based probabilities on the interior grid.

    Args:
        dataset: Loaded dataset and its grid/split metadata.
        weighted: Use volume weights instead of uniform probabilities.
        device: Device used for tensor computations.
    """
    indices = candidates(dataset)
    if not weighted:
        return np.full(len(indices), 1 / len(indices))
    from microscope.cyclic_dimensions import get_difference_intervals
    grid = dataset.grid[dataset.volume_slices]
    grid = grid.reshape(*grid.shape[:-dataset.feature_ndim], -1)
    intervals = dataset.difference_intervals or get_difference_intervals(
        list(grid.shape[1:-1]), dataset.range_sizes, dataset.cyclic_dimensions)
    volumes = [volume_element(
        d,
        intervals,
        dataset.cyclic_dimensions,
        dataset.patch_sizes,
        device=device
    ) for d in grid]
    weights = np.stack([v / v.sum() for v in volumes])
    if weights.size != len(indices):
        raise ValueError('Volume weights and the interior split grid do not align.')
    return weights.reshape(-1) / weights.sum()


def split_indices(
    points: np.ndarray,
    weights: np.ndarray,
    spec: dict,
    device: str = 'cpu',
    fixed_test: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return local train/test indices; restore the caller's NumPy RNG state.

    Args:
        points: Candidate samples of shape (n ...).
        weights: Sampling probabilities of shape (n).
        spec: Dataset or split specification.
        device: Device used for tensor computations.
        fixed_test: Fixed local test indices of shape (t), if supplied.
    """
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
            pool.reshape(-1, 1, 1),
            pool,
            probabilities,
            spec['train_fraction']
        )
        train = pool[selected]
        if not len(train):
            raise ValueError('The selected fraction has no training points.')
        return train, np.sort(test)
    finally:
        np.random.set_state(state)


def prepare_split(
    dataset: Dataset,
    spec: dict,
    device: str = 'cpu',
    weights: np.ndarray | None = None,
    fixed_test: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Select flat train/test indices using the requested split method.

    Args:
        dataset: Loaded dataset and its grid/split metadata.
        spec: Dataset or split specification.
        device: Device used for tensor computations.
        weights: Sampling probabilities of shape (n), or None to compute them.
        fixed_test: Fixed local test indices of shape (t), if supplied.
    """
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
