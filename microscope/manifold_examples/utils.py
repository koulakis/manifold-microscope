import multiprocessing
from collections.abc import Callable
from functools import partial
from typing import Any

import scipy
from joblib import Parallel, delayed

import numpy as np
from tqdm import tqdm


def parallel_apply(
    array: np.ndarray,
    f: Callable,
    processes: int | None = None,
    element_wise: bool = False,
    **kwargs: Any
) -> np.ndarray:
    """Apply a function on an array in multiple threads. Just a convenience function.

    Args:
        array: Input array, shape (n ...).
        f: Function to apply to each value or leading-axis item.
        processes: Number of worker processes; None uses the available CPU count.
        element_wise: Apply f to scalars instead of leading-axis items.
        **kwargs: Additional keyword arguments forwarded to f.
    """
    if processes is None:
        processes = multiprocessing.cpu_count()

    array = np.array(array)
    shape = array.shape

    if element_wise:
        array = array.flatten()

    results_gen = Parallel(
        n_jobs=processes,
        return_as="generator"
    )(
        delayed(partial(f, **kwargs))(x)
        for x in array
    )

    results = [r for r in tqdm(results_gen, total=len(array))]
    results = np.array(results)

    if element_wise:
        results = results.reshape(shape)

    return results


def embed_to_ambient_space(grid: np.ndarray, ambient_space_dim: int) -> np.ndarray:
    """Append zero coordinates to a point grid.

    Args:
        grid: Embedded point grid, shape (s1 ... sd f).
        ambient_space_dim: Number of embedding coordinates.
    """
    return np.concatenate(
        [
            grid,
            np.zeros((*grid.shape[:-1], ambient_space_dim - grid.shape[-1]))
        ],
        axis=-1
    )


def random_isometry_transform(
    grid: np.ndarray,
    limit: float,
    seed: int | None = None
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Rotate and translate a point grid.

    Args:
        grid: Embedded point grid, shape (s1 ... sd f).
        limit: Symmetric bound for the random translation coordinates.
        seed: Optional random seed for the isometry.
    """
    ambient_space_dim = grid.shape[-1]

    random_orthogonal_matrix = scipy.stats.ortho_group(ambient_space_dim, seed=seed).rvs()
    random_translation_vector = np.random.uniform(-limit, limit, size=ambient_space_dim)

    return (
        grid @ random_orthogonal_matrix + random_translation_vector,
        (random_orthogonal_matrix, random_translation_vector)
    )
