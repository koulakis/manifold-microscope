import multiprocessing
from functools import partial

import scipy
from joblib import Parallel, delayed

import numpy as np
from tqdm import tqdm


def parallel_apply(array, f, processes=None, element_wise=False, **kwargs):
    """Apply a function on an array in multiple threads. Just a convenience function."""
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


def embed_to_ambient_space(grid, ambient_space_dim):
    return np.concatenate(
        [
            grid,
            np.zeros((*grid.shape[:-1], ambient_space_dim - grid.shape[-1]))
        ],
        axis=-1
    )


def random_isometry_transform(grid, limit, seed=None) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    ambient_space_dim = grid.shape[-1]

    random_orthogonal_matrix = scipy.stats.ortho_group(ambient_space_dim, seed=seed).rvs()
    random_translation_vector = np.random.uniform(-limit, limit, size=ambient_space_dim)

    return (
        grid @ random_orthogonal_matrix + random_translation_vector,
        (random_orthogonal_matrix, random_translation_vector)
    )
