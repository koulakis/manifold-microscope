import numpy as np
from tqdm import tqdm

from microscope.computations_grid.basic import partial_derivatives_across_all_dims_batched
from microscope.cyclic_dimensions import get_difference_intervals


def null_space(A: np.ndarray) -> np.ndarray:
    """Compute the null space of a matrix. The function decomposes the matrice with SVD to U, S, V and
    then keeps the last n - m vectors of V which span the orthogonal space.

    Args:
        A: A (d1 ... dk m n) array of matrices describing linear transformations whose null spaces will be computed.

    Returns:
        A (d1 ... dk n - m n) array with k orthogonal vectors spanning the null space. Note those are parts of the
        n x n matrices V of the SVD decompositions U S V.
    """
    m, n = A.shape[-2:]

    U, S, V = np.linalg.svd(A, full_matrices=True)

    nullspace = V[..., m:, :].conj()

    return nullspace


def cotangent_gaussian_noise(
    data: np.ndarray,
    sigma: float,
    range_sizes: list[float],
    patch_sizes: list[int],
    cyclic_dimensions: list[int],
    batch_size: int = 20,
    check_orthogonality: bool = False,
    orthogonality_check_sample_size: int = 1000
) -> np.ndarray:
    # Estimate the tangent vectors of the data.
    dims_shape = data.shape[:-1]

    difference_intervals = get_difference_intervals(
        n_samples_per_dim=list(dims_shape),
        range_sizes=range_sizes,
        cyclic_dimensions=cyclic_dimensions
    )
    tangent_vectors = partial_derivatives_across_all_dims_batched(
        data,
        cyclic_dimensions=cyclic_dimensions,
        difference_intervals=difference_intervals,
        patch_sizes=patch_sizes
    )
    tangent_vectors = np.swapaxes(tangent_vectors, -1, -2)

    # Compute a cotangent base per point and sample it.
    tangent_vectors_flat = tangent_vectors.reshape(np.prod(tangent_vectors.shape[:-2]), *tangent_vectors.shape[-2:])

    batch_results = []
    n_batches = tangent_vectors_flat.shape[0] // batch_size
    batch_residue = tangent_vectors_flat.shape[0] % batch_size
    for i in tqdm(list(range(n_batches))):
        s = batch_size*i
        e = s + batch_size
        bases = null_space(tangent_vectors_flat[s:e]).astype(np.float32)
        batch_results.append(bases)
    if batch_residue != 0:
        s = batch_size * n_batches
        e = s + batch_residue
        bases = null_space(tangent_vectors_flat[s:e]).astype(np.float32)
        batch_results.append(bases)
    cotangent_bases_flat = np.concatenate(batch_results)
    cotangent_bases = cotangent_bases_flat.reshape(*tangent_vectors.shape[:-2], *cotangent_bases_flat.shape[-2:])

    if check_orthogonality:
        tangent_test_idx = np.random.choice(
            len(tangent_vectors_flat),
            size=orthogonality_check_sample_size,
            replace=False
        )
        cotangent_test_idx = np.random.choice(
            len(cotangent_bases_flat),
            size=orthogonality_check_sample_size,
            replace=False
        )
        tangent_vectors_flat_test = tangent_vectors_flat[tangent_test_idx]
        cotangent_vectors_flat_test = cotangent_bases_flat[cotangent_test_idx]
        error = (tangent_vectors_flat_test @ np.swapaxes(cotangent_vectors_flat_test, -1, -2)).max()
        if error > 1e-7:
            print(f"Orthogonality of the cotangent space failed with error: {error}")

    codim = data.shape[-1] - len(dims_shape)

    gaussian_noise = np.random.normal(
        size=(*tangent_vectors.shape[:-2], codim),
        loc=0,
        scale=sigma
    )
    cotangent_noise = gaussian_noise[..., None] @ cotangent_bases

    return cotangent_noise[..., 0, :]


def compute_gaussian_noise(shape: tuple[int, ...], sigma: float):
    return np.random.normal(size=shape, loc=0, scale=sigma)
