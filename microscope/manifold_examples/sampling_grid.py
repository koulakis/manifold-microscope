from typing import Optional

import numpy as np

from microscope.manifold_examples.utils import embed_to_ambient_space, random_isometry_transform


def sample_ellipsoid_on_grid(
    semi_axes: list[float],
    n_samples_per_dim: list[int],
    ambient_space_dim: int,
    fraction_of_angles: float = 0.7,
    apply_random_isometry: bool = True,
    seed: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[tuple[np.ndarray, np.ndarray]]]:
    """Sample points on a spherical coordinate grid on an n-ellipsoid. All angles, except from the polar one, are
    sampled on a fraction of their domain to create a simple cylindrical grid.
    """
    assert len(semi_axes) - 1 == len(n_samples_per_dim), "Inconsistent number of semi-axes and samples per dimension."
    dim = len(n_samples_per_dim)
    assert ambient_space_dim >= dim + 1

    phis_per_dim = [
        *[
            np.linspace((1 - fraction_of_angles)*np.pi, fraction_of_angles*np.pi, num=n_samples)
            for n_samples in n_samples_per_dim[:-1]
        ],
        np.linspace(0, 2 * np.pi, num=n_samples_per_dim[-1], endpoint=False)
    ]
    phis = np.meshgrid(*phis_per_dim, indexing="ij")

    grid = np.stack([
        a
        * (np.ones_like(phis[0]) if i == dim else np.cos(phis[i]))
        * (
            np.prod(np.stack([np.sin(phis[i]) for i in range(i)], axis=-1), axis=-1)
            if i > 0
            else np.ones_like(phis[0])
        )
        for i, a in enumerate(semi_axes)
    ], axis=-1)

    if ambient_space_dim > dim + 1:
        grid = embed_to_ambient_space(grid, ambient_space_dim)

    isometry = None
    if apply_random_isometry:
        grid, isometry = random_isometry_transform(grid, limit=max(semi_axes), seed=seed)

    return grid, np.array(semi_axes), np.stack(phis, axis=-1), isometry


def sample_hyperboloid_on_grid(
    semi_axes: list[float],
    n_samples_per_dim: list[int],
    ambient_space_dim: int,
    apply_random_isometry: bool = True,
    height_variable_limit: float = 1.,
    seed: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[tuple[np.ndarray, np.ndarray]]]:
    """Sample points on a coordinate grid on an n-hyperboloid.
    """
    assert len(semi_axes) - 1 == len(n_samples_per_dim), "Inconsistent number of semi-axes and samples per dimension."
    dim = len(n_samples_per_dim)
    assert ambient_space_dim >= dim + 1

    phis_per_dim = [
        *[
            np.linspace(-height_variable_limit, height_variable_limit, num=n_samples)
            for n_samples in n_samples_per_dim[:-1]
        ],
        np.linspace(0, 2 * np.pi, num=n_samples_per_dim[-1], endpoint=False)
    ]
    phis = np.meshgrid(*phis_per_dim, indexing="ij")

    grid = np.stack([
        a
        * (
            np.ones_like(phis[0]) if i == dim
            else (
                np.sin(phis[i])
                if i == dim - 1
                else np.sinh(phis[i])
            ))
        * (
            np.prod(np.stack([
                np.cos(phis[i]) if i == dim - 1 else np.cosh(phis[i])
                for i in range(i)
            ], axis=-1), axis=-1)
            if i > 0
            else np.ones_like(phis[0])
        )
        for i, a in enumerate(semi_axes)
    ], axis=-1)

    if ambient_space_dim > 3:
        grid = embed_to_ambient_space(grid, ambient_space_dim)

    isometry = None
    if apply_random_isometry:
        grid, isometry = random_isometry_transform(grid, limit=max(semi_axes), seed=seed)

    return grid, np.array(semi_axes), np.stack(phis, axis=-1), isometry
