import numpy as np
import pytest

from microscope.manifold_examples.sampling_uniform import invert_integral
from microscope.manifold_examples.utils import embed_to_ambient_space, random_isometry_transform


def test_invert_integral_accepts_explicit_quadrature_options():
    result = invert_integral(
        lambda x: 1.0,
        y=0.3,
        lower_limit=0.0,
        t0=0.0,
        quad_kwargs={"epsabs": 1e-9},
    )

    assert result == pytest.approx(0.3)


def test_embedding_and_isometry_preserve_pairwise_distances():
    points = np.array([[0.0, 0.0], [3.0, 4.0], [-1.0, 2.0]])
    embedded = embed_to_ambient_space(points, ambient_space_dim=5)
    transformed, (rotation, _) = random_isometry_transform(embedded, limit=2.0, seed=7)

    assert embedded.shape == (3, 5)
    np.testing.assert_array_equal(embedded[:, :2], points)
    np.testing.assert_array_equal(embedded[:, 2:], 0.0)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(5), atol=1e-14)
    np.testing.assert_allclose(
        np.linalg.norm(transformed[:, None] - transformed[None, :], axis=-1),
        np.linalg.norm(points[:, None] - points[None, :], axis=-1),
        atol=1e-14,
    )
