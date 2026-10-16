import numpy as np
import pytest

from microscope.datasets.toy_manifolds import Circle, Moons, Sphere, Torus


@pytest.mark.parametrize("manifold", [Circle(2.0), Moons(2.0), Sphere(2.0), Torus(2.0, 0.5)])
def test_normalized_manifolds_have_unit_volume(manifold):
    assert manifold.measures(normalized=True).volume == pytest.approx(1.0)


@pytest.mark.parametrize("manifold", [Circle(2.0), Sphere(2.0), Torus(2.0, 0.5)])
def test_sampled_normals_are_unit_vectors(manifold):
    np.random.seed(3)
    points, normals = manifold.sample(7, return_unitary_normal=True)

    assert points.shape == normals.shape
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=1e-6)
