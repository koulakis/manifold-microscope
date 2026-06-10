import unittest

import numpy as np
from sympy import cos, sin
from scipy.special import ellipeinc, ellipkinc

from microscope.manifold_examples.ellipsoid import ellipsoid_volume_element, ellipsoid_scalar_curvature


def ellipsoid_2d_surface_area(a, b, c):
    """Adapted from here: https://www.johndcook.com/blog/2014/07/06/ellipsoid-surface-area."""
    a, b, c = sorted([a, b, c])[::-1]
    if a == c:
        # If all semi-axes are same, return the sphere area.
        return 4 * np.pi * a**2

    m = (a**2 * (b**2 - c**2)) / (b**2 * (a**2 - c**2))
    phi = np.arccos(c / a)

    temp = ellipeinc(phi, m) * sin(phi) ** 2 + ellipkinc(phi, m) * cos(phi) ** 2
    return 2 * np.pi * (c ** 2 + a * b * temp / sin(phi))


def ellipsoid_2d_curvature(a, b, c, phis):
    """From https://www.johndcook.com/blog/2019/10/07/curvature-of-an-ellipsoid."""
    phi1, phi2 = phis[..., 0], phis[..., 1]
    x = a * np.cos(phi1)
    y = b * np.sin(phi1) * np.cos(phi2)
    z = c * np.sin(phi1) * np.sin(phi2)

    return 2 / ((a*b*c) * (x**2/a**4 + y**2/b**4 + z**2/c**4))**2


def _scalar_curvature_3_ellipsoid_3111(phi_2):
    return 18 * (8 * np.sin(phi_2) ** 2 + 3) / (8 * np.sin(phi_2) ** 2 + 1) ** 2


class TestVolume(unittest.TestCase):
    def test_volume_element_integrates_to_2_ellipsoid_area(self):
        n_sample_ellipsoids = 20
        dim = 2

        for sample_as in np.random.uniform(1, 5, (n_sample_ellipsoids, dim + 1)):
            # Known area formula for the ellipsoid.
            gt_area = ellipsoid_2d_surface_area(*sample_as)

            # Create a grid to estimate the area.
            grid_size_per_dim = 2000
            phis_per_dim = [
                *[np.linspace(0, np.pi, num=grid_size_per_dim, endpoint=False) for _ in range(dim - 1)],
                np.linspace(0, 2 * np.pi, num=grid_size_per_dim, endpoint=False)
            ]
            mesh = np.meshgrid(*phis_per_dim, indexing="ij")
            grid_phis = np.stack(mesh, axis=-1).reshape(-1, len(phis_per_dim))

            element_area = np.pi**(dim - 1) * 2*np.pi / len(grid_phis)
            volume_elements = ellipsoid_volume_element(sample_as, grid_phis)

            pred_area = (volume_elements * element_area).sum()

            np.testing.assert_almost_equal(
                gt_area,
                pred_area,
                decimal=4,
                err_msg=f"Different area predictions: gt {gt_area}, pred: {pred_area} for {dim}-ellipsoid."
            )


class TestScalarCurvature(unittest.TestCase):
    def test_scalar_curvature_coincides_with_2_ellipsoid_formula(self):
        np.random.seed(42)
        n_sample_ellipsoids = 20
        dim = 2

        for sample_as in np.random.uniform(1, 5, (n_sample_ellipsoids, dim + 1)):
            # Sample points to compute curvature on.
            n_samples = 100
            phis = np.hstack([
                np.random.uniform(0, np.pi, size=(n_samples, dim - 1)),
                np.random.uniform(0, 2 * np.pi, size=(n_samples, 1))
            ])

            gt_curvatures = ellipsoid_2d_curvature(*sample_as, phis)
            pred_curvatures = ellipsoid_scalar_curvature(sample_as, phis)

            np.testing.assert_almost_equal(
                gt_curvatures,
                pred_curvatures,
                err_msg=
                f"Different scalar curvature predictions: gt {gt_curvatures}, pred: {pred_curvatures} "
                f"for {dim}-ellipsoid."
            )

    def test_scalar_curvature_on_isotropic_3_ellipsoid(self):
        dim = 3
        n_sample_ellipsoids = 20

        for r in np.random.uniform(1, 5, n_sample_ellipsoids):
            sample_as = np.repeat(r, dim + 1)

            # Sample points to compute curvature on.
            n_samples = 100
            phis = np.hstack([
                np.random.uniform(0, np.pi, size=(n_samples, dim - 1)),
                np.random.uniform(0, 2 * np.pi, size=(n_samples, 1))
            ])

            gt_curvatures = np.repeat(6/r**2, n_samples)
            pred_curvatures = ellipsoid_scalar_curvature(sample_as, phis)

            np.testing.assert_almost_equal(
                gt_curvatures,
                pred_curvatures,
                err_msg=
                f"Different scalar curvature predictions: gt {gt_curvatures}, pred: {pred_curvatures} "
                f"for {dim}-ellipsoid."
            )

    def test_scalar_curvature_coincides_with_3_ellipsoid_formula_on_simple_anisotropic_case(self):
        dim = 3
        sample_as = np.array([3, 1, 1, 1])

        # Sample points to compute curvature on.
        n_samples = 100
        phis = np.hstack([
            np.random.uniform(0, np.pi, size=(n_samples, dim - 1)),
            np.random.uniform(0, 2 * np.pi, size=(n_samples, 1))
        ])

        gt_curvatures = _scalar_curvature_3_ellipsoid_3111(phis[..., 0])
        pred_curvatures = ellipsoid_scalar_curvature(sample_as, phis)

        np.testing.assert_almost_equal(
            gt_curvatures,
            pred_curvatures,
            err_msg=
            f"Different scalar curvature predictions: gt {gt_curvatures}, pred: {pred_curvatures} "
            f"for {dim}-ellipsoid."
        )
