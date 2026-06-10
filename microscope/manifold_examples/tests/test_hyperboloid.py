import unittest

import numpy as np

from microscope.manifold_examples.hyperboloid import hyperboloid_scalar_curvature


def _2_hyperboloid_curvature(a, b, c, phis):
    """From https://www.johndcook.com/blog/2019/10/07/curvature-of-an-hyperboloid."""
    phi1, phi2 = phis[..., 0], phis[..., 1]
    x = a * np.sinh(phi1)
    y = b * np.cosh(phi1) * np.sin(phi2)
    z = c * np.cosh(phi1) * np.cos(phi2)

    return - 2 / ((a*b*c) * (x**2/a**4 + y**2/b**4 + z**2/c**4))**2


def _scalar_curvature_3_hyperboloid_3111(phis) -> float:
    """Computed with einsteinpy."""
    phi_1, phi_2 = phis[..., 0], phis[..., 1]

    return -9*(np.cosh(2*phi_1 - 2*phi_2) - np.cosh(2*phi_1 + 2*phi_2))**2/((-18*np.cosh(phi_1)**2*np.cosh(phi_2)**2 + 8*np.cosh(phi_1)**2 + 1)**2*(20*(np.cosh(4*phi_1) - 1)*np.cosh(2*phi_2) + 32*np.sinh(phi_1)**2*np.sinh(phi_2)**2*np.cosh(phi_1)**2*np.cosh(2*phi_2) + 144*np.cosh(phi_1)**2*np.cosh(2*phi_2) + np.cosh(4*phi_1) + np.cosh(4*phi_2) - np.cosh(2*phi_1 - 2*phi_2)**2 - np.cosh(2*phi_1 + 2*phi_2)**2)) + 18432*(np.cosh(2*phi_1 - 2*phi_2) - np.cosh(2*phi_1 + 2*phi_2))*(648*np.sinh(phi_1)**6*np.sinh(phi_2)**4 + 720*np.sinh(phi_1)**6*np.sinh(phi_2)**2 + 200*np.sinh(phi_1)**6 + 1944*np.sinh(phi_1)**4*np.sinh(phi_2)**4 + 2142*np.sinh(phi_1)**4*np.sinh(phi_2)**2 + 590*np.sinh(phi_1)**4 + 1944*np.sinh(phi_1)**2*np.sinh(phi_2)**4 + 2124*np.sinh(phi_1)**2*np.sinh(phi_2)**2 + 581*np.sinh(phi_1)**2 - 648*np.sinh(phi_2)**4*np.cosh(phi_1)**6 + 648*np.sinh(phi_2)**4 - 720*np.sinh(phi_2)**2*np.cosh(phi_1)**6 + 702*np.sinh(phi_2)**2 - 200*np.cosh(phi_1)**6 + 191)*np.sinh(phi_1)*np.sinh(phi_2)*np.cosh(phi_1)**3*np.cosh(phi_2)/((-18*np.cosh(phi_1)**2*np.cosh(phi_2)**2 + 8*np.cosh(phi_1)**2 + 1)*(20*(np.cosh(4*phi_1) - 1)*np.cosh(2*phi_2) + 32*np.sinh(phi_1)**2*np.sinh(phi_2)**2*np.cosh(phi_1)**2*np.cosh(2*phi_2) + 144*np.cosh(phi_1)**2*np.cosh(2*phi_2) + np.cosh(4*phi_1) + np.cosh(4*phi_2) - np.cosh(2*phi_1 - 2*phi_2)**2 - np.cosh(2*phi_1 + 2*phi_2)**2)**3) - (2*np.sinh(phi_1)**2*np.sinh(phi_2)**2 + 10*np.sinh(phi_1)**2 + 9)*((18*np.sinh(phi_2)**2 + 9)*np.cosh(phi_1)**2/(-18*np.cosh(phi_1)**2*np.cosh(phi_2)**2 + 8*np.cosh(phi_1)**2 + 1)**2 - 1 + 144*np.sinh(2*phi_2)*np.cosh(phi_1)**4*np.tanh(phi_2)/(20*(np.cosh(4*phi_1) - 1)*np.cosh(2*phi_2) + 32*np.sinh(phi_1)**2*np.sinh(phi_2)**2*np.cosh(phi_1)**2*np.cosh(2*phi_2) + 144*np.cosh(phi_1)**2*np.cosh(2*phi_2) + np.cosh(4*phi_1) + np.cosh(4*phi_2) - np.cosh(2*phi_1 - 2*phi_2)**2 - np.cosh(2*phi_1 + 2*phi_2)**2) - np.sinh(2*phi_1)*np.tanh(phi_1)/(-36*np.cosh(phi_1)**2*np.cosh(phi_2)**2 + 16*np.cosh(phi_1)**2 + 2))/(-18*np.cosh(phi_1)**4*np.cosh(phi_2)**2 + 8*np.cosh(phi_1)**4 + np.cosh(phi_1)**2) + 1536*(378*np.sinh(phi_1)**6*np.sinh(phi_2)**4 + 435*np.sinh(phi_1)**6*np.sinh(phi_2)**2 + 125*np.sinh(phi_1)**6 + 1134*np.sinh(phi_1)**4*np.sinh(phi_2)**4 + 1308*np.sinh(phi_1)**4*np.sinh(phi_2)**2 + 375*np.sinh(phi_1)**4 + 1134*np.sinh(phi_1)**2*np.sinh(phi_2)**4 + 1311*np.sinh(phi_1)**2*np.sinh(phi_2)**2 + 375*np.sinh(phi_1)**2 - 378*np.sinh(phi_2)**4*np.cosh(phi_1)**6 + 378*np.sinh(phi_2)**4 - 411*np.sinh(phi_2)**2*np.cosh(phi_1)**6 + 438*np.sinh(phi_2)**2 - 125*np.cosh(phi_1)**6 + 125)*np.cosh(2*phi_2)/((-18*np.cosh(phi_1)**2*np.cosh(phi_2)**2 + 8*np.cosh(phi_1)**2 + 1)*(20*(np.cosh(4*phi_1) - 1)*np.cosh(2*phi_2) + 32*np.sinh(phi_1)**2*np.sinh(phi_2)**2*np.cosh(phi_1)**2*np.cosh(2*phi_2) + 144*np.cosh(phi_1)**2*np.cosh(2*phi_2) + np.cosh(4*phi_1) + np.cosh(4*phi_2) - np.cosh(2*phi_1 - 2*phi_2)**2 - np.cosh(2*phi_1 + 2*phi_2)**2)**2) + (9216*(4*np.cosh(2*phi_1) + 5)*(-18*np.cosh(phi_1)**2*np.cosh(phi_2)**2 + 8*np.cosh(phi_1)**2 + 1)**2*np.sinh(phi_2)**2*np.cosh(phi_1)**6*np.cosh(phi_2) + (4*np.cosh(2*phi_1) + 5)*(-18*np.cosh(phi_1)**2*np.cosh(phi_2)**2 + 8*np.cosh(phi_1)**2 + 1)*(20*(np.cosh(4*phi_1) - 1)*np.cosh(2*phi_2) + 32*np.sinh(phi_1)**2*np.sinh(phi_2)**2*np.cosh(phi_1)**2*np.cosh(2*phi_2) + 144*np.cosh(phi_1)**2*np.cosh(2*phi_2) + np.cosh(4*phi_1) + np.cosh(4*phi_2) - np.cosh(2*phi_1 - 2*phi_2)**2 - np.cosh(2*phi_1 + 2*phi_2)**2)**2*np.cosh(phi_2) + 144*(4*np.cosh(2*phi_1) + 5)*(-18*np.cosh(phi_1)**2*np.cosh(phi_2)**2 + 8*np.cosh(phi_1)**2 + 1)*(20*(np.cosh(4*phi_1) - 1)*np.cosh(2*phi_2) + 32*np.sinh(phi_1)**2*np.sinh(phi_2)**2*np.cosh(phi_1)**2*np.cosh(2*phi_2) + 144*np.cosh(phi_1)**2*np.cosh(2*phi_2) + np.cosh(4*phi_1) + np.cosh(4*phi_2) - np.cosh(2*phi_1 - 2*phi_2)**2 - np.cosh(2*phi_1 + 2*phi_2)**2)*np.sinh(phi_2)*np.sinh(2*phi_2)*np.cosh(phi_1)**4 + 256*(-18*np.cosh(phi_1)**2*np.cosh(phi_2)**2 + 8*np.cosh(phi_1)**2 + 1)**3*np.sinh(phi_1)**2*np.cosh(phi_1)**4*np.cosh(phi_2) - 9*(20*(np.cosh(4*phi_1) - 1)*np.cosh(2*phi_2) + 32*np.sinh(phi_1)**2*np.sinh(phi_2)**2*np.cosh(phi_1)**2*np.cosh(2*phi_2) + 144*np.cosh(phi_1)**2*np.cosh(2*phi_2) + np.cosh(4*phi_1) + np.cosh(4*phi_2) - np.cosh(2*phi_1 - 2*phi_2)**2 - np.cosh(2*phi_1 + 2*phi_2)**2)**2*np.cosh(phi_1)**2*np.cosh(phi_2)*np.cosh(2*phi_2))/((-18*np.cosh(phi_1)**2*np.cosh(phi_2)**2 + 8*np.cosh(phi_1)**2 + 1)**2*(20*(np.cosh(4*phi_1) - 1)*np.cosh(2*phi_2) + 32*np.sinh(phi_1)**2*np.sinh(phi_2)**2*np.cosh(phi_1)**2*np.cosh(2*phi_2) + 144*np.cosh(phi_1)**2*np.cosh(2*phi_2) + np.cosh(4*phi_1) + np.cosh(4*phi_2) - np.cosh(2*phi_1 - 2*phi_2)**2 - np.cosh(2*phi_1 + 2*phi_2)**2)**2*np.cosh(phi_1)**2*np.cosh(phi_2))


class TestScalarCurvature(unittest.TestCase):
    def test_scalar_curvature_coincides_with_2_hyperboloid_formula(self):
        n_sample_hyperboloids = 20
        dim = 2

        for sample_as in np.random.uniform(1, 5, (n_sample_hyperboloids, dim + 1)):
            # Sample points to compute curvature on.
            n_samples = 100
            phis = np.hstack([
                np.random.uniform(0, np.pi, size=(n_samples, dim - 1)),
                np.random.uniform(0, 2 * np.pi, size=(n_samples, 1))
            ])

            gt_curvatures = _2_hyperboloid_curvature(*sample_as, phis)
            pred_curvatures = hyperboloid_scalar_curvature(sample_as, phis)

            np.testing.assert_almost_equal(
                gt_curvatures,
                pred_curvatures,
                err_msg=
                f"Different scalar curvature predictions: gt {gt_curvatures}, pred: {pred_curvatures} "
                f"for {dim}-hyperboloid."
            )

    def test_scalar_curvature_coincides_with_2_hyperboloid_when_one_semi_axis_is_zero(self):
        n_sample_hyperboloids = 20
        dim = 3

        for sample_as in np.random.uniform(1, 5, (n_sample_hyperboloids, dim + 1)):
            # Sample points to compute curvature on.
            n_samples = 100
            phis = np.hstack([
                np.random.uniform(0, np.pi, size=(n_samples, dim - 1)),
                np.random.uniform(0, 2 * np.pi, size=(n_samples, 1))
            ])
            phis[:, 0] = 0
            # Note that we need to move far away from the origin on the first semi-axis so that the changes in phi1
            # have a negligible contribution to the curvature.
            sample_as[0] = 1_000_000

            gt_curvatures = _2_hyperboloid_curvature(*sample_as[1:], phis[:, 1:])
            pred_curvatures = hyperboloid_scalar_curvature(sample_as, phis)

            np.testing.assert_almost_equal(
                gt_curvatures,
                pred_curvatures,
                err_msg=
                f"Different scalar curvature predictions: gt {gt_curvatures}, pred: {pred_curvatures} "
                f"for {dim}-hyperboloid."
            )

    def test_scalar_curvature_coincides_with_3_hyperboloid_formula_on_simple_anisotropic_case(self):
        dim = 3
        np.random.seed(42)
        sample_as = np.array([3, 1, 1, 1]).astype(np.float64)

        # Sample points to compute curvature on.
        n_samples = 100
        phis = np.hstack([
            np.random.uniform(0, np.pi, size=(n_samples, dim - 1)),
            np.random.uniform(0, 2 * np.pi, size=(n_samples, 1))
        ]).astype(np.float64)

        gt_curvatures = _scalar_curvature_3_hyperboloid_3111(phis)
        pred_curvatures = hyperboloid_scalar_curvature(sample_as, phis)

        np.testing.assert_almost_equal(
            gt_curvatures,
            pred_curvatures,
            decimal=2,
            err_msg=
            f"Different scalar curvature predictions: gt {gt_curvatures}, pred: {pred_curvatures} "
            f"for {dim}-hyperboloid."
        )
