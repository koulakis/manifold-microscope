import unittest

import numpy as np
import sympy as sp
from sympy import Matrix, cos, sin

from microscope.manifold_examples.symbolic_computations import volume_element_formula, scalar_curvature_formula


class TestBasicFunctions(unittest.TestCase):
    def test_volume_element_formula(self):
        for dim in [2, 3, 5]:
            # Define the parametrization.
            R = sp.symbols("R")
            phis = sp.symbols(" ".join([f"phi_{i}" for i in range(1, dim + 1)]))

            coordinates = Matrix(phis)
            u = Matrix([
                R
                * (1 if i == dim else cos(phis[i]))
                * sp.prod([sin(phis[i]) for i in range(i)])
                for i in range(dim + 1)
            ])

            # Known volume element formula for the sphere.
            n = len(coordinates)
            components = np.concatenate([[R], np.vectorize(sin)((coordinates[:-1]))])
            gt_element = np.prod(components ** np.arange(n, 0, -1))

            pred_element = volume_element_formula(u, coordinates)

            # Lambdify and check equality for some random inputs.
            gt_element_np = sp.lambdify([R, *phis], gt_element)
            pred_element_np = sp.lambdify([R, *phis], pred_element)

            n_samples = 100
            random_phis = [
                *[np.random.uniform(low=0, high=np.pi, size=n_samples) for _ in range(dim - 1)],
                np.random.uniform(low=0, high=2*np.pi, size=n_samples)
            ]
            random_Rs = np.random.uniform(0, 10, n_samples)

            np.testing.assert_almost_equal(
                gt_element_np(random_Rs, *random_phis),
                pred_element_np(random_Rs, *random_phis),
                err_msg=f"Different predictions: gt {gt_element}, pred: {pred_element}."
            )

    def test_scalar_curvature_formula(self):
        for dim in [2, 3, 5]:
            # Define the parametrization.
            R = sp.symbols("R")
            phis = sp.symbols(" ".join([f"phi_{i}" for i in range(1, dim + 1)]))

            coordinates = Matrix(phis)
            u = Matrix([
                R
                * (1 if i == dim else cos(phis[i]))
                * sp.prod([sin(phis[i]) for i in range(i)])
                for i in range(dim + 1)
            ])

            # Known scalar curvature formula for the sphere.
            gt_curvature = dim * (dim - 1) / R**2

            pred_curvature, _, _ = scalar_curvature_formula(u, coordinates)

            # Lambdify and check equality for some random inputs.
            gt_curvature_np = sp.lambdify([R, *phis], gt_curvature)
            pred_curvature_np = sp.lambdify([R, *phis], pred_curvature)

            n_samples = 100
            random_phis = [
                *[np.random.uniform(low=0, high=np.pi, size=n_samples) for _ in range(dim - 1)],
                np.random.uniform(low=0, high=2*np.pi, size=n_samples)
            ]
            random_Rs = np.random.uniform(0, 10, n_samples)

            np.testing.assert_almost_equal(
                gt_curvature_np(random_Rs, *random_phis),
                pred_curvature_np(random_Rs, *random_phis),
                err_msg=f"Different predictions: gt {gt_curvature}, pred: {pred_curvature}."
            )
