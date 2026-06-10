import unittest

import numpy as np
import pytest

from microscope.computations_grid.curvature import scalar_curvature, compute_total_curvature
from microscope.computations_grid.volume import volume_element
from microscope.cyclic_dimensions import get_difference_intervals
from microscope.manifold_examples.ellipsoid import ellipsoid_scalar_curvature
from microscope.manifold_examples.hyperboloid import hyperboloid_scalar_curvature
from microscope.manifold_examples.sampling_grid import sample_ellipsoid_on_grid, sample_hyperboloid_on_grid
from microscope.manifold_examples.sphere import sphere_scalar_curvature


class TestEllipsoids(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0"

    def test_scalar_curvature_2d_ellipsoids(self):
        for semi_axes in [[2, 2, 2], [1, 1.5, 3]]:
            # Sample a relatively dense grid.
            fraction_of_angles = 0.8
            n_samples_per_dim = [1_006, 1_000]
            cyclic_dimensions = [1]

            points, semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=3,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Compute the ground truth values.
            curvature_gt = ellipsoid_scalar_curvature(semi_axes, phis)

            # Estimate the curvature.
            patch_sizes = [56, 56]
            range_sizes = [(2 * fraction_of_angles - 1) * np.pi, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            curvature_pred = scalar_curvature(
                points,
                cyclic_dimensions=cyclic_dimensions,
                difference_intervals=difference_intervals,
                patch_sizes=patch_sizes,
                device=self.device
            )

            np.testing.assert_almost_equal(
                curvature_gt[3:-3, ...],
                curvature_pred,
                decimal=4,
                err_msg=f"Failed on an ellipsoid with semi axes: {semi_axes}."
            )

    def test_scalar_curvature_2d_ellipsoids_irregular_grid(self):
        for semi_axes in [[2, 2, 2], [1, 1.5, 3]]:
            # Sample a relatively dense grid.
            fraction_of_angles = 0.8
            n_samples_per_dim = [1_223, 1_344]
            cyclic_dimensions = [1]

            points, semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=3,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Compute the ground truth values.
            curvature_gt = ellipsoid_scalar_curvature(semi_axes, phis)

            # Estimate the curvature.
            patch_sizes = [34, 42]
            range_sizes = [(2 * fraction_of_angles - 1) * np.pi, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            curvature_pred = scalar_curvature(
                points,
                cyclic_dimensions=cyclic_dimensions,
                difference_intervals=difference_intervals,
                patch_sizes=patch_sizes,
                device=self.device
            )

            np.testing.assert_almost_equal(
                curvature_gt[3:-3, ...],
                curvature_pred,
                decimal=4,
                err_msg=f"Failed on an ellipsoid with semi axes: {semi_axes}."
            )

    def test_total_scalar_curvature_2d_ellipsoids(self):
        for semi_axes in [[2, 2, 2], [1, 1.5, 3]]:
            # Sample a relatively dense grid.
            fraction_of_angles = 0.9999
            n_samples_per_dim = [10_223, 10_344]
            cyclic_dimensions = [1]

            points, semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=3,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Compute the ground truth via the Gauss-Bonnet theorem: Int(K dA) = 2 pi chi(M) = 4 pi.
            # The scalar curvature is 2*K, thus the total scalar curvature is 8 pi.
            total_curvature_gt = 8 * np.pi

            # Estimate the curvature.
            patch_sizes = [1034, 1042]
            range_sizes = [(2 * fraction_of_angles - 1) * np.pi, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            volume_pred = volume_element(
                points,
                cyclic_dimensions=cyclic_dimensions,
                difference_intervals=difference_intervals,
                patch_sizes=patch_sizes,
                device=self.device
            )[2:-2]

            curvature_pred = scalar_curvature(
                points,
                cyclic_dimensions=cyclic_dimensions,
                difference_intervals=difference_intervals,
                patch_sizes=patch_sizes,
                device=self.device
            )

            total_curvature_pred = compute_total_curvature(curvature_pred, volume_pred, range_sizes)

            np.testing.assert_almost_equal(
                total_curvature_gt,
                total_curvature_pred,
                decimal=2,
                err_msg=f"Failed on an ellipsoid with semi axes: {semi_axes}."
            )

    @pytest.mark.slow
    def test_scalar_curvature_3d_ellipsoid(self):
        semi_axes = [1, 1.5, 3, 1.8]
        for ambient_dim in [4, 100, 4096]:
            # Sample a relatively dense grid.
            fraction_of_angles = 0.8
            n_samples_per_dim = [56, 56, 50] if ambient_dim > 100 else [106, 106, 100]
            cyclic_dimensions = [2]

            points, semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=ambient_dim,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Compute the ground truth values.
            curvature_gt = ellipsoid_scalar_curvature(semi_axes, phis)

            # Estimate the metric.
            patch_sizes = [16, 16, 16]
            range_sizes = [(2*fraction_of_angles - 1) * np.pi, (2*fraction_of_angles - 1) * np.pi, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            curvature_pred = scalar_curvature(
                points,
                cyclic_dimensions=cyclic_dimensions,
                difference_intervals=difference_intervals,
                patch_sizes=patch_sizes,
                device=self.device
            )

            np.testing.assert_almost_equal(
                curvature_gt[3:-3, 3:-3, ...],
                curvature_pred,
                decimal=1 if ambient_dim > 100 else 2,
                err_msg=f"Failed on an ellipsoid with semi axes: {semi_axes} and ambient dim {ambient_dim}."
            )

    @pytest.mark.slow
    def test_scalar_curvature_4d_sphere(self):
        semi_axes = 5*[2]
        for ambient_dim in [5, 10]:
            # Sample a relatively dense grid.
            fraction_of_angles = 0.8
            n_samples_per_dim = 3*[106] + [100]
            cyclic_dimensions = [3]

            points, semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=ambient_dim,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Compute the ground truth values.
            curvature_gt = sphere_scalar_curvature(2., phis)

            # Estimate the metric.
            patch_sizes = 4*[16]
            range_sizes = 3*[(2*fraction_of_angles - 1) * np.pi] + [2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            curvature_pred = scalar_curvature(
                points,
                cyclic_dimensions=cyclic_dimensions,
                difference_intervals=difference_intervals,
                patch_sizes=patch_sizes,
                device=self.device
            )

            np.testing.assert_almost_equal(
                curvature_gt[3:-3, 3:-3, 3:-3, ...],
                curvature_pred,
                decimal=3,
                err_msg=f"Failed on an ellipsoid with semi axes: {semi_axes}."
            )


class TestHyperboloids(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0"

    def test_scalar_curvature_2d_hyperboloid(self):
        semi_axes = [5, 1, 3]

        # Sample a relatively dense grid.
        n_samples_per_dim = [1_006, 1_000]
        cyclic_dimensions = [1]

        points, semi_axes, phis, _ = sample_hyperboloid_on_grid(
            semi_axes=semi_axes,
            n_samples_per_dim=n_samples_per_dim,
            ambient_space_dim=3,
            apply_random_isometry=True
        )

        # Compute the ground truth values.
        curvature_gt = hyperboloid_scalar_curvature(semi_axes, phis)

        # Estimate the metric.
        patch_sizes = 2 * [56]
        range_sizes = [2, 2 * np.pi]
        difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

        curvature_pred = scalar_curvature(
            points,
            cyclic_dimensions=cyclic_dimensions,
            difference_intervals=difference_intervals,
            patch_sizes=patch_sizes,
            device=self.device
        )

        np.testing.assert_almost_equal(
            curvature_gt[3:-3, ...],
            curvature_pred,
            decimal=5,
            err_msg=f"Failed on a hyperboloid with semi axes: {semi_axes}."
        )

    @pytest.mark.slow
    def test_scalar_curvature_3d_hyperboloid(self):
        semi_axes = [5, 1, 3, 1]
        for ambient_dim in [4, 100]:
            # Sample a relatively dense grid.
            n_samples_per_dim = [156, 156, 150]
            cyclic_dimensions = [2]

            points, semi_axes, phis, _ = sample_hyperboloid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=ambient_dim,
                apply_random_isometry=True
            )

            # Compute the ground truth values.
            curvature_gt = hyperboloid_scalar_curvature(semi_axes, phis)

            # Estimate the metric.
            patch_sizes = 3 * [56]
            range_sizes = [2, 2, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            curvature_pred = scalar_curvature(
                points,
                cyclic_dimensions=cyclic_dimensions,
                difference_intervals=difference_intervals,
                patch_sizes=patch_sizes,
                device=self.device
            )

            np.testing.assert_almost_equal(
                curvature_gt[3:-3, 3:-3, ...],
                curvature_pred,
                decimal=2,
                err_msg=f"Failed on a hyperboloid with semi axes: {semi_axes} and ambient dim {ambient_dim}."
            )
