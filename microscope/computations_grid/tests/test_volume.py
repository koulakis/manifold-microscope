import unittest

import numpy as np
import torch

from microscope.computations_grid.volume import volume_element, compute_total_volume
from microscope.cyclic_dimensions import get_difference_intervals
from microscope.manifold_examples.ellipsoid import ellipsoid_volume_element
from microscope.manifold_examples.hyperboloid import hyperboloid_volume_element
from microscope.manifold_examples.sampling_grid import sample_ellipsoid_on_grid, sample_hyperboloid_on_grid
from microscope.manifold_examples.sphere import sphere_volume_element
from microscope.manifold_examples.tests.test_ellipsoid import ellipsoid_2d_surface_area


class TestEllipsoids(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0"

    def test_volume_element_2d_ellipsoids(self):
        for semi_axes in [[2, 2, 2], [1, 1.5, 3]]:
            # Sample a relatively dense grid.
            fraction_of_angles = 0.8
            n_samples_per_dim = [3_002, 3_000]
            cyclic_dimensions = [1]

            points, semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=3,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Compute the ground truth values.
            volume_gt = ellipsoid_volume_element(semi_axes, phis)

            # Estimate the volume.
            patch_sizes = [52, 52]
            range_sizes = [(2 * fraction_of_angles - 1) * np.pi, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            volume_pred = volume_element(
                points,
                cyclic_dimensions=cyclic_dimensions,
                difference_intervals=difference_intervals,
                patch_sizes=patch_sizes,
                device=self.device
            )

            np.testing.assert_almost_equal(
                volume_gt[1:-1, ...],
                volume_pred,
                decimal=4,
                err_msg=f"Failed on an ellipsoid with semi axes: {semi_axes}."
            )

    def test_volume_element_2d_ellipsoids_irregular_grid(self):
        for semi_axes in [[2, 2, 2], [1, 1.5, 3]]:
            # Sample a relatively dense grid.
            fraction_of_angles = 0.8
            n_samples_per_dim = [2_993, 2_853]
            cyclic_dimensions = [1]

            points, semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=3,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Compute the ground truth values.
            volume_gt = ellipsoid_volume_element(semi_axes, phis)

            # Estimate the volume.
            patch_sizes = [37, 29]
            range_sizes = [(2 * fraction_of_angles - 1) * np.pi, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            volume_pred = volume_element(
                points,
                cyclic_dimensions=cyclic_dimensions,
                difference_intervals=difference_intervals,
                patch_sizes=patch_sizes,
                device=self.device
            )

            np.testing.assert_almost_equal(
                volume_gt[1:-1, ...],
                volume_pred,
                decimal=4,
                err_msg=f"Failed on an ellipsoid with irregular grid and semi axes: {semi_axes}."
            )

    def test_total_volume_2d_ellipsoid(self):
        for semi_axes in [[2, 2, 2], [1, 1.5, 3]]:
            # Sample a relatively dense grid.
            fraction_of_angles = 0.999
            n_samples_per_dim = [5_993, 5_853]
            cyclic_dimensions = [1]

            points, semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=3,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Ground truth from the area formula.
            total_volume_gt = ellipsoid_2d_surface_area(*semi_axes)

            # Estimate the volume.
            patch_sizes = [1037, 1029]
            range_sizes = [(2 * fraction_of_angles - 1) * np.pi, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            volume_pred = volume_element(
                points,
                cyclic_dimensions=cyclic_dimensions,
                difference_intervals=difference_intervals,
                patch_sizes=patch_sizes,
                device=self.device
            )

            total_volume_pred = compute_total_volume(volume_pred, range_sizes)

            np.testing.assert_almost_equal(
                total_volume_gt,
                total_volume_pred,
                decimal=2,
                err_msg=f"Failed on an ellipsoid with irregular grid and semi axes: {semi_axes}."
            )

    def test_volume_element_3d_ellipsoid(self):
        semi_axes = [1, 1.5, 3, 1.8]
        for ambient_dim in [4, 100]:
            # Sample a relatively dense grid.
            fraction_of_angles = 0.8
            # n_samples_per_dim = [56, 56, 50] if ambient_dim > 100 else [106, 106, 100]
            n_samples_per_dim = [102, 102, 100]
            cyclic_dimensions = [2]

            points, semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=ambient_dim,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Compute the ground truth values.
            volume_gt = ellipsoid_volume_element(semi_axes, phis)

            # Estimate the metric.
            patch_sizes = [12, 12, 12]
            range_sizes = [(2 * fraction_of_angles - 1) * np.pi, (2 * fraction_of_angles - 1) * np.pi, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            volume_pred = volume_element(
                points,
                cyclic_dimensions=cyclic_dimensions,
                difference_intervals=difference_intervals,
                patch_sizes=patch_sizes,
                device=self.device
            )

            np.testing.assert_almost_equal(
                volume_gt[1:-1, 1:-1, ...],
                volume_pred,
                decimal=2,
                err_msg=f"Failed on an ellipsoid with semi axes: {semi_axes} and ambient dim {ambient_dim}."
            )

    def test_volume_element_4d_sphere(self):
        semi_axes = 5*[2]
        for ambient_dim in [5, 10]:
            # Sample a relatively dense grid.
            fraction_of_angles = 0.8
            n_samples_per_dim = 3 * [52] + [50]
            cyclic_dimensions = [3]

            points, semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=ambient_dim,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Compute the ground truth values.
            volume_gt = sphere_volume_element(2., phis)

            # Estimate the metric.
            patch_sizes = 4 * [12]
            range_sizes = 3 * [(2 * fraction_of_angles - 1) * np.pi] + [2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            volume_pred = volume_element(
                points,
                cyclic_dimensions=cyclic_dimensions,
                difference_intervals=difference_intervals,
                patch_sizes=patch_sizes,
                device=self.device
            )

            np.testing.assert_almost_equal(
                volume_gt[1:-1, 1:-1, 1:-1, ...],
                volume_pred,
                decimal=1,
                err_msg=f"Failed on an ellipsoid with semi axes: {semi_axes}."
            )


class TestHyperboloids(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0"

    def test_volume_element_2d_hyperboloid(self):
        semi_axes = [5, 1, 3]

        # Sample a relatively dense grid.
        n_samples_per_dim = [1_002, 1_000]
        cyclic_dimensions = [1]

        points, semi_axes, phis, _ = sample_hyperboloid_on_grid(
            semi_axes=semi_axes,
            n_samples_per_dim=n_samples_per_dim,
            ambient_space_dim=3,
            apply_random_isometry=True
        )

        # Compute the ground truth values.
        volume_gt = hyperboloid_volume_element(semi_axes, phis)

        # Estimate the metric.
        patch_sizes = 2 * [52]
        range_sizes = [2, 2 * np.pi]
        difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

        volume_pred = volume_element(
            points,
            cyclic_dimensions=cyclic_dimensions,
            difference_intervals=difference_intervals,
            patch_sizes=patch_sizes,
            device=self.device
        )

        np.testing.assert_almost_equal(
            volume_gt[1:-1, ...],
            volume_pred,
            decimal=3,
            err_msg=f"Failed on a hyperboloid with semi axes: {semi_axes}."
        )

    def test_volume_element_3d_hyperboloid(self):
        semi_axes = [5, 1, 3, 1]
        for ambient_dim in [4]:
            # Sample a relatively dense grid.
            n_samples_per_dim = [302, 302, 300]
            cyclic_dimensions = [2]

            points, semi_axes, phis, _ = sample_hyperboloid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=ambient_dim,
                apply_random_isometry=True
            )

            # Compute the ground truth values.
            volume_gt = hyperboloid_volume_element(semi_axes, phis)

            # Estimate the metric.
            patch_sizes = 3 * [52]
            range_sizes = [2, 2, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            volume_pred = volume_element(
                points,
                cyclic_dimensions=cyclic_dimensions,
                difference_intervals=difference_intervals,
                patch_sizes=patch_sizes,
                device=self.device
            )

            np.testing.assert_almost_equal(
                volume_gt[1:-1, 1:-1, ...],
                volume_pred,
                decimal=2,
                err_msg=f"Failed on a hyperboloid with semi axes: {semi_axes} and ambient dim {ambient_dim}."
            )
