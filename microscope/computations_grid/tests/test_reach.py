import unittest
from typing import cast

import numpy as np
import pytest

from microscope.computations_grid.reach import reach_per_point
from microscope.computations_grid.tests._device import get_test_device
from microscope.manifold_examples.sampling_grid import sample_ellipsoid_on_grid, sample_hyperboloid_on_grid


class TestEllipsoids(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_test_device()

    def test_reach_2d_ellipsoids(self):
        semi_axes_cases: list[list[float]] = [
            [1, 2, 0.5],
            [1, 1/2, 5],
            [1, 1.3, 0.86]
        ]
        for semi_axes in semi_axes_cases:
            b = min(semi_axes)
            a = max(semi_axes)
            reach_gt = b**2/a

            # Sample a relatively dense grid.
            fraction_of_angles = 0.7
            points, sampled_semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=[152, 150],
                ambient_space_dim=3,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Estimate the reach.
            cyclic_dimensions = [1]
            range_sizes = [(2 * fraction_of_angles - 1) * np.pi, 2 * np.pi]

            reach_pred = cast(
                np.ndarray,
                reach_per_point(
                    points,
                    cyclic_dimensions=cyclic_dimensions,
                    range_sizes=range_sizes,
                    patch_sizes=len(range_sizes)*[52],
                    device=self.device
                )
            ).min()

            np.testing.assert_almost_equal(
                reach_gt,
                reach_pred,
                decimal=3,
                err_msg=f"Failed on an ellipsoid with semi axes: {sampled_semi_axes}."
            )

    def test_reach_2d_ellipsoids_irregular_grid(self):
        semi_axes_cases: list[list[float]] = [
            [1, 2, 0.5],
            [1, 1 / 2, 5],
            [1, 1.3, 0.86]
        ]
        for semi_axes in semi_axes_cases:
            b = min(semi_axes)
            a = max(semi_axes)
            reach_gt = b ** 2 / a

            # Sample a relatively dense grid.
            fraction_of_angles = 0.8
            points, sampled_semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=[129, 113],
                ambient_space_dim=3,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Estimate the reach.
            cyclic_dimensions = [1]
            range_sizes = [(2 * fraction_of_angles - 1) * np.pi, 2 * np.pi]

            reach_pred = cast(
                np.ndarray,
                reach_per_point(
                    points,
                    cyclic_dimensions=cyclic_dimensions,
                    range_sizes=range_sizes,
                    patch_sizes=len(range_sizes)*[47],
                    batch_size=11,
                    device=self.device
                )
            ).min()

            np.testing.assert_almost_equal(
                reach_gt,
                reach_pred,
                decimal=3,
                err_msg=f"Failed on an ellipsoid with semi axes: {sampled_semi_axes}."
            )

    @pytest.mark.slow
    def test_reach_3d_ellipsoid(self):
        semi_axes: list[float] = [1, 1, 2, 1/2]

        b = min(semi_axes)
        a = max(semi_axes)
        reach_gt = b ** 2 / a

        for ambient_dim in [4, 100]:
            # Sample a relatively dense grid.
            fraction_of_angles = 0.8
            points, sampled_semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=[22, 62, 60],
                ambient_space_dim=ambient_dim,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Estimate the reach.
            cyclic_dimensions = [2]
            range_sizes = [(2*fraction_of_angles - 1) * np.pi, (2*fraction_of_angles - 1) * np.pi, 2 * np.pi]

            reach_pred = cast(
                np.ndarray,
                reach_per_point(
                    points,
                    cyclic_dimensions=cyclic_dimensions,
                    range_sizes=range_sizes,
                    patch_sizes=len(range_sizes) * [62],
                    batch_size=50,
                    device=self.device
                )
            ).min()

            np.testing.assert_almost_equal(
                reach_gt,
                reach_pred,
                decimal=2,
                err_msg=f"Failed on an ellipsoid with semi axes: {sampled_semi_axes} and ambient dim {ambient_dim}."
            )

    def test_reach_4d_sphere(self):
        semi_axes: list[float] = [2, 2, 2, 2, 2]
        reach_gt = 2
        for ambient_dim in [5]:
            # Sample a relatively dense grid.
            fraction_of_angles = 0.8
            points, sampled_semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=3*[8]+[10],
                ambient_space_dim=ambient_dim,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Estimate the reach.
            cyclic_dimensions = [3]
            range_sizes = 3*[(2*fraction_of_angles - 1) * np.pi] + [2 * np.pi]
            reach_pred = cast(
                np.ndarray,
                reach_per_point(
                    points,
                    cyclic_dimensions=cyclic_dimensions,
                    range_sizes=range_sizes,
                    patch_sizes=len(range_sizes) * [12],
                    device=self.device
                )
            ).min()

            np.testing.assert_almost_equal(
                reach_gt,
                reach_pred,
                err_msg=f"Failed on an ellipsoid with semi axes: {sampled_semi_axes} and ambient dim {ambient_dim}."
            )


class TestHyperboloids(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_test_device()

    def test_reach_2d_hyperboloid(self):
        semi_axes_cases: list[list[float]] = [
            [10, 2, 0.5],
            [10, 1 / 2, 5],
            [10, 1.3, 0.86]
        ]
        for semi_axes in semi_axes_cases:
            c = semi_axes[0]
            b, a = sorted(semi_axes[1:])
            reach_gt = min(b**2/a, c**2/a)

            # Sample a relatively dense grid.
            points, sampled_semi_axes, phis, _ = sample_hyperboloid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=[52, 250],
                ambient_space_dim=3,
                apply_random_isometry=True
            )

            # Estimate the reach.
            cyclic_dimensions = [1]
            range_sizes = [2, 2 * np.pi]
            reach_pred = cast(
                np.ndarray,
                reach_per_point(
                    points,
                    cyclic_dimensions=cyclic_dimensions,
                    range_sizes=range_sizes,
                    patch_sizes=len(range_sizes) * [52],
                    batch_size=50,
                    device=self.device
                )
            ).min()

            np.testing.assert_almost_equal(
                reach_gt,
                reach_pred,
                decimal=3,
                err_msg=f"Failed on a hyperboloid with semi axes: {sampled_semi_axes}."
            )

    @pytest.mark.slow
    def test_reach_3d_hyperboloid(self):
        semi_axes: list[float] = [5, 1, 1/2, 1]
        ambient_dim = 4

        c = semi_axes[0]
        b, a = sorted(semi_axes[2:])
        reach_gt = min(b ** 2 / a, c ** 2 / a)

        # Sample a relatively dense grid.
        points, sampled_semi_axes, phis, _ = sample_hyperboloid_on_grid(
            semi_axes=semi_axes,
            n_samples_per_dim=[20, 52, 100],
            ambient_space_dim=ambient_dim,
            apply_random_isometry=True
        )

        # Estimate the reach.
        cyclic_dimensions = [2]
        range_sizes = [2, 2, 2 * np.pi]
        reach_pred = cast(
            np.ndarray,
            reach_per_point(
                points,
                cyclic_dimensions=cyclic_dimensions,
                range_sizes=range_sizes,
                patch_sizes=len(range_sizes) * [52],
                batch_size=50,
                device=self.device
            )
        ).min()

        np.testing.assert_almost_equal(
            reach_gt,
            reach_pred,
            decimal=3,
            err_msg=f"Failed on a hyperboloid with semi axes: {sampled_semi_axes}."
        )
