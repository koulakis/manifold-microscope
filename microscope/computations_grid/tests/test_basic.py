import unittest

import numpy as np
import pytest
import sympy as sp
import torch

from microscope.computations_grid.basic import partial_derivatives_across_all_dims, crop_dim_borders, riemannian_metric
from microscope.cyclic_dimensions import get_difference_intervals, pad_cyclic_dimensions
from microscope.manifold_examples.sampling_grid import sample_ellipsoid_on_grid, sample_hyperboloid_on_grid
from microscope.manifold_examples.symbolic_computations import ellipsoid_parametrization, hyperboloid_parametrization


class TestEllipsoids(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0"

    def test_partial_derivatives_across_all_dims_2d_ellipsoids(self):
        for semi_axes in [[2, 2, 2], [1, 1.5, 3]]:
            # Compute the formula of the true tangent vectors.
            u, coordinates = ellipsoid_parametrization(dim=2)
            u = u.subs({f"a_{i+1}": sa for i, sa in enumerate(semi_axes)})

            tangent_vectors_formula = u.jacobian(coordinates)
            tangent_vectors_formula_arr = np.array(tangent_vectors_formula).T

            # Sample a relatively dense grid.
            fraction_of_angles = 0.8
            cyclic_dimensions = [1]
            n_samples_per_dim = [1_000, 1_000]

            points, semi_axes, phis, [rotation, _] = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=3,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Compute the ground truth values.
            shape = tangent_vectors_formula_arr.shape
            tangent_vectors_gt = list(map(
                lambda x:
                    np.ones_like(phis[..., 0])*float(x)
                    if x.is_constant()
                    else sp.lambdify(coordinates, x)(phis[..., 0], phis[..., 1]),
                np.array(tangent_vectors_formula_arr.flatten())
            ))

            tangent_vectors_gt = np.stack(tangent_vectors_gt, axis=-1).reshape((*phis.shape[:-1], *shape))

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(tangent_vectors_gt, axis=-1), axis=-1)[..., None]
            tangent_vectors_gt = np.take_along_axis(tangent_vectors_gt, sort_idx, axis=-2)

            # Estimate the tangent vectors.
            points_padded = pad_cyclic_dimensions(points, cyclic_dimensions=cyclic_dimensions, pad_sizes=[1])
            points_pt = torch.from_numpy(points_padded).to(self.device)
            range_sizes = [(2*fraction_of_angles - 1) * np.pi, 2 * np.pi]

            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)
            tangent_vectors_pred = partial_derivatives_across_all_dims(
                points_pt,
                manifold_dim=len(points.shape) - 1,
                difference_intervals=difference_intervals
            ).cpu().numpy()
            tangent_vectors_pred = np.moveaxis(tangent_vectors_pred, -1, -2)

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(tangent_vectors_pred, axis=-1), axis=-1)[..., None]
            tangent_vectors_pred = np.take_along_axis(tangent_vectors_pred, sort_idx, axis=-2)

            # Apply the random rotation.
            tangent_vectors_gt = tangent_vectors_gt @ rotation

            np.testing.assert_almost_equal(
                tangent_vectors_gt[1:-1, ...],
                tangent_vectors_pred,
                decimal=4,
                err_msg=f"Failed on an ellipsoid with semi axes: {semi_axes}."
            )

    def test_partial_derivatives_across_all_dims_3d_ellipsoid(self):
        semi_axes = [1, 1.5, 3, 1.8]
        for ambient_dim in [4, 100]:
            # Compute the formula of the true tangent vectors.
            u, coordinates = ellipsoid_parametrization(dim=3)
            u = u.subs({f"a_{i+1}": sa for i, sa in enumerate(semi_axes)})

            tangent_vectors_formula = u.jacobian(coordinates)
            tangent_vectors_formula_arr = np.array(tangent_vectors_formula).T

            # Sample a relatively dense grid.
            fraction_of_angles = 0.8
            cyclic_dimensions = [2]
            n_samples_per_dim = [102, 102, 100]

            points, semi_axes, phis, [rotation, _] = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=ambient_dim,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Compute the ground truth values.
            shape = tangent_vectors_formula_arr.shape
            tangent_vectors_gt = list(map(
                lambda x:
                    np.ones_like(phis[..., 0])*float(x)
                    if x.is_constant()
                    else sp.lambdify(coordinates, x)(phis[..., 0], phis[..., 1], phis[..., 2]),
                np.array(tangent_vectors_formula_arr.flatten())
            ))

            tangent_vectors_gt = np.stack(tangent_vectors_gt, axis=-1).reshape((*phis.shape[:-1], *shape))

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(tangent_vectors_gt, axis=-1), axis=-1)[..., None]
            tangent_vectors_gt = np.take_along_axis(tangent_vectors_gt, sort_idx, axis=-2)

            # Estimate the tangent vectors.
            points_padded = pad_cyclic_dimensions(points, cyclic_dimensions=cyclic_dimensions, pad_sizes=[1])
            points_pt = torch.from_numpy(points_padded).to(self.device)
            range_sizes = [(2*fraction_of_angles - 1) * np.pi, (2*fraction_of_angles - 1) * np.pi, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)
            tangent_vectors_pred = partial_derivatives_across_all_dims(
                points_pt,
                manifold_dim=len(points.shape) - 1,
                difference_intervals=difference_intervals
            ).cpu().numpy()
            tangent_vectors_pred = np.moveaxis(tangent_vectors_pred, -1, -2)

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(tangent_vectors_pred, axis=-1), axis=-1)[..., None]
            tangent_vectors_pred = np.take_along_axis(tangent_vectors_pred, sort_idx, axis=-2)

            # Embed to the ambient space and apply the random rotation.
            if ambient_dim > 4:
                tangent_vectors_gt = np.concatenate(
                    [
                        tangent_vectors_gt,
                        np.zeros((*tangent_vectors_gt.shape[:-1], ambient_dim - tangent_vectors_gt.shape[-1]))
                    ],
                    axis=-1
                )
            tangent_vectors_gt = tangent_vectors_gt @ rotation

            diff = (tangent_vectors_gt[1:-1, 1:-1, ...] - tangent_vectors_pred).flatten()

            epsilon_error = 5e-3
            quantile_error = 1e-2
            perc_high_error = (diff > epsilon_error).sum() / len(diff)

            if perc_high_error > quantile_error:
                raise ValueError(
                    f"Failed on an ellipsoid with semi axes: {semi_axes} and ambient dim {ambient_dim}."
                    f"The ground truth and predictions differ by at least {epsilon_error} "
                    f"on {perc_high_error} of the values."
                )

    def test_crop_dim_borders(self):
        tensor = torch.rand((10, 24, 17))
        cropped_tensor = crop_dim_borders(tensor, dims={0, 2}, crop=2)

        torch.testing.assert_close(tensor[2:-2, :, 2:-2], cropped_tensor)

    def test_riemannian_metrics_2d_ellipsoids(self):
        for semi_axes in [[2, 2, 2], [1, 1.5, 3]]:
            # Compute the formula of the true metric.
            u, coordinates = ellipsoid_parametrization(dim=2)
            u = u.subs({f"a_{i+1}": sa for i, sa in enumerate(semi_axes)})

            tangent_vectors_formula = u.jacobian(coordinates)
            metric_formula = sp.simplify(tangent_vectors_formula.T @ tangent_vectors_formula)
            metric_formula_arr = np.array(metric_formula)

            # Sample a relatively dense grid.
            fraction_of_angles = 0.8
            n_samples_per_dim = [1_000, 1_000]
            cyclic_dimensions = [1]

            points, semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=3,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Compute the ground truth values.
            metric_gt = list(map(
                lambda x:
                    np.ones_like(phis[..., 0])*float(x)
                    if x.is_constant()
                    else sp.lambdify(coordinates, x)(phis[..., 0], phis[..., 1]),
                np.array(metric_formula_arr.flatten())
            ))

            metric_gt = np.stack(metric_gt, axis=-1).reshape((*phis.shape[:-1], *metric_formula_arr.shape))

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(metric_gt, axis=-1), axis=-1)[..., None]
            metric_gt = np.take_along_axis(metric_gt, sort_idx, axis=-2)

            # Estimate the metric.
            points_padded = pad_cyclic_dimensions(points, cyclic_dimensions=cyclic_dimensions, pad_sizes=[1])
            points_pt = torch.from_numpy(points_padded).to(self.device)
            range_sizes = [(2*fraction_of_angles - 1) * np.pi, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            metric_pred = riemannian_metric(
                points_pt,
                difference_intervals=difference_intervals
            ).cpu().numpy()

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(metric_pred, axis=-1), axis=-1)[..., None]
            metric_pred = np.take_along_axis(metric_pred, sort_idx, axis=-2)

            np.testing.assert_almost_equal(
                metric_gt[1:-1, ...],
                metric_pred,
                decimal=4,
                err_msg=f"Failed on an ellipsoid with semi axes: {semi_axes}."
            )

    @pytest.mark.slow
    def test_riemannian_metrics_3d_ellipsoid(self):
        semi_axes = [1, 1.5, 3, 1.8]
        for ambient_dim in [4, 100, 4096]:
            # Compute the formula of the true metric.
            u, coordinates = ellipsoid_parametrization(dim=3)
            u = u.subs({f"a_{i+1}": sa for i, sa in enumerate(semi_axes)})

            tangent_vectors_formula = u.jacobian(coordinates)
            metric_formula = sp.simplify(tangent_vectors_formula.T @ tangent_vectors_formula)
            metric_formula_arr = np.array(metric_formula)

            # Sample a relatively dense grid.
            fraction_of_angles = 0.8
            n_samples_per_dim = [100, 100, 100] if ambient_dim <= 100 else [50, 50, 50]
            cyclic_dimensions = [2]

            points, semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=ambient_dim,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Compute the ground truth values.
            metric_gt = list(map(
                lambda x:
                    np.ones_like(phis[..., 0])*float(x)
                    if x.is_constant()
                    else sp.lambdify(coordinates, x)(phis[..., 0], phis[..., 1], phis[..., 2]),
                np.array(metric_formula_arr.flatten())
            ))

            metric_gt = np.stack(metric_gt, axis=-1).reshape((*phis.shape[:-1], *metric_formula_arr.shape))

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(metric_gt, axis=-1), axis=-1)[..., None]
            metric_gt = np.take_along_axis(metric_gt, sort_idx, axis=-2)

            # Estimate the metric.
            points_padded = pad_cyclic_dimensions(points, cyclic_dimensions=cyclic_dimensions, pad_sizes=[1])
            points_pt = torch.from_numpy(points_padded).to(self.device if ambient_dim <= 100 else "cpu")
            range_sizes = [(2*fraction_of_angles - 1) * np.pi, (2*fraction_of_angles - 1) * np.pi, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            metric_pred = riemannian_metric(
                points_pt,
                difference_intervals=difference_intervals
            ).cpu().numpy()

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(metric_pred, axis=-1), axis=-1)[..., None]
            metric_pred = np.take_along_axis(metric_pred, sort_idx, axis=-2)

            diff = (metric_gt[1:-1, 1:-1, ...] - metric_pred).flatten()

            epsilon_error = 8e-3 if ambient_dim <= 100 else 5e-2
            quantile_error = 2e-2
            perc_high_error = (diff > epsilon_error).sum() / len(diff)

            if perc_high_error > quantile_error:
                raise ValueError(
                    f"Failed on an ellipsoid with semi axes: {semi_axes} and ambient dim {ambient_dim}."
                    f"The ground truth and predictions differ by at least {epsilon_error} "
                    f"on {perc_high_error} of the values."
                )

    def test_riemannian_metrics_4d_sphere(self):
        semi_axes = 5*[2]
        for ambient_dim in [5, 10]:
            # Compute the formula of the true metric.
            u, coordinates = ellipsoid_parametrization(dim=4)
            u = u.subs({f"a_{i+1}": sa for i, sa in enumerate(semi_axes)})

            tangent_vectors_formula = u.jacobian(coordinates)
            metric_formula = sp.simplify(tangent_vectors_formula.T @ tangent_vectors_formula)
            metric_formula_arr = np.array(metric_formula)

            # Sample a relatively dense grid.
            fraction_of_angles = 0.8
            n_samples_per_dim = 4*[50]
            cyclic_dimensions = [3]

            points, semi_axes, phis, _ = sample_ellipsoid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=4*[50],
                ambient_space_dim=ambient_dim,
                apply_random_isometry=True,
                fraction_of_angles=fraction_of_angles
            )

            # Compute the ground truth values.
            metric_gt = list(map(
                lambda x:
                    np.ones_like(phis[..., 0])*float(x)
                    if x.is_constant()
                    else sp.lambdify(coordinates, x)(phis[..., 0], phis[..., 1], phis[..., 2], phis[..., 3]),
                np.array(metric_formula_arr.flatten())
            ))

            metric_gt = np.stack(metric_gt, axis=-1).reshape((*phis.shape[:-1], *metric_formula_arr.shape))

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(metric_gt, axis=-1), axis=-1)[..., None]
            metric_gt = np.take_along_axis(metric_gt, sort_idx, axis=-2)

            # Estimate the metric.
            points_padded = pad_cyclic_dimensions(points, cyclic_dimensions=cyclic_dimensions, pad_sizes=[1])
            points_pt = torch.from_numpy(points_padded).to(self.device if ambient_dim <= 100 else "cpu")
            range_sizes = 3*[(2*fraction_of_angles - 1) * np.pi] + [2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            metric_pred = riemannian_metric(
                points_pt,
                difference_intervals=difference_intervals
            ).cpu().numpy()

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(metric_pred, axis=-1), axis=-1)[..., None]
            metric_pred = np.take_along_axis(metric_pred, sort_idx, axis=-2)

            diff = (metric_gt[1:-1, 1:-1, 1:-1, ...] - metric_pred).flatten()

            epsilon_error = 1e-2
            quantile_error = 3e-2
            perc_high_error = (diff > epsilon_error).sum() / len(diff)

            if perc_high_error > quantile_error:
                raise ValueError(
                    f"Failed on an ellipsoid with semi axes: {semi_axes} and ambient dim {ambient_dim}."
                    f"The ground truth and predictions differ by at least {epsilon_error} "
                    f"on {perc_high_error} of the values."
                )


class TestHyperboloids(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0"

    def test_partial_derivatives_across_all_dims_2d_hyperboloid(self):
        for semi_axes in [[5, 1, 3]]:
            # Compute the formula of the true tangent vectors.
            u, coordinates = hyperboloid_parametrization(dim=2)
            u = u.subs({f"a_{i+1}": sa for i, sa in enumerate(semi_axes)})

            tangent_vectors_formula = u.jacobian(coordinates)
            tangent_vectors_formula_arr = np.array(tangent_vectors_formula).T

            # Sample a relatively dense grid.
            n_samples_per_dim = [1_000, 1_000]
            cyclic_dimensions = [1]

            points, semi_axes, phis, [rotation, _] = sample_hyperboloid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=3,
                apply_random_isometry=True
            )

            # Compute the ground truth values.
            shape = tangent_vectors_formula_arr.shape
            tangent_vectors_gt = list(map(
                lambda x:
                    np.ones_like(phis[..., 0])*float(x)
                    if x.is_constant()
                    else sp.lambdify(coordinates, x)(phis[..., 0], phis[..., 1]),
                np.array(tangent_vectors_formula_arr.flatten())
            ))

            tangent_vectors_gt = np.stack(tangent_vectors_gt, axis=-1).reshape((*phis.shape[:-1], *shape))

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(tangent_vectors_gt, axis=-1), axis=-1)[..., None]
            tangent_vectors_gt = np.take_along_axis(tangent_vectors_gt, sort_idx, axis=-2)

            # Estimate the tangent vectors.
            points_padded = pad_cyclic_dimensions(points, cyclic_dimensions=cyclic_dimensions, pad_sizes=[1])
            points_pt = torch.from_numpy(points_padded).to(self.device)
            range_sizes = [2, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            tangent_vectors_pred = partial_derivatives_across_all_dims(
                points_pt,
                manifold_dim=len(points.shape) - 1,
                difference_intervals=difference_intervals
            ).cpu().numpy()
            tangent_vectors_pred = np.moveaxis(tangent_vectors_pred, -1, -2)

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(tangent_vectors_pred, axis=-1), axis=-1)[..., None]
            tangent_vectors_pred = np.take_along_axis(tangent_vectors_pred, sort_idx, axis=-2)

            # Apply the random rotation.
            tangent_vectors_gt = tangent_vectors_gt @ rotation

            np.testing.assert_almost_equal(
                tangent_vectors_gt[1:-1, ...],
                tangent_vectors_pred,
                decimal=4,
                err_msg=f"Failed on a hyperboloid with semi axes: {semi_axes}."
            )

    def test_partial_derivatives_across_all_dims_3d_hyperboloid(self):
        semi_axes = [5, 1, 3, 1]
        for ambient_dim in [4, 100]:
            # Compute the formula of the true tangent vectors.
            u, coordinates = hyperboloid_parametrization(dim=3)
            u = u.subs({f"a_{i+1}": sa for i, sa in enumerate(semi_axes)})

            tangent_vectors_formula = u.jacobian(coordinates)
            tangent_vectors_formula_arr = np.array(tangent_vectors_formula).T

            # Sample a relatively dense grid.
            n_samples_per_dim = [100, 100, 100]
            cyclic_dimensions = [2]

            points, semi_axes, phis, [rotation, _] = sample_hyperboloid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=ambient_dim,
                apply_random_isometry=True
            )

            # Compute the ground truth values.
            shape = tangent_vectors_formula_arr.shape
            tangent_vectors_gt = list(map(
                lambda x:
                    np.ones_like(phis[..., 0])*float(x)
                    if x.is_constant()
                    else sp.lambdify(coordinates, x)(phis[..., 0], phis[..., 1], phis[..., 2]),
                np.array(tangent_vectors_formula_arr.flatten())
            ))

            tangent_vectors_gt = np.stack(tangent_vectors_gt, axis=-1).reshape((*phis.shape[:-1], *shape))

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(tangent_vectors_gt, axis=-1), axis=-1)[..., None]
            tangent_vectors_gt = np.take_along_axis(tangent_vectors_gt, sort_idx, axis=-2)

            # Estimate the tangent vectors.
            points_padded = pad_cyclic_dimensions(points, cyclic_dimensions=cyclic_dimensions, pad_sizes=[1])
            points_pt = torch.from_numpy(points_padded).to(self.device)
            range_sizes = [2, 2, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            tangent_vectors_pred = partial_derivatives_across_all_dims(
                points_pt,
                manifold_dim=len(points.shape) - 1,
                difference_intervals=difference_intervals
            ).cpu().numpy()
            tangent_vectors_pred = np.moveaxis(tangent_vectors_pred, -1, -2)

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(tangent_vectors_pred, axis=-1), axis=-1)[..., None]
            tangent_vectors_pred = np.take_along_axis(tangent_vectors_pred, sort_idx, axis=-2)

            # Embed to the ambient space and apply the random rotation.
            if ambient_dim > 4:
                tangent_vectors_gt = np.concatenate(
                    [
                        tangent_vectors_gt,
                        np.zeros((*tangent_vectors_gt.shape[:-1], ambient_dim - tangent_vectors_gt.shape[-1]))
                    ],
                    axis=-1
                )
            tangent_vectors_gt = tangent_vectors_gt @ rotation

            diff = (tangent_vectors_gt[1:-1, 1:-1, ...] - tangent_vectors_pred).flatten()

            epsilon_error = 5e-3
            quantile_error = 1e-2
            perc_high_error = (diff > epsilon_error).sum() / len(diff)

            if perc_high_error > quantile_error:
                raise ValueError(
                    f"Failed on a hyperboloid with semi axes: {semi_axes} and ambient dim {ambient_dim}."
                    f"The ground truth and predictions differ by at least {epsilon_error} "
                    f"on {perc_high_error} of the values."
                )

    def test_riemannian_metrics_2d_hyperboloids(self):
        for semi_axes in [[5, 1, 3]]:
            # Compute the formula of the true metric.
            u, coordinates = hyperboloid_parametrization(dim=2)
            u = u.subs({f"a_{i+1}": sa for i, sa in enumerate(semi_axes)})

            tangent_vectors_formula = u.jacobian(coordinates)
            metric_formula = sp.simplify(tangent_vectors_formula.T @ tangent_vectors_formula)
            metric_formula_arr = np.array(metric_formula)

            # Sample a relatively dense grid.
            n_samples_per_dim = [1_000, 1_000]
            cyclic_dimensions = [1]

            points, semi_axes, phis, _ = sample_hyperboloid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=3,
                apply_random_isometry=True
            )

            # Compute the ground truth values.
            metric_gt = list(map(
                lambda x:
                    np.ones_like(phis[..., 0])*float(x)
                    if x.is_constant()
                    else sp.lambdify(coordinates, x)(phis[..., 0], phis[..., 1]),
                np.array(metric_formula_arr.flatten())
            ))

            metric_gt = np.stack(metric_gt, axis=-1).reshape((*phis.shape[:-1], *metric_formula_arr.shape))

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(metric_gt, axis=-1), axis=-1)[..., None]
            metric_gt = np.take_along_axis(metric_gt, sort_idx, axis=-2)

            # Estimate the metric.
            points_padded = pad_cyclic_dimensions(points, cyclic_dimensions=cyclic_dimensions, pad_sizes=[1])
            points_pt = torch.from_numpy(points_padded).to(self.device)
            range_sizes = [2, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            metric_pred = riemannian_metric(
                points_pt,
                difference_intervals=difference_intervals
            ).cpu().numpy()

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(metric_pred, axis=-1), axis=-1)[..., None]
            metric_pred = np.take_along_axis(metric_pred, sort_idx, axis=-2)

            np.testing.assert_almost_equal(
                metric_gt[1:-1, ...],
                metric_pred,
                decimal=3,
                err_msg=f"Failed on a hyperboloid with semi axes: {semi_axes}."
            )

    @pytest.mark.slow
    def test_riemannian_metrics_3d_hyperboloid(self):
        semi_axes = [5, 1, 3, 1]
        for ambient_dim in [4, 100, 4096]:
            # Compute the formula of the true metric.
            u, coordinates = hyperboloid_parametrization(dim=3)
            u = u.subs({f"a_{i+1}": sa for i, sa in enumerate(semi_axes)})

            tangent_vectors_formula = u.jacobian(coordinates)
            metric_formula = sp.simplify(tangent_vectors_formula.T @ tangent_vectors_formula)
            metric_formula_arr = np.array(metric_formula)

            # Sample a relatively dense grid.
            n_samples_per_dim = [100, 100, 100] if ambient_dim <= 100 else [50, 50, 50]
            cyclic_dimensions = [2]

            points, semi_axes, phis, _ = sample_hyperboloid_on_grid(
                semi_axes=semi_axes,
                n_samples_per_dim=n_samples_per_dim,
                ambient_space_dim=ambient_dim,
                apply_random_isometry=True
            )

            # Compute the ground truth values.
            metric_gt = list(map(
                lambda x:
                    np.ones_like(phis[..., 0])*float(x)
                    if x.is_constant()
                    else sp.lambdify(coordinates, x)(phis[..., 0], phis[..., 1], phis[..., 2]),
                np.array(metric_formula_arr.flatten())
            ))

            metric_gt = np.stack(metric_gt, axis=-1).reshape((*phis.shape[:-1], *metric_formula_arr.shape))

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(metric_gt, axis=-1), axis=-1)[..., None]
            metric_gt = np.take_along_axis(metric_gt, sort_idx, axis=-2)

            # Estimate the metric.
            points_padded = pad_cyclic_dimensions(points, cyclic_dimensions=cyclic_dimensions, pad_sizes=[1])
            points_pt = torch.from_numpy(points_padded).to(self.device if ambient_dim <= 100 else "cpu")
            range_sizes = [2, 2, 2 * np.pi]
            difference_intervals = get_difference_intervals(n_samples_per_dim, range_sizes, cyclic_dimensions)

            metric_pred = riemannian_metric(
                points_pt,
                difference_intervals=difference_intervals
            ).cpu().numpy()

            # Sort the vectors by their norm to handle a potential permutation of them.
            sort_idx = np.argsort(np.linalg.norm(metric_pred, axis=-1), axis=-1)[..., None]
            metric_pred = np.take_along_axis(metric_pred, sort_idx, axis=-2)

            diff = (metric_gt[1:-1, 1:-1, ...] - metric_pred).flatten()

            epsilon_error = 1e-2 if ambient_dim <= 100 else 5e-2
            quantile_error = 7e-2
            perc_high_error = (diff > epsilon_error).sum() / len(diff)

            if perc_high_error > quantile_error:
                raise ValueError(
                    f"Failed on a hyperboloid with semi axes: {semi_axes} and ambient dim {ambient_dim}."
                    f"The ground truth and predictions differ by at least {epsilon_error} "
                    f"on {perc_high_error} of the values."
                )
