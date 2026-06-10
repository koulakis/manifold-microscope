import numpy as np
import torch

from microscope.patches import extract_patches, stack_patches


def partial_derivative_approximation(
    tensor: torch.Tensor,
    dim: int,
    difference_intervals: float
) -> torch.Tensor:
    """Estimate the partial derivative of a tensor along a dimension and a given sample size, assuming the samples are
    equidistantly placed on a grid.

    Args:
        tensor: An input tensor of shape (s1 ... sk f1 ... fl).
        dim: The index of the dimension along which the partial derivative will be computed.
        difference_intervals: The h value for each dimension of the grid.

    Returns:
        A tensor of dimension (s1 ... si_ ... sk f1 ... fl) with the partial derivative along the i-th dimension.
    """
    device = tensor.device
    length = tensor.shape[dim]

    t_plus = torch.index_select(tensor, dim=dim, index=torch.arange(2, length, device=device))
    t_minus = torch.index_select(tensor, dim=dim, index=torch.arange(0, length - 2, device=device))

    return (t_plus - t_minus) / (2 * difference_intervals)


def crop_dim_borders(tensor: torch.Tensor, dims: set[int], crop: int = 1) -> torch.Tensor:
    """Crop the borders of a given set of dimensions by a given value. This can be used to ensure that tensors which
    have been cropped along different dimension during estimations can be further cropped to have the same shape.

    Args:
        tensor: The tensor to be cropped.
        dims: A set of dimensions to crop the tensor on.
        crop: A integer indicating the amount of cropping at the start and end of a dimension of the tensor.

    Returns:
        The cropped tensor.
    """
    device = tensor.device

    for d in dims:
        length = tensor.shape[d]
        tensor = torch.index_select(tensor, dim=d, index=torch.arange(crop, length - crop, device=device))

    return tensor


def partial_derivatives_across_all_dims(
    tensor: torch.Tensor,
    manifold_dim: int,
    difference_intervals: list[float]
) -> torch.Tensor:
    """Given a tensor of shape (s1 ... sk f1 ... fl) it computes its partial derivatives with respect to
     all its dimensions.

        Args:
            tensor: A tensor of shape (s1 ... sk f1 ... fl), where s_i is the number of points of the i-th dimension of
                its grid and f_i is the number of points on additional feature dimensions.
            manifold_dim: The dimension of the manifold. Any dimensions after those in the tensor are treated as
                features.
            difference_intervals: The h value for each dimension of the grid.

        Returns:
            A tensor of shape (s1_ ... sk_ k) which stacks the partial derivatives for each dimension on the last
            coordinate. The new dimensions s_i_ equal to s_i if the i-th dimension is cyclic, else to s_i - 2 as
            the computations cannot be performed on the borders.
     """
    dim_idxs = set(range(manifold_dim))

    # This tensor had shape (s1 ... sk f1 ... fl k)
    partial_derivatives = torch.stack(
        [
            crop_dim_borders(
                partial_derivative_approximation(
                    tensor,
                    dim,
                    difference_intervals=range_size
                ),
                dim_idxs.difference({dim})
            )
            for dim, range_size in enumerate(difference_intervals)
        ],
        dim=-1
    )

    return partial_derivatives


def riemannian_metric(
        features_on_grid: torch.Tensor,
        difference_intervals: list[float]
) -> torch.Tensor:
    """Given a grid with features on the space of the manifold, it computes an estimate of the Riemannian metric per
    point of the grid, excluding border points.

    Args:
        features_on_grid: A tensor of shape (s1 ... sk f), where s_i is the number of points of the i-th dimension
            of its grid and f the number of features.
        difference_intervals: The h value for each dimension of the grid.

    Returns:
        A tensor of shape (s1_ ... sk_ k k) which has a metric tensor per point. The new dimensions s_i_ equal to
        s_i if the i-th dimension is cyclic, else to s_i - 2 as the computations cannot be performed on the borders.
     """
    partial_derivatives = partial_derivatives_across_all_dims(
        features_on_grid,
        manifold_dim=len(features_on_grid.shape) - 1,
        difference_intervals=difference_intervals,
    )

    return partial_derivatives.transpose(-1, -2) @ partial_derivatives


def partial_derivatives_across_all_dims_batched(
    features_on_grid: np.ndarray,
    difference_intervals: list[float],
    patch_sizes: list[int],
    cyclic_dimensions: list[int],
    device: str = "cuda:0"
):
    dims = len(patch_sizes)
    overlaps = dims * [2]

    patches = extract_patches(
        features_on_grid,
        patch_sizes,
        overlaps,
        cyclic_dimensions=cyclic_dimensions
    )

    derivative_patches = []
    for patch in patches.reshape(-1, *patches.shape[dims:]):
        points_pt_patch = torch.from_numpy(patch).to(device)

        volume_pred = partial_derivatives_across_all_dims(
            points_pt_patch,
            difference_intervals=difference_intervals,
            manifold_dim=dims
        ).cpu().detach().numpy()
        derivative_patches.append(volume_pred)

    derivative_patches = np.stack(derivative_patches, axis=0).reshape(
        *patches.shape[:dims],
        *list(np.array(patches.shape[dims:2 * dims]) - np.array(overlaps)),
        *patches.shape[2 * dims:],
        dims
    )

    derivative_array = stack_patches(derivative_patches, n_features=2)

    # Truncate any padded zeros introduced in the patch extraction.
    expected_array_shape = tuple(
        slice(
            0,
            s if i in cyclic_dimensions else s - 2
        )
        for i, s in enumerate(features_on_grid.shape[:-1])
    )

    return derivative_array[expected_array_shape]

