import numpy as np
import torch

from microscope.patches import extract_patches, stack_patches


def partial_derivative_approximation(
    tensor: torch.Tensor,
    dim: int,
    difference_intervals: float
) -> torch.Tensor:
    """Estimate a central finite difference along one equally spaced grid axis.

    Args:
        tensor: Input tensor of shape (s0, ..., s_{n-1}).
        dim: Zero-based axis to differentiate.
        difference_intervals: Scalar coordinate spacing along that axis.

    Returns:
        A tensor of shape (s0, ..., s_dim - 2, ..., s_{n-1}) on the input
        device. Only axis dim is shortened; all other axes are preserved.
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
    """Differentiate along the first manifold_dim axes of an equally spaced grid.

    Args:
        tensor: Shape (s1, ..., sk, f1, ..., fl), with k = manifold_dim.
        manifold_dim: Number of parameter axes; remaining axes are features.
        difference_intervals: One coordinate spacing per parameter axis.

    Returns:
        A tensor of shape (s1 - 2, ..., sk - 2, f1, ..., fl, k) on the
        input device. All parameter axes are cropped; pad periodic axes first."""
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
    """Estimate the induced metric J.T @ J using central finite differences.

    Args:
        features_on_grid: Tensor of shape (s1, ..., sk, features).
        difference_intervals: One coordinate spacing per grid axis.

    Returns:
        A tensor of shape (s1 - 2, ..., sk - 2, k, k) on the input device.
        All grid axes are cropped; pad periodic axes first."""
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
    device: str | torch.device | None = None
):
    """Compute a NumPy grid's Jacobian in patches.

    Args:
        features_on_grid: Floating array of shape (s1, ..., sk, features).
        difference_intervals: One coordinate spacing per grid axis.
        patch_sizes: One patch size per grid axis, each greater than two.
        cyclic_dimensions: Zero-based periodic grid-axis indices.
        device: Torch device; None selects CUDA when available, otherwise CPU.

    Returns:
        A NumPy array of shape (s1_out, ..., sk_out, features, k), where
        s_i_out = s_i for periodic axes and s_i - 2 otherwise. The last axis
        indexes parameter derivatives. Results are detached from autograd.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

