"""Conditioned metric inversion and reduced-QR tangent geometry."""
import torch

from microscope._grid_validation import spacings


def validate_tensor(tensor, intervals, border):
    if tensor.dtype not in (torch.float32, torch.float64):
        raise TypeError("Curvature requires float32 or float64 inputs.")
    dims = tensor.ndim - 1
    if dims < 1 or tensor.shape[-1] < dims:
        raise ValueError("A grid needs at least one coordinate and ambient dimension >= intrinsic dimension.")
    intervals = spacings(intervals, dims)
    if any(n <= 2*border for n in tensor.shape[:-1]):
        raise ValueError(f"Each batch grid axis needs at least {2*border + 1} points.")
    if not torch.isfinite(tensor).all():
        raise ValueError("Curvature inputs must contain only finite values.")
    return intervals


def tangent_geometry(tangent, *, grid_indices=None):
    """Return g, an orthonormal tangent frame and the inverse QR factor.

    QR preserves full-rank tangents whose tiny components disappear in J.T J.
    A small intrinsic-size SVD diagnoses numerical rank before any solve.
    No ambient-by-ambient projector is formed and no regularization is added.
    """
    if not torch.isfinite(tangent).all():
        raise FloatingPointError("Nonfinite finite-difference tangent vectors.")
    frame, factor = torch.linalg.qr(tangent, mode="reduced")
    if not torch.isfinite(factor).all():
        raise FloatingPointError("Tangent QR overflow at the working dtype; rescale the embedding.")
    singular_values = torch.linalg.svdvals(factor)
    if not torch.isfinite(singular_values).all():
        raise FloatingPointError("Tangent singular-value overflow; rescale the embedding.")
    largest, smallest = singular_values[..., 0], singular_values[..., -1]
    tolerance = torch.finfo(tangent.dtype).eps*tangent.shape[-1]*largest
    invalid = smallest <= tolerance
    if invalid.any():
        index = tuple(torch.nonzero(invalid, as_tuple=False)[0].tolist())
        ratio = (smallest[index]/largest[index]).item() if largest[index] > 0 else 0.
        if grid_indices is not None:
            index = tuple(grid_indices[index[0]].tolist())
        raise torch.linalg.LinAlgError(
            f"Tangent rank deficient at working precision at grid index {index}; "
            f"smallest/largest singular value = {ratio:.3g}."
        )
    identity = torch.eye(factor.shape[-1], dtype=factor.dtype, device=factor.device)
    inverse_factor = torch.linalg.solve_triangular(
        factor, identity.expand_as(factor), upper=True
    )
    metric = tangent.transpose(-1, -2) @ tangent
    if not torch.isfinite(metric).all():
        raise FloatingPointError("Metric overflow at the working dtype; rescale the embedding.")
    return metric, frame, inverse_factor


def metric_and_inverse(tangent):
    """Use the small Gram metric directly when its spectrum is well-conditioned.

    Accept lambda_min(g) > sqrt(eps)*lambda_max(g), bounding the estimated
    condition number by 1/sqrt(eps). This is a roundoff safeguard, not an error
    bound for the subsequent curvature calculation. Eigenvalues are used only
    to choose the branch, outside autograd. Only rejected points go through QR;
    this recovers tangent information that a rounded Gram matrix can lose.
    """
    metric = tangent.transpose(-1, -2) @ tangent
    # Any nonfinite tangent component makes its Gram diagonal nonfinite too.
    # Checking the small metric avoids another scan of the full pixel Jacobian.
    if not torch.isfinite(metric).all():
        raise FloatingPointError("Metric overflow at the working dtype; rescale the embedding.")
    eigenvalues = torch.linalg.eigvalsh(metric.detach())
    threshold = torch.finfo(metric.dtype).eps**.5
    good = eigenvalues[..., 0] > threshold*eigenvalues[..., -1]
    if good.all():
        return metric, torch.linalg.inv(metric)
    if not good.any():
        _, _, inverse_factor = tangent_geometry(tangent)
        return metric, inverse_factor @ inverse_factor.transpose(-1, -2)

    # Invert only safe metrics. Inverting the rejected entries and masking
    # afterward would still raise or contaminate gradients at singular points.
    inverse = torch.empty_like(metric)
    inverse[good] = torch.linalg.inv(metric[good])
    _, _, inverse_factor = tangent_geometry(
        tangent[~good], grid_indices=torch.nonzero(~good, as_tuple=False)
    )
    inverse[~good] = inverse_factor @ inverse_factor.transpose(-1, -2)
    return metric, inverse
