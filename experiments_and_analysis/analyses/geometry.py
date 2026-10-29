"""Full-grid geometry; numerical implementations remain in microscope."""

from __future__ import annotations

from typing import TYPE_CHECKING

from microscope.computations_grid.data_analysis.data_analysis import compute_measures_multiclass

if TYPE_CHECKING:
    import numpy as np
    from ..data.preparation import Dataset
    from microscope.computations_grid.data_analysis.data_analysis import AnalysisResults


def compute(
    features: np.ndarray,
    dataset: Dataset,
    normalize_for_volume: bool = True,
    normalize_curvatures: bool = False,
    reach_subsample: int = 2,
    reach_batch_size: int = 2,
    device: str = 'cpu'
) -> AnalysisResults:
    """Compute geometry on the dataset's full structured grid.

    Args:
        features: Representations of shape (n ...), reshaped to the dataset grid.
        dataset: Loaded dataset and its grid/split metadata.
        normalize_for_volume: Rescale features to unit total volume before computing geometry.
        normalize_curvatures: Normalize derivative directions in curvature computations.
        reach_subsample: Stride used when selecting reach candidates.
        reach_batch_size: Number of reach query points per batch.
        device: Device used for tensor computations.

    Returns:
        AnalysisResults containing per-class measure grids, scalar aggregates and cropping metadata.
    """
    features = features.reshape(*dataset.grid_shape, -1)
    print(f'[geometry] computing volume, curvature and reach on {device}')
    return compute_measures_multiclass(
        features,
        dataset.class_names,
        dataset.range_sizes,
        dataset.cyclic_dimensions,
        dataset.patch_sizes,
        normalize_for_volume=normalize_for_volume,
        normalize_curvatures=normalize_curvatures,
        reach_subsample=reach_subsample,
        reach_batch_size=reach_batch_size,
        device=device
    )
