"""Full-grid geometry; numerical implementations remain in microscope."""
import numpy as np

from microscope.computations_grid.data_analysis.data_analysis import AnalysisResults, compute_measures_multiclass

from ..data.preparation import Dataset


def _validate_finite_measures(results: AnalysisResults) -> None:
    """Reject nonfinite measures while allowing empty sign-subset aggregate NaNs.

    Args:
        results: Computed geometry with per-class and class-pair measure arrays.
    """
    measures = results.measures
    aggregates = results.measure_aggregates
    for name, groups in (
        ('volume_elements', measures.volume_elements),
        ('total_volumes', measures.total_volumes),
        ('normalized_volume_elements', measures.normalized_volume_elements),
        ('normalized_curvatures', measures.normalized_curvatures),
        ('normalized_reaches', measures.normalized_reaches),
        ('normalized_distances', measures.normalized_distances),
        ('normalized_total_volumes', aggregates.normalized_total_volumes),
        ('normalized_total_curvatures', aggregates.normalized_total_curvatures),
        ('normalized_min_reaches', aggregates.normalized_min_reaches),
        ('normalized_total_reaches', aggregates.normalized_total_reaches),
        ('normalized_total_distances', aggregates.normalized_total_distances),
        ('normalized_average_distances', aggregates.normalized_average_distances),
    ):
        for label, values in groups.items():
            if not np.isfinite(values).all():
                raise FloatingPointError(f'Nonfinite geometry measure {name} for {label}.')
    for label, curvatures in measures.normalized_curvatures.items():
        for subset, total in (
            (curvatures >= 0, aggregates.normalized_total_curvatures_positive[label]),
            (curvatures < 0, aggregates.normalized_total_curvatures_negative[label]),
        ):
            if subset.any() and not np.isfinite(total):
                raise FloatingPointError(f'Nonfinite sign-subset curvature aggregate for {label}.')


def compute(
    features: np.ndarray,
    dataset: Dataset,
    normalize_for_volume: bool = True,
    normalize_curvatures: bool = False,
    reach_subsample: int = 2,
    reach_batch_size: int = 2,
    device: str = 'cpu',
    fallback_to_float64_on_nan_measures: bool = False
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
        fallback_to_float64_on_nan_measures: Retry nonfinite float32 geometry once in float64, before normalization.

    Returns:
        AnalysisResults containing per-class measure grids, scalar aggregates and cropping metadata.
    """
    features = features.reshape(*dataset.grid_shape, -1)
    print(f'[geometry] computing volume, curvature and reach on {device}')

    def evaluate(values: np.ndarray) -> AnalysisResults:
        """Compute all classes together at the supplied precision.

        Args:
            values: Unnormalized feature grid of shape (classes, s1, ..., sd, features).
        """
        return compute_measures_multiclass(
            values,
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

    try:
        results = evaluate(features)
        if fallback_to_float64_on_nan_measures:
            _validate_finite_measures(results)
        return results
    except (FloatingPointError, ValueError) as error:
        nonfinite = (isinstance(error, FloatingPointError)
                     or str(error) == 'Curvature inputs must contain only finite values.')
        if not fallback_to_float64_on_nan_measures or features.dtype != np.float32 or not nonfinite:
            raise
        print(f'[geometry] float32 geometry failed: {error} Retrying all classes in float64.', flush=True)
    # Leave the exception handler so its traceback releases failed float32 buffers before the retry.
    results = evaluate(features.astype(np.float64))
    _validate_finite_measures(results)
    return results
