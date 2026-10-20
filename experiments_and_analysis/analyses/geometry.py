"""Full-grid geometry; numerical implementations remain in microscope."""
from microscope.computations_grid.data_analysis.data_analysis import compute_measures_multiclass


def compute(features, dataset, *, normalize_for_volume=True, normalize_curvatures=False,
            reach_subsample=2, reach_batch_size=2, device='cpu'):
    features = features.reshape(*dataset.grid_shape, -1)
    return compute_measures_multiclass(
        features, dataset.class_names, dataset.range_sizes, dataset.cyclic_dimensions,
        dataset.patch_sizes, normalize_for_volume=normalize_for_volume,
        normalize_curvatures=normalize_curvatures, reach_subsample=reach_subsample,
        reach_batch_size=reach_batch_size, device=device)
