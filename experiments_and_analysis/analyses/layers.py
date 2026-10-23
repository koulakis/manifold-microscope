import pickle
from pathlib import Path

import numpy as np
import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt

from microscope.computations_grid.data_analysis.data_analysis import (
    compute_measures_multiclass,
    plot_pairs_multiclass,
    plot_3d_pca_projections_multiclass,
    clip_results_multiclass,
    MeasureAggregates
)



def run_single_output(
    name: str,
    features_on_grid: np.ndarray,
    range_sizes: list[float],
    cyclic_dimensions: list[int],
    patch_sizes: list[int],
    class_names: list[str],
    distributions_dir: Path,
    projection_measures_dir: Path,
    projection_classes_dir: Path,
    measures_dir: Path,
    normalize_for_volume: bool,
    dpi: int = 300,
    export_plots: bool = True,
    n_samples_for_plots: int = 50_000,
    device: str | None = None
) -> MeasureAggregates:
    measures = compute_measures_multiclass(
        data=features_on_grid,
        class_names=class_names,
        range_sizes=range_sizes,
        cyclic_dimensions=cyclic_dimensions,
        patch_sizes=patch_sizes,
        reach_subsample=2,
        normalize_for_volume=normalize_for_volume,
        device=device
    )

    with open(measures_dir / f"{name}.pkl", "wb") as f:
        pickle.dump(measures, f)

    measures_clipped = clip_results_multiclass(measures)

    if export_plots:
        features_on_grid_trimmed = features_on_grid[measures_clipped.trim_slices_data]

        plot_pairs_multiclass(
            features_on_grid_trimmed,
            analysis_results=measures_clipped,
            distributions_dir=distributions_dir,
            name=name,
            dpi=dpi,
            n_samples_for_plots=n_samples_for_plots
        )

        plot_3d_pca_projections_multiclass(
            features_on_grid_trimmed,
            measures_clipped,
            projection_measures_dir=projection_measures_dir,
            projection_classes_dir=projection_classes_dir,
            case_name=name,
            show=False,
            dpi=dpi,
            n_samples_for_plots=n_samples_for_plots
        )

    return measures_clipped.measure_aggregates

