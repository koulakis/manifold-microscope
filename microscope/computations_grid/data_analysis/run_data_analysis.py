import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
import typer
import matplotlib

from microscope.datasets.generic_dataset_loader import DatasetName, load_dataset
from microscope.datasets import custom_dsprites
from microscope.datasets import coil20

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from microscope.computations_grid.data_analysis.data_analysis import (
    compute_measures_multiclass,
    plot_pairs_multiclass,
    plot_3d_pca_projections_multiclass,
    clip_results_multiclass,
    MeasureAggregates
)

app = typer.Typer(pretty_exceptions_enable=False)


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
    n_samples_for_plots: int = 50_000
) -> MeasureAggregates:
    measures = compute_measures_multiclass(
        data=features_on_grid,
        class_names=class_names,
        range_sizes=range_sizes,
        cyclic_dimensions=cyclic_dimensions,
        patch_sizes=patch_sizes,
        reach_subsample=2,
        normalize_for_volume=normalize_for_volume
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


@app.command()
def main(
    inference_path: Path = typer.Option(...),
    output_path: Path = typer.Option(...),
    dataset: DatasetName = typer.Option(...),
    model_type: str = typer.Option(...),
    number_of_dims: int = typer.Option(...),
    normalize_for_volume: bool = typer.Option(...),
    dpi: int = 300,
    only_evolution: bool = False,
    scales: list[str] = ("linear", "symlog"),
    n_samples_for_plots: int = 50_000
):
    distributions_dir = output_path / "distributions"
    projection_measures_dir = output_path / "projection_measures"
    projection_classes_dir = output_path / "projection_classes"
    measures_dir = output_path / "measures"
    evolution_dir = output_path / "layer_evolution"
    evolution_subdirs = {scale: evolution_dir / scale for scale in scales}
    for d in [
        distributions_dir,
        projection_measures_dir,
        projection_classes_dir,
        measures_dir,
        *evolution_subdirs.values()
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Load the dataset.
    data, target, range_sizes, cyclic_dimensions, patch_sizes = load_dataset(
        dataset_name=dataset,
        number_of_dims=number_of_dims,
        ratio_per_dim=False,
        training_ratio=1.0,
        noise_sigma=0.0,
        save_train_idx=False,
        weight_subsampling_by_manifold_volume=False,
        return_full_datasets=True,
        return_dataset_unflat_with_metadata=True,
        full_datasets_unclipped=True
    )
    data_shape = data.shape
    if model_type == "beta_vae":
        data = (data - data.min()) / (data.max() - data.min())
    range_sizes, cyclic_dimensions, patch_sizes = list(range_sizes), list(cyclic_dimensions), list(patch_sizes)
    data = data.reshape(*data.shape[:-2], np.prod(data.shape[-2:]))

    # Load some basic inference and dataset information.
    inference_output_paths = list(inference_path.glob("*.npz"))
    aggregate_names = [p.stem for p in inference_output_paths]
    match dataset:
        case (
            DatasetName.dsprites
            | DatasetName.dsprites_single_size
            | DatasetName.custom_dsprites
            | DatasetName.custom_dsprites_single_size
            | DatasetName.custom_dsprites_balanced
        ):
            class_names = list(custom_dsprites.LABEL_MAPPING.values())
        case DatasetName.extended_coil20:
            class_names = list(coil20.LABEL_MAPPING.values())
        case _:
            raise ValueError(f"Undefined label mapping for {dataset}.")

    aggregates = {agg_name: [] for agg_name in aggregate_names}

    # Run over intermediate outputs for different checkpoints.
    pbar = None
    layer_names = None
    for checkpoint_name, path in zip(aggregate_names, inference_output_paths):
        intermediate_outputs = np.load(path)

        if layer_names is None:
            layer_names = list(intermediate_outputs.keys())

        if pbar is None:
            pbar = tqdm(total=len(inference_output_paths) * len(layer_names))

        for layer in layer_names:
            # if (number_of_dims < 4) or (layer in ["input", "output"]) or (model_type != "mae"):
            #     print(layer)
            features_on_grid = (
                intermediate_outputs
                [layer]
                .reshape((*data_shape[:-2], np.prod(intermediate_outputs[layer].shape[1:])))
            )

            intermediate_aggregates = run_single_output(
                features_on_grid=features_on_grid,
                range_sizes=range_sizes,
                cyclic_dimensions=cyclic_dimensions,
                patch_sizes=patch_sizes,
                class_names=class_names,
                distributions_dir=distributions_dir,
                projection_measures_dir=projection_measures_dir,
                projection_classes_dir=projection_classes_dir,
                measures_dir=measures_dir,
                name=f"{checkpoint_name}__{layer}",
                dpi=dpi,
                export_plots=not only_evolution,
                normalize_for_volume=normalize_for_volume,
                n_samples_for_plots=n_samples_for_plots
            )

            aggregates[checkpoint_name].append(intermediate_aggregates)
            pbar.update(1)

            if layer == "output":
                hausdorff_distances = np.linalg.norm(data - features_on_grid, axis=-1)
                np.savez(measures_dir / "hausdorff_distances.npz", hausdorff_distances=hausdorff_distances)

    # Export measures' evolution through layers.
    sprites_combinations = list(list(aggregates.values())[0][0].normalized_total_distances.keys())

    measure_names_fns = {
        "Volume": lambda a: a.total_volumes,
        "Normalized volume": lambda a: a.normalized_total_volumes,
        "Curvature": lambda a: a.normalized_total_curvatures,
        "Curvature negative": lambda a: a.normalized_total_curvatures_negative,
        "Curvature positive": lambda a: a.normalized_total_curvatures_positive,
        "Reach": lambda a: a.normalized_min_reaches,
        "Average reach": lambda a: a.normalized_total_reaches,
        "Total distances": lambda a: a.normalized_total_distances,
        "Average distances": lambda a: a.normalized_average_distances
    }
    cols = len(measure_names_fns)

    for scale in scales:
        for name, aggregate in aggregates.items():
            _, ax = plt.subplots(1, cols, figsize=(5*cols, 4))
            plt.suptitle(name)

            for i, [measure_name, measure_fn] in enumerate(measure_names_fns.items()):
                ax[i].set_title(measure_name)
                for sprite in (sprites_combinations if "distances" in measure_name else class_names):
                    ax[i].plot([measure_fn(a)[sprite] for a in aggregate], label=str(sprite))
                ax[i].legend()
                ax[i].set_xticks(range(len(layer_names)), layer_names, rotation=90)
                ax[i].set_yscale(scale)

            plt.tight_layout()
            plt.savefig(evolution_subdirs[scale] / f"{name}.png", dpi=dpi)


if __name__ == "__main__":
    app()
