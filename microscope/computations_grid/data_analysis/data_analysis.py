from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from microscope.computations_grid.curvature import scalar_curvature, compute_total_curvature
from microscope.computations_grid.reach import reach_per_point
from microscope.computations_grid.volume import volume_element, compute_total_volume
from microscope.cyclic_dimensions import get_difference_intervals


@dataclass
class Measures:
    volume_elements: dict[str, np.ndarray]
    total_volumes: dict[str, float]
    normalized_volume_elements: dict[str, np.ndarray]
    normalized_curvatures: dict[str, np.ndarray]
    normalized_reaches: dict[str, np.ndarray]
    normalized_distances: dict[tuple[str, str], np.ndarray]


@dataclass
class MeasureAggregates:
    total_volumes: dict[str, float]
    normalized_total_volumes: dict[str, float]
    normalized_total_curvatures: dict[str, float]
    normalized_total_curvatures_positive: dict[str, float]
    normalized_total_curvatures_negative: dict[str, float]
    normalized_min_reaches: dict[str, float]
    normalized_total_reaches: dict[str, float]
    normalized_total_distances: dict[tuple[str, str], float]
    normalized_average_distances: dict[tuple[str, str], float]


@dataclass
class AnalysisResults:
    measures: Measures
    measure_aggregates: MeasureAggregates
    range_sizes: list[float]
    trim_slices_data: tuple[slice, ...]


def compute_measure_aggregates_multiclass(measures: Measures, range_sizes: list[float]):
    class_names = list(measures.normalized_volume_elements.keys())
    grid_volume = np.prod(range_sizes)

    normalized_volume_elements = measures.normalized_volume_elements
    normalized_curvatures = measures.normalized_curvatures
    normalized_reaches = measures.normalized_reaches
    normalized_distances = measures.normalized_distances

    # Compute aggregates of the measures.
    normalized_total_volumes = {
        name: compute_total_volume(vols, range_sizes)
        for name, vols in normalized_volume_elements.items()
    }
    normalized_total_curvatures = {
        name: compute_total_curvature(
            curvature=normalized_curvatures[name],
            element=normalized_volume_elements[name],
            range_sizes=range_sizes
        )
        for name in class_names
    }
    normalized_total_curvatures_positive = {
        name: (
                compute_total_curvature(
                    curvature=normalized_curvatures[name].flatten()[normalized_curvatures[name].flatten() >= 0],
                    element=normalized_volume_elements[name].flatten()[normalized_curvatures[name].flatten() >= 0],
                    range_sizes=range_sizes
                )
        )
        for name in class_names
    }
    normalized_total_curvatures_negative = {
        name: (
                compute_total_curvature(
                    curvature=normalized_curvatures[name].flatten()[normalized_curvatures[name].flatten() < 0],
                    element=normalized_volume_elements[name].flatten()[normalized_curvatures[name].flatten() < 0],
                    range_sizes=range_sizes
                )
        )
        for name in class_names
    }
    normalized_min_reaches = {
        name: normalized_reaches[name].min()
        for name in class_names
    }
    normalized_total_reaches = {
        name: (normalized_reaches[name] * normalized_volume_elements[name]).mean() * grid_volume
        for name in class_names
    }
    normalized_total_distances = {
        (name1, name2):
            (
                normalized_distances[(name1, name2)]
                * (normalized_volume_elements[name1] + normalized_volume_elements[name2]) / 2
            ).mean()
            * grid_volume
        for name1, name2 in normalized_distances.keys()
    }
    normalized_average_distances = {
        (name1, name2):
            (
                normalized_distances[(name1, name2)]
                * (normalized_volume_elements[name1] + normalized_volume_elements[name2]) / 2
            ).mean()
        for name1, name2 in normalized_distances.keys()
    }

    return MeasureAggregates(
        total_volumes=measures.total_volumes,
        normalized_total_volumes=normalized_total_volumes,
        normalized_total_curvatures=normalized_total_curvatures,
        normalized_total_curvatures_positive=normalized_total_curvatures_positive,
        normalized_total_curvatures_negative=normalized_total_curvatures_negative,
        normalized_min_reaches=normalized_min_reaches,
        normalized_total_reaches=normalized_total_reaches,
        normalized_total_distances=normalized_total_distances,
        normalized_average_distances=normalized_average_distances
    )


def compute_measures_multiclass(
    data: np.ndarray,
    class_names: list[str],
    range_sizes: list[float],
    cyclic_dimensions: list[int],
    patch_sizes: list[int],
    normalize_for_volume: bool,
    normalize_curvatures: bool = False,
    reach_subsample: Optional[int] = None,
    reach_batch_size: int = 2,
    device: str = "cuda:0"
) -> AnalysisResults:
    """Given a multi-class dataset in grid format, compute its volume, curvature and reach per point and generate
    plots for the analysis of the manifold. Note that the data is rescaled to have a unitary total volume.

    Args:
        data: A data array of shape (n_classes d1 ... dk f).
        class_names: A list of the class names.
        range_sizes: The sizes of the value range of each dimension of the grid.
        patch_sizes: The sizes of the patches used along each dimension of the grid.
        cyclic_dimensions: A set of dimensions where the grid is cyclic.
        normalize_for_volume: If true, then the data manifold is scaled to have total volume 1 before computing the
            rest of the measures.
        normalize_curvatures: If true, it computes a normalized version of the scalar curvature, like in the definition
            in Do Carmo.
        reach_subsample: If set to some integer, then the local reach will be computed only on every
            n-th point. The tangent spaces will be approximated though with all points.
        reach_batch_size: The size of batches on which the local reach is computed.
        device: The torch device.

    Returns:
        A dictionary with all the computed measures.
    """
    # Check the number of classes agrees with the data array.
    if len(class_names) != data.shape[0]:
        raise ValueError(
            f"The number of class names ({len(class_names)} is different than the "
            f"first data dimension ({data.shape[0]})."
        )
    dim_shapes = list(data.shape[1:-1])
    n_dims = len(dim_shapes)

    difference_intervals = get_difference_intervals(dim_shapes, range_sizes, cyclic_dimensions)

    # First compute the volume element to normalize.
    volume_elements = {}
    for i, name in enumerate(class_names):
        volume_elements[name] = volume_element(
            features_on_grid=data[i],
            difference_intervals=difference_intervals,
            cyclic_dimensions=cyclic_dimensions,
            patch_sizes=patch_sizes,
            device=device
        )

    total_volumes = {
        name: compute_total_volume(volume_elements[name], range_sizes)
        for name in class_names
    }
    total_volume = sum(total_volumes.values())

    if normalize_for_volume:
        normalized_data = data / total_volume**(1 / n_dims)
    else:
        normalized_data = data

    # Compute all measures with the normalized data.
    normalized_volume_elements = {}
    normalized_curvatures = {}
    normalized_reaches: dict[str, np.ndarray] = {}
    for i, name in enumerate(class_names):
        i: int
        name: str

        print(f"Computing measures for {name}.")
        normalized_curvatures[name] = scalar_curvature(
            features_on_grid=normalized_data[i],
            difference_intervals=difference_intervals,
            cyclic_dimensions=cyclic_dimensions,
            patch_sizes=patch_sizes,
            normalize=normalize_curvatures,
            device=device
        )

        normalized_volume_elements[name] = volume_element(
            features_on_grid=normalized_data[i],
            difference_intervals=difference_intervals,
            cyclic_dimensions=cyclic_dimensions,
            patch_sizes=patch_sizes,
            device=device
        )
        # Create slices to trim the volume to the area of the curvature which shrinks the most.
        trim_slices_volume = tuple(
            slice((vol_s - curv_s) // 2, vol_s - (vol_s - curv_s) // 2)
            for curv_s, vol_s
            in zip(normalized_curvatures[name].shape, normalized_volume_elements[name].shape)
        )
        normalized_volume_elements[name] = normalized_volume_elements[name][trim_slices_volume]

        normalized_reaches[name] = reach_per_point(
            features_on_grid=normalized_data[i],
            range_sizes=range_sizes,
            patch_sizes=patch_sizes,
            cyclic_dimensions=cyclic_dimensions,
            subsample_points=reach_subsample,
            batch_size=reach_batch_size,
            return_witnesses=False,
            device=device
        )
        # Create slices to trim the reach to the area of the curvature which shrinks the most.
        trim_slices_reach = tuple(
            slice((reach_s - curv_s) // 2, reach_s - (reach_s - curv_s) // 2)
            for curv_s, reach_s
            in zip(normalized_curvatures[name].shape, normalized_reaches[name].shape)
        )
        normalized_reaches[name] = normalized_reaches[name][trim_slices_reach]

    # Create slices to trim the data to the area of the curvature which shrinks the most.
    trim_slices_data = (slice(normalized_data.shape[0]),) + tuple(
        slice((dat_s - curv_s) // 2, dat_s - (dat_s - curv_s) // 2)
        for curv_s, dat_s
        in zip(list(normalized_curvatures.values())[0].shape, normalized_data.shape[1:-1])
    )
    normalized_data_trimmed = normalized_data[trim_slices_data]
    normalized_distances: dict[tuple[str, str], np.ndarray] = {
        (class_names[i], class_names[j]): np.linalg.norm(
            normalized_data_trimmed[i] - normalized_data_trimmed[j],
            axis=-1
        )
        for i in range(len(class_names))
        for j in range(i + 1, len(class_names))
    }

    measures = Measures(
        volume_elements=volume_elements,
        total_volumes=total_volumes,
        normalized_volume_elements=normalized_volume_elements,
        normalized_curvatures=normalized_curvatures,
        normalized_reaches=normalized_reaches,
        normalized_distances=normalized_distances
    )

    measure_aggregates = compute_measure_aggregates_multiclass(measures, range_sizes)

    return AnalysisResults(
        measures=measures,
        measure_aggregates=measure_aggregates,
        range_sizes=range_sizes,
        trim_slices_data=trim_slices_data
    )


def clip_measure_multiclass(
        measure: dict[Union[str, tuple[str, str]], np.ndarray],
        quantile_margin: float = 0.05
) -> dict[Union[str, tuple[str, str]], np.ndarray]:
    clipped_measure = {}

    for k, v in measure.items():
        q_left, q_right = np.quantile(v, [quantile_margin, 1 - quantile_margin])
        clipped_measure[k] = np.clip(v, q_left, q_right)

    return clipped_measure


def clip_results_multiclass(results: AnalysisResults, quantile_margin: float = 0.01) -> AnalysisResults:
    clipped_measures = Measures(
        volume_elements=clip_measure_multiclass(results.measures.volume_elements, quantile_margin),
        total_volumes=results.measures.total_volumes,
        normalized_volume_elements=clip_measure_multiclass(
            results.measures.normalized_volume_elements, quantile_margin),
        normalized_curvatures=clip_measure_multiclass(results.measures.normalized_curvatures, quantile_margin),
        normalized_reaches=clip_measure_multiclass(results.measures.normalized_reaches, quantile_margin),
        normalized_distances=clip_measure_multiclass(results.measures.normalized_distances, quantile_margin),
    )

    clipped_measure_aggregates = compute_measure_aggregates_multiclass(clipped_measures, results.range_sizes)

    return AnalysisResults(
        measures=clipped_measures,
        measure_aggregates=clipped_measure_aggregates,
        range_sizes=results.range_sizes,
        trim_slices_data=results.trim_slices_data
    )


def _plot_measure_pairs(
    title: str,
    volumes: np.ndarray,
    curvatures: np.ndarray,
    reaches: np.ndarray,
    classes=None,
    height: float = 2.5,
    alpha=0.5
):
    if classes is None:
        df = pd.DataFrame(dict(
            volume=volumes,
            curvature=curvatures,
            reach=reaches
        ))
    else:
        df = pd.DataFrame(dict(
            volume=volumes,
            curvature=curvatures,
            reach=reaches,
            classes=classes
        ))

    g = sns.pairplot(
        df,
        kind="kde",
        diag_kind="kde",
        plot_kws=dict(fill=True, alpha=alpha),
        height=height,
        corner=True,
        hue=None if classes is None else "classes"
    )
    g.fig.suptitle(title, y=1.02)


def plot_pairs_multiclass(
    data: np.ndarray,
    analysis_results: AnalysisResults,
    distributions_dir: Optional[Path] = None,
    name: Optional[str] = None,
    plot_per_class: bool = False,
    height: float = 2.5,
    alpha: float = 0.5,
    dpi: int = 300,
    n_samples_for_plots: int = 50_000
):
    class_names = list(analysis_results.measures.normalized_volume_elements.keys())
    if len(class_names) != data.shape[0]:
        raise ValueError(
            f"The number of class names ({len(class_names)} is different than the "
            f"first data dimension ({data.shape[0]})."
        )

    # Pair plot of measures per class and together.
    all_volumes = np.concatenate(
        [analysis_results.measures.normalized_volume_elements[name] for name in class_names],
        axis=0
    )
    all_curvatures = np.concatenate(
        [analysis_results.measures.normalized_curvatures[name] for name in class_names],
        axis=0
    )
    all_reaches = np.concatenate(
        [analysis_results.measures.normalized_reaches[name] for name in class_names],
        axis=0
    )
    classes = np.concatenate([
        np.array(np.prod(analysis_results.measures.normalized_reaches[name].shape)*[name])
        for name in class_names
    ])
    n_elements = len(all_volumes.flatten())
    N = min(n_samples_for_plots, n_elements)
    idx = np.random.choice(n_elements, size=N, replace=False)

    _plot_measure_pairs(
        title="All classes",
        volumes=all_volumes.flatten()[idx],
        curvatures=all_curvatures.flatten()[idx],
        reaches=all_reaches.flatten()[idx],
        classes=classes[idx],
        height=height,
        alpha=alpha
    )

    if distributions_dir is not None and name is not None:
        plt.savefig(distributions_dir / f"{name}.png", dpi=dpi)

    if plot_per_class:
        for name in class_names:
            _plot_measure_pairs(
                title=name,
                volumes=analysis_results.measures.normalized_volume_elements[name].flatten(),
                curvatures=analysis_results.measures.normalized_curvatures[name].flatten(),
                reaches=analysis_results.measures.normalized_reaches[name].flatten(),
                height=height,
                alpha=alpha
            )


def _plot_feature_measure_pairs(
    title: str,
    data: np.ndarray,
    volumes: np.ndarray,
    curvatures: np.ndarray,
    reaches: np.ndarray,
    classes=None,
    height: float = 2.5,
    alpha=0.5
):
    n_classes = len(np.unique(classes))
    rotations, x_translations, y_translations = np.meshgrid(*[np.arange(n) for n in data.shape[1:-1]])
    rotations = 360 * rotations / rotations.max()
    rotations = rotations.flatten()
    rotations = np.tile(rotations, n_classes)
    x_translations = x_translations.flatten()
    x_translations = np.tile(x_translations, n_classes)
    y_translations = y_translations.flatten()
    y_translations = np.tile(y_translations, n_classes)

    if classes is None:
        df = pd.DataFrame(dict(
            volume=volumes,
            curvature=curvatures,
            reach=reaches,
            rotation=rotations,
            x_translation=x_translations,
            y_translation=y_translations,
        ))
    else:
        df = pd.DataFrame(dict(
            volume=volumes,
            curvature=curvatures,
            reach=reaches,
            rotation=rotations,
            x_translation=x_translations,
            y_translation=y_translations,
            classes=classes
        ))

    g = sns.pairplot(
        df,
        kind="kde",
        diag_kind="kde",
        plot_kws=dict(fill=True, alpha=alpha),
        height=height,
        corner=False,
        hue=None if classes is None else "classes",
        y_vars=["volume", "curvature", "reach"],
        x_vars=["rotation", "x_translation", "y_translation"]
    )
    g.fig.suptitle(title, y=1.02)


def plot_features_vs_measures_multiclass(
    data: np.ndarray,
    analysis_results: AnalysisResults,
    plot_per_class: bool = False,
    height: float = 2.5,
    alpha=0.5
):
    class_names = list(analysis_results.measures.normalized_volume_elements.keys())
    if len(class_names) != data.shape[0]:
        raise ValueError(
            f"The number of class names ({len(class_names)} is different than the "
            f"first data dimension ({data.shape[0]})."
        )

    # Pair plot of measures per class and together.
    all_volumes = np.concatenate(
        [analysis_results.measures.normalized_volume_elements[name] for name in class_names],
        axis=0
    )
    all_curvatures = np.concatenate(
        [analysis_results.measures.normalized_curvatures[name] for name in class_names],
        axis=0
    )
    all_reaches = np.concatenate(
        [analysis_results.measures.normalized_reaches[name] for name in class_names],
        axis=0
    )
    classes = np.concatenate([
        np.array(np.prod(analysis_results.measures.normalized_reaches[name].shape)*[name])
        for name in class_names
    ])

    _plot_feature_measure_pairs(
        title="All classes",
        data=data,
        volumes=all_volumes.flatten(),
        curvatures=all_curvatures.flatten(),
        reaches=all_reaches.flatten(),
        classes=classes,
        height=height,
        alpha=alpha
    )

    if plot_per_class:
        for name in class_names:
            _plot_feature_measure_pairs(
                title=name,
                data=data,
                volumes=analysis_results.measures.normalized_volume_elements[name].flatten(),
                curvatures=analysis_results.measures.normalized_curvatures[name].flatten(),
                reaches=analysis_results.measures.normalized_reaches[name].flatten(),
                height=height,
                alpha=alpha
            )


def _plot_with_scalar_values(
    title,
    data,
    scalar_values,
    size_mult=5,
    alpha=0.6,
    s=1,
    cmap="coolwarm",
    colorbar_fraction=0.03,
    dpi: int = 300,
    output_path: Optional[Path] = None,
    show: bool = True
):
    n_plots = len(scalar_values)

    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(size_mult * n_plots, size_mult),
        subplot_kw=dict(projection='3d'),
        squeeze=False
    )
    fig.suptitle(title)
    axes = axes[0]

    for i, [name, value] in enumerate(scalar_values.items()):
        ax = axes[i]
        ax.set_title(name)
        plot = ax.scatter(*data.T, c=value, s=s, alpha=alpha, cmap=cmap)
        plt.colorbar(plot, ax=ax, fraction=colorbar_fraction)

    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=dpi)
    if show:
        plt.show()


def plot_3d_pca_projections_multiclass(
    data: np.ndarray,
    analysis_results: AnalysisResults,
    projection_measures_dir: Optional[Path] = None,
    projection_classes_dir: Optional[Path] = None,
    case_name: Optional[str] = None,
    alpha=0.3,
    s=0.2,
    dpi: int = 300,
    show: bool = True,
    n_samples_for_plots: int = 50_000
):
    class_names = list(analysis_results.measures.normalized_volume_elements.keys())
    if len(class_names) != data.shape[0]:
        raise ValueError(
            f"The number of class names ({len(class_names)} is different than the "
            f"first data dimension ({data.shape[0]})."
        )

    data_flat = data.reshape(-1, data.shape[-1])
    projection_pca = PCA(n_components=3).fit_transform(data_flat)

    size_per_class = int(np.prod(data.shape[1:-1]))
    proj_per_class = [
        projection_pca[i * size_per_class:(i + 1) * size_per_class]
        for i in range(len(class_names))
    ]

    output_path = None
    for name, proj in zip(class_names, proj_per_class):
        if projection_measures_dir is not None and case_name is not None:
            output_path = projection_measures_dir / f"{case_name}_{name}.png"

        n_elements = len(analysis_results.measures.normalized_volume_elements[name].flatten())
        N = min(n_samples_for_plots, n_elements)
        idx = np.random.choice(n_elements, size=N, replace=False)

        # noinspection PyTypeChecker
        _plot_with_scalar_values(
            name,
            proj[idx],
            {
                "Volume element": analysis_results.measures.normalized_volume_elements[name].flatten()[idx],
                "Scalar curvature": analysis_results.measures.normalized_curvatures[name].flatten()[idx],
                "Absolute scalar curvature": np.abs(
                    analysis_results.measures.normalized_curvatures[name].flatten()[idx]
                ),
                "Local reach": analysis_results.measures.normalized_reaches[name].flatten()[idx]
            },
            size_mult=4,
            s=s,
            alpha=alpha,
            dpi=dpi,
            show=show,
            output_path=output_path
        )

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    fig.suptitle("Classes")
    for name, proj in zip(class_names, proj_per_class):
        ax.scatter(*proj.T, s=s, alpha=alpha, label=name)

    plt.legend()
    plt.tight_layout()

    if projection_classes_dir is not None and case_name is not None:
        plt.savefig(projection_classes_dir / f"{case_name}.png", dpi=dpi)
    if show:
        plt.show()
