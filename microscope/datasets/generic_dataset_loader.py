from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from microscope.computations_grid.volume import volume_element
from microscope.cyclic_dimensions import get_difference_intervals
from microscope.datasets import original_dsprites, custom_dsprites
from microscope.datasets.coil20 import extended_coil20
from microscope.datasets.dataset_split import weighted_uniform_splitting, maximal_distance_selection_torch
from microscope.datasets.noise_adding import compute_gaussian_noise


class DatasetName(str, Enum):
    dsprites = "dsprites"
    dsprites_single_size = "dsprites_single_size"
    custom_dsprites_single_size = "custom_dsprites_single_size"
    custom_dsprites = "custom_dsprites"
    custom_dsprites_balanced = "custom_dsprites_balanced"
    extended_coil20 = "extended_coil20"


def load_dataset_no_split(
    dataset_name: DatasetName,
    number_of_dims: int,
    return_data_slice_info: bool = True,
    scale_to_0_1: bool = True
):
    """Load one of the datasets applying different transforms and splits.

    Args:
        dataset_name: The name of the base dataset.
        number_of_dims: The number of grid dimensions to keep in the dataset. On each dataset there is an explicit
            way to remove dimensions, preferring to keep cyclic vs non-cyclic.
        return_data_slice_info: If true, it returns information needed to handle cyclic dimensions of the data.
        scale_to_0_1: If true scale to [0, 1].

    Returns:
        If return_data_slice_info is False:
            data, target
        else:
            data, target, patch_sizes, difference_intervals, volume_clip_slices, full_clip_slices
    """
    difference_intervals, cyclic_dimensions, patch_sizes = 3*[None]
    init_flat = False
    slices, indices, range_sizes, volume_clip_slices, full_clip_slices = 5*[None]
    b, bt = 2, 3

    match dataset_name:
        case DatasetName.dsprites:
            data, target = original_dsprites.dsprites_original(flat=init_flat)

        case DatasetName.dsprites_single_size:
            data, target = original_dsprites.dsprites_original_single_size(flat=init_flat)

        case DatasetName.custom_dsprites_single_size:
            data, target = custom_dsprites.dsprites_original_remake_single_size(flat=init_flat)

        case DatasetName.custom_dsprites:
            data, target = custom_dsprites.dsprites_original_remake(flat=init_flat)

        case DatasetName.custom_dsprites_balanced:
            data, target = custom_dsprites.dsprites_balanced(
                flat=init_flat,
                buffer_for_measure_estimation=3
            )  # Shape: (3 22 16 22 22 64 64)
            volume_clip_slices = (slice(None), slice(b, -b), slice(None), slice(b, -b), slice(b, -b))
            full_clip_slices = (slice(None), slice(bt, -bt), slice(None), slice(bt, -bt), slice(bt, -bt))

            range_sizes = [1, 2 * np.pi, 1, 1]
            cyclic_dimensions = [1]  # The rotation is cyclic.
            patch_sizes = 4*[10]
            difference_intervals = get_difference_intervals(list(data.shape[1:5]), range_sizes, cyclic_dimensions)

        case DatasetName.extended_coil20:
            data, target = extended_coil20(
                flat=init_flat,
                buffer_for_measure_estimation=3
            )  # Shape: (20 16 22 16 64 64)

            volume_clip_slices = (slice(None), slice(None), slice(b, -b), slice(None))
            full_clip_slices = (slice(None), slice(None), slice(bt, -bt), slice(None))

            range_sizes = [2 * np.pi, 1, 2 * np.pi]
            cyclic_dimensions = [0, 2]  # The rotations are cyclic.
            patch_sizes = 3*[10]
            difference_intervals = get_difference_intervals(list(data.shape[1:4]), range_sizes, cyclic_dimensions)

        case _:
            raise NotImplementedError()

    # Subsample dims. We always select the feature on the largest index outside the buffer margin.
    match dataset_name:
        case DatasetName.custom_dsprites_balanced:
            match number_of_dims:
                case 3:
                    slices = (slice(None), slice(None), slice(None), slice(None), 3)
                    indices = [0, 1, 2]
                    cyclic_dimensions = [1]  # The rotation is cyclic.
                case 2:
                    slices = (slice(None), slice(None), slice(None), 3, 3)
                    indices = [0, 1]
                    cyclic_dimensions = [1]  # The rotation is cyclic.
                case 1:
                    slices = (slice(None), -4, slice(None), 3, 3)
                    indices = [1]
                    cyclic_dimensions = [0]  # The rotation is cyclic.
        case DatasetName.extended_coil20:
            match number_of_dims:
                case 2:
                    slices = (slice(None), slice(None), -4, slice(None))
                    indices = [0, 2]
                    cyclic_dimensions = [0, 1]  # The rotations are cyclic.
                case 1:
                    slices = (slice(None), slice(None), -4, 0)
                    indices = [0]
                    cyclic_dimensions = [0]  # The rotations are cyclic.

    # If applicable, reduce the dimension of the data.
    if slices is not None:
        data = data[slices]
        target = target[slices]

        range_sizes = list(np.array(range_sizes)[indices])
        patch_sizes = list(np.array(patch_sizes)[indices])
        difference_intervals = get_difference_intervals(
            list(data.shape[1:1+len(indices)]),
            range_sizes,
            cyclic_dimensions
        )

        clip_slice_indices = [0] + list(np.array(indices) + 1)
        volume_clip_slices = tuple(np.array(volume_clip_slices)[clip_slice_indices])
        full_clip_slices = tuple(np.array(full_clip_slices)[clip_slice_indices])

    if scale_to_0_1:
        data = (data - data.min()) / (data.max() - data.min())

    if return_data_slice_info:
        return (
            data,
            target,
            patch_sizes,
            difference_intervals,
            volume_clip_slices,
            full_clip_slices,
            range_sizes,
            cyclic_dimensions
        )
    else:
        return data, target


def load_dataset(
    dataset_name: DatasetName,
    number_of_dims: int,
    ratio_per_dim: bool,
    training_ratio: float,
    noise_sigma: float,
    save_train_idx: bool = False,
    weight_subsampling_by_manifold_volume: bool = True,
    output_dir: Optional[Path] = None,
    return_full_datasets: bool = False,
    return_dataset_unflat_with_metadata: bool = False,
    full_datasets_unclipped: bool = False,
    device: str = "cuda:0"
) -> tuple[np.ndarray, ...]:
    """Load one of the datasets applying different transforms and splits.

    Args:
        dataset_name: The name of the base dataset.
        number_of_dims: The number of grid dimensions to keep in the dataset. On each dataset there is an explicit
            way to remove dimensions, preferring to keep cyclic vs non-cyclic.
        ratio_per_dim: If true, then the training ratio is applied per dimension, i.e. the dataset is subsampled so that
            each dimension keep this point ratio. This can be useful in generating ratios which are comparable between
            datasets of different dimensions. It did not prove to be very successful in practice as the number of
            samples grows exponentially with the dimension.
        training_ratio: A training sampling ratio which is applied directly or per dimension.
        noise_sigma: The standard deviation of an ambient Gaussian noise added to the data.
        save_train_idx: If true, then a mask which selects the training data is saved to recover train or test datasets.
        weight_subsampling_by_manifold_volume: If true, then the subsampling is weighted based on the volume in order
            to simulate a uniform sampling on the volume measure.
        output_dir: The output directory for the train index.
        return_full_datasets: If true, no data split happens and the full data and targets are returned.
        return_dataset_unflat_with_metadata: If ture, return dataset metadata such as cyclic dimensions and ranges.
        full_datasets_unclipped: If the return_full_datasets is true, then this option allows to return the datasets
            with the margin buffer needed to compute measures such as curvature.
        device: Torch device.

    Returns:
        The following arrays with the corresponding splits. Note that the grid dimensions are flattened.
            data_train
            data_test
            target_train
            target_test
    """
    (
        data,
        target,
        patch_sizes,
        difference_intervals,
        volume_clip_slices,
        full_clip_slices,
        range_sizes,
        cyclic_dimensions
    ) = load_dataset_no_split(
        dataset_name,
        number_of_dims,
        return_data_slice_info=True
    )

    if return_full_datasets:
        if not full_datasets_unclipped:
            data, target = data[full_clip_slices], target[full_clip_slices]

        if return_dataset_unflat_with_metadata:
            return data, target, range_sizes, cyclic_dimensions, patch_sizes
        else:
            data = data.reshape(-1, *data.shape[-2:])
            target = target.flatten()
            return data, target

    match dataset_name:
        case DatasetName.custom_dsprites_balanced | DatasetName.extended_coil20:
            if weight_subsampling_by_manifold_volume:
                # Get weights which simulate the volume distribution.
                data_volume_clipped = data[volume_clip_slices]
                data_volume_clipped = data_volume_clipped.reshape([
                    *data_volume_clipped.shape[:-2],
                    np.prod(data_volume_clipped.shape[-2:])
                ])

                volumes = [
                    volume_element(
                        d,
                        difference_intervals,
                        cyclic_dimensions,
                        patch_sizes,
                        device=device
                    )
                    for d in data_volume_clipped
                ]
                volumes = np.stack([v / v.sum() for v in volumes], axis=0)
                weights = volumes / volumes.sum()
                data = data[full_clip_slices]
                target = target[full_clip_slices]
            else:
                data = data[full_clip_slices]
                target = target[full_clip_slices]
                n_examples = np.prod(data.shape[:-2])
                weights = np.ones(data.shape[:-2]) / n_examples
        case (
            DatasetName.dsprites
            | DatasetName.dsprites_single_size
            | DatasetName.custom_dsprites
            | DatasetName.custom_dsprites_single_size
        ):
            n_examples = np.prod(data.shape[:-2])
            weights = np.ones(data.shape[:-2]) / n_examples
        case _:
            raise ValueError(f"Unhandled dataset: {dataset_name}")

    # Split to train/test and export the split
    intrinsic_dim = len(data.shape[1:-2])
    if ratio_per_dim:
        subsampling_ratio = training_ratio ** intrinsic_dim
    else:
        subsampling_ratio = training_ratio

    data_train, target_train, data_test, target_test, train_idx = weighted_uniform_splitting(
        data,
        target,
        weights,
        train_sample_ratio=subsampling_ratio
    )

    if save_train_idx:
        np.savez(output_dir / "train_idx.npz", train_idx=train_idx)

    # Add Gaussian noise.
    noise = compute_gaussian_noise(data_train.shape, sigma=noise_sigma).astype(data_train.dtype)
    data_train = data_train + noise

    return (
        data_train.astype(np.float32),
        data_test.astype(np.float32),
        target_train.astype(np.int64),
        target_test.astype(np.int64)
    )


def export_fixed_grid_test_set_and_rest_for_train(
    dataset_name: DatasetName,
    number_of_dims: int,
    output_dir: Path,
    max_test_size: int = 1_000,
    device: str = "cuda:0"
):
    output_path = output_dir / f"{dataset_name}__{number_of_dims}.npz"
    if output_path.exists():
        print(f"Skipping dataset generation for {dataset_name}, {number_of_dims} dimensions as it exists.")
        return

    # Load the whole dataset.
    (
        data_unclipped,
        target_unclipped,
        patch_sizes,
        difference_intervals,
        volume_clip_slices,
        full_clip_slices,
        range_sizes,
        cyclic_dimensions
    ) = load_dataset_no_split(
        dataset_name,
        number_of_dims,
        return_data_slice_info=True
    )

    # Compute volume weights.
    match dataset_name:
        case DatasetName.custom_dsprites_balanced | DatasetName.extended_coil20:
            # Get weights which simulate the volume distribution.
            data_volume_clipped = data_unclipped[volume_clip_slices]
            data_volume_clipped = data_volume_clipped.reshape([
                *data_volume_clipped.shape[:-2],
                np.prod(data_volume_clipped.shape[-2:])
            ])

            volumes = [
                volume_element(
                    d,
                    difference_intervals,
                    cyclic_dimensions,
                    patch_sizes,
                    device=device
                )
                for d in data_volume_clipped
            ]
            volumes = np.stack([v / v.sum() for v in volumes], axis=0)
            weights = volumes / volumes.sum()
            data = data_unclipped[full_clip_slices]
            target = target_unclipped[full_clip_slices]
        case (
            DatasetName.dsprites
            | DatasetName.dsprites_single_size
            | DatasetName.custom_dsprites
            | DatasetName.custom_dsprites_single_size
        ):
            data = data_unclipped
            target = target_unclipped
            n_examples = np.prod(data_unclipped.shape[:-2])
            weights = np.ones(data_unclipped.shape[:-2]) / n_examples
        case _:
            raise ValueError(f"Unhandled dataset: {dataset_name}")

    # Flatten the data.
    data_flat = data.reshape(-1, *data.shape[-2:])
    target_flat = target.flatten()
    weights_flat = weights.flatten()

    # Extract a test dataset of adequate size.
    test_size = min(max_test_size, int(0.3 * len(data_flat)))
    test_idx = np.zeros(len(data_flat), dtype=bool)
    _, test_idx_integers = maximal_distance_selection_torch(
        data_flat.reshape(-1, np.prod(data_flat.shape[1:])),
        test_size,
        verbose=True
    )
    test_idx[test_idx_integers] = True
    data_test = data_flat[test_idx]

    train_idx = ~test_idx
    data_train = data_flat[train_idx]

    weights_train = weights_flat[train_idx]
    weights_test = weights_flat[test_idx]
    # Normalize the weights as they are now split to train/test.
    weights_train = weights_train / weights_train.sum()
    weights_test = weights_test / weights_test.sum()

    target_train = target_flat[train_idx]
    target_test = target_flat[test_idx]

    # Export the train, test datasets, indices and weights.
    np.savez(
        output_path,
        dataset_name=dataset_name,
        number_of_dims=number_of_dims,
        data_unclipped=data_unclipped,
        full_clip_slices=full_clip_slices,
        weights=weights,
        data_train=data_train,
        data_test=data_test,
        train_idx=train_idx,
        test_idx=test_idx,
        weights_train=weights_train,
        weights_test=weights_test,
        target_train=target_train,
        target_test=target_test
    )


def load_dataset_fixed_test_split(
    datasets_dir: Path,
    dataset_name: DatasetName,
    number_of_dims: int,
    ratio_per_dim: bool,
    training_ratio: float,
    noise_sigma: float,
    weight_subsampling_by_manifold_volume: bool = True
) -> tuple[np.ndarray, ...]:
    """Load one of the datasets applying different transforms and splits.

    Args:
        datasets_dir: The directory with the exported datasets.
        dataset_name: The name of the base dataset.
        number_of_dims: The number of grid dimensions to keep in the dataset. On each dataset there is an explicit
            way to remove dimensions, preferring to keep cyclic vs non-cyclic.
        ratio_per_dim: If true, then the training ratio is applied per dimension, i.e. the dataset is subsampled so that
            each dimension keep this point ratio. This can be useful in generating ratios which are comparable between
            datasets of different dimensions. It did not prove to be very successful in practice as the number of
            samples grows exponentially with the dimension.
        training_ratio: A training sampling ratio which is applied directly or per dimension.
        noise_sigma: The standard deviation of an ambient Gaussian noise added to the data.
        weight_subsampling_by_manifold_volume: If true, then the subsampling is weighted based on the volume in order
            to simulate a uniform sampling on the volume measure.

    Returns:
        The following arrays with the corresponding splits. Note that the grid dimensions are flattened.
            data_train
            data_test
            target_train
            target_test
    """
    # Load the dataset info.
    if isinstance(dataset_name, Enum):
        dataset_name = dataset_name.value
    filename = f"{dataset_name}__{number_of_dims}.npz"
    data_info = np.load(datasets_dir / filename, allow_pickle=True)

    number_of_dims = data_info["number_of_dims"]
    data_train = data_info["data_train"]
    data_test = data_info["data_test"]
    weights_train = data_info["weights_train"]
    target_train = data_info["target_train"]
    target_test = data_info["target_test"]

    # Subsample the training set.
    if ratio_per_dim:
        subsampling_ratio = training_ratio ** number_of_dims
    else:
        subsampling_ratio = training_ratio

    if not weight_subsampling_by_manifold_volume:
        weights_train = np.ones_like(weights_train)
        weights_train = weights_train / weights_train.sum()

    data_train, target_train, _, _, _ = weighted_uniform_splitting(
        data_train,
        target_train,
        weights_train,
        train_sample_ratio=subsampling_ratio
    )

    # Add Gaussian noise.
    noise = compute_gaussian_noise(data_train.shape, sigma=noise_sigma).astype(data_train.dtype)
    data_train = data_train + noise

    return (
        data_train.astype(np.float32),
        data_test.astype(np.float32),
        target_train.astype(np.int64),
        target_test.astype(np.int64)
    )
