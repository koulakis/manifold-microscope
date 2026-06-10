import pickle
from pathlib import Path

import numpy as np

from experiment_scripts.model_configs import MMLSConfig
from experiment_scripts.toy_manifolds_experiment.manifold_fitting_no_noise import ANNMMLSProjector
from microscope.datasets.generic_dataset_loader import DatasetName, load_dataset_fixed_test_split


def fit_mmls(model_config: MMLSConfig) -> None:
    output_dir = Path(model_config.output_dir)
    exported_datasets_dir = Path(model_config.exported_datasets_dir)
    dataset = DatasetName[model_config.dataset]
    number_of_dims = model_config.number_of_dims
    training_ratio = model_config.training_ratio
    ratio_per_dim = model_config.ratio_per_dim
    noise_sigma = model_config.noise_sigma
    number_of_neighbors = model_config.number_of_neighbors
    verbose = model_config.verbose
    device = model_config.device

    config = locals()

    # Load the exported dataset.
    data_train, data_test, _, _ = load_dataset_fixed_test_split(
        datasets_dir=exported_datasets_dir,
        dataset_name=dataset,
        number_of_dims=number_of_dims,
        ratio_per_dim=ratio_per_dim,
        training_ratio=training_ratio,
        noise_sigma=noise_sigma,
        weight_subsampling_by_manifold_volume=True
    )

    # Load the dataset.
    data_train = (data_train - data_train.min()) / (data_train.max() - data_train.min())
    data_test = (data_test - data_test.min()) / (data_test.max() - data_test.min())
    data_train = data_train.reshape(data_train.shape[0], np.prod(data_train.shape[1:]))
    data_test = data_test.reshape(data_test.shape[0], np.prod(data_test.shape[1:]))

    if len(data_train) < number_of_neighbors:
        print(
            f"Skipping the training ratio {training_ratio} as it results to {len(data_train)} points which are less "
            f"than the number of neighbors {number_of_neighbors}."
        )
        return

    projector = ANNMMLSProjector(data_train, d=number_of_dims, k=number_of_neighbors, verbose=verbose, device=device)
    prediction = projector.project(data_test)

    distances = np.linalg.norm(data_test - prediction, axis=-1)
    hausdorff_distance = distances.max()

    results = dict(
        number_of_train_points=len(data_train),
        number_of_test_points=len(data_test),
        pointwise_distances=distances,
        hausdorff_distance=hausdorff_distance
    )

    with open(output_dir / f"distance_results_{dataset}.pkl", "wb") as f:
        pickle.dump(results, f, -1)
