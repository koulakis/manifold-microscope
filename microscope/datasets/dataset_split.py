import numpy as np
import torch
from tqdm import tqdm


def interpolation_splitting(
    data: np.ndarray,
    target: np.ndarray,
    stride: int,
    subsample_dims: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stride over the transformation dimensions to select data on a thinner grid. This can be used to measure the
    interpolation performance of a supervised model."""
    subsample_slices = [slice(None)] * data.ndim
    for d in subsample_dims:
        subsample_slices[d] = slice(0, data.shape[d], stride)
    subsample_slices = tuple(subsample_slices)

    train_mask = np.zeros(data.shape, dtype=bool)
    train_mask[subsample_slices] = True

    train_data = data[subsample_slices]
    train_target = target[subsample_slices[:-2]]

    test_data = data[~train_mask]
    test_target = target[~train_mask[..., 0, 0]]

    train_data = train_data.reshape(-1, *train_data.shape[-2:])
    test_data = test_data.reshape(-1, *test_data.shape[-2:])
    train_target = train_target.flatten()
    test_target = test_target.flatten()
    train_mask = train_mask[..., 0, 0].flatten()

    return train_data, train_target, test_data, test_target, train_mask


def weighted_uniform_splitting(
    data: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    train_sample_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly sample points from the data grid with some weights.

    Args:
        data: The data in an array of shape (d1 ... dk f1 f2).
        target: The target in an array of shape (d1 ... dk).
        weights: Weights in an array of shape (d1 ... dk). They should sum up to 1.
        train_sample_ratio: Ratio of the data kept in the training set.

    Returns:
        array_train: The train data in an array of shape (n_train f).
        target_train: The train target in an array of shape (n_train).
        array_test: The test data in an array of shape ((d1*...*dk - n_train) f1 f2).
        target_test: The test target in an array of shape ((d1*...*dk - n_train)).
        train_mask: A boolean mask of shape ((d1*...*dk)) which defines which indices will be used for training.
    """
    if not 0 <= train_sample_ratio <= 1:
        raise ValueError(f"The train ratio must have a value in [0, 1]. Found value {train_sample_ratio}.")

    data_flat = data.reshape(-1, *data.shape[-2:])
    weights_flat = weights.flatten()
    n_train = int(train_sample_ratio * len(data_flat))
    train_idx = np.random.choice(
        len(data_flat),
        size=n_train,
        replace=False,
        p=weights_flat
    )
    train_mask = np.zeros(len(data_flat), dtype=bool)
    train_mask[train_idx] = True

    array_train = data_flat[train_mask]
    array_test = data_flat[~train_mask]

    target_flat = target.flatten()
    target_train = target_flat[train_mask]
    target_test = target_flat[~train_mask]

    return array_train, target_train, array_test, target_test, train_mask


def maximal_distance_selection_torch(data: np.ndarray, size: int, device=None, verbose: bool = False):
    """
    Farthest point sampling (a.k.a. maximal distance selection).

    Args:
        data: Tensor of shape (N, D), must fit in memory.
        size: Number of points to select.
        device: Optional "cuda" or "cpu". If None, will try GPU first, fallback to CPU.
        verbose: If true, show progress bar.
    Returns:
        selected_data: (size, D) tensor
        selected_idxs: (size, ) tensor of indices
    """
    N, D = data.shape

    data = torch.from_numpy(data).to(device)

    selected_idxs = torch.zeros(size, dtype=torch.long, device=device)
    selected_data = torch.empty((size, D), dtype=data.dtype, device=device)

    # pick first point
    selected_idxs[0] = 0
    selected_data[0] = data[0]

    # track nearest distances to selected set
    min_distances = torch.norm(data - data[0], dim=1)

    for i in tqdm(range(1, size), total=size, disable=not verbose):
        # pick the farthest point
        new_point_idx = torch.argmax(min_distances)
        selected_idxs[i] = new_point_idx
        selected_data[i] = data[new_point_idx]

        # update nearest distances (only vs new point)
        new_distances = torch.norm(data - data[new_point_idx], dim=1)
        min_distances = torch.minimum(min_distances, new_distances)

    return selected_data.cpu().numpy(), selected_idxs.cpu().numpy()
