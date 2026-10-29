"""MMLS fitting and projector-state IO, separate from projection and metrics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dataclasses import asdict
import pickle

from ..config import MMLSConfig
from .projector import ANNMMLSProjector
from ...outputs import artifact_path, load_array, read_json, save_array, write_json

if TYPE_CHECKING:
    import numpy as np
    from ...bundles import BundlePath


def fit(
    data: np.ndarray,
    config: MMLSConfig,
    device: str = 'cpu'
) -> ANNMMLSProjector:
    """Construct a projector from the selected training samples.

    Args:
        data: Input samples of shape (n ...), with sample axis first.
        config: MMLS projection parameters.
        device: Device used for tensor computations.
    """
    points = data.reshape(len(data), -1)
    return ANNMMLSProjector(
        points,
        d=config.number_of_dims,
        k=config.number_of_neighbors,
        sigma=config.sigma,
        exact_nn_threshold=config.exact_nn_threshold,
        verbose=config.verbose,
        device=device,
        projection_method=config.projection_method,
        projection_batch_size=config.projection_batch_size
    )


def save_projector(
    projector: ANNMMLSProjector,
    config: MMLSConfig,
    directory: Path | BundlePath,
    split_id: str | None = None,
    export_points: bool = False
) -> None:
    """Normally reference saved split indices; optionally export a standalone fit.

    Args:
        projector: Fitted MMLS projector.
        config: MMLS projection parameters.
        directory: Directory or virtual bundle path for this stage.
        split_id: Identifier of the training split, when available.
        export_points: Save standalone training points instead of relying only on the split identifier.
    """
    directory = artifact_path(directory)
    write_json(directory / 'projector.json', dict(parameters=asdict(config), split_id=split_id,
                                                 exported_points=export_points))
    if export_points:
        save_array(directory / 'training_points.npy', projector.X.detach().cpu().numpy())
    if projector.ann_index is not None:
        with (directory / 'neighbors.pkl').open('wb') as stream:
            pickle.dump(projector.ann_index, stream, -1)


def load_projector(
    directory: str | Path | BundlePath,
    training_points: np.ndarray | None = None,
    device: str = 'cpu'
) -> ANNMMLSProjector:
    """Restore projector parameters and any saved neighbor index.

    Args:
        directory: Directory or virtual bundle path for this stage.
        training_points: Training samples of shape (n f), if not exported with the fit.
        device: Device used for tensor computations.
    """
    directory = artifact_path(directory)
    state = read_json(directory / 'projector.json')
    if state['exported_points']:
        training_points = load_array(directory / 'training_points.npy')
    if training_points is None:
        raise ValueError('Supply the training points selected by the saved split.')
    projector = fit(training_points, MMLSConfig(**state['parameters']), device)
    if (directory / 'neighbors.pkl').is_file():
        with (directory / 'neighbors.pkl').open('rb') as stream:
            projector.ann_index = pickle.load(stream)
    return projector
