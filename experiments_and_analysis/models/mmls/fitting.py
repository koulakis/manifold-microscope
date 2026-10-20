"""MMLS fitting and projector-state IO, separate from projection and metrics."""
from dataclasses import asdict
from pathlib import Path
import pickle

import numpy as np

from ..config import MMLSConfig
from .projector import ANNMMLSProjector
from ...outputs import read_json, write_json


def fit(data, config: MMLSConfig, device='cpu'):
    points = data.reshape(len(data), -1)
    return ANNMMLSProjector(points, d=config.number_of_dims, k=config.number_of_neighbors,
                            sigma=config.sigma, exact_nn_threshold=config.exact_nn_threshold,
                            verbose=config.verbose, device=device,
                            projection_method=config.projection_method,
                            projection_batch_size=config.projection_batch_size)


def save_projector(projector, config, directory, *, split_id=None, export_points=False):
    """Normally reference saved split indices; optionally export a standalone fit."""
    directory = Path(directory)
    write_json(directory / 'projector.json', dict(parameters=asdict(config), split_id=split_id,
                                                 exported_points=export_points))
    if export_points:
        np.save(directory / 'training_points.npy', projector.X.detach().cpu().numpy())
    if projector.ann_index is not None:
        with (directory / 'neighbors.pkl').open('wb') as stream:
            pickle.dump(projector.ann_index, stream, -1)


def load_projector(directory, *, training_points=None, device='cpu'):
    directory = Path(directory)
    state = read_json(directory / 'projector.json')
    if state['exported_points']:
        training_points = np.load(directory / 'training_points.npy')
    if training_points is None:
        raise ValueError('Supply the training points selected by the saved split.')
    projector = fit(training_points, MMLSConfig(**state['parameters']), device)
    if (directory / 'neighbors.pkl').is_file():
        with (directory / 'neighbors.pkl').open('rb') as stream:
            projector.ann_index = pickle.load(stream)
    return projector
