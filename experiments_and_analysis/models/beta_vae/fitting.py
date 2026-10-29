"""Fit the existing beta-VAE solver to an explicitly supplied training array."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..config import BetaVAEConfig
from .dataset import training_loader
from .solver import Solver
from ...outputs import write_json

if TYPE_CHECKING:
    from .model import BetaVAE_H


def fit(
    data: np.ndarray,
    directory: str | Path,
    parameters: dict,
    seed: int = 42,
    device: str = 'cpu',
    num_workers: int = 0
) -> BetaVAE_H:
    """Training/checkpoint IO boundary. No dataset loading or sampling here.

    Args:
        data: Training images of shape (n 64 64).
        directory: Directory or virtual bundle path for this stage.
        parameters: Model parameters.
        seed: Random seed.
        device: Device used for tensor computations.
        num_workers: Number of data-loader workers.

    Returns:
        Trained beta-VAE model from the completed solver run.
    """
    directory = Path(directory)
    config = BetaVAEConfig(
        output_dir=directory,
        ckpt_dir=directory / 'checkpoints',
        device=device,
        num_workers=num_workers,
        **parameters
    )
    write_json(directory / 'model.json', asdict(config))
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader = training_loader(data, config.batch_size, num_workers, device)
    solver = Solver(config, loader)
    solver.train()
    return solver.net
