"""Fit the existing beta-VAE solver to an explicitly supplied training array."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..config import BetaVAEConfig
from .dataset import training_loader
from .solver import Solver
from ...outputs import write_json


def fit(data, directory, parameters, *, seed=42, device='cpu', num_workers=0):
    """Training/checkpoint IO boundary. No dataset loading or sampling here."""
    directory = Path(directory)
    config = BetaVAEConfig(output_dir=directory, ckpt_dir=directory / 'checkpoints',
                           device=device, num_workers=num_workers, **parameters)
    write_json(directory / 'model.json', asdict(config))
    np.random.seed(seed)
    torch.manual_seed(seed)
    loader = training_loader(data, config.batch_size, num_workers, device)
    solver = Solver(config, loader)
    solver.train()
    return solver.net
