from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional


@dataclass
class BetaVAEConfig:
    dataset: str
    number_of_dims: int
    training_ratio: float
    ratio_per_dim: bool
    noise_sigma: float

    ckpt_dir: Path
    output_dir: Path
    exported_datasets_dir: Path

    cuda: bool = True  # Deprecated: use device instead
    device: Optional[str] = None  # New: "cpu", "cuda", "cuda:0", "mps", etc.
    max_iter: float = 2e6
    max_epochs: int = 100_000
    strict_budget: bool = False  # Stop at exact batch/epoch limits when requested.
    plot_interval: int = 500
    batch_size: int = 64

    z_dim: int = 10
    beta: float = 4
    objective: str = "B"
    model: str = "B"
    gamma: float = 100
    C_max: float = 20
    C_stop_iter: float = 1e5
    lr: float = 5e-4
    beta1: float = 0.9
    beta2: float = 0.999
    loss_threshold: float = 35
    patience_percentage_epochs: float = 0.005
    patience_num: int = 5

    image_size: int = 64
    num_workers: int = 2
    save_output: bool = True
    update_pbar_on_epochs: bool = True

    gather_step: int = 1_000
    display_step: int = 10_000
    save_step: int = 10_000

    ckpt_name: Optional[str] = None


@dataclass
class MMLSConfig:
    output_dir: str
    exported_datasets_dir: Path
    dataset: str
    number_of_dims: int
    training_ratio: float
    ratio_per_dim: bool
    noise_sigma: float

    number_of_neighbors: int = 6
    sigma: float = 1.0
    exact_nn_threshold: int = 10_000
    device: str = "cuda:0"
    verbose: bool = True
    projection_method: str = "thin_svd"
    projection_batch_size: int = 32


def apply_model_overrides(config, overrides):
    """Apply model parameters without changing the selected inputs or output paths."""
    fixed = {"dataset", "number_of_dims", "training_ratio", "ratio_per_dim", "noise_sigma",
             "output_dir", "ckpt_dir", "exported_datasets_dir", "device", "cuda"}
    if fixed.intersection(overrides):
        raise ValueError("Set dataset, device and paths through the experiment arguments.")
    return replace(config, **overrides)
