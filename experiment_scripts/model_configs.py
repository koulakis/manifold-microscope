from dataclasses import dataclass
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
class MAEConfig:
    output_dir: str
    exported_datasets_dir: Path
    dataset: str
    number_of_dims: int
    training_ratio: float
    ratio_per_dim: bool
    noise_sigma: float

    max_epochs: int = 4_000
    loss_threshold: float = 0.01
    n_consecutive_epochs_below_loss_threshold_till_stop: int = 5
    batch_size: int = 4096
    num_workers: int = 2
    epoch_examples: Optional[int] = None
    plot_interval: Optional[int] = 50
    base_lr: float = 1.5e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    use_scheduler: bool = False
    in_chans: int = 1
    patch_size: int = 8
    img_size: int = 64
    embed_dim: int = 128
    depth: int = 4
    num_heads: int = 8
    pretrained: bool = False
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    mask_ratio: float = 0.5
    latent_dim: int = 10
    decoder_depth: int = 2
    device: str = "cuda:0"
    use_bottleneck: bool = False


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
