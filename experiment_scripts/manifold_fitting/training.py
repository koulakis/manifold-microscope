"""Configurable image MMLS and beta-VAE experiments."""
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
import yaml

from experiment_scripts.manifold_fitting.mmls import fit_mmls
from experiment_scripts.model_configs import BetaVAEConfig, MMLSConfig, apply_model_overrides
from experiment_scripts.reproduction.common import run_experiment, write_json
from microscope.datasets.generic_dataset_loader import DatasetName, export_fixed_grid_test_set_and_rest_for_train

app = typer.Typer(pretty_exceptions_enable=False)
FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def update_max_epochs(config: BetaVAEConfig, mini_test_run: bool, training_ratio: float) -> BetaVAEConfig:
    if mini_test_run:
        config = replace(config, max_epochs=1, plot_interval=2)
    elif training_ratio < 1.0:
        ratio_correction_coeff = 1 / training_ratio
        config = replace(config, max_epochs=int(ratio_correction_coeff * config.max_epochs),
                         plot_interval=int(ratio_correction_coeff * config.plot_interval))
    return config


def train_model(output_path: Path, exported_datasets_dir: Path, seed: int,
                dataset_name: str, model_type: str, training_ratio: float,
                ratio_per_dim: bool, number_of_dims: int, noise_sigma: float,
                skip_done: bool = True, device: str = "cpu", num_workers: int = 2,
                model_overrides: Optional[dict] = None) -> bool:
    """Fit one fresh directory. skip_done is retained for older callers.

    Completed and partial directories always skip, as in the paper runner.
    """
    exported_datasets_dir = Path(exported_datasets_dir).expanduser().resolve()

    def action(directory):
        parameters = dict(dataset=dataset_name, number_of_dims=number_of_dims,
                          training_ratio=training_ratio, ratio_per_dim=ratio_per_dim,
                          noise_sigma=noise_sigma, exported_datasets_dir=exported_datasets_dir,
                          device=device)
        if model_type == "beta_vae":
            config = BetaVAEConfig(**parameters, output_dir=directory,
                                   ckpt_dir=directory / "checkpoints", num_workers=num_workers)
            if dataset_name == DatasetName.extended_coil20:
                config = replace(config, objective="H", model="H", lr=1e-4, loss_threshold=10)
        elif model_type == "MMLS":
            config = MMLSConfig(**parameters, output_dir=str(directory),
                                number_of_neighbors=2 * 2**number_of_dims)
        else:
            raise ValueError(f"Unknown model type: {model_type}.")
        config = apply_model_overrides(config, model_overrides or {})
        saved = dict(seed=seed, model_config=asdict(config), experiment_config=dict(
            dataset_name=dataset_name, model_type=model_type, device=device))
        write_json(directory / "config.json", saved)
        # Keep a portable YAML copy for callers using this older entry point.
        (directory / "config.yml").write_text(yaml.safe_dump(json.loads(
            (directory / "config.json").read_text())))
        if model_type == "beta_vae":
            from representation_learning.beta_vae.solver import Solver
            np.random.seed(seed)
            torch.manual_seed(seed)
            Solver(config).train()
            return [config.ckpt_dir / "last.pth"]
        fit_mmls(config)
        return list(directory.glob("distance_results_*.pkl"))

    return run_experiment(output_path, action)


@app.command()
def run_trainings(
    output_path: Path = typer.Option(...),
    n_experiment_repetitions: int = 1,
    max_test_size: int = 500,
    seed: int = 42,
    skip_done: bool = True,
    device: str = "cuda:0",
    datasets: Optional[list[DatasetName]] = None,
    models: Optional[list[str]] = None,
    dimensions: Optional[list[int]] = None,
    fractions: Optional[list[float]] = None,
    repetitions: Optional[list[int]] = None,
    seeds: Optional[list[int]] = None,
    num_workers: int = 2,
    max_iter: Optional[int] = None,
    max_epochs: Optional[int] = None,
    strict_budget: bool = False,
    model_overrides: Optional[Path] = None
) -> None:
    """Run selected jobs; model_overrides is a JSON file keyed by beta_vae/MMLS."""
    datasets = datasets if datasets is not None else [DatasetName.custom_dsprites_balanced, DatasetName.extended_coil20]
    models = models if models is not None else ["beta_vae", "MMLS"]
    fractions = fractions if fractions is not None else FRACTIONS
    repetitions = repetitions if repetitions is not None else list(range(n_experiment_repetitions))
    # Keep the historical seed default; explicit seeds correspond to selected repetitions.
    seeds = seeds if seeds is not None else [seed] * len(repetitions)
    if len(seeds) != len(repetitions):
        raise ValueError("Provide one seed for each selected repetition.")
    if any(model not in ("beta_vae", "MMLS") for model in models):
        raise ValueError("models must contain beta_vae or MMLS.")
    overrides = json.loads(model_overrides.read_text()) if model_overrides is not None else {}
    vae_overrides = dict(overrides.get("beta_vae", {}))
    for name, value in [("max_iter", max_iter), ("max_epochs", max_epochs)]:
        if value is not None:
            vae_overrides[name] = value
    if strict_budget:
        vae_overrides["strict_budget"] = True
    exported_datasets_dir = output_path / "datasets"
    exported_datasets_dir.mkdir(exist_ok=True, parents=True)
    for dataset in datasets:
        dataset = DatasetName(dataset)
        max_dimension = 3 if dataset == DatasetName.extended_coil20 else 4
        selected_dimensions = dimensions if dimensions is not None else range(1, max_dimension + 1)
        for model, dimension in itertools.product(models, selected_dimensions):
            if not 1 <= dimension <= max_dimension:
                continue
            export_fixed_grid_test_set_and_rest_for_train(
                dataset_name=dataset, number_of_dims=dimension, output_dir=exported_datasets_dir,
                max_test_size=max_test_size, device=device)
            for fraction, (repetition, repetition_seed) in itertools.product(fractions, zip(repetitions, seeds)):
                if dimension == 1 and fraction < 0.1:
                    continue
                name = "__".join([dataset.value, model, str(round(fraction, 3)), str(dimension), "0", str(repetition)])
                train_model(output_path / name, exported_datasets_dir, repetition_seed,
                            dataset.value, model, fraction, False, dimension, 0,
                            skip_done=skip_done, device=device, num_workers=num_workers,
                            model_overrides=vae_overrides if model == "beta_vae" else overrides.get("MMLS", {}))


if __name__ == "__main__":
    app()
