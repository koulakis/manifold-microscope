import itertools
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import typer
import yaml

from experiment_scripts.manifold_fitting.mmls import fit_mmls
from experiment_scripts.model_configs import BetaVAEConfig, MMLSConfig
from microscope.datasets.generic_dataset_loader import DatasetName, export_fixed_grid_test_set_and_rest_for_train
from representation_learning.beta_vae.solver import Solver

app = typer.Typer(pretty_exceptions_enable=False)


def update_max_epochs(
    config: BetaVAEConfig,
    mini_test_run: bool,
    training_ratio: float
) -> BetaVAEConfig:
    if mini_test_run:
        config = replace(config, max_epochs=1, plot_interval=2)
    elif training_ratio < 1.0:
        ratio_correction_coeff = 1 / training_ratio
        max_epochs = int(ratio_correction_coeff * config.max_epochs)
        plot_interval = int(ratio_correction_coeff * config.plot_interval)
        config = replace(
            config,
            max_epochs=max_epochs,
            plot_interval=plot_interval
        )

    return config


def train_model(
    output_path: Path,
    exported_datasets_dir: Path,
    seed: int,
    dataset_name: str,
    model_type: str,
    training_ratio: float,
    ratio_per_dim: bool,
    number_of_dims: int,
    noise_sigma: float,
    skip_done: bool,
    device: str = "cpu"
) -> None:
    if output_path.exists() and skip_done:
        print(f"Skipping {output_path.name} as it exists.")
        return None

    print(f"Training on {output_path.name}.")
    output_path.mkdir(parents=True)
    experiment_config = dict(
        output_path=str(output_path),
        seed=seed,
        dataset_name=dataset_name,
        model_type=model_type,
        training_ratio=training_ratio,
        ratio_per_dim=ratio_per_dim,
        number_of_dims=number_of_dims,
        noise_sigma=noise_sigma,
        device=device,
    )

    match model_type:
        case "beta_vae":
            model_config = BetaVAEConfig(
                dataset=dataset_name,
                ckpt_dir=output_path / "checkpoints",
                exported_datasets_dir=exported_datasets_dir,
                output_dir=output_path,
                training_ratio=training_ratio,
                ratio_per_dim=ratio_per_dim,
                number_of_dims=number_of_dims,
                noise_sigma=noise_sigma,
                device=device
            )

            if dataset_name == DatasetName.extended_coil20:
                model_config = replace(
                    model_config,
                    max_epochs=int(1e5),
                    objective="H",
                    model="H",
                    lr=1e-4,
                    loss_threshold=10
                )

            # Save config.
            config = dict(
                experiment_config=experiment_config,
                model_config=asdict(model_config)
            )
            with open(output_path / "config.yml", "w") as f:
                yaml.dump(config, f)

            torch.manual_seed(seed)
            if device == "mps" and torch.backends.mps.is_available():
                torch.mps.manual_seed(seed)
            elif device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            np.random.seed(seed)

            net = Solver(args=model_config)
            net.train()
        case "MMLS":
            model_config = MMLSConfig(
                output_dir=str(output_path),
                exported_datasets_dir=exported_datasets_dir,
                dataset=dataset_name,
                number_of_dims=number_of_dims,
                training_ratio=training_ratio,
                ratio_per_dim=ratio_per_dim,
                noise_sigma=noise_sigma,
                number_of_neighbors=2*2**number_of_dims,
                device=device
            )
            fit_mmls(model_config)
        case _:
            raise ValueError(f"Unknown model type: {model_type}.")


@app.command()
def run_trainings(
    output_path: Path = typer.Option(...),
    n_experiment_repetitions: int = 1,
    max_test_size: int = 500,
    seed: int = 42,
    skip_done: bool = True,
    device: str = "cuda:0"
) -> None:
    exported_datasets_dir = output_path / "datasets"
    exported_datasets_dir.mkdir(exist_ok=True, parents=True)
    dataset_name_list = [
        "custom_dsprites_balanced",
        "extended_coil20"
    ]
    model_type_list = [
        "beta_vae",
        "MMLS"
    ]
    ratio_per_dim = False
    # No noise.
    noise_sigma_list = [
        0
    ]

    hyperparameter_grid = list(itertools.product(
        dataset_name_list,
        model_type_list,
        noise_sigma_list
    ))

    for dataset_name, model_type, noise_sigma in hyperparameter_grid:
        match dataset_name:
            case "extended_coil20":
                training_ratio_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

                number_of_dims_list = [
                    1,
                    2,
                    3
                ]
            case "custom_dsprites_balanced":
                training_ratio_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                number_of_dims_list = [
                    1,
                    2,
                    3,
                    4
                ]
            case _:
                raise ValueError(f"Unknown dataset name: {dataset_name}.")

        second_level_hyperparameter_grid = list(itertools.product(
            number_of_dims_list,
            training_ratio_list
        ))
        for number_of_dims, training_ratio in second_level_hyperparameter_grid:
            export_fixed_grid_test_set_and_rest_for_train(
                dataset_name=dataset_name,
                number_of_dims=number_of_dims,
                output_dir=exported_datasets_dir,
                max_test_size=max_test_size,
                device=device
            )

            for repetition_n in range(n_experiment_repetitions):
                if number_of_dims == 1:
                    if training_ratio < 0.1:
                        continue
                model_dir = "__".join([
                    dataset_name,
                    model_type,
                    str(round(training_ratio, 3)),
                    str(number_of_dims),
                    str(noise_sigma),
                    str(repetition_n)
                ])
                train_model(
                    output_path=output_path / model_dir,
                    exported_datasets_dir=exported_datasets_dir,
                    seed=seed,
                    dataset_name=dataset_name,
                    model_type=model_type,
                    training_ratio=training_ratio,
                    ratio_per_dim=ratio_per_dim,
                    number_of_dims=number_of_dims,
                    noise_sigma=noise_sigma,
                    skip_done=skip_done,
                    device=device
                )


if __name__ == "__main__":
    app()
