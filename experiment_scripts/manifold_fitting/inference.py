import itertools
from pathlib import Path

import typer
from tqdm import tqdm

from microscope.datasets.generic_dataset_loader import DatasetName
from representation_learning.mae.inference_intermediate_layers import main as inference_mae
from representation_learning.beta_vae.inference_intermediate_layers import main as inference_beta_vae

app = typer.Typer(pretty_exceptions_enable=False)


def inference_on_model(
    output_path: Path,
    model_path: Path,
    only_final_model: bool,
    only_latent_and_output: bool,
    dataset_name: DatasetName,
    model_type: str,
    number_of_dims: int,
    skip_done: bool
) -> None:
    if output_path.exists() and skip_done:
        print(f"Skipping {output_path.name} as it exists.")
        return None

    print(f"Inference on {output_path.name}.")
    if model_type == "beta_vae":
        inference_fn = inference_beta_vae
    elif model_type == "mae":
        inference_fn = inference_mae
    else:
        raise ValueError(f"Unknown model type: {model_type}.")

    if only_final_model:
        checkpoints_path = model_path / "checkpoints"
        final_candidates = [
            ckpt for ckpt in checkpoints_path.glob("*")
            if "last" in ckpt.name or "final" in ckpt.name
        ]
        if len(final_candidates) != 1:
            raise ValueError(f"Found the following final model candidates {final_candidates}. Expected as single one.")
        checkpoint_path = final_candidates[0]
        random_model = False

        inference_fn(
            dataset=dataset_name,
            number_of_dims=number_of_dims,
            only_latent_and_output=only_latent_and_output,
            checkpoint_path=checkpoint_path,
            output_dir=output_path,
            random_model=random_model
        )
    else:
        checkpoints_path = model_path / "checkpoints"
        checkpoint_paths = [p for p in checkpoints_path.glob("*") if "npz" not in p.suffix]
        final_candidates = [
            ckpt for ckpt in checkpoint_paths
            if "last" in ckpt.name or "final" in ckpt.name
        ]
        if len(final_candidates) != 1:
            raise ValueError(f"Found the following final model candidates {final_candidates}. Expected as single one.")
        final_checkpoint = final_candidates[0]
        middle_checkpoint_idx = len(checkpoint_paths) // 2
        middle_checkpoint = checkpoint_paths[middle_checkpoint_idx]
        first_checkpoint = checkpoint_paths[0]

        for checkpoint_path in [first_checkpoint, middle_checkpoint, final_checkpoint]:
            inference_fn(
                dataset=dataset_name,
                number_of_dims=number_of_dims,
                only_latent_and_output=only_latent_and_output,
                checkpoint_path=checkpoint_path,
                output_dir=output_path,
                random_model=False
            )


@app.command()
def run_inferences(
    training_path: Path = typer.Option(...),
    output_path: Path = typer.Option(...),
    only_final_model: bool = True,
    only_latent_and_output: bool = True,
    skip_done: bool = True
) -> None:
    dataset_name_list = [
        "custom_dsprites_balanced",
        "extended_coil20"
    ]
    model_type_list = [
        "beta_vae",
        "mae"
    ]
    training_ratio_per_dim_list = [
        0.4,
        0.5,
        0.6,
        1.0
    ]
    number_of_dims_list = [
        1,
        2,
        3,
        4
    ]
    # No noise.
    noise_sigma_list = [
        0
    ]

    hyperparameter_grid = list(itertools.product(
        dataset_name_list,
        model_type_list,
        training_ratio_per_dim_list,
        number_of_dims_list,
        noise_sigma_list
    ))

    for dataset_name, model_type, training_ratio_per_dim, number_of_dims, noise_sigma in tqdm(hyperparameter_grid):
        # Skip dimension 4 for COIL20.
        if (number_of_dims == 4) and (dataset_name == "extended_coil20"):
            continue
        if (number_of_dims == 4) and (model_type == "mae"):
            continue
        model_dir = "__".join([
            dataset_name,
            model_type,
            str(training_ratio_per_dim),
            str(number_of_dims),
            str(noise_sigma)
        ])
        inference_on_model(
            output_path=output_path / model_dir,
            model_path=training_path / model_dir,
            only_final_model=only_final_model,
            only_latent_and_output=only_latent_and_output,
            dataset_name=dataset_name,
            model_type=model_type,
            number_of_dims=number_of_dims,
            skip_done=skip_done
        )


if __name__ == "__main__":
    app()
