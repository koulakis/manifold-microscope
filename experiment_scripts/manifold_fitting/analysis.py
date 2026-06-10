import itertools
from pathlib import Path

import typer
from tqdm import tqdm

from microscope.datasets.generic_dataset_loader import DatasetName
from microscope.computations_grid.data_analysis.run_data_analysis import main as run_analysis

app = typer.Typer(pretty_exceptions_enable=False)


def analysis_on_model(
    inference_path: Path,
    output_path: Path,
    dataset_name: DatasetName,
    model_type: str,
    number_of_dims: int,
    only_evolution: bool,
    normalize_for_volume: bool,
    skip_done: bool
) -> None:
    if output_path.exists() and skip_done:
        print(f"Skipping {output_path.name} as it exists.")
        return None

    print(f"Analysis on {output_path.name}.")
    run_analysis(
        inference_path=inference_path,
        output_path=output_path,
        dataset=dataset_name,
        model_type=model_type,
        number_of_dims=number_of_dims,
        only_evolution=only_evolution,
        normalize_for_volume=normalize_for_volume
    )


@app.command()
def run_analyses(
    inference_path: Path = typer.Option(...),
    output_path: Path = typer.Option(...),
    only_evolution: bool = True,
    normalize_for_volume: bool = True,
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
        analysis_on_model(
            output_path=output_path / model_dir,
            inference_path=inference_path / model_dir,
            dataset_name=dataset_name,
            model_type=model_type,
            number_of_dims=number_of_dims,
            only_evolution=only_evolution,
            normalize_for_volume=normalize_for_volume,
            skip_done=skip_done
        )


if __name__ == "__main__":
    app()
