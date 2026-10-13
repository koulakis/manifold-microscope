"""Analyze actual completed beta-VAE inference directories."""
from pathlib import Path

import typer
from tqdm import tqdm

from experiment_scripts.manifold_fitting.paths import model_directories
from experiment_scripts.reproduction.common import run_experiment
from microscope.datasets.generic_dataset_loader import DatasetName
from microscope.computations_grid.data_analysis.run_data_analysis import main as run_analysis

app = typer.Typer(pretty_exceptions_enable=False)


def analysis_on_model(inference_path: Path, output_path: Path, dataset_name: DatasetName,
                      model_type: str, number_of_dims: int, only_evolution: bool,
                      normalize_for_volume: bool, skip_done: bool = True,
                      n_samples_for_plots: int = 50_000, device: str = "cuda:0") -> bool:
    """Analyze one model. Existing outputs always skip; skip_done is retained for callers."""
    inference_path = Path(inference_path).expanduser().resolve()

    def action(directory):
        run_analysis(inference_path=inference_path, output_path=directory,
                     dataset=DatasetName(dataset_name), model_type=model_type,
                     number_of_dims=number_of_dims, only_evolution=only_evolution,
                     normalize_for_volume=normalize_for_volume, device=device,
                     n_samples_for_plots=n_samples_for_plots)
        return list((directory / "measures").glob("intermediate_outputs_*.pkl"))

    return run_experiment(output_path, action, requires=[inference_path])


@app.command()
def run_analyses(inference_path: Path = typer.Option(...), output_path: Path = typer.Option(...),
                 only_evolution: bool = True, normalize_for_volume: bool = True,
                 skip_done: bool = True, n_samples_for_plots: int = 50_000,
                 device: str = "cuda:0") -> None:
    for model_path, dataset_name, number_of_dims in tqdm(list(model_directories(inference_path))):
        analysis_on_model(model_path, output_path / model_path.name, DatasetName(dataset_name),
                          "beta_vae", number_of_dims, only_evolution, normalize_for_volume,
                          skip_done, n_samples_for_plots=n_samples_for_plots, device=device)


if __name__ == "__main__":
    app()
