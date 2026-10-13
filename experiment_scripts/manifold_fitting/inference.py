"""Infer beta-VAE layers for actual completed training directories."""
import json
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from experiment_scripts.manifold_fitting.paths import model_directories
from experiment_scripts.reproduction.common import run_experiment
from microscope.datasets.generic_dataset_loader import DatasetName
from representation_learning.beta_vae.inference_intermediate_layers import main as inference_beta_vae

app = typer.Typer(pretty_exceptions_enable=False)


def inference_on_model(output_path: Path, model_path: Path, only_final_model: bool,
                       only_latent_and_output: bool, dataset_name: DatasetName,
                       model_type: str, number_of_dims: int, skip_done: bool = True,
                       device: str = "cuda:0", num_workers: int = 2,
                       seed: Optional[int] = None, decoder_sampling: bool = True) -> bool:
    """Infer one model. Existing outputs always skip; skip_done is retained for callers."""
    model_path = Path(model_path).expanduser().resolve()

    def action(directory):
        if model_type != "beta_vae":
            raise ValueError(f"Unknown model type: {model_type}.")
        config = json.loads((model_path / "config.json").read_text())["model_config"]
        checkpoints = sorted(
            (p for p in (model_path / "checkpoints").iterdir() if p.is_file() and p.suffix != ".npz"),
            key=lambda p: int(p.name.split("_")[1]) if p.name.startswith("iter_") else float("inf"))
        final = [p for p in checkpoints if "last" in p.name or "final" in p.name]
        if len(final) != 1:
            raise ValueError(f"Expected one final checkpoint, found {final}.")
        selected = final if only_final_model else list(dict.fromkeys(
            [checkpoints[0], checkpoints[len(checkpoints) // 2], final[0]]))
        for checkpoint in selected:
            inference_beta_vae(
                dataset=DatasetName(dataset_name), number_of_dims=number_of_dims,
                only_latent_and_output=only_latent_and_output, checkpoint_path=checkpoint,
                output_dir=directory, random_model=False, latent_dim=config["z_dim"],
                model_type=config["model"], batch_size=config["batch_size"],
                device=device, num_workers=num_workers, seed=seed, decoder_sampling=decoder_sampling)
        return [directory / f"intermediate_outputs_{p.name}.npz" for p in selected]

    return run_experiment(output_path, action, requires=[model_path])


@app.command()
def run_inferences(training_path: Path = typer.Option(...), output_path: Path = typer.Option(...),
                   only_final_model: bool = True, only_latent_and_output: bool = True,
                   skip_done: bool = True, device: str = "cuda:0", num_workers: int = 2,
                   seed: Optional[int] = None, decoder_sampling: bool = True) -> None:
    for model_path, dataset_name, number_of_dims in tqdm(list(model_directories(training_path))):
        inference_on_model(output_path / model_path.name, model_path, only_final_model,
                           only_latent_and_output, DatasetName(dataset_name), "beta_vae",
                           number_of_dims, skip_done, device=device, num_workers=num_workers,
                           seed=seed, decoder_sampling=decoder_sampling)


if __name__ == "__main__":
    app()
