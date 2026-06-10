import pickle
from pathlib import Path

import numpy as np
import seaborn as sns
import pandas as pd
from PIL import Image
from tqdm import tqdm
import typer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from microscope.computations_grid.data_analysis.data_analysis import clip_results_multiclass

app = typer.Typer(pretty_exceptions_enable=False)


def merge_layer_evolution(
    plots_intermediate_outputs_dir: Path, steps: list[str],
    merge_output_dir: Path,
    dpi: int,
    figsize: tuple[int, int] = (40, 20)
) -> None:
    for scale in ["linear", "symlog"]:
        image_paths = list((plots_intermediate_outputs_dir / f"layer_evolution/{scale}").glob("*.png"))
        step_to_image_paths = {"_".join(p.stem.split("_")[1:]): p for p in image_paths}

        images = [np.array(Image.open(step_to_image_paths[step])) for step in steps]
        image_height = images[0].shape[0]

        plt.figure(figsize=figsize)
        plt.imshow(np.vstack(images))
        plt.yticks(np.arange(start=image_height // 2, stop=image_height * len(steps), step=image_height), steps)
        plt.savefig(merge_output_dir / f"layer_evolution_{scale}.png", dpi=dpi)


def merge_distributions(
    plots_intermediate_outputs_dir: Path,
    merge_output_dir: Path,
    steps: list[str],
    encoder_layers: list[str],
    sprites: list[str],
    dpi: int,
    figsize: tuple[int, int] = (40, 40),
    n_measures: int = 3,
    alpha: float = 0.1,
    quantile_margin: float = 0.1,
    cmap: matplotlib.colors.LinearSegmentedColormap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "red_to_blue",
        ["b", "r"]
    )
) -> None:
    measures_paths = list((plots_intermediate_outputs_dir / "measures").glob("*.pkl"))

    paths_per_step_layer = dict(
        [
            (
                (
                    path.stem.replace("1_5M", "15M").split("_")[1].replace("15M", "1_5M"),
                    "_".join(path.stem.replace("1_5M", "15M").split("_")[2:])
                ),
                path
            )
            for path in measures_paths
            if path.stem != "inputs"
        ]
        + [((step, "inputs"), [path for path in measures_paths if path.stem == "inputs"][0]) for step in steps]
    )

    # noinspection PyUnresolvedReferences

    sns.set(font_scale=1.5)
    sns.set_theme(style='white')

    for scale in ["linear", "symlog"]:
        _, axes = plt.subplots(nrows=n_measures * len(sprites), ncols=len(steps), figsize=figsize)

        for col, step in tqdm(enumerate(steps), total=len(steps)):
            for layer_n, layer in enumerate(encoder_layers):
                path = paths_per_step_layer[(step, layer)]
                with open(path, "rb") as f:
                    analysis_results = pickle.load(f)

                analysis_results = clip_results_multiclass(analysis_results, quantile_margin=quantile_margin)

                for n_sprite, sprite in enumerate(sprites):
                    volumes = analysis_results.measures.normalized_volume_elements[sprite].flatten()
                    curvatures = analysis_results.measures.normalized_curvatures[sprite].flatten()
                    reaches = analysis_results.measures.normalized_reaches[sprite].flatten()

                    measures = dict(
                        volume=volumes,
                        curvature=curvatures,
                        reach=reaches
                    )
                    for row, measure in enumerate(["volume", "curvature", "reach"]):
                        ax = axes[len(sprites) * row + n_sprite][col]
                        ax.set_title(f"{step}, {sprite}")

                        color = cmap(layer_n / len(encoder_layers))
                        sns.kdeplot(
                            pd.DataFrame({measure: measures[measure]}),
                            x=measure,
                            fill=True,
                            alpha=alpha,
                            color=color,
                            linewidth=0.2,
                            label=encoder_layers[layer_n],
                            ax=ax
                        )

        for ax in axes:
            for a in ax:
                a.legend()
                a.set_xscale(scale)
                a.set_yscale(scale)
        plt.tight_layout()
        plt.savefig(merge_output_dir / f"distributions_{scale}.png", dpi=dpi)


def merge_projection_classes(
    plots_intermediate_outputs_dir: Path,
    merge_output_dir: Path,
    steps: list[str],
    layers: list[str],
    dpi: int,
    figsize: tuple[int, int] = (45, 20)
) -> None:
    projection_class_paths = list((plots_intermediate_outputs_dir / "projection_classes").glob("*.png"))

    paths_per_step_layer_proj_class = dict(
        [
            (
                (
                    path.stem.replace("1_5M", "15M").split("_")[1].replace("15M", "1_5M"),
                    "_".join(path.stem.replace("1_5M", "15M").split("_")[2:])
                ),
                path
            )
            for path in projection_class_paths
            if path.stem != "inputs"
        ]
        + [((step, "inputs"), [path for path in projection_class_paths if path.stem == "inputs"][0]) for step in steps]
    )

    _, axes = plt.subplots(nrows=len(steps), ncols=len(layers), figsize=figsize)

    for row, step in tqdm(enumerate(steps), total=len(steps)):
        for col, layer in enumerate(layers):
            ax = axes[row][col]
            path = paths_per_step_layer_proj_class[(step, layer)]
            image = np.array(Image.open(path))

            ax.set_title(f"{step}, {layer}")
            ax.imshow(image)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(merge_output_dir / "projection_classes.png", dpi=dpi)


def merge_projection_measures(
    plots_intermediate_outputs_dir: Path,
    merge_output_dir: Path,
    steps: list[str],
    layers: list[str],
    sprites: list[str],
    dpi: int,
    figsize: tuple[int, int] = (80, 40)
) -> None:
    projection_measure_paths = list((plots_intermediate_outputs_dir / "projection_measures").glob("*.png"))

    paths_per_step_layer_proj_measure = dict(
        [
            (
                (
                    path.stem.replace("1_5M", "15M").split("_")[1].replace("15M", "1_5M"),
                    "_".join(path.stem.replace("1_5M", "15M").split("_")[2:-1]),
                    path.stem.split("_")[-1]
                ),
                path
            )
            for path in projection_measure_paths
            if "inputs" not in path.stem
        ]
        + [((step, "inputs", path.stem.split("_")[-1]), path) for step in steps for path in projection_measure_paths if
           "inputs" in path.stem]
    )

    _, axes = plt.subplots(nrows=len(steps) * len(sprites), ncols=len(layers), figsize=figsize)

    for n_step, step in tqdm(enumerate(steps), total=len(steps)):
        for col, layer in enumerate(layers):
            for n_sprite, sprite in enumerate(sprites):
                row = n_sprite * len(steps) + n_step
                ax = axes[row][col]
                path = paths_per_step_layer_proj_measure[(step, layer, sprite)]
                image = np.array(Image.open(path))

                ax.set_title(f"{sprite}, {step}, {layer}")
                ax.imshow(image)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticks([], minor=True)
                ax.set_yticks([], minor=True)
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])

    plt.tight_layout()
    plt.savefig(merge_output_dir / "projection_measures.png", dpi=dpi)


@app.command()
def main(
    data_dir: Path = typer.Option(...),
    dpi: int = 150,
    model: str = "beta-vae"
):
    # Handle directories.
    plots_intermediate_outputs_dir = data_dir / "plots_intermediate_outputs"
    merge_output_dir = plots_intermediate_outputs_dir / "merged_results"
    merge_output_dir.mkdir(exist_ok=True)

    # Define experiment parameters
    match model:
        case "beta-vae":
            steps = ["random", "50k", "450k", "1_5M"]

            layers = [
                "inputs",
                "encoder_relu_5",
                "encoder_relu_7",
                "encoder_relu_10",
                "encoder_relu_12",
                "mu",
                "decoder_relu_1",
                "decoder_relu_3",
                "decoder_relu_5",
                "output"
            ]

            encoder_layers = layers[:6]
        case "mae":
            steps = ["random", "002", "020", "100"]

            layers = [
                "inputs",
                "normalized_tokens",
                "encoder_block_01",
                "encoder_block_03",
                "latent",
                "decoder_block_01",
                "predicted_tokens",
                "reconstructed_images"
            ]

            encoder_layers = layers[:5]
        case "cnn":
            steps = ["random", "50k", "250k", "450k", "1_5M"]

            layers = [
                "inputs",
                "encoder_relu_5",
                "encoder_relu_7",
                "encoder_relu_10",
                "encoder_relu_12"
            ]

            encoder_layers = layers
        case _:
            raise ValueError(f"Unknown model type: {model}.")

    # TODO: This is temporary given the discrepancy with the triangle instead of ellipse. When resolved, remove it.
    sprites = (
        ["square", "ellipse", "heart"]
        if ("_custom_" not in data_dir.name) and model in ["mae", "cnn"]
        else ["square", "triangle", "heart"]
    )

    # Run merge functions
    merge_layer_evolution(
        plots_intermediate_outputs_dir=plots_intermediate_outputs_dir,
        merge_output_dir=merge_output_dir,
        steps=steps,
        dpi=dpi
    )
    merge_distributions(
        plots_intermediate_outputs_dir=plots_intermediate_outputs_dir,
        merge_output_dir=merge_output_dir,
        steps=steps,
        encoder_layers=encoder_layers,
        sprites=sprites,
        dpi=dpi
    )
    merge_projection_classes(
        plots_intermediate_outputs_dir=plots_intermediate_outputs_dir,
        merge_output_dir=merge_output_dir,
        steps=steps,
        layers=layers,
        dpi=dpi
    )
    merge_projection_measures(
        plots_intermediate_outputs_dir=plots_intermediate_outputs_dir,
        merge_output_dir=merge_output_dir,
        steps=steps,
        layers=layers,
        sprites=sprites,
        dpi=dpi
    )


if __name__ == "__main__":
    app()
