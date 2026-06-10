import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import typer
from tqdm import tqdm

from experiment_scripts.toy_manifolds_experiment.manifold_fitting_denoising_autoencoder import DenoisingAutoencoder
from microscope.datasets import toy_manifolds
from experiment_scripts.toy_manifolds_experiment.manifold_fitting_no_noise import ANNMMLSProjector

app = typer.Typer(pretty_exceptions_enable=False)


def compute_distances(
    manifold: toy_manifolds.Manifold,
    ground_truth: np.ndarray,
    N: int,
    sigma: float = 0.0001,
    fitting_method: str = "MMLS",
    normalized: bool = False,
    error_weight: float = 1.0
):
    match fitting_method:
        case "MMLS":
            samples = manifold.sample(N, normalized=normalized)

            d = manifold.dimension
            k = 5
            projector = ANNMMLSProjector(samples, d=d, k=k)
            prediction = projector.project(ground_truth)
        case "denoising_autoencoder_normal_noise":
            samples, unitary_normals = manifold.sample(
                N,
                normalized=normalized,
                return_unitary_normal=True
            )

            match manifold.__class__:
                case toy_manifolds.Circle | toy_manifolds.Sphere:
                    manifold: toy_manifolds.Circle | toy_manifolds.Sphere
                    R = manifold.R
                    noise_std = 0.2*R
                case toy_manifolds.Moons:
                    manifold: toy_manifolds.Moons
                    R = manifold.R
                    noise_std = 0.1 * R
                case toy_manifolds.Torus:
                    manifold: toy_manifolds.Torus
                    R, r = manifold.R, manifold.r
                    noise_std = 0.2 * min(R - r, r)
                case _:
                    raise ValueError(f"Unknown manifold class: {manifold.__class__}")

            denoising_ae = DenoisingAutoencoder(
                input_dim=manifold.dimension + 1,
                noise_std=noise_std,
                lr=1e-3,
                hidden_channels=[128, 128, 128],
            )
            denoising_ae.fit(
                samples,
                unitary_normals,
                # train_loss_stop_threshold=1e-2 * noise_std ** 2
                train_loss_stop_threshold=error_weight * noise_std ** 2
            )
            prediction = denoising_ae.predict(ground_truth)
        case "denoising_autoencoder_random_noise":
            samples = manifold.sample(
                N,
                normalized=normalized,
                return_unitary_normal=False
            )

            match manifold.__class__:
                case toy_manifolds.Circle | toy_manifolds.Sphere:
                    manifold: toy_manifolds.Circle | toy_manifolds.Sphere
                    R = manifold.R
                    noise_std = 0.1 * R
                case toy_manifolds.Moons:
                    manifold: toy_manifolds.Moons
                    R = manifold.R
                    noise_std = 0.1 * R
                case toy_manifolds.Torus:
                    manifold: toy_manifolds.Torus
                    R, r = manifold.R, manifold.r
                    noise_std = 0.1 * min(R - r, r)
                case _:
                    raise ValueError(f"Unknown manifold class: {manifold.__class__}")

            denoising_ae = DenoisingAutoencoder(
                input_dim=manifold.dimension + 1,
                noise_std=noise_std,
                lr=1e-3,
                hidden_channels=[128, 128, 128],
            )
            denoising_ae.fit(
                samples,
                unitary_normals=None,
                train_loss_stop_threshold=error_weight*noise_std ** 2
            )
            prediction = denoising_ae.predict(ground_truth)
        case _:
            raise ValueError(f"Unknown manifold fitting method: {fitting_method}.")

    distances = np.linalg.norm(ground_truth - prediction, axis=-1)

    return distances, distances.max()


def run_size_ablation(
    manifold: toy_manifolds.Manifold,
    ground_truth: np.ndarray,
    N_range: np.ndarray = np.arange(10, 200),
    n_examples_per_size=20,
    sigma: float = 0.0001,
    fitting_method: str = "MMLS",
    normalized: bool = True,
    error_weight: float = 1.0,
    parallel=True,
    max_workers=None
):
    avg_all, haus_all = {}, {}

    for N in tqdm(N_range):
        N: int
        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        compute_distances,
                        manifold=manifold,
                        ground_truth=ground_truth,
                        N=N,
                        sigma=sigma,
                        fitting_method=fitting_method,
                        normalized=normalized,
                        error_weight=error_weight
                    )
                    for _ in range(n_examples_per_size)
                ]
                results = [f.result() for f in as_completed(futures)]
        else:
            results = [
                compute_distances(
                    manifold=manifold,
                    ground_truth=ground_truth,
                    N=N,
                    sigma=sigma,
                    fitting_method=fitting_method,
                    normalized=normalized,
                    error_weight=error_weight
                )
                for _ in range(n_examples_per_size)
            ]

        avg, haus = zip(*results)
        avg_all[N] = avg
        haus_all[N] = haus

    return avg_all, haus_all


@app.command()
def fit_and_get_measures(
    output_path: Path = typer.Option(...),
    N_range: tuple[int, int, int] = (25, 505, 10),
    n_examples_per_size: int = 20,
    N_ground_truth: int = 500,
    max_workers: int = 8,
    fitting_method: str = "MMLS",
    normalized: bool = True,
    error_weight: float = 1.0
):
    output_path.mkdir(exist_ok=True, parents=True)

    results = {}
    for manifold_class in [
        toy_manifolds.Circle,
        toy_manifolds.Moons,
        toy_manifolds.Sphere,
        toy_manifolds.Torus
    ]:
        manifold = manifold_class()
        name = str(manifold)
        print(f"Fitting on {name}.")

        ground_truth = manifold.tesselation(N=N_ground_truth, normalized=normalized)
        measures = manifold.measures(normalized=normalized)
        measures_pointwise_on_gt = manifold.measures_pointwise(ground_truth, normalized=normalized)

        avg_all, haus_all = run_size_ablation(
            manifold=manifold,
            ground_truth=ground_truth,
            N_range=np.concatenate([np.arange(5, N_range[0]), np.arange(*N_range)]),
            n_examples_per_size=n_examples_per_size,
            fitting_method=fitting_method,
            normalized=normalized,
            error_weight=error_weight,
            max_workers=max_workers
        )

        results[name] = dict(
            average_distances=avg_all,
            hausdorff_distances=haus_all,
            measures=measures,
            measures_pointwise_on_gt=measures_pointwise_on_gt
        )
        with open(output_path / f"distance_and_measure_results_{name}.pkl", "wb") as f:
            pickle.dump(results, f, -1)

    with open(output_path / "distance_and_measure_results.pkl", "wb") as f:
        pickle.dump(results, f, -1)


if __name__ == "__main__":
    app()
