# The Data Manifold Under the Microscope

`manifold-microscope` provides a framework for studying and benchmarking data manifolds through densely sampled
grid-based datasets and finite-difference geometric computations. The goal of the library is to make it practical to
construct measurable reference manifolds, compute geometric quantities such as curvature, reach, and volume, and use
those quantities when evaluating manifold fitting, generalization bounds, and geometric estimation methods.

## Installation

Install the library from PyPI with:

```bash
pip install "manifold-microscope"
```

To also install the benchmark dependencies, use:

```bash
pip install "manifold-microscope[benchmark]"
```

For local development from a cloned repository, use `pip install -e .` at the top level of the project. A GPU with at
least 5 GB of GPU memory and 30 GB of system RAM is recommended for dense image-dataset experiments. For the paper
experiments, we used an RTX 4090 with 24 GB of GPU memory on a server with 96 GB of system RAM. To run the unit tests,
run `pytest` at the top level of the project.

## Paper

This repository accompanies the paper **The Data Manifold under the Microscope**, accepted to **ICML 2026**. The work
will also be presented at the **Mechanistic Interpretability Workshop** at ICML 2026: https://mechinterpworkshop.com

A temporary arXiv version is available here: https://arxiv.org/abs/2606.15760

> [!CAUTION]
> **This project is under active construction.** The code runs and can be used to explore the framework, but it still
> needs more updates before it can be relied on to reproduce the original paper results or to support new research
> projects.

## Repository contents and API scope

This repository provides a usable Python library for working with grid-sampled data manifolds and finite-difference
geometric measurements. The most stable, library-facing parts of the codebase are the dataset generation tools and the
grid-based geometry computations. The manifold fitting and bounds evaluation code is also included, mainly as reference
material from the accompanying paper and as example usage of the framework, but it is less polished as a public API.

Here is a map of the main components:

- Datasets
  - The notebooks under `notebooks/datasets_and_measures` contain examples of loading and visualizing the toy and image datasets.
  - The toy dataset code is in `microscope/datasets/toy_manifolds.py`.
  - The image dataset code is in `microscope/datasets/custom_dsprites.py` and `microscope/datasets/coil20.py`.

- Geometric measures
  - The notebook `notebooks/datasets_and_measures/toy_manifold_datasets.ipynb` contains example computations and visualizations of the geometric measures.
  - The finite-difference computations are under `microscope/computations_grid`. `basic.py` contains basic operations such as partial derivatives and the Riemannian metric; `volume.py`, `curvature.py`, and `reach.py` build on top of it.
  - The functions are tested in `microscope/computations_grid/tests`. These tests use symbolic reference computations from `microscope/manifold_examples` for simple manifolds.

- Manifold fitting and bounds
  - The MMLS fitting method is in `experiment_scripts/manifold_fitting/mmls.py`.
  - The denoising autoencoder used for the toy datasets is in `experiment_scripts/toy_manifolds_experiment/manifold_fitting_denoising_autoencoder.py`.
  - The beta-VAE components are under `representation_learning/beta_vae`.
  - The scripts running the main experiments are `experiment_scripts/manifold_fitting/training.py` for the image datasets and `experiment_scripts/toy_manifolds_experiment/fit_and_get_measures.py` for the toy datasets.
  - The notebooks under `notebooks/manifold_fitting` show how to generate some of the paper results, including the bound curves for MMLS, the dSprites dimensionality experiments, and comparisons between fitting methods. These notebooks are mainly intended as reference material; the final paper plots were generated separately from the exported curves.

---

## Why Use the Microscope?

In most research settings, one must choose between idealized mathematical manifolds (e.g., spheres, ellipsoids) with unrealistic simplicity, or real-world datasets where true geometric quantities are unknown or hard to measure accurately for testing. This framework bridges that gap by offering datasets that are both structured and realistic, yet fully measurable.

For instance, if you derive a new generalization or manifold fitting bound involving curvature or volume, you can directly test how tight it is under controlled geometric conditions. Similarly, if you develop a curvature estimation algorithm, you can benchmark its performance on datasets where the true curvature is exactly known. Randomly sampling points from the provided grids lets you simulate realistic sparse sampling scenarios and directly compare estimates to ground truth.

## Datasets

The two main image datasets included are:

- **dSprites (grid generator)** – generates synthetic images with controllable grid density, image size, and transformations.  
- **COIL-20 (augmented generator)** – extends the original dataset (to be downloaded separately) with controlled xy-rotations and rescaling.

Additionally, four toy datasets are provided for smaller experiments: 

- **Circle** - A circle embedded in 2D.
- **Moons** - Two semicircles close to each other. It is practically the same as the moons dataset from sklearn. 
- **Sphere** – A sphere embedded in 3D.  
- **Torus** – A torus in 3D. Has slightly more complex topology and nonuniform geometric measures.  


All datasets are densely sampled on a grid, where each grid direction corresponds to a transformation axis. This limits practical dimensionality to about 4–5 directions but provides highly accurate geometric values which can be used as ground truth. Datasets can be loaded as full grids or sampled uniformly with respect to the local volume element. For the toy datasets, the geometric measures are computed directly using the corresponding closed-form formulas.

For example usages and visualizations of the datasets look in the notebooks in `notebooks/datasets_and_measures`.

## Geometric Measures

Finite-difference operators are used to compute geometric quantities directly on the grid—making the framework accurate, stable, and differentiable.

Available measures include:

- Volume and volume element
- Tangent spaces and the Riemannian metric tensor 
- Scalar curvature
- Reach along with a pointwise version of it.

All computations can be executed on GPU, allowing fast analysis even for dense grids.

Example computations of the measures can be found in the following notebook: `notebooks/datasets_and_measures/toy_manifold_datasets.ipynb`.

Long term, the goal is to extend this module to include geodesic distances, exponential maps, and other advanced differential quantities.

## Manifold Fitting Bounds

This section provides reference experiments used in the accompanying paper to validate theoretical manifold fitting
bounds. These scripts and notebooks are kept in this repository for now so that the original experiments remain visible,
but this part of the codebase is still being reorganized and may move to a separate public reproducibility repository.

Two types of models are used to approximate the reference manifolds:

- Moving Least Squares (MMLS): A classical local manifold fitting algorithm used to recover smooth embeddings from sampled data.
- $\beta$-VAE: A deep generative model trained to learn a low-dimensional latent manifold consistent with the data geometry and using it to reconstruct a full data manifold.

The $\beta$-VAE implementation in `representation_learning/beta_vae` is copied and adapted from
[1Konny/Beta-VAE](https://github.com/1Konny/Beta-VAE), a PyTorch reproduction of the $\beta$-VAE models from Higgins
et al. (2017) and Burgess et al. (2018). The upstream project is distributed under the MIT License; the original
copyright and license text are included in `representation_learning/beta_vae/LICENSE`, and a package-level attribution
notice is included in `representation_learning/beta_vae/NOTICE.md`.

The results are compared to theoretical bounds proposed by Fefferman, Narayanan & Mitter (2016) and Genovese et al. (2012), assessing their tightness and dependence on curvature, reach, and sample density.

To reproduce the fitting of the manifolds:

- For the toy datasets run:
  ```
  python experiment_scripts/toy_manifolds_experiment/fit_and_get_measures.py \
      --output-path <path to output dir>/toy_datasets_fitting_mmls \
      --n-range 25 505 5 \
      --n-examples-per-size 20 \
      --n-ground-truth 1_000 \
      --max-workers 5 \
      --fitting-method MMLS
  
  python experiment_scripts/toy_manifolds_experiment/fit_and_get_measures.py \
      --output-path <path to output dir>/toy_dataset_fitting_denoising_autoencoder \
      --n-examples-per-size 5 \
      --n-ground-truth 1_000 \
      --max-workers 5 \
      --fitting-method denoising_autoencoder_random_noise
  ```
- For the image datasets run:
  ```
  COIL20_PATH=<path where you extracted coil-20-proc> python experiment_scripts/manifold_fitting/training.py --output-path <output path>
  ```

Examples of generated results can be found in the three notebooks under `notebooks/manifold_fitting`.
