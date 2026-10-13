# Paper reproduction

This repository accompanies [*The Data Manifold under the Microscope*, Marios Koulakis and Constantin Seibold, ICML 2026](https://proceedings.mlr.press/v306/koulakis26a.html). Cite the paper when using the framework; cite source datasets and adapted methods as appropriate.

**Status:** Figure 5 has been rerun, checked and reviewed by the author; Figures 3, 4 and 6 still require end-to-end validation. Reproduction aims to obtain very similar plots and trends across runs, allowing small sampling and numerical deviations without requiring seed-level agreement. Package version `0.0.1` alone does not distinguish subsequent source changes: record the Git revision. A validated reproduction release will need frozen source, inputs, configurations and environment details.

Recent maintenance added CPU fallback to device-bearing grid APIs and corrected second-kind Riemann differentiation, signed curvature integration and subsampled reach output shapes. Comparisons with older results must account for these changes; their effect on the paper results has not yet been measured.

## Setup and entry points

**Warning:** MMLS now defaults to the faster batched thin SVD (`thin_svd`).
This can produce slight numerical deviations from the original covariance SVD
and the published plots. To select the preserved original implementation, pass
`--projection-method original` to a paper, fitting, training or toy command.

From a clone, install experiment dependencies with `python -m pip install -e '.[benchmark]'`. This extra is declared in [pyproject.toml](../pyproject.toml); it is not needed for the README example. Dependencies are not an exact reproduction lockfile.

Custom dSprites is generated locally. For COIL-20, download the processed images from [Columbia's dataset page](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php), extract them and set `COIL20_PATH` to the directory containing `obj1__0.png` and the other images. The loader expects 20 objects × 72 grayscale 128×128 views.

| Stage | Source |
| --- | --- |
| Toy fitting and measures | [fit_and_get_measures.py](../experiment_scripts/toy_manifolds_experiment/fit_and_get_measures.py) |
| Image MMLS / β-VAE training | [training.py](../experiment_scripts/manifold_fitting/training.py), [model_configs.py](../experiment_scripts/model_configs.py) |
| Grid measures and exports | [run_data_analysis.py](../microscope/computations_grid/data_analysis/run_data_analysis.py) |
| Figures 3–6 recipe | [paper.py](../experiment_scripts/reproduction/paper.py), [commands and options](../experiment_scripts/reproduction/README.md) |
| Bound curves | [bounds.py](../experiment_scripts/reproduction/bounds.py), [methods notebook](../notebooks/manifold_fitting/bound_curves_all_methods.ipynb) |

Existing experiment commands, retained as starting configurations (not verified paper-reproduction recipes):

```bash
python -m experiment_scripts.toy_manifolds_experiment.fit_and_get_measures \
    --output-path results/toy_mmls --n-range 25 505 5 \
    --n-examples-per-size 20 --n-ground-truth 1000 --max-workers 5 \
    --fitting-method MMLS

python -m experiment_scripts.toy_manifolds_experiment.fit_and_get_measures \
    --output-path results/toy_autoencoder --n-examples-per-size 5 \
    --n-ground-truth 1000 --max-workers 5 \
    --fitting-method denoising_autoencoder_random_noise

COIL20_PATH=/path/to/coil-20-proc \
python -m experiment_scripts.manifold_fitting.training \
    --output-path results/image_fitting --device cuda:0
```

The toy script writes pickled distance/measure results; the image script exports datasets and loops over selected datasets, fitting methods, dimensions, fractions and repetitions. The inference/analysis wrappers discover the actual six-field directories and require completion markers; incomplete directories warn and skip. The modular Figures 3–6 recipe and configurable training/inference options are documented in [the reproduction entry points](../experiment_scripts/reproduction/README.md). Device selection is forwarded through the reproduction fitting, toy and analysis paths. Focused CPU regressions, reduced CUDA toy runs and the full recovered Figure 5 recipe passed on 2026-10-13 using `thin_svd`, now the default. Figure 5's output counts/finiteness, rendering and resume behavior were checked; the author found the plots very similar and accepted the small differences. Figures 3, 4 and 6 have not been run end to end. The methods notebook delegates to the native modules and its stale saved outputs were cleared; the other historical bound notebooks still contain local paths and stored outputs.

For each retained run, save the commit and dirty diff, exact package versions, command/configuration, seeds, input sources/checksums, hardware/device and outputs. Preserve original and corrected results separately and compare target figures/tables quantitatively. Current scripts do not capture all this provenance automatically.

## Validation

With pytest available, run from the repository root:

```bash
# Small analytic, device and regression checks
python -m pytest microscope/computations_grid/tests/test_device.py \
    microscope/computations_grid/tests/test_geometry_small.py \
    microscope/computations_grid/tests/test_reach_small.py \
    microscope/computations_grid/tests/test_data_analysis.py \
    microscope/computations_grid/tests/test_analysis_outputs.py -q

# Full grid suite; some unmarked dense tests take several minutes
python -m pytest microscope/computations_grid -q
# Add --runslow to include the seven normally skipped cases.
```

The README/usage examples and the focused selection above were checked on CPU during the documentation refresh (2026-10-09): **44 passed**, with one existing invalid-patch-size warning. Earlier full-grid validation of the code at `b2f6a40`: **67 passed, 7 skipped**, with the same warning. CUDA, MPS, slow tests and full paper experiments were not validated. These geometry tests establish a numerical baseline, not paper reproduction.
