# Paper reproduction

This repository accompanies [*The Data Manifold under the Microscope*, Marios Koulakis and Constantin Seibold, ICML 2026](https://proceedings.mlr.press/v306/koulakis26a.html). Cite the paper when using the framework; cite source datasets and adapted methods as appropriate.

**Status:** the current checkout is under maintenance, not a validated reproduction release. Original final-experiment artifacts and an end-to-end figure/table reproduction remain to be checked. Package version `0.0.1` alone does not distinguish subsequent source changes: record the Git revision. A future verified paper-compatible release will need frozen source, inputs, configurations and environment details.

Recent maintenance added CPU fallback to device-bearing grid APIs and corrected second-kind Riemann differentiation, signed curvature integration and subsampled reach output shapes. Comparisons with older results must account for these changes; their effect on the paper results has not yet been measured.

## Setup and entry points

From a clone, install experiment dependencies with `python -m pip install -e '.[benchmark]'`. This extra is declared in [pyproject.toml](../pyproject.toml); it is not needed for the README example. Dependencies are not an exact reproduction lockfile.

Custom dSprites is generated locally. For COIL-20, download the processed images from [Columbia's dataset page](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php), extract them and set `COIL20_PATH` to the directory containing `obj1__0.png` and the other images. The loader expects 20 objects × 72 grayscale 128×128 views.

| Stage | Source |
| --- | --- |
| Toy fitting and measures | [fit_and_get_measures.py](../experiment_scripts/toy_manifolds_experiment/fit_and_get_measures.py) |
| Image MMLS / β-VAE training | [training.py](../experiment_scripts/manifold_fitting/training.py), [model_configs.py](../experiment_scripts/model_configs.py) |
| Grid measures and exports | [run_data_analysis.py](../microscope/computations_grid/data_analysis/run_data_analysis.py) |
| Bound curves | [manifold_fitting notebooks](../notebooks/manifold_fitting) |

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

The toy script writes pickled distance/measure results; the image script exports datasets and loops over both datasets, fitting methods, dimensions and training ratios. Inspect its configuration before launching: these are substantial runs, and some helpers still assume CUDA. Existing inference/analysis scripts are not a complete pipeline: `manifold_fitting/inference.py` imports the absent `representation_learning.mae` module and uses experiment naming that differs from training. Bound notebooks contain local artifact paths; final paper plots were generated separately from exported curves. Stored notebook outputs have not been regenerated for the fixes.

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
