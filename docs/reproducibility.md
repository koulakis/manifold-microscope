# Paper reproduction

This repository accompanies [*The Data Manifold under the Microscope*, Marios Koulakis and Constantin Seibold, ICML 2026](https://proceedings.mlr.press/v306/koulakis26a.html). Cite the paper when using the framework; cite source datasets and adapted methods as appropriate.

**Status:** Figures 3–6 have now been rerun and checked at the agreed practical scope, and the author accepts their plots and trends as a reasonable paper reproduction. Figure 4 used the documented quick experiment—six sizes and one repetition, with completed Figure 3/6 fits reused—rather than the full 13-size, three-repetition workload. Its full entry point remains available. Reproduction aims for similar plots and trends across runs, allowing sampling and numerical deviations without requiring seed-level agreement. Package version `0.0.1` alone does not distinguish subsequent source changes: record the Git revision. A reproduction release still needs frozen source, inputs, configurations and environment details.

Recent maintenance added CPU fallback to device-bearing grid APIs and corrected second-kind Riemann differentiation and subsampled reach output shapes. Their effect on paper results needs separate assessment. Positive/negative curvature aggregation now uses the original paper definition again: each weighted sum is divided by its sign-subset count, then multiplied by the full parameter-domain volume. Zeros belong to the nonnegative subset; empty subsets produce NaN and NumPy warnings. These two quantities need not sum to total curvature. The corrected full-grid contribution definition is deferred until after the reproduction release.

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

The toy script writes pickled distance/measure results; the image script exports datasets and loops over selected datasets, fitting methods, dimensions, fractions and repetitions. The inference/analysis wrappers discover the actual six-field directories and require completion markers; incomplete directories warn and skip. The modular Figures 3–6 recipe and configurable training/inference options are documented in [the reproduction entry points](../experiment_scripts/reproduction/README.md). Device selection is forwarded through the reproduction fitting, toy and analysis paths. Focused CPU regressions, reduced CUDA toy runs and full recovered Figure 3/5 recipes passed on 2026-10-13 using `thin_svd`, now the default. Output counts/finiteness, rendering and resume behavior were checked. Figure 3's count grids match the retained original curves and its full-training mean errors agree within 3.4e-6; partial-data differences can be larger (about 43% at the 2D 90% point). The author found Figure 5 very similar and accepted its small differences. Figure 6 completed through per-layer GPU recovery and was replotted with the restored historical curvature aggregation; the author accepted the remaining learned-layer variation as plausible training variation. Figure 4's quick run completed on 2026-10-15: five new VAE trainings, six evaluations, export and rendering passed structural, finite-value and visual checks. Its fitted decay exponents are 0.240 versus 0.220 in the retained paper data for MMLS and 0.127 versus 0.114 for beta-VAE. Differences in the small fitted lower curve are expected because the quick run omits the difficult 1%, 2% and 10% points and uses one repetition. The full 13x3 recipe remains unrun. The methods notebook delegates to the native modules and its stale saved outputs were cleared; the other historical bound notebooks still contain local paths and stored outputs.

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
