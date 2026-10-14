# Paper experiments

This package adds small entry points around the existing dataset, MMLS,
beta-VAE and geometry implementations.
The high-level recipe targets Figures 3–6 and retains the final submission
plotting styles. Focused CPU regressions and reduced CUDA toy runs have passed.
The full Figure 3 and Figure 5 recipes with the faster projector completed on
CUDA. Figure 6 training and recovery analysis/rendering also completed; its
learned-layer agreement with the paper still needs review. Figure 4 remains unrun.

Run commands from the new repository root using the project's environment.
On the desktop use `PYENV_VERSION=curvatures pyenv exec python`; on the Mac
use the corresponding `microscope` environment. The existing benchmark extra
declares the runtime dependencies; nothing here installs packages.

## High-level recipe

**Warning:** MMLS uses the faster batched `thin_svd` projector by default.
Expect slight numerical deviations from the published plots due to SVD
roundoff and different random samples. The goal is to rerun the experiments
and obtain very similar plots and trends; seed-level agreement is not required.

```bash
PYENV_VERSION=curvatures pyenv exec python -m experiment_scripts.reproduction.paper \
  --output-root ../outputs/PAPER_REPRODUCTION --figures 3 4 5 6
```

This command selects the full paper workloads and defaults to CUDA device 0.
`--device cpu` is forwarded through fitting, fixed splits, toy surface relaxation,
inference and grid analysis. Full workloads remain substantial on either device.
Supply `COIL20_PATH`
pointing to the processed COIL-20 PNG directory when using that dataset.

Run stages separately against the same root:

```bash
python -m experiment_scripts.reproduction.paper --output-root ../outputs/PAPER_REPRODUCTION --stages fit
python -m experiment_scripts.reproduction.paper --output-root ../outputs/PAPER_REPRODUCTION --stages evaluate
python -m experiment_scripts.reproduction.paper --output-root ../outputs/PAPER_REPRODUCTION --stages infer analyze
python -m experiment_scripts.reproduction.paper --output-root ../outputs/PAPER_REPRODUCTION --stages export plot
```

To use the preserved historical covariance SVD, add `--projection-method original`
to the paper, fitting, training or toy command, or set
`MMLSConfig.projection_method='original'`. The default `thin_svd` path uses a
batched thin SVD of weighted centered neighbors; its right singular vectors
span the same tangent space on regular inputs. Weights and neighbor selection
are retained, and an ANN index is built only when the existing rule uses it.
Roundoff and degenerate eigenspaces can differ. `MMLSConfig.projection_batch_size`
defaults to 32 and can be set through JSON overrides. Each MMLS/toy config
records the method. Use a fresh output root when switching implementations.

On 2026-10-13, the recovered Figure 5 workload completed on an RTX 4090:
65 image fits, four toy classes with 1,016 sizes and 20 repetitions each,
bound export and PNG/PDF rendering. Reusing completed dataset exports, it took
about 26 minutes 38 seconds. Output counts/finiteness and resume preservation
were checked. Small real-image comparisons with the original projector differed
by at most about 1.4e-5 per pixel. The author reviewed the generated Figure 5 PDF
and accepted its small differences as consistent with the reproduction goal.
The original full run was stopped before its first fit
completed. Torus sampling still performs scalar root solves on CPU.

The Figure 3 dimension ablation also completed on 2026-10-13 in about 6 minutes
6 seconds, reusing the 4D dataset export. It produced 141 fits: 27/36/39/39 for
dimensions 1/2/3/4, with three repetitions per feasible training size. The 1D
10% and 2D 1% fractions have too few neighbors and are skipped, matching the
retained original plot's count grids. Test counts are 14/230/500/500. Export,
rendering and resume preservation passed. Full-training mean errors differ
from retained originals by less than 3.4e-6. Partial-training differences are
larger: the largest measured mean error difference is about 43% at the 2D 90%
point (2.865 versus 2.000). Subsampling is unseeded; exact agreement between
individual partial-training points is not required by the reproduction goal.

For runs importing optional KeOps, `KEOPS_CACHE_FOLDER` can point to a directory
inside the output root, alongside any configured Matplotlib/Numba/Python caches.

`--figures` selects a subset. Figure 3 uses dSprites MMLS dimensions 1–4;
Figure 4 adds 4D beta-VAE fitting/evaluation; Figure 5 uses sphere, torus,
3D COIL and 4D dSprites MMLS; Figure 6 uses dSprites repetition 2 and COIL
repetition 0 at fraction 1.0. `--device` and `--num-workers` are configurable.
Training, inference, export and plotting outputs share a root with these folders:

```text
datasets/<dataset>__<dimension>/
fits/<dataset>__<method>__<fraction>__<dimension>__0__<repetition>/
evaluation/<same-model-name>/
toy/
inference/{dsprites,coil}/
analysis/{dsprites,coil}/
bounds/figure_{3,4,5}/
figures/figure_{3,4,5,6}/figure_<number>.{png,pdf}
```

Figure 6 also keeps the separate row PDFs/PNGs. The combined PDF is assembled
from 300-DPI row images; the separate row PDFs retain the renderer's vector
output. Bounds export includes actual repetition counts and regression values.

## Individual experiments

### Recover Figure 6 after completed training

For machines with limited host RAM, use a fresh recovery root and run one layer
per process. This reuses completed inference when available, performs missing
inference from the completed checkpoints, then analyzes each layer and renders
Figure 6. It never trains a model or writes into the source run:

```bash
PYENV_VERSION=curvatures pyenv exec python -u -m experiment_scripts.reproduction.recover_figure6 run \
  --input ../outputs/PAPER_REPRODUCTION \
  --output ../outputs/PAPER_FIGURE6_RECOVERY --device cuda:0
```

On Linux with a systemd user manager, prefix the command with
`systemd-run --user --scope -p MemoryMax=64G -p MemorySwapMax=1G` to contain
its memory usage; choose limits that leave ample RAM for the desktop. A limit
failure stops the job and leaves its partial output for inspection.

`analysis_parts/<dataset>/<layer>/COMPLETED` marks each completed layer;
`analysis/<dataset>/measures/` collects links only after all layers finish.
Existing partial directories warn and skip, causing recovery to stop. Preserve
them under a different name before retrying that layer. Completed layers skip
on the next invocation. The recovery path produces the measure pickles and
final Figure 6, omitting the auxiliary reconstruction-distance NPZ and generic
layer-evolution diagnostic plots. Geometry formulas, normalization, patch sizes
and reach subsampling are unchanged. Patch iteration avoids materializing the
whole overlapping patch grid, and derivative patches fill one output array.

The memory fix passed 22 focused CPU cases. On 2026-10-14 the GPU recovery
completed in 23 minutes 56 seconds, excluding the previously completed training
and dSprites inference. All 19 layer outputs (10 dSprites, 9 COIL), class/pair
counts, grid shapes, finite arrays/aggregates and final images were checked.
The original training checksums are unchanged. Both datasets' saved input-layer geometry matches the retained results exactly.
That completed run used full-grid positive/negative curvature contributions,
which differ from the historical sign-subset averages even for identical inputs.
The default now restores the paper definition: average volume-weighted curvature
over points with curvature >= 0 or < 0, then multiply by the full parameter-domain
volume. Zeros count in the nonnegative subset; empty subsets yield NaN and NumPy
warnings. These quantities need not sum to total curvature. The corrected
full-grid definition is deferred until after the reproduction release. Existing
figures and serialized aggregates remain unchanged; replot saved measures into
a fresh output directory to apply the restored definition. The Figure 6 plotter
recomputes aggregates after clipping; no retraining or geometry rerun is needed. Some learned dSprites
magnitudes differ substantially (mean latent volume about 9.4 times the retained
value); scientific agreement needs review. Successful execution alone does not
establish paper equivalence. Actual peak memory was not recorded.

For example, prepare a 1D dataset and fit one model:

```bash
python -m experiment_scripts.reproduction.fitting dataset \
  --dataset custom_dsprites_balanced --dimension 1 --max-test-size 16 \
  --output ../outputs/MANUAL_EXPERIMENT/dataset

python -m experiment_scripts.reproduction.fitting mmls \
  --dataset custom_dsprites_balanced --dimension 1 --fraction 0.5 \
  --datasets-dir ../outputs/MANUAL_EXPERIMENT/dataset \
  --output ../outputs/MANUAL_EXPERIMENT/mmls

python -m experiment_scripts.reproduction.fitting vae \
  --dataset custom_dsprites_balanced --dimension 1 --fraction 0.5 --seed 439 \
  --datasets-dir ../outputs/MANUAL_EXPERIMENT/dataset \
  --max-iter 1 --max-epochs 1 --strict-budget --num-workers 0 \
  --output ../outputs/MANUAL_EXPERIMENT/vae

python -m experiment_scripts.reproduction.evaluation \
  --model-dir ../outputs/MANUAL_EXPERIMENT/vae \
  --output ../outputs/MANUAL_EXPERIMENT/evaluation
```

Separate modules also expose toy fitting (`toy`), layer inference/analysis
(`layers infer` / `layers analyze`), bound export (`bounds`) and rendering
(`plots`). They can be called as ordinary functions from another experiment.
By default Solver keeps historical epoch-finish stopping. Add `--strict-budget`
to VAE fitting to stop at the requested batch/epoch limits. Full trusted local
solver checkpoints load onto the selected device with an explicit loading policy.
`--model-overrides` accepts a JSON object of numerical model parameters for one
fit; dataset/device/path selection stays in the explicit arguments.

Inference defaults to unseeded sampling, storing mu and decoding sampled z.
The layer module accepts `--seed`; the paper coordinator accepts `--inference-seed`.
Both accept `--no-decoder-sampling` to decode mu instead. Mean decoding changes
the experiment from the historical recipe. Seed and sampling choices are saved
with inference outputs; a seed alone does not establish cross-device agreement.

The older training wrapper also supports repeated `--datasets`, `--models`,
`--dimensions`, `--fractions`, `--repetitions` and `--seeds` options, plus workers
and budgets. Seeds correspond in order to selected repetitions; absent explicit
seeds, its historical single seed is reused. For example, a separate repetition:

```bash
python -m experiment_scripts.manifold_fitting.training \
  --output-path ../outputs/SELECTED_EXPERIMENT --datasets custom_dsprites_balanced \
  --models beta_vae --dimensions 1 --fractions 0.5 --repetitions 2 --seeds 473959 \
  --device cpu --num-workers 0 --max-iter 1 --max-epochs 1 --strict-budget
```

Its `--model-overrides` file is keyed by `beta_vae` and/or `MMLS`, for example
`{"beta_vae": {"z_dim": 4}, "MMLS": {"sigma": 0.5}}`. Explicit budget flags
override the file. Older inference/analysis wrappers discover actual six-field
directories, including separate repetitions, and use native JSON configs.
Their compatibility `skip_done` argument cannot force an overwrite: existing
outputs always skip. The methods notebook now calls these reproduction modules.

## Completion and partial results

Each successful experiment writes a one-line `COMPLETED` file after its
essential outputs exist. A directory with that file is skipped. Any existing
directory without it prints a warning and is skipped without changing files.
Exceptions leave the experiment unmarked. Missing completed prerequisites
skip downstream jobs before creating their directories.

For a partial experiment, choose a new output directory or deal with its
contents explicitly on a later step; these scripts never delete/restart it.
Some MMLS fractions can have fewer points than neighbors: the existing fitter
skips them, and their directories remain unmarked. Export uses completed
results only and reports the actual repetitions present. Export requires at
least two distinct positive-error training sizes for every panel.

Markers do not compare configurations. Use a new output root when changing
settings or adding results after an export/plot has already completed.
At the lower-level dataset exporter, a cached split is reused only when its
effective test count matches `min(max_test_size, floor(0.3 * dataset_size))`.
An incompatible cache raises and remains untouched; select a fresh directory.

## Historical scope and follow-ups

The recovered defaults are three dSprites repetitions, two retained COIL
MMLS repetitions, one COIL VAE and twenty toy runs per size. dSprites VAE seeds
are 439/437875/473959; COIL's seed is 438. MMLS and inference preserve their
historically unseeded behavior; inference stores mu but decodes sampled z.
The caption/repetition discrepancies and unknown historical fixes remain.
The cleanup branch also includes fixes to subsampled reach shapes and
second-kind Riemann differentiation; their effect on historical figure values
needs separate assessment. Signed-curvature aggregation follows the original
paper definition described above.
The bound constants use the actual notebook formulas; the paper prose describes
a different envelope procedure. Record these differences when comparing plot
trends. Further figure validation remains necessary; exact numerical agreement
with historical random draws is outside the reproduction goal.

Completed source fixes and remaining historical/validation work are listed in
[FOLLOW_UPS.md](FOLLOW_UPS.md). Focused regression sources cover FPS/cache behavior,
strict budgets, full checkpoint loading, mean decoding and partial-directory
warnings. Twenty-two focused CPU cases passed across the suite and final option
checks; broader geometry and layer pipelines
still require validation against retained paper artifacts.
