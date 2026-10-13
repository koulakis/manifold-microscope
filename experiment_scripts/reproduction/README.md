# Paper experiments

This package adds small entry points around the existing dataset, MMLS,
beta-VAE and geometry implementations.
The high-level recipe targets Figures 3–6 and retains the final submission
plotting styles. **These new scripts have not been executed or tested.**

Run commands from the new repository root using the project's environment.
On the desktop use `PYENV_VERSION=curvatures pyenv exec python`; on the Mac
use the corresponding `microscope` environment. The existing benchmark extra
declares the runtime dependencies; nothing here installs packages.

## High-level recipe

```bash
PYENV_VERSION=curvatures pyenv exec python -m experiment_scripts.reproduction.paper \
  --output-root ../outputs/PAPER_REPRODUCTION --figures 3 4 5 6
```

This command selects the full paper workloads and defaults to CUDA device 0.
`--device cpu` is forwarded through fitting, fixed splits, toy surface relaxation,
inference and grid analysis. These newly connected paths have not been run;
full workloads remain substantial on either device. Supply `COIL20_PATH`
pointing to the processed COIL-20 PNG directory when using that dataset.

Run stages separately against the same root:

```bash
python -m experiment_scripts.reproduction.paper --output-root ../outputs/PAPER_REPRODUCTION --stages fit
python -m experiment_scripts.reproduction.paper --output-root ../outputs/PAPER_REPRODUCTION --stages evaluate
python -m experiment_scripts.reproduction.paper --output-root ../outputs/PAPER_REPRODUCTION --stages infer analyze
python -m experiment_scripts.reproduction.paper --output-root ../outputs/PAPER_REPRODUCTION --stages export plot
```

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
The cleanup branch also includes fixes to signed curvature integration,
subsampled reach shapes and second-kind Riemann differentiation; their effect
on the historical figure values has not yet been measured.
The bound constants use the actual notebook formulas, not the different
envelope procedure described in the paper prose. Exact numerical agreement
with the original paper is therefore an unverified goal, not a claim.

Completed source fixes and remaining historical/validation work are listed in
[FOLLOW_UPS.md](FOLLOW_UPS.md). Focused regression sources cover FPS/cache behavior,
strict budgets, full checkpoint loading, mean decoding and partial-directory
warnings. They have not been executed.
