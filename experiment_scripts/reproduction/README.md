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

This command selects the full paper workloads. The reproduction toy/analysis
entry points currently require CUDA device 0; explicit device forwarding through
their existing wrappers remains a follow-up. The underlying grid APIs on the
cleanup branch already select CUDA or CPU automatically. Supply `COIL20_PATH`
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
  --max-iter 1 --max-epochs 1 --num-workers 0 \
  --output ../outputs/MANUAL_EXPERIMENT/vae

python -m experiment_scripts.reproduction.evaluation \
  --model-dir ../outputs/MANUAL_EXPERIMENT/vae \
  --output ../outputs/MANUAL_EXPERIMENT/evaluation
```

Separate modules also expose toy fitting (`toy`), layer inference/analysis
(`layers infer` / `layers analyze`), bound export (`bounds`) and rendering
(`plots`). They can be called as ordinary functions from another experiment.
The existing Solver finishes an epoch before stopping, so `--max-iter` is
not a strict number of batches. Individual fitting/evaluation/inference can
select CPU; the reproduction analysis and toy entry points still require CUDA.

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

Required follow-up work in existing files is listed in [FOLLOW_UPS.md](FOLLOW_UPS.md).
