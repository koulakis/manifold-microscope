# Experiments and analysis

The components are `data`, `models`, `analyses` and `plotting`. Numerical functions
accept arrays and return results. The scripts handle configuration, files and
completion. Geometry's numerical core remains in `microscope`.

Run from the repository root with the `benchmark` extra installed:

```bash
python -m experiments_and_analysis.scripts.run \
  --config experiments_and_analysis/scripts/configs/single.yaml \
  --output-dir ../outputs/new-experiments

python -m experiments_and_analysis.scripts.sweep \
  --config experiments_and_analysis/scripts/configs/paper/figure4_quick.yaml \
  --output-dir ../outputs/new-experiments --dry-run
```

Omit `--dry-run` to execute the sweep and render its requested figures.
Set `base.device: cuda:0` in a sweep to use CUDA; the default is CPU. Full paper
configurations can take many hours; dry runs only expand configurations. Files
for Figures 3–6 and quick Figure 4 are together in `scripts/configs/paper`.
Quick Figure 4 now computes all missing fits in the new working directory;
it does not import old reproduction outputs.

The same functions accept dictionaries from Python:

```python
from experiments_and_analysis.scripts.run import run
from experiments_and_analysis.scripts.sweep import run as run_sweep

result = run({
    "dataset": {"name": "custom_dsprites_balanced", "dimension": 4},
    "split": {"method": "random", "test_fraction": 0.1, "seed": 42},
    "model": {"name": "beta_vae", "parameters": {"z_dim": 10}},
    "training": {"seed": 439, "trial": 0},
    "analyses": ["errors", {"name": "geometry", "representations": ["mu"]}],
}, "../outputs/new-experiments")
```

`config.load_config` reads YAML or JSON. Defaults are resolved before execution
and exported as JSON. `model.name` accepts `beta_vae`, `mmls` or null. Set
`training.enabled: false` to require an existing matching fit, or use a null
model and input-only geometry to omit fitting/inference entirely. See
`scripts/configs/dataset_geometry.yaml`.

Splits are `random`, `fixed_test` (the paper's farthest-point test selection), or
`full`. `train_fraction` samples from the remaining training pool; `weighted`
uses the dataset's volume weights. Toy paper presets use their provided
sample/tessellation split. Only indices and small sampling metadata are saved.
Normalization retains the historical independent min/max scaling of each
training/evaluation domain; analytic toy coordinates retain their own scaling.

Errors use held-out data by default. Geometry always uses the full ordered grid,
including derivative buffers, even when training used a split. It requires grid
ranges, cyclic axes and patch sizes. A custom NPZ may supply a `data` grid, with
`path`, `feature_ndim`, `range_sizes`, `cyclic_dimensions`, `patch_sizes` and
`class_names` in its dataset configuration. `dimension` is the intrinsic grid
dimension. Inputs have a class axis, parameter axes and trailing feature axes.

Inference exposes `input`, `output`, `mu`, and the existing `encoder_relu_N` /
`decoder_relu_N` names for beta-VAE; MMLS exposes input/output. Geometry accepts
`representations: all`. `inference.decoder_sampling` selects sampled decoding
or the latent mean; `inference.seed` records its randomness. Error summaries
retain the paper's corresponding-point Hausdorff **proxy**.

The working directory contains:

```text
experiment-configurations/   Resolved experiments, sweep job lists and results
splits/                     Index files, shared weights and fixed-test indices
trained-models/             Checkpoints or MMLS projector state
inference-outputs/          Metadata; arrays only if persist is true
analysis-outputs/           Errors, geometry and plot-ready aggregate curves
plots/                      Requested PNG/PDF figures
```

Stage identities include their inputs and relevant parameters. Training seeds
and `training.trial` distinguish repetitions. New analyses/layers reuse matching
fits. Stage implementations use an explicit format/behavior version in
`outputs.py`; increment it when changing numerical behavior. Completed stages
verify their manifest and declared files; partial
directories stop execution and remain untouched. Move a partial directory aside
before retrying. These outputs use a new layout; old directories are unsupported.

Inference feeds analyses directly. Errors accumulate per batch; geometry handles
one representation at a time. `inference.buffer: temporary` uses a temporary
memory-mapped layer array and removes it afterward. `persist: true` keeps arrays
for later analysis; without persistence, a new analysis may rerun inference.
MMLS normally references its saved split; `model.parameters.export_points: true`
exports a self-contained training-point state instead.

Sweep configurations have `base`, dotted-path `grid` choices, optional `groups`,
`exclude` matching rules, and additional experiments under `include`. Choose
coupled seeds/trial numbers with a `repetition` dictionary in the grid. Outputs
select explicit results using `where` conditions; available rendering names are
`bounds` and `layers`. The paper YAML files provide complete examples.
