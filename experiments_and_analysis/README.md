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
Sweeps show an overall progress bar alongside each experiment's output.
Use `--quiet-experiments` to show only the sweep bar, including cached and
skipped jobs. Figures 3 and 5 enable this with top-level
`quiet_experiments: true`; use `--no-quiet-experiments` to restore detailed
output. Python callers can pass `quiet_experiments=True` or `False` to
`run_sweep`. These display options do not change experiment/cache identities.
The paper presets explicitly set `base.device: cuda:0` for training,
inference and the volume/curvature/reach kernels. CUDA initialization errors
stop execution before dataset loading or new stage outputs; there is no CPU
fallback for these kernels. Use `--device cpu` or change `base.device` for a
CPU run. Custom configurations still default to CPU. Geometry logs its device,
and the reach progress bar includes it. NumPy preparation, aggregation,
inter-class distances and plotting still use CPU. Full paper
configurations can take many hours; dry runs only expand configurations. Files
for Figures 3–6 and quick Figure 4 are together in `scripts/configs/paper`.
Quick Figure 4 now computes all missing fits in the new working directory;
it does not import old reproduction outputs.

Add `--quick-run` to either runner, or pass `quick_run=True` from Python. All
quick settings live in `quick.py:QUICK_RUN`: VAE training stops at at most 10,000
optimizer iterations (`strict_budget: true`); size sweeps use 20%, 50%, 100%
and one repetition; held-out evaluation uses at most 100 examples. Toy sweeps
use 25, 250, 1000 training points, one repetition and 100 reference points.
Smaller existing budgets/size grids are retained. Quick mode keeps model/dataset
choices, full structured grids and all configured geometry layers, using
`reach_subsample: 4`. It adds no preview dataset. Repetitions keep the last
configured trial, allowing Figures 4 and 6 to share their full-size dSprites fit.

For a fresh quick paper run, choose an unused output directory and run:

```bash
for figure in 3 4 5 6; do
  python -m experiments_and_analysis.scripts.sweep \
    --config experiments_and_analysis/scripts/configs/paper/figure${figure}.yaml \
    --output-dir ../outputs/paper-quick-01 --quick-run --device cuda:0 || break
done
```

These presets produce 12, 6, 12 and 2 jobs respectively. Use `--dry-run` to
export their job lists without executing them. Figure 6 still computes all
geometry measures on the full grids for every configured layer, so it can take
substantially longer. Completed compatible work is reused in the shared root;
quick and full VAE budgets have distinct fit identities. Stage elapsed seconds
are printed and saved in `manifest.json`; reuse reports the original duration.
Analysis timing excludes the inference stream; nested split/plot subtasks are
included in their parent's elapsed time. Dataset loading is timed in the log.
Both runners check accelerator initialization before loading data or creating
stage outputs. Dry runs and fully cached runs do not require the accelerator.
After a failure, preserve/move incomplete stage directories before retrying;
the runner reports their paths. The loop above stops at the first failed figure.

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
experiment-bundles/         Compact pickle outputs for small toy MMLS runs
```

Figure 5's Sphere/Torus groups use `output_layout: bundled`. Each dataset,
fixed configuration and repetition gets one readable bundle folder containing
all its sample sizes: **40 toy bundle folders for the full preset**, plus the
ordinary image-dataset outputs. Experiments still run separately and
sequentially. Each has its own hashes, seeds, results and completion state;
the single-experiment runner also accepts this layout. Other workflows default
to `output_layout: directories`. Bundling currently supports toy MMLS with a
`provided` split and reconstruction-error analyses.

Each bundle contains:

- `results.pkl`: a normal pickle dictionary with every experiment's resolved
  configuration, result, stage metadata/timings, split indices, model state,
  pointwise errors and any requested persisted arrays. NumPy payloads are
  decoded to arrays/dictionaries when exported.
- `summary.csv`: one row per error analysis, with sample size, seed, trial,
  counts, error metrics and elapsed time; skipped jobs are included.
- `family.pkl`: the bundle's fixed configuration and export checkpoint.
- `progress.pkl`: an incremental, framed pickle journal for resuming. It
  commits whole experiments without rewriting the growing export each time.

Shared reference points remain in the normal `splits/reference__.../points.npy`
cache. No per-experiment subdirectories or temporary experiment trees are
created for bundled runs. Exports refresh at sweep exit, including Ctrl+C,
or after a direct single run. A hard kill can leave an older export; rerun the
command to reuse committed work, or call `export_bundle(bundle_directory)`
from `experiments_and_analysis.bundles` to refresh it without executing jobs.

Load results with standard pickle:

```python
import pickle
from pathlib import Path

for directory in Path("../outputs/new-experiments/experiment-bundles").iterdir():
    with (directory / "results.pkl").open("rb") as stream:
        bundle = pickle.load(stream)
    for experiment in bundle["experiments"].values():
        config = experiment["configuration"]
        for analysis in experiment["stages"].get("analysis-outputs", {}).values():
            metrics = analysis["metrics.json"]
            distances = analysis["pointwise_distances.npy"]  # NumPy array
```

New output folders use descriptive fields separated by `__`, followed by their
existing 20-character content hash. Fields depend on the stage:

| Output | Information in the name |
| --- | --- |
| Experiment configurations and fits | Dataset, dimension, model, training fraction or toy sample count, seed, trial; MMLS neighbor count/projection method or VAE architecture/latent size/iteration budget |
| Splits | Dataset, dimension, split method, training percentage, weighting, test size setting and sampling seed |
| Shared weights/test/reference sets | Dataset and dimension where applicable, weighting or test/reference size; reference seed |
| Inference and analyses | Fit details, train/test/grid domain, representation/layer, analysis type; input grid geometry uses only dataset details |
| Sweeps, aggregates and plots | Requested figure/filename, followed by the hash |

For example, a fit may be named
`custom_dsprites_balanced__dim4__mmls__k32__thin_svd__fixed_test__train20pct__weighted__testmax500__seed439__trial0__<hash>`.
`train20pct` refers to 20% of the available training pool after holding out the
test set. Toy runs use `n250`, for example, for 250 sampled training points.
Names are sanitized and length-limited; the hash still distinguishes all
configuration details. Full parameters remain in the JSON files.
Only the descriptive naming convention is supported; hash-only output folders
from earlier runs are ignored.

Stage identities include their inputs and relevant parameters. Training seeds
and `training.trial` distinguish repetitions. New analyses/layers reuse matching
fits. Stage implementations use an explicit format/behavior version in
`outputs.py`; increment it when changing numerical behavior. Completed stages
verify their manifest and declared files; partial
directories stop execution and remain untouched. Move a partial directory aside
before retrying. Output layouts from before the `experiments_and_analysis`
pipeline are unsupported.

Inference feeds analyses directly. Errors accumulate per batch; geometry handles
one representation at a time. Only requested representations are copied from
the network; inference stops after the last required layer. Optional
`inference.evaluation_size` limits held-out inference without shrinking the
training set or geometry grid. `inference.buffer: temporary` uses a temporary
memory-mapped layer array and removes it afterward. `persist: true` keeps arrays
for later analysis; without persistence, a new analysis may rerun inference.
MMLS normally references its saved split; `model.parameters.export_points: true`
exports a self-contained training-point state instead.

Sweep configurations have `base`, dotted-path `grid` choices, optional `groups`,
`exclude` matching rules, and additional experiments under `include`. Choose
coupled seeds/trial numbers with a `repetition` dictionary in the grid. Outputs
select explicit results using `where` conditions; available rendering names are
`bounds` and `layers`. The paper YAML files provide complete examples.
