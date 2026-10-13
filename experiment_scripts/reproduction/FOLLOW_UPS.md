# Existing-code issues deferred to the next step

The reproduction entry points reuse the existing implementations. The following
issues remain in the older public entry points or their implementations:

| Existing file(s) | Follow-up |
|---|---|
| `experiment_scripts/manifold_fitting/inference.py`, both analysis wrappers | Include repetition in experiment-directory naming, or accept explicit model/input directories. They currently build five fields while training writes six. |
| `experiment_scripts/manifold_fitting/training.py` | Expose dataset/model/dimension/fraction lists, separate repetition seeds, workers and model-config overrides. The new fitting module builds configurations directly and reuses Solver/MMLS. |
| `representation_learning/beta_vae/solver.py` | Review checkpoint loading's bare torch.load for device/loading-policy portability, and make any strict smoke-budget behavior explicit. Current max_iter can finish the whole epoch. New evaluation uses the existing mapped network-only loader instead. |
| `microscope/computations_grid/data_analysis/run_data_analysis.py` | Expose/forward device through the CLI and run_single_output. Grid APIs already select CUDA or CPU automatically, but the wrapper does not pass an explicit selection. |
| `microscope/datasets/generic_dataset_loader.py` | Review explicit device forwarding to farthest-point selection and define behavior when reusing a cached split with a different requested test size. |
| Toy sweep/projector and `microscope/datasets/toy_manifolds.py` | Thread device through MMLS and CVT calls. Move unused denoising/Lightning imports behind the selected method so MMLS does not require unrelated code. |
| `representation_learning/beta_vae/inference_intermediate_layers.py` | Make pin_memory conditional on device; explicitly expose seed/decoder-sampling choices if deterministic new runs are wanted. Preserve sampled decoding when comparing with historical outputs. |
| `notebooks/manifold_fitting/bound_curves_all_methods.ipynb` | Replace obsolete geometry imports and historical absolute paths; eventually refer to the new evaluation/export functions. |

Additional decisions/validation, rather than automatic existing-code edits:

- The cleanup branch already forwards the reach device to derivative computation
  and fixes signed curvature integration and subsampled reach shapes. Preserve
  those fixes and measure their effect when comparing with older paper outputs.
- Confirm toy reach/measure schema changes when comparing with OLD artifacts.
  Importing old geometry/curvatures pickles and old YAML configs is deliberately
  not implemented by these minimal new-run entry points.
- Reconcile the paper captions' repetition counts with retained runs and the
  implemented notebook bound-constant fitting with the paper description.
- Validate datasets/splits, checkpoint errors, bound curves and layer aggregates
  before claiming exact reproduction. No new scripts have been executed yet.
- For partial directories, recovery remains a manual next step. Do not add a
  termination file just because a checkpoint or some plots happen to exist.
