Outputs and reuse
=================

The output directory holds shared stages and figure outputs:

.. code-block:: text

   experiment-configurations/   Resolved configs, sweep jobs and results
   splits/                     Indices, sampling weights and test/reference caches
   trained-models/             VAE checkpoints or MMLS projector state
   inference-outputs/          Metadata; arrays when persistence is enabled
   analysis-outputs/           Errors, geometry and aggregate curves
   plots/                      PNG/PDF figures
   experiment-bundles/         Compact outputs for small toy MMLS runs

Directory names
---------------

Folders contain readable fields separated by ``__``, followed by a 20-character
content hash. Fits identify dataset, dimension, model, training fraction or toy
sample count, seed and trial, plus relevant model options. For example:

.. code-block:: text

   custom_dsprites_balanced__dim4__mmls__k32__thin_svd__fixed_test__train20pct__weighted__testmax500__seed439__trial0__<hash>

``train20pct`` means 20% of the training pool after holding out the test set;
``n250`` means 250 sampled toy training points. Split names identify sampling
settings; inference/analysis names add domain, representation and analysis type.
Shared caches and input geometry identify the dataset and relevant settings.
Sweeps, aggregates and plots identify the figure or filename.

Names are sanitized and length-limited; hashes distinguish the full parameter
sets. Directory jobs save full configuration and metadata as JSON. Old
hash-only names and output layouts predating this pipeline are unsupported.

What is reused
--------------

Use one output directory for related figures. Fits are shared only when the
dataset, split, model parameters, training seed and trial match. A different
VAE budget creates a different fit. New analyses or layers can reuse a fit.
Stage identities include relevant inputs, parameters and the implementation
version in ``experiments_and_analysis/outputs.py``.

Messages about reused **weights** refer to geometric sampling probabilities,
not learned model weights. These are shared for matching dataset/weighting
settings; fixed test sets are shared for matching dataset/test-size settings.
Fractions and split seeds distinguish training selections. At training fraction
one, every repetition selects the entire remaining pool.

Completed directory stages verify their manifests and declared files.
Incomplete directories stop execution and remain available for inspection;
move the reported directories aside before retrying. For directory outputs,
``training.enabled: false`` requires a matching existing fit.

Bundled toy results
-------------------

Figure 5 uses ``output_layout: bundled`` for Sphere/Torus jobs. Each fixed
configuration and repetition gets one folder covering all sample sizes:
**40 toy bundle folders** for the full preset, plus ordinary image outputs.
Experiments still run separately and sequentially, with their own identities,
seeds, results and completion state.

The single-run runner also accepts this layout. It supports toy MMLS,
``provided`` splits and reconstruction-error analyses. Other workflows default
to ``output_layout: directories``. Each bundle contains:

* ``results.pkl``: a normal dictionary of experiments, resolved configs,
  results, stage metadata/timings, split indices, model state, pointwise errors
  and any persisted arrays. Numerical payloads are native NumPy arrays.
* ``summary.csv``: one row per error analysis, including sample size, seed,
  trial, counts, metrics and elapsed time. Skipped jobs are included.
* ``family.pkl``: fixed configuration and export checkpoint.
* ``progress.pkl``: an incremental framed-pickle journal that commits whole
  experiments without rewriting the growing export each time.

Shared reference points live in ``splits/reference__.../points.npy``. Bundled
runs create no per-experiment subdirectories or temporary experiment trees.
Exports refresh on sweep exit, including Ctrl+C, or after a direct single run.
After a hard kill, rerun to reuse committed work, or refresh the export without
running experiments:

.. code-block:: python

   from experiments_and_analysis.bundles import export_bundle

   export_bundle(bundle_directory)

Load exports with standard pickle. Configs and metadata are already
dictionaries; stage keys such as ``metrics.json`` do not require JSON parsing:

.. code-block:: python

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

Inference storage and timing
----------------------------

Inference feeds analyses directly: errors accumulate per batch, while geometry
handles one representation at a time. Only requested representations are
copied; inference stops after the last required layer.
``inference.evaluation_size`` limits held-out inference without reducing
training data or geometry grids.

``inference.buffer: temporary`` uses a memory-mapped layer array for directory
jobs and removes it afterward. Small bundled jobs buffer in memory.
``inference.persist: true`` keeps arrays; without persistence, a new analysis
may rerun inference. MMLS normally references its saved split;
``model.parameters.export_points: true`` saves self-contained training points.

Stage timings are saved in directory manifests or bundle metadata. Analysis
timing excludes the inference stream; nested split/plot tasks are included in
their parent's elapsed time. Verbose logs report original durations for cached
work and the current dataset-loading duration.
