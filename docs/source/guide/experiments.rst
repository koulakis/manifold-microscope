Experiments and sweeps
======================

Each experiment loads data, selects a split, fits a model, runs inference and
computes the requested analyses. A sweep resolves a list of experiments and
runs them sequentially. Matching completed stages are reused.

.. _experiment-run-preview:

Run and preview
---------------

From the repository root, with the ``benchmark`` extra installed:

.. code-block:: bash

   python -m experiments_and_analysis.scripts.run \
     --config experiments_and_analysis/scripts/configs/single.yaml \
     --output-dir ../outputs/new-experiments

   python -m experiments_and_analysis.scripts.sweep \
     --config experiments_and_analysis/scripts/configs/paper/figure4_quick.yaml \
     --output-dir ../outputs/new-experiments --dry-run

The sweep preview saves ``config.json`` and ``jobs.json`` under
``experiment-configurations/sweep__<figure-or-datasets>__<hash>/``. It expands
configuration without loading data or fitting models. Remove ``--dry-run``
to execute and render the requested figures. The single-run CLI has no
dry-run flag. Both runners support ``--help``.

Device and progress
-------------------

Every sweep shows overall progress alongside the individual experiment's
output. ``--quiet-experiments`` keeps only the overall bar;
``--no-quiet-experiments`` restores details. Figures 3 and 5 enable quiet mode
in their presets. Python callers can pass ``quiet_experiments=True`` to the
sweep runner. Display settings do not change cache identities.

Paper presets request ``base.device: cuda:0``; pass ``--device cpu`` to override.
See :ref:`device-selection` for initialization and cache behavior. Geometry
reports its device, including in the reach progress bar.

Quick runs
----------

``figure4_quick.yaml`` reduces fractions and repetitions but keeps its configured
VAE budget. The separate ``--quick-run`` flag applies the shared quick profile:

* VAE training: at most 10,000 optimizer steps, with strict stopping.
* Larger fraction sweeps: 20%, 50%, 100%, one repetition and at most 100
  held-out evaluation examples.
* Toy sweeps: 25, 250 and 1,000 training points, one repetition and 100
  reference points.
* Geometry: all configured layers and full grids, with reach subsample four.

Smaller existing budgets are retained. Repetitions keep the last configured
trial. Settings live in ``experiments_and_analysis/quick.py:QUICK_RUN``.
For a fresh quick paper run:

.. code-block:: bash

   for figure in 3 4 5 6; do
     python -m experiments_and_analysis.scripts.sweep \
       --config experiments_and_analysis/scripts/configs/paper/figure${figure}.yaml \
       --output-dir ../outputs/paper-quick-01 --quick-run --device cuda:0 || break
   done

This gives 12, 6, 12 and 2 jobs. Figure 6 can still take substantially longer
because geometry retains the full grids and layers. Compatible fits can be
shared across figures; their resolved training budgets must match.

Python calls and configuration
------------------------------

The runners also accept dictionaries:

.. code-block:: python

   from experiments_and_analysis.scripts.run import run

   result = run({
       "dataset": {"name": "custom_dsprites_balanced", "dimension": 4},
       "split": {"method": "random", "test_fraction": 0.1, "seed": 42},
       "model": {"name": "beta_vae", "parameters": {
           "z_dim": 10, "max_iter": 50000, "strict_budget": True,
       }},
       "training": {"seed": 439, "trial": 0},
       "analyses": ["errors", {"name": "geometry", "representations": ["mu"]}],
       "device": "cuda:0",
   }, "../outputs/new-experiments")

See :doc:`configuration` for defaults, VAE budgets and model-specific sweep
choices; :doc:`outputs` for result loading and recovery; and
:doc:`../api_reference/experiments` for Python entry points and input formats.

Caller-supplied models
----------------------

Python callers can pass a ``ModelAdapter`` from
``experiments_and_analysis.models.interface`` to the single-run API. The caller
owns the native framework, objective, preprocessing and checkpoint format:

.. code-block:: python

   result = run({
       "dataset": dataset_spec,
       "model": {"name": "external", "adapter": "my-model/v1",
                 "parameters": native_recipe},
       "analyses": ["errors"],
   }, output_dir, dataset=loaded_dataset, model_adapter=adapter)

The adapter's ``identifier`` must match ``model.adapter``. Implement ``fit``
(save files in the supplied directory and return their relative names), ``load``,
``representation_names`` and ``batches``. Batches yield consecutive
``(start_index, floating_array)`` pairs in input order. Representations must be
deterministic for the supplied inference seed, finite, and have a fixed feature
shape. ``output`` must use the dataset's ambient coordinates for error analysis.
Keep validation points inside the supplied training pool; the pipeline owns the
held-out test split. Record every model/preprocessing choice and implementation
version in the model configuration so incompatible fits receive distinct IDs.

Set ``dataset.normalize: false`` when the caller supplies calibrated units and
the native adapter owns preprocessing. The historical normalization remains the
default. Geometry consumes the full structured grid in either case.

An adapter may raise ``TrainingPaused`` after saving a resumable native
checkpoint. Repeat with ``resume_training=True`` to continue the matching
incomplete fit. The stage is marked complete only after ``fit`` returns all
required files. Completed fits and analyses reuse the ordinary pipeline cache.
External checkpoints currently require directory outputs. Layer-plot panels can
set ``bottleneck`` to the representation at the encoder/decoder boundary;
the default remains ``mu``.
