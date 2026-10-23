Experiment API
==============

The ``experiments_and_analysis`` package separates ``data``, ``models``,
``analyses`` and ``plotting``. Numerical functions accept arrays and return
results; runners handle configuration, files and completion. Geometry's core
remains in ``microscope``.

Runners
-------

.. py:function:: run(config, output_dir, *, dataset=None, quick_run=False)
   :module: experiments_and_analysis.scripts.run

   Executes one configuration dictionary. An optional loaded ``Dataset``
   avoids repeated data loading. ``quick_run=True`` applies shared caps.

.. py:function:: run(config, output_dir, *, dry_run=False, quick_run=False, device=None, quiet_experiments=None)
   :module: experiments_and_analysis.scripts.sweep

   Expands and sequentially executes a sweep. ``dry_run=True`` saves resolved
   jobs only. ``device`` overrides all jobs; ``quiet_experiments`` overrides
   the sweep's display setting.

.. py:function:: expand(config, *, quick_run=False, device=None)
   :module: experiments_and_analysis.scripts.sweep

   Returns resolved experiments from grid choices, exclusions and inclusions.

.. py:function:: load_config(path)
   :module: experiments_and_analysis.config

   Reads a YAML or JSON dictionary.

.. py:function:: resolve(config)
   :module: experiments_and_analysis.config

   Validates and fills defaults without mutating the caller's dictionary.
   See :doc:`../guide/configuration` for default sources and precedence.

Models, splits and analyses
---------------------------

``model.name`` accepts ``beta_vae``, ``mmls`` or null. Use a null model and
input-only geometry to analyze a dataset without fitting or inference;
``scripts/configs/dataset_geometry.yaml`` is an example.

Split methods are ``random``, ``fixed_test`` (the paper's farthest-point test
selection), ``full`` and ``provided`` (toy training/reference points).
``train_fraction`` samples the remaining training pool; ``weighted`` uses
volume weights. Only indices and small sampling metadata are saved.
Normalization retains independent min/max scaling of training/evaluation
domains; analytic toy coordinates retain their own scaling.

Errors use held-out data by default and retain the paper's corresponding-point
Hausdorff **proxy**. Geometry uses the full ordered grid, including derivative
buffers, even when training uses a split. It requires ranges, periodic axes
and patch sizes.

Custom NPZ inputs can supply a ``data`` grid. Set ``path``, ``feature_ndim``,
``range_sizes``, ``cyclic_dimensions``, ``patch_sizes`` and ``class_names`` in the
dataset configuration. ``dimension`` is the intrinsic grid dimension. Inputs
have a class axis, parameter axes and trailing feature axes; the chosen model
must accept that feature shape.

Beta-VAE exposes ``input``, ``output``, ``mu`` and
``encoder_relu_N`` / ``decoder_relu_N`` representations through its
``inference.batches`` function; training uses ``fitting.fit``. MMLS exposes
input/output. Geometry accepts ``representations: all``.
``inference.decoder_sampling`` selects sampled decoding or the latent mean,
and ``inference.seed`` records randomness.

See `experiment source and example configurations
<https://github.com/koulakis/manifold-microscope/tree/main/experiments_and_analysis>`_
and :doc:`../guide/outputs` for result formats and cache sharing.
