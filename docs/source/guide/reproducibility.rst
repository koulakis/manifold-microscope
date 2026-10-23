Paper reproduction
==================

Install from the repository checkout with ``python -m pip install -e '.[benchmark]'``.
Custom dSprites is generated locally; COIL-20 needs ``COIL20_PATH`` set as
described in :doc:`installation`.

Preview and run
---------------

From the repository root:

.. code-block:: bash

   python -m experiments_and_analysis.scripts.sweep \
     --config experiments_and_analysis/scripts/configs/paper/figure4_quick.yaml \
     --output-dir ../outputs/new-paper-run --dry-run

The preview saves resolved jobs under
``experiment-configurations/sweep__figure_4__<hash>/jobs.json`` without loading
data or initializing CUDA. Remove ``--dry-run`` to execute.

All presets live in ``experiments_and_analysis/scripts/configs/paper/``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Preset
     - Configured experiments
   * - ``figure3.yaml``
     - 147 dSprites MMLS runs across dimensions, fractions and repetitions
   * - ``figure4.yaml``
     - 78 MMLS/VAE runs: thirteen fractions, three repetitions
   * - ``figure4_quick.yaml``
     - 12 MMLS/VAE runs: six fractions, one repetition
   * - ``figure5.yaml``
     - 40,705 runs: 40,640 toy jobs and 65 image-dataset jobs
   * - ``figure6.yaml``
     - Two VAEs with geometry on all configured representations

Counts include MMLS jobs skipped for having fewer training points than
neighbors. Full configurations can take many hours and substantial memory.

Practical settings
------------------

* All paper presets request ``cuda:0`` and stop new computation if it cannot
  initialize. ``--device cpu`` explicitly selects CPU. Dry runs and fully
  cached runs need no accelerator.
* ``figure4_quick.yaml`` reduces the sweep without shortening VAE training.
  The separate ``--quick-run`` profile caps training at 10K steps and reduces
  sweep/evaluation sizes. Full grids and all geometry layers are retained.
  See :doc:`experiments` for the quick-run loop and exact caps.
* For a custom VAE budget, use ``model.parameters.max_iter`` with
  ``strict_budget: true``. In Figure 4, apply these parameters only to the VAE
  model choice. Both YAML examples are in :doc:`configuration`.
* Figures 3 and 5 show only overall progress by default. Add
  ``--no-quiet-experiments`` for detail; use ``--quiet-experiments`` on other
  figures to suppress individual experiment output.
* Share one output directory across figures to reuse compatible work. Figure
  5's toy jobs produce 40 bundle folders with native pickle/NumPy data and CSV
  summaries. See :doc:`outputs` for naming, loading and interrupted-run recovery.

Scope of reproducibility
------------------------

The pipeline retains historical normalization, bound formulas and signed-curvature
aggregation. Original paper experiments were not fully seeded. Current presets
record seeds, but small variation across runs and devices remains possible;
seeded execution does not establish exact equivalence to the original results.

For retained results, record the source revision and local changes, resolved
configuration, seeds, input provenance, environment versions, hardware, command
and output location. Keep reduced/quick runs distinct from full paper runs.
