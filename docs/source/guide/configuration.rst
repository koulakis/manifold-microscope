Configuration and defaults
==========================

Single runs and sweeps use ``experiments_and_analysis.config.resolve``.
Explicit values override its defaults. Sweeps merge ``base`` with each
group's ``base``, then apply grid choices. ``--device`` overrides every job's
device; ``--quick-run`` applies caps after resolution.

Default values
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Settings
     - Selected defaults
     - Source under ``experiments_and_analysis/``
   * - Runner and split
     - CPU; random split; test fraction 0.1; training fraction 1.0;
       seed 42/trial 0 unless set by ``repetition``; directory layout
     - ``config.py:resolve``
   * - Inference
     - Batch size 64; no persisted arrays; memory buffer; sampled decoding
     - ``config.py:resolve``
   * - VAE training
     - ``max_iter: 2000000``; ``max_epochs: 100000``;
       ``strict_budget: false``; batch size 64; latent dimension 10
     - ``models/config.py:BetaVAEConfig``
   * - Dataset-specific models
     - COIL: VAE model/objective H, learning rate 0.0001, loss threshold 10.
       MMLS neighbors: ``2 * 2**dataset.dimension``
     - ``config.py:resolve``
   * - Other MMLS options
     - ``sigma: 1.0``; ``projection_method: thin_svd``;
       projection batch size 32
     - ``models/config.py:MMLSConfig``
   * - Geometry
     - Volume normalization on; curvature normalization off;
       reach subsample 2; reach batch size 2
     - ``analyses/geometry.py:compute``
   * - Quick profile
     - Training/evaluation, repetition and reach caps
     - ``quick.py:QUICK_RUN``

Use ``--dry-run`` on a sweep to inspect resolved runner and model values in
``jobs.json``. Omitted geometry arguments use the function defaults; those
implicit values are not inserted into ``jobs.json``. The full definitions are
in the `configuration resolver
<https://github.com/koulakis/manifold-microscope/blob/main/experiments_and_analysis/config.py>`_,
`model dataclasses
<https://github.com/koulakis/manifold-microscope/blob/main/experiments_and_analysis/models/config.py>`_
and `geometry function
<https://github.com/koulakis/manifold-microscope/blob/main/experiments_and_analysis/analyses/geometry.py>`_.

Set ``device`` and ``num_workers`` at the experiment level. Pass the output
directory to the runner; paths and workers do not belong in ``model.parameters``.

VAE step budgets
----------------

Use ``max_iter``, not ``max_iterations``. For a VAE-only sweep such as Figure 6,
update the existing ``base.model`` section and keep the other settings:

.. code-block:: yaml

   base:
     model:
       name: beta_vae
       parameters:
         max_iter: 50000
         strict_budget: true

``strict_budget: true`` stops at the step limit within an epoch. Early stopping
or the epoch limit can finish sooner. The historical ``false`` default finishes
the epoch and can exceed ``max_iter``. Omit ``--quick-run`` for a 50K cap;
quick mode would reduce it to 10K.

Figures 4 and 4 quick mix MMLS and VAE models. Replace ``grid.model.name``
(or update the existing ``grid.model``) with full model choices:

.. code-block:: yaml

   grid:
     model:
     - name: mmls
     - name: beta_vae
       parameters:
         max_iter: 50000
         strict_budget: true

Keep the existing fraction and repetition entries under ``grid``. This applies
training parameters only to the VAE; MMLS does not accept them. Preview the
modified configuration with ``--dry-run``.

Sweep structure
---------------

Sweeps accept ``base``, dotted-path ``grid`` choices, optional ``groups``,
``exclude`` matching rules, and additional experiments under ``include``.
Use ``repetition`` dictionaries in the grid to couple seeds and trial numbers.
Top-level ``quiet_experiments: true`` suppresses individual experiment output.
``outputs`` selects results with ``where`` conditions and requests ``bounds``
or ``layers`` rendering.

See the `paper YAML presets
<https://github.com/koulakis/manifold-microscope/tree/main/experiments_and_analysis/scripts/configs/paper>`_
for complete examples and :doc:`reproducibility` for their workloads.
