Grid geometry
=============

These NumPy wrappers accept a floating grid ``(s1, ..., sd, features)`` and
return NumPy arrays detached from autograd. ``difference_intervals`` gives one
coordinate spacing per axis; ``range_sizes`` gives full parameter-domain lengths.
Periodic axes retain their length. Nonperiodic axes lose boundary samples as
shown below. See :doc:`../guide/getting_started` for a complete example.

Derivatives and volume
----------------------

.. py:function:: partial_derivatives_across_all_dims_batched(features_on_grid, difference_intervals, patch_sizes, cyclic_dimensions, device=None)
   :module: microscope.computations_grid.basic

   Returns a Jacobian shaped ``(s1_out, ..., sd_out, features, d)``. Each
   nonperiodic axis has ``s_out = s - 2``. Patch sizes must exceed two.

.. py:function:: volume_element(features_on_grid, difference_intervals, cyclic_dimensions, patch_sizes, device=None)
   :module: microscope.computations_grid.volume

   Returns ``sqrt(det(g))`` shaped ``(s1_out, ..., sd_out)``, with
   ``s_out = s - 2`` for each nonperiodic axis. Patch sizes must exceed two.

.. py:function:: compute_total_volume(element, range_sizes)
   :module: microscope.computations_grid.volume

   Integrates the volume element using its grid mean and domain lengths.

Curvature
---------

.. py:function:: scalar_curvature(features_on_grid, difference_intervals, cyclic_dimensions, patch_sizes, normalize=False, device=None)
   :module: microscope.computations_grid.curvature

   Returns scalar curvature shaped ``(s1_out, ..., sd_out)``, with
   ``s_out = s - 6`` on nonperiodic axes. Patch sizes must exceed six.
   ``normalize=True`` divides by ``d * (d - 1)`` and requires ``d >= 2``;
   on a surface, this gives Gaussian curvature. It does not normalize volume.

.. py:function:: compute_total_curvature(element, curvature, range_sizes)
   :module: microscope.computations_grid.curvature

   Integrates scalar curvature weighted by volume. Align the arrays first:
   on each nonperiodic axis, remove two more samples from each end of the
   volume grid to match the curvature grid.

Reach
-----

.. py:function:: reach_per_point(features_on_grid, range_sizes, patch_sizes, cyclic_dimensions=None, subsample_points=None, batch_size=10, return_witnesses=False, device=None)
   :module: microscope.computations_grid.reach

   Returns local-reach estimates shaped ``(s1_out, ..., sd_out)``, with
   ``s_out = s - 2`` for nonperiodic axes. The minimum estimates global reach.
   Patch sizes must exceed two.

   ``subsample_points`` is a stride applied to queries and candidates along
   every axis; tangents still use the full grid. Subsampled estimates are
   repeated and truncated back to the output grid shape.

   With ``return_witnesses=True``, returns ``(estimates, witnesses)``.
   Witnesses have shape ``(N,)``, where
   ``N = prod(ceil(s_out / stride))`` and ``stride = subsample_points or 1``.
   Values index the flattened, cropped, subsampled candidate array;
   witnesses are not repeated or reshaped.

Multiple classes
----------------

.. py:function:: compute_measures_multiclass(data, class_names, range_sizes, cyclic_dimensions, patch_sizes, normalize_for_volume, normalize_curvatures=False, reach_subsample=None, reach_batch_size=2, device=None)
   :module: microscope.computations_grid.data_analysis.data_analysis

   Accepts ``(classes, s1, ..., sd, features)`` with one unique name per class.
   Returns ``AnalysisResults`` with ``measures``, ``measure_aggregates``,
   ``range_sizes`` and ``trim_slices_data``; it creates no plots.

   Raw per-class ``volume_elements`` use ``s - 2`` on nonperiodic axes.
   The ``normalized_volume_elements``, ``normalized_curvatures``,
   ``normalized_reaches`` and class-pair ``normalized_distances`` use the
   common curvature-grid shape, ``s - 6``. Applying ``trim_slices_data`` to
   the input gives ``(classes, c1, ..., cd, features)`` on that common grid.

   ``normalize_for_volume=True`` rescales all classes together to combined
   volume one. Curvature normalization is independent. Fields keep their
   ``normalized_`` names even when volume normalization is disabled.
   Historical signed-curvature aggregates use sign-subset averages, so their
   contributions need not sum to total curvature; empty subsets yield NaN.

Device and source
-----------------

``device=None`` selects CUDA when available, otherwise CPU. Explicit
``device="cuda:0"`` raises if CUDA cannot initialize. Array preparation,
aggregation and returned arrays use CPU. Lower-level tensor functions use
the tensor's device and require manual periodic padding.

See `geometry source
<https://github.com/koulakis/manifold-microscope/tree/main/microscope/computations_grid>`_
for lower-level operations and implementation details.
