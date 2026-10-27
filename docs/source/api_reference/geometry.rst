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

.. py:function:: reach_per_point(features_on_grid, range_sizes, patch_sizes, cyclic_dimensions=None, subsample_points=None, batch_size=10, return_witnesses=False, device=None, *, candidate_batch_size=1024, min_pair_distance=None)
   :module: microscope.computations_grid.reach

   Returns local-reach estimates shaped ``(s1_out, ..., sd_out)``, with
   ``s_out = s - 2`` for nonperiodic axes. The minimum estimates global reach.
   Inputs must be finite float32/float64 arrays. Patch sizes must exceed two;
   batch sizes and subsampling strides must be positive integers. Negative
   NumPy strides are supported. Tangent frames reuse the shared reduced-QR
   geometry and diagnose numerical rank loss with an input-grid location.

   ``subsample_points`` is a stride applied to queries and candidates along
   every axis; tangents still use the full grid. Subsampled estimates are
   repeated and truncated back to the output grid shape.

   With ``return_witnesses=True``, returns ``(estimates, witnesses)``.
   Witnesses have shape ``(N,)``, where
   ``N = prod(ceil(s_out / stride))`` and ``stride = subsample_points or 1``.
   Values index the flattened, cropped, subsampled candidate array;
   witnesses are not repeated or reshaped. Infinite flat estimates retain a
   distinct eligible witness. No distinct eligible candidate raises ValueError.

   ``candidate_batch_size`` bounds candidate-side device storage in addition
   to query-side ``batch_size``. Only these chunks move to the device. The
   search still visits every eligible sampled pair and forms no ambient by
   ambient projection matrix.

   Exact duplicate coordinates are excluded. The default
   ``min_pair_distance=None`` raises on positive separations below a roundoff
   threshold: eight times dtype epsilon times the Euclidean norm of the
   maximum absolute sampled coordinate in each ambient axis. An explicit
   nonnegative value excludes pairs at or below that distance; zero includes
   all distinct coordinates and disables the automatic diagnostic. A positive
   cutoff is an approximation that can omit a real bottleneck. Invalid data
   and nonfinite arithmetic are reported, not replaced with finite sentinels.

   This tangent-space formula estimates an underlying smooth embedded
   manifold without boundary. It omits boundary normal-cone contributions to
   the reach of a bounded parameter image, such as an annulus. Approximate
   tangents and roundoff can underestimate true reach; convergence must be
   checked for generic parameterizations. Normal fractions at the QR
   projection's roundoff scale (eight times dtype epsilon times intrinsic
   dimension) are treated as zero, preserving infinite flat ratios.

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
