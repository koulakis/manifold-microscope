Grid geometry
=============

These NumPy wrappers accept a floating grid ``(s1, ..., sd, features)``.
Array results are detached from autograd. ``difference_intervals`` gives one
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

.. py:function:: reach_per_point(features_on_grid, range_sizes, patch_sizes, cyclic_dimensions=None, subsample_points=None, batch_size=10, return_witnesses=False, device=None, *, candidate_batch_size=1024, bottleneck_tolerance=0.1, n_directions=32, curvature_iterations=4, min_pair_distance=None, return_components=False)
   :module: microscope.computations_grid.reach_two_step

   Experimental **pointwise array** of the minimum of curvature and bottleneck
   radii. Its minimum estimates global reach. Default return forms match the
   original function above: an array, or ``(reach_grid, witnesses)`` when
   ``return_witnesses=True``. Existing multiclass pipelines continue to use
   the original pairwise estimator.

   ``return_components=True`` always returns three items:
   ``(reach_grid, witnesses, components)``, including witnesses even when
   ``return_witnesses=False``. All values use the input float32/float64 dtype,
   except int64 witnesses. ``components.shape == reach_grid.shape + (2,)``;
   ``components[..., 0]`` is curvature radius, ``components[..., 1]`` is
   bottleneck radius, both in distance units. These pointwise diagnostics
   generally differ from the original exhaustive tangent-ratio field; the
   theoretical curvature/global decomposition concerns the global minimum.

   Reuses the extrinsic second fundamental form and QR metric coordinates.
   Curvature means the maximum of ``norm(II(v, v))`` over unit tangent vectors,
   rather than scalar curvature or a flattened matrix norm. Axis/pair directions
   and ``n_directions`` fixed-seed starts undergo ``curvature_iterations``
   spectral refinements. This is exact for the estimated form on curves and,
   after one refinement, hypersurfaces; higher codimension requires convergence
   checks and has no certified global optimizer.

   Bottleneck candidates have unit-chord tangential projection at **both**
   endpoints at most ``bottleneck_tolerance``, a sine in ``[0, 1)``. The larger
   of the two endpoint tangent ratios corrects the half-chord value for angular
   error. An exact doubly normal chord gives half its length. No accepted pair
   yields infinity and witness ``-1``; this does not establish absence
   of bottlenecks. Check grid and angular-tolerance convergence. Search remains
   quadratic with query/candidate chunking; no ambient projection matrix is made.

   In exact arithmetic, the angular condition is symmetric: a pair failing at
   either end is rejected in both directions. Staggering between sampled
   branches and finite-difference tangent errors can reject an otherwise
   resolved bottleneck. Density alone is insufficient: the tangential offset
   must be small relative to the separation times the sine tolerance. Very
   tight tolerances can reject all nearby chords; marginal directional
   differences can also arise from roundoff at the threshold.

   Second differences require patches at least five and retain ``s - 4``
   centers on nonperiodic axes, or ``s`` on periodic axes. Subsampling selects
   every kth center; derivatives use the original adjacent samples. Reach and
   component arrays repeat/truncate to the cropped center-grid shape, while
   witnesses stay flat, with ``N = prod(ceil(s_out / stride))`` entries.
   Witnesses index the best bottleneck partner on the flattened sampled grid,
   even when curvature gives the smaller combined estimate. They are ``-1``
   when no chord is accepted. This new method crops two samples at each
   nonperiodic end, versus one in the original API; align grids before comparing.
   The input,
   rank-loss and ``min_pair_distance`` policies match the pairwise API. At least
   two sampled centers with a distinct distance-eligible pair are required.
   Flat curvature gives an infinite radius; overflow of a finite positive
   curvature radius in the input dtype raises ``FloatingPointError``.

   Inspired by the curvature/bottleneck dichotomy in `Aamari et al. (2019)
   <https://arxiv.org/pdf/1705.04565>`_ and the curvature/weak-feature-size
   decomposition in `Aamari et al. (2022)
   <https://arxiv.org/pdf/2207.06074>`_. This angular chord search is a simpler
   approximation, not the latter paper's geodesic/spherical-distortion estimator.
   The intended setting is smooth compact embedded manifolds without boundary.
   Finite differences, sampling, optimization and filtering introduce separate
   errors; the result is not a certified bound on true reach.

   Example::

      from microscope.computations_grid.reach_two_step import reach_per_point

      reach, witnesses, components = reach_per_point(
          points, ranges, patches, cyclic, device="cuda:0",
          return_components=True,
      )
      curvature_radius = components[..., 0]
      bottleneck_radius = components[..., 1]
      global_reach = reach.min()

.. py:function:: reach_curvature_bottleneck(features_on_grid, range_sizes, patch_sizes, cyclic_dimensions=None, subsample_points=None, batch_size=10, device=None, *, candidate_batch_size=1024, bottleneck_tolerance=0.1, n_directions=32, curvature_iterations=4, min_pair_distance=None, return_components=False)
   :module: microscope.computations_grid.reach_two_step

   Compatibility wrapper for the initial global scalar API, using the same
   pointwise computation. ``return_components=True`` retains its original
   ``ReachComponents(reach, curvature_radius, bottleneck_radius, bottleneck_pair)``
   scalar result. Prefer the new pointwise API for field analysis.

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
