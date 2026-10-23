Getting started
===============

Geometry routines need an ordered, uniformly spaced parameter grid shaped
``(s1, ..., sd, features)``. For images, flatten the image axes into ``features``
and keep the class axis separate. An unordered point cloud does not provide
the grid needed for finite differences.

Scalar curvature on a sphere
----------------------------

This CPU example estimates the scalar curvature of a sphere of radius two:

.. literalinclude:: ../../../README.md
   :language: python
   :start-after: ```python
   :end-before: ```

Avoid coordinate singularities such as the poles. ``cyclic_dimensions`` lists
periodic parameter axes, starting at zero; omit duplicate endpoints there.
``difference_intervals`` contains the coordinate spacings. The induced metric
must be nonsingular.

Measures on your own grid
-------------------------

Save an ordered floating array as ``my_grid.npy``. The following assumes a
parameter domain [0, 1] × [0, 2π), periodic on the second axis, with at least
seven samples on the first:

.. code-block:: python

   import numpy as np
   from microscope.cyclic_dimensions import get_difference_intervals
   from microscope.computations_grid.volume import volume_element, compute_total_volume
   from microscope.computations_grid.curvature import scalar_curvature, compute_total_curvature
   from microscope.computations_grid.reach import reach_per_point

   points = np.load("my_grid.npy").astype(np.float64)  # (s1, s2, features)
   ranges, cyclic, patches = [1.0, 2 * np.pi], [1], [13, 17]
   spacing = get_difference_intervals(points.shape[:-1], ranges, cyclic)
   volume = volume_element(points, spacing, cyclic_dimensions=cyclic,
                           patch_sizes=patches, device="cpu")
   curvature = scalar_curvature(points, spacing, cyclic_dimensions=cyclic,
                                patch_sizes=patches, device="cpu")
   reach = reach_per_point(points, ranges, patches, cyclic,
                           subsample_points=4, device="cpu")
   print("Volume:", compute_total_volume(volume, ranges))
   print("Sampled reach:", reach.min())
   aligned_volume = volume[2:-2, :]  # Align the volume and curvature grids.
   print("Total scalar curvature:", compute_total_curvature(
       aligned_volume, curvature, ranges,
   ))

Volume and reach lose one sample at each nonperiodic end; curvature loses
three. Periodic axes retain their size. Patch sizes must exceed two for
derivatives/volume/reach and six for curvature. Smaller patches reduce
temporary memory; the full arrays remain in RAM.

Integrals use cropped-grid means and full domain lengths. Resolution and
boundary treatment affect accuracy. Reach searches sampled pairs; subsampling
can miss bottlenecks. Begin with small grids.

See :doc:`../api_reference/geometry` for shapes and normalization conventions,
and :doc:`../api_reference/datasets` for toy and image loading examples.
The `dataset notebooks
<https://github.com/koulakis/manifold-microscope/tree/main/notebooks/datasets_and_measures>`_
provide visualizations. For model training and representation analysis, see
:doc:`experiments`.
