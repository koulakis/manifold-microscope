Datasets
========

Toy manifolds
-------------

``microscope.datasets.toy_manifolds`` provides ``Circle``, ``Moons``, ``Sphere``
and ``Torus``. Each exposes ``sample``, ``tesselation``, ``measures`` and
``measures_pointwise``:

.. code-block:: python

   import numpy as np
   from microscope.datasets.toy_manifolds import Circle

   np.random.seed(0)
   manifold = Circle(R=1.0)
   points = manifold.sample(100)  # (100, 2), uniform in arc length
   print(round(manifold.measures().volume, 3))  # 6.283

``normalized=True`` requests unit-volume toy geometry. For sphere/torus
``tesselation``, pass ``device="cpu"`` to override their CUDA default.

Custom dSprites
---------------

``microscope.datasets.custom_dsprites.dsprites`` renders images locally.
This small grid is suitable for loading checks, but too coarse for reliable
curvature estimates:

.. code-block:: python

   from microscope.datasets.custom_dsprites import dsprites

   images, labels = dsprites(
       sprites=("square",), n_sizes=3, n_angles=8,
       max_translation=2, translation_stride=2,
       processes=(None, None, None, None), verbose=False,
   )
   print(images.shape)  # (1, 3, 8, 2, 2, 64, 64)

Axes are class, scale, rotation, y translation, x translation, height and width.

COIL-20
-------

Set ``COIL20_PATH`` as described in :doc:`../guide/installation`, then load views:

.. code-block:: python

   from microscope.datasets.coil20 import load_coil20

   views, labels = load_coil20(views_stride=4)
   print(views.shape)  # (20, 18, 128, 128), float32 intensities divided by 256

``extended_coil20`` adds scale and in-plane rotation.

Paper grids and metadata
------------------------

The generic loader returns image grids and finite-difference metadata:

.. code-block:: python

   from microscope.datasets.generic_dataset_loader import DatasetName, load_dataset_no_split

   (images, labels, patch_sizes, difference_intervals, volume_clip_slices,
    full_clip_slices, range_sizes, cyclic_dimensions) = load_dataset_no_split(
       DatasetName.custom_dsprites_balanced, number_of_dims=2,
   )
   # images: (classes, scale, rotation, height, width), scaled to [0, 1].
   grid = images.reshape(*images.shape[:-2], -1)
   # grid[0] is a single class with flattened image features.

Balanced custom dSprites supports dimensions 1–4; extended COIL-20 supports
1–3. These loaders construct full image grids before slicing and can need many
GiB of RAM. Clipping slices select the interior after derivative estimation.

See `dataset source
<https://github.com/koulakis/manifold-microscope/tree/main/microscope/datasets>`_
for parameter options and :doc:`experiments` for custom NPZ inputs to the runners.
