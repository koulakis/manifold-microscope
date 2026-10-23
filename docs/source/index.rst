The Data Manifold under the Microscope
======================================

Manifold Microscope provides toy and image datasets and finite-difference
estimates of volume, scalar curvature and reach on ordered parameter grids.
It accompanies `Koulakis and Seibold, The Data Manifold under the Microscope
(ICML 2026) <https://proceedings.mlr.press/v306/koulakis26a.html>`_.

Start with :doc:`guide/installation` and the CPU example in
:doc:`guide/getting_started`. To run the paper experiments, see
:doc:`guide/reproducibility`.

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   guide/installation
   guide/getting_started
   guide/experiments
   guide/configuration
   guide/outputs
   guide/reproducibility
   guide/building

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_reference/datasets
   api_reference/geometry
   api_reference/experiments

.. toctree::
   :maxdepth: 1
   :caption: Project

   third_party/beta_vae

Citation
--------

If you use this code, please cite the paper:

.. literalinclude:: ../../README.md
   :language: bibtex
   :start-after: ```bibtex
   :end-before: ```

`Source and issue tracker <https://github.com/koulakis/manifold-microscope>`_.
The main package uses the BSD-3-Clause license; adapted beta-VAE code uses
the MIT license. See :doc:`third_party/beta_vae` for attribution and license texts.
