Licenses and beta-VAE attribution
=================================

Manifold Microscope
-------------------

The main package uses the BSD-3-Clause license:

.. literalinclude:: ../../../LICENSE
   :language: text

Third-party notice for beta-VAE code
------------------------------------

The code in ``experiments_and_analysis/models/beta_vae`` is copied and adapted
from `1Konny/Beta-VAE <https://github.com/1Konny/Beta-VAE>`_. The original
author and copyright holder is WonKwang Lee. It is licensed under the MIT
license, retained in that directory and reproduced below.

The upstream project is a PyTorch implementation/reproduction of models
associated with:

* Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a Constrained
  Variational Framework", ICLR 2017.
* Burgess et al., "Understanding disentangling in beta-VAE", arXiv:1804.03599, 2018.

Local modifications integrate the dataset loaders and experiment configuration,
update device handling, checkpoints and plotting, replace Visdom reporting,
and expose intermediate representations during inference. Architectures and
training losses are retained. See :doc:`../api_reference/experiments` for
training and inference entry points.

MIT license
-----------

.. literalinclude:: ../../../experiments_and_analysis/models/beta_vae/LICENSE
   :language: text
