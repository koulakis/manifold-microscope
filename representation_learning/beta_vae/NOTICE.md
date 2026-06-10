# Third-party notice for beta-VAE code

The code in this package is copied and adapted from:

- **1Konny/Beta-VAE**
- Repository: https://github.com/1Konny/Beta-VAE
- Original author/copyright holder: WonKwang Lee
- License: MIT License, included in `LICENSE`

The upstream project is a PyTorch implementation/reproduction of beta-VAE models associated with:

- Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework", ICLR 2017.
- Burgess et al., "Understanding disentangling in beta-VAE", arXiv:1804.03599, 2018.

Local modifications in this repository include integration with the manifold-microscope dataset loaders and experiment
configuration, device handling updates, checkpoint and plotting changes, removal/replacement of Visdom-based reporting,
and inference utilities for extracting intermediate representations.
