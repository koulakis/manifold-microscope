"""beta-VAE model definitions.

Copied and adapted from https://github.com/1Konny/Beta-VAE. The upstream project is MIT licensed; see this package's
`LICENSE` and the repository's `docs/source/third_party/beta_vae.rst` for copyright and attribution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample latent codes using the Gaussian reparameterization.

    Args:
        mu: Latent means of shape (b z), or (b z 1 1).
        logvar: Latent log variances with the same shape as mu.
    """
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size: tuple[int, ...]) -> None:
        """Store the target tensor shape.

        Args:
            size: Target dimensions, optionally including one inferred dimension (-1).
        """
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape the input tensor to the stored size.

        Args:
            tensor: Tensor of shape (...).
        """
        return tensor.view(self.size)


def kaiming_init(m: nn.Module) -> None:
    """Apply Kaiming initialization to a supported module.

    Args:
        m: Layer whose weights and biases are initialized.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(
    m: nn.Module,
    mean: float,
    std: float
) -> None:
    """Apply normal initialization to a supported module.

    Args:
        m: Layer whose weights and biases are initialized.
        mean: Mean of the normal weight distribution.
        std: Standard deviation of the normal weight distribution.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()



class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def weight_init(self) -> None:
        """Initialize the encoder and decoder weights."""
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def __init__(
        self,
        z_dim: int = 10,
        nc: int = 3
    ) -> None:
        """Construct the encoder and decoder and initialize their weights.

        Args:
            z_dim: Number of latent coordinates.
            nc: Number of image channels.
        """
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into latent Gaussian parameters.

        Args:
            x: Images of shape (b c 64 64).
        """
        return self.encoder(x)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes into image logits.

        Args:
            z: Latent codes of shape (b z).
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction logits, latent means and log variances.

        Args:
            x: Images of shape (b c 64 64).
        """
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar


class BetaVAE_B(BetaVAE_H):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def weight_init(self) -> None:
        """Initialize the encoder and decoder weights."""
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def __init__(
        self,
        z_dim: int = 10,
        nc: int = 1
    ) -> None:
        """Construct the encoder and decoder and initialize their weights.

        Args:
            z_dim: Number of latent coordinates.
            nc: Number of image channels.
        """
        super(BetaVAE_B, self).__init__()
        self.nc = nc
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.ReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1), # B,  nc, 64, 64
        )
        self.weight_init()

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into latent Gaussian parameters.

        Args:
            x: Images of shape (b c 64 64).
        """
        return self.encoder(x)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes into image logits.

        Args:
            z: Latent codes of shape (b z).
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reconstruction logits, latent means and log variances.

        Args:
            x: Images of shape (b c 64 64).
        """
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z).view(x.size())

        return x_recon, mu, logvar
