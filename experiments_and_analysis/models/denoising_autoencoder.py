from __future__ import annotations

from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import MLP
from torch import nn
import torch


def _validate_training_batch(batch: tuple[torch.Tensor, ...] | list[torch.Tensor], noise_along_normals: bool) -> None:
    """Require points and normals for normal noise, or points alone for isotropic noise.

    Args:
        batch: Point tensors of shape (b f), with a second normal tensor when required.
        noise_along_normals: Perturb points along supplied normals instead of isotropically.
    """
    if noise_along_normals:
        assert len(batch) == 2
    else:
        assert len(batch) == 1


class DenoisingAutoencoderPL(pl.LightningModule):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_channels: list[int] | tuple[int, ...] = (128, 128, 128),
        lr: float = 1e-3,
        noise_std: float = 0.1,
        noise_along_normals: bool = False
    ) -> None:
        """Initialize the denoising autoencoder and its training settings.

        Args:
            input_dim: Number of input coordinates.
            hidden_channels: Widths of the hidden layers.
            lr: Optimizer learning rate.
            noise_std: Standard deviation of the added Gaussian noise.
            noise_along_normals: Perturb points along supplied normals instead of isotropically.
        """
        super().__init__()
        self.lr = lr
        self.noise_std = noise_std
        self.noise_along_normals = noise_along_normals

        self.encoder = MLP(in_channels=input_dim, hidden_channels=hidden_channels)
        self.decoder = MLP(in_channels=hidden_channels[-1], hidden_channels=[*hidden_channels[-2::-1], input_dim])

        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input points.

        Args:
            x: Points of shape (b f).
        """
        return self.decoder(self.encoder(x))

    def add_noise(
        self,
        points: torch.Tensor,
        normals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Perturb points with Gaussian noise.

        Args:
            points: Points of shape (b f).
            normals: Unit normals of shape (b f), required for normal-directed noise.
        """
        if self.noise_along_normals:
            noise_norm = torch.normal(
                mean=0,
                std=self.noise_std,
                size=(len(points), 1),
                dtype=points.dtype
            )
            noise = noise_norm * normals
        else:
            noise = torch.normal(
                mean=0,
                std=self.noise_std,
                size=points.shape,
                dtype=points.dtype
            )
        return points + noise.to(points.device)

    def training_step(
        self,
        batch: tuple[torch.Tensor, ...] | list[torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Train on a batch containing points and any required normals.

        Args:
            batch: Point tensors of shape (b f), followed by normals when used.
            batch_idx: Batch index supplied by Lightning.
        """
        _validate_training_batch(batch, self.noise_along_normals)
        if self.noise_along_normals:
            points, normals = batch
            noisy_points = self.add_noise(points, normals)
        else:
            points = batch[0]
            noisy_points = self.add_noise(points)
        out = self(noisy_points)
        loss = self.criterion(out, points)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False
        )

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create the AdamW optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


class DenoisingAutoencoder:
    def __init__(
        self,
        input_dim: int = 3,
        hidden_channels: list[int] | tuple[int, ...] = (32, 16),
        lr: float = 1e-3,
        noise_std: float = 0.1,
        noise_along_normals: bool = False,
        device: int = 0
    ) -> None:
        """Initialize the denoising autoencoder and its training settings.

        Args:
            input_dim: Number of input coordinates.
            hidden_channels: Widths of the hidden layers.
            lr: Optimizer learning rate.
            noise_std: Standard deviation of the added Gaussian noise.
            noise_along_normals: Perturb points along supplied normals instead of isotropically.
            device: Device used for tensor computations.
        """
        self.noise_along_normals = noise_along_normals
        self.model = DenoisingAutoencoderPL(
            input_dim=input_dim,
            hidden_channels=hidden_channels,
            lr=lr,
            noise_std=noise_std,
            noise_along_normals=noise_along_normals
        )
        self.device = f"cuda:{device}"
        self.devices = [device]

    def fit(
        self,
        samples: np.ndarray,
        unitary_normals: Optional[np.ndarray],
        max_epochs: int = 1_000,
        train_loss_stop_threshold: float = 1e-3,
        batch_size: int = 32,
        num_workers: int = 1
    ) -> None:
        """Fit with Gaussian corruption and loss-based early stopping.

        Args:
            samples: Training points of shape (n f).
            unitary_normals: Unit normals of shape (n f), required for normal-directed noise.
            max_epochs: Maximum number of training epochs.
            train_loss_stop_threshold: Training loss threshold for early stopping.
            batch_size: Number of points or images per batch.
            num_workers: Number of data-loader workers.
        """
        if self.noise_along_normals:
            dataset = TensorDataset(
                torch.from_numpy(samples),
                torch.from_numpy(unitary_normals)
            )
        else:
            dataset = TensorDataset(torch.from_numpy(samples))

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="train_loss",
            mode="min",
            min_delta=-1,
            stopping_threshold=train_loss_stop_threshold
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            devices=self.devices,
            callbacks=[early_stopping_callback],
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=True
        )
        trainer.fit(self.model, train_dataloaders=dataloader)

    def predict(
        self,
        points: np.ndarray,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> np.ndarray:
        """Reconstruct points in batches.

        Args:
            points: Query points of shape (n f).
            batch_size: Number of points or images per batch.
            num_workers: Number of data-loader workers.
        """
        dataset = TensorDataset(torch.from_numpy(points))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        self.model.eval()
        self.model.to(self.device)

        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                predictions.append(self.model(batch[0].to(self.device)).cpu().numpy())

        predictions = np.concatenate(predictions)

        self.model.train()

        return predictions
