from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torchvision.ops import MLP
from torch import nn
import torch


class DenoisingAutoencoderPL(pl.LightningModule):
    def __init__(
        self,
        input_dim=3,
        hidden_channels=(128, 128, 128),
        lr=1e-3,
        noise_std: float = 0.1,
        noise_along_normals: bool = False
    ):
        super().__init__()
        self.lr = lr
        self.noise_std = noise_std
        self.noise_along_normals = noise_along_normals

        self.encoder = MLP(in_channels=input_dim, hidden_channels=hidden_channels)
        self.decoder = MLP(in_channels=hidden_channels[-1], hidden_channels=[*hidden_channels[-2::-1], input_dim])

        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def add_noise(
        self,
        points: torch.Tensor,
        normals: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor] | torch.Tensor, batch_idx: int) -> float:
        if self.noise_along_normals:
            assert len(batch) == 2
            points, normals = batch
            noisy_points = self.add_noise(points, normals)
        else:
            assert len(batch) == 1
            points = batch[0]
            noisy_points = self.add_noise(points)
        out = self(noisy_points)
        loss = self.criterion(out, points)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


class DenoisingAutoencoder:
    def __init__(
        self,
        input_dim: int = 3,
        hidden_channels: list[int] = (32, 16),
        lr: float = 1e-3,
        noise_std: float = 0.1,
        noise_along_normals: bool = False,
        device: int = 0
    ) -> None:
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

    def predict(self, points: np.ndarray, batch_size: int = 32, num_workers: int = 4) -> np.ndarray:
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
