"""Checkpoint loading and functional beta-VAE inference."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from experiments_and_analysis.models.beta_vae.dataset import CustomTensorDataset
from experiments_and_analysis.models.beta_vae.model import reparametrize, BetaVAE_B, BetaVAE_H


def get_dataloader(
    data: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: str = "cuda:0"
) -> DataLoader:
    """Build a deterministic-order image loader.

    Args:
        data: Grayscale images of shape (n h w).
        batch_size: Number of points or images per batch.
        num_workers: Number of data-loader workers.
        device: Device used for tensor computations.

    Returns:
        DataLoader yielding float32 grayscale image tensors of shape (b 1 h w) in input order.
    """
    data_tensor = torch.from_numpy(data).unsqueeze(1).float()
    train_kwargs = {'data_tensor': data_tensor}

    dataset = CustomTensorDataset(**train_kwargs)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.device(device).type == "cuda",
        drop_last=False
    )

    data_loader = train_loader

    return data_loader


def load_autoencoder_from_checkpoint(
    model_type: str,
    checkpoint_path: Path,
    random_model: bool = False,
    latent_dim: int = 10,
    device: str = "cuda:0"
) -> BetaVAE_B | BetaVAE_H:
    """Load a beta-VAE checkpoint or initialize random weights.

    Args:
        model_type: Beta-VAE architecture identifier, B or H.
        checkpoint_path: Checkpoint file to load unless random_model is enabled.
        random_model: Use newly initialized model weights.
        latent_dim: Number of latent coordinates.
        device: Device used for tensor computations.

    Returns:
        Initialized or checkpoint-loaded beta-VAE on the requested device, in evaluation mode.
    """
    if model_type == "B":
        autoencoder = BetaVAE_B(z_dim=latent_dim, nc=1)
    elif model_type == "H":
        autoencoder = BetaVAE_H(z_dim=latent_dim, nc=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}.")

    if not random_model:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        autoencoder.load_state_dict(checkpoint['model_states']['net'])

    autoencoder.to(device)
    autoencoder.eval()
    return autoencoder


def get_intermediate_from_autoencoder(
    autoencoder: BetaVAE_H,
    images: torch.Tensor,
    only_latent_and_output: bool,
    decoder_sampling: bool = True,
    representations: list[str] | None = None
) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """Collect selected activations from an image batch.

    Args:
        autoencoder: Autoencoder whose activations or reconstructions are collected.
        images: Image batch of shape (b c h w).
        only_latent_and_output: Collect the latent mean and reconstruction without hidden layers.
        decoder_sampling: Sample latent codes instead of decoding the latent mean.
        representations: Activation names to collect, or None for all eligible layers.

    Returns:
        (reconstruction, activations). Reconstruction has shape (b c h w), or is None when skipped; activation arrays
        have shape (b, *layer_shape), including latent means (b z).
    """
    intermediate_outputs = {}
    requested = set(representations) if representations is not None else None
    x = images
    for i, layer in enumerate(autoencoder.encoder):
        x = layer(x)
        if (isinstance(layer, nn.ReLU) and i > 3) and (not only_latent_and_output):
            name = f"encoder_relu_{i}"
            if representations is None or name in representations:
                intermediate_outputs[name] = x.cpu().numpy().copy()
                if requested is not None and requested.issubset(intermediate_outputs):
                    return None, intermediate_outputs

    distributions = x
    mu = distributions[:, :autoencoder.z_dim]
    logvar = distributions[:, autoencoder.z_dim:]
    if representations is None or "mu" in representations:
        intermediate_outputs["mu"] = mu.cpu().numpy().copy()
        if requested is not None and requested.issubset(intermediate_outputs):
            return None, intermediate_outputs

    z = reparametrize(mu, logvar) if decoder_sampling else mu
    x = z
    for i, layer in enumerate(autoencoder.decoder):
        x = layer(x)
        if (isinstance(layer, nn.ReLU) and i < 7) and (not only_latent_and_output):
            name = f"decoder_relu_{i}"
            if representations is None or name in representations:
                intermediate_outputs[name] = x.cpu().numpy().copy()
                if requested is not None and requested.issubset(intermediate_outputs):
                    return None, intermediate_outputs

    x_recon = F.sigmoid(x).cpu().numpy() if requested is None or 'output' in requested else None

    return x_recon, intermediate_outputs


def gather_intermediate_outputs(
    autoencoder: BetaVAE_H,
    data_loader: DataLoader,
    only_latent_and_output: bool,
    device: str = "cuda:0",
    decoder_sampling: bool = True
) -> dict[str, np.ndarray]:
    """Collect image, latent, layer and reconstruction arrays over a loader.

    Args:
        autoencoder: Autoencoder whose activations or reconstructions are collected.
        data_loader: Loader yielding image batches of shape (b c h w).
        only_latent_and_output: Collect the latent mean and reconstruction without hidden layers.
        device: Device used for tensor computations.
        decoder_sampling: Sample latent codes instead of decoding the latent mean.

    Returns:
        Dictionary of representation arrays of shape (n, *layer_shape), including input/output images (n c h w) and
        latent means (n z).
    """
    autoencoder.to(device)
    autoencoder.eval()

    intermediate_outputs = {}

    with torch.no_grad():
        for images in tqdm(data_loader):
            if "input" not in intermediate_outputs:
                intermediate_outputs["input"] = []
            intermediate_outputs["input"].append(images.cpu().numpy())

            images = images.to(device)

            reconstruction, intermediate = get_intermediate_from_autoencoder(
                autoencoder,
                images,
                only_latent_and_output,
                decoder_sampling=decoder_sampling
            )

            for name, tensor_np in intermediate.items():
                if name not in intermediate_outputs:
                    intermediate_outputs[name] = []
                intermediate_outputs[name].append(tensor_np)

            if "output" not in intermediate_outputs:
                intermediate_outputs["output"] = []
            intermediate_outputs["output"].append(reconstruction)

    # Stack all intermediate outputs along the 0 axis
    for name in intermediate_outputs:
        intermediate_outputs[name] = np.concatenate(intermediate_outputs[name], axis=0)

    return intermediate_outputs

def representation_names(model: BetaVAE_H) -> list[str]:
    """List the available input, hidden, latent and output representations.

    Args:
        model: Autoencoder used for inference.

    Returns:
        Available representation names in input, encoder, latent, decoder and output order.
    """
    return (['input'] + [f'encoder_relu_{i}' for i, layer in enumerate(model.encoder)
                         if isinstance(layer, nn.ReLU) and i > 3] + ['mu'] +
            [f'decoder_relu_{i}' for i, layer in enumerate(model.decoder)
             if isinstance(layer, nn.ReLU) and i < 7] + ['output'])


def batches(
    model: BetaVAE_H,
    data: np.ndarray,
    representations: list[str],
    batch_size: int = 64,
    device: str = 'cpu',
    num_workers: int = 0,
    seed: int = 42,
    decoder_sampling: bool = True
) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    """Yield selected representation arrays and their starting sample indices.

    Args:
        model: Autoencoder used for inference.
        data: Grayscale images of shape (n h w).
        representations: Names of the representations to collect.
        batch_size: Number of points or images per batch.
        device: Device used for tensor computations.
        num_workers: Number of data-loader workers.
        seed: Random seed.
        decoder_sampling: Sample latent codes instead of decoding the latent mean.

    Yields:
        (start_index, outputs), with representation arrays of shape (b, *layer_shape); input/output images have shape (b
        1 h w), and latent means have shape (b z).
    """
    unknown = set(representations) - set(representation_names(model))
    if unknown:
        raise ValueError(f'Unknown beta-VAE representations: {sorted(unknown)}')
    torch.manual_seed(seed)
    model.to(device).eval()
    loader = get_dataloader(data, batch_size, num_workers, device)
    start = 0
    with torch.no_grad():
        for images in loader:
            outputs = {}
            selected = [name for name in representations if name != 'input']
            if selected:
                reconstruction, outputs = get_intermediate_from_autoencoder(
                    model,
                    images.to(device),
                    False,
                    decoder_sampling,
                    selected
                )
                if 'output' in selected:
                    outputs['output'] = reconstruction
            if 'input' in representations:
                outputs['input'] = images.numpy()
            yield start, outputs
            start += len(images)
