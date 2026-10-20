"""Checkpoint loading and functional beta-VAE inference."""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from experiments_and_analysis.models.beta_vae.dataset import CustomTensorDataset
from experiments_and_analysis.models.beta_vae.model import reparametrize, BetaVAE_B, BetaVAE_H



def get_dataloader(data, batch_size, num_workers, device="cuda:0"):
    data = torch.from_numpy(data).unsqueeze(1).float()
    train_kwargs = {'data_tensor': data}

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
    random_model=False,
    latent_dim=10,
    device="cuda:0"
) -> BetaVAE_B | BetaVAE_H:
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


def get_intermediate_from_autoencoder(autoencoder, images, only_latent_and_output: bool,
                                     decoder_sampling: bool = True, representations=None):
    intermediate_outputs = {}
    x = images
    for i, layer in enumerate(autoencoder.encoder):
        x = layer(x)
        if (isinstance(layer, nn.ReLU) and i > 3) and (not only_latent_and_output):
            name = f"encoder_relu_{i}"
            if representations is None or name in representations:
                intermediate_outputs[name] = x.cpu().numpy().copy()

    distributions = x
    mu = distributions[:, :autoencoder.z_dim]
    logvar = distributions[:, autoencoder.z_dim:]
    z = reparametrize(mu, logvar) if decoder_sampling else mu

    if representations is None or "mu" in representations:
        intermediate_outputs["mu"] = mu.cpu().numpy().copy()

    x = z
    for i, layer in enumerate(autoencoder.decoder):
        x = layer(x)
        if (isinstance(layer, nn.ReLU) and i < 7) and (not only_latent_and_output):
            name = f"decoder_relu_{i}"
            if representations is None or name in representations:
                intermediate_outputs[name] = x.cpu().numpy().copy()

    x_recon = F.sigmoid(x).cpu().numpy()

    return x_recon, intermediate_outputs


def gather_intermediate_outputs(
    autoencoder,
    data_loader,
    only_latent_and_output: bool,
    device="cuda:0",
    decoder_sampling: bool = True
) -> dict[str, np.ndarray]:
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

def representation_names(model):
    return (['input'] + [f'encoder_relu_{i}' for i, layer in enumerate(model.encoder)
                         if isinstance(layer, nn.ReLU) and i > 3] + ['mu'] +
            [f'decoder_relu_{i}' for i, layer in enumerate(model.decoder)
             if isinstance(layer, nn.ReLU) and i < 7] + ['output'])


def batches(model, data, representations, *, batch_size=64, device='cpu',
            num_workers=0, seed=42, decoder_sampling=True):
    unknown = set(representations) - set(representation_names(model))
    if unknown:
        raise ValueError(f'Unknown beta-VAE representations: {sorted(unknown)}')
    torch.manual_seed(seed)
    model.to(device).eval()
    loader = get_dataloader(data, batch_size, num_workers, device)
    start = 0
    with torch.no_grad():
        for images in loader:
            reconstruction, outputs = get_intermediate_from_autoencoder(
                model, images.to(device), False, decoder_sampling, representations)
            outputs.update(input=images.numpy(), output=reconstruction)
            yield start, {name: outputs[name] for name in representations}
            start += len(images)
