from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import typer

from microscope.datasets.generic_dataset_loader import DatasetName, load_dataset
from representation_learning.beta_vae.dataset import CustomTensorDataset
from representation_learning.beta_vae.model import reparametrize, BetaVAE_B, BetaVAE_H

app = typer.Typer(pretty_exceptions_enable=False)


def get_dataloader(data, batch_size, num_workers):
    data = torch.from_numpy(data).unsqueeze(1).float()
    train_kwargs = {'data_tensor': data}

    dataset = CustomTensorDataset(**train_kwargs)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
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


def get_intermediate_from_autoencoder(autoencoder, images, only_latent_and_output: bool):
    intermediate_outputs = {}
    x = images
    for i, layer in enumerate(autoencoder.encoder):
        x = layer(x)
        if (isinstance(layer, nn.ReLU) and i > 3) and (not only_latent_and_output):
            intermediate_outputs[f"encoder_relu_{i}"] = x.cpu().numpy()

    distributions = x
    mu = distributions[:, :autoencoder.z_dim]
    logvar = distributions[:, autoencoder.z_dim:]
    z = reparametrize(mu, logvar)

    intermediate_outputs["mu"] = mu.cpu().numpy()

    x = z
    for i, layer in enumerate(autoencoder.decoder):
        x = layer(x)
        if (isinstance(layer, nn.ReLU) and i < 7) and (not only_latent_and_output):
            intermediate_outputs[f"decoder_relu_{i}"] = x.cpu().numpy()

    x_recon = F.sigmoid(x).cpu().numpy()

    return x_recon, intermediate_outputs


def gather_intermediate_outputs(
    autoencoder,
    data_loader,
    only_latent_and_output: bool,
    device="cuda:0"
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
                only_latent_and_output
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


@app.command()
def main(
    dataset: DatasetName = typer.Option(...),
    number_of_dims: int = typer.Option(...),
    only_latent_and_output: bool = True,
    checkpoint_path: Path = typer.Option(...),
    output_dir: Path = typer.Option(...),
    random_model: bool = False,
    batch_size: int = 64,
    num_workers: int = 2,
    latent_dim: int = 10,
    device: str = "cuda:0"
):
    output_dir.mkdir(exist_ok=True, parents=True)

    data, _ = load_dataset(
        dataset_name=dataset,
        number_of_dims=number_of_dims,
        ratio_per_dim=False,
        training_ratio=1.0,
        noise_sigma=0,
        save_train_idx=False,
        weight_subsampling_by_manifold_volume=False,
        return_full_datasets=True,
        full_datasets_unclipped=True
    )
    data = (data - data.min()) / (data.max() - data.min())

    if len(data) < batch_size:
        batch_size = len(data)
    dataloader = get_dataloader(data, batch_size, num_workers)

    match dataset:
        case DatasetName.extended_coil20:
            model_type = "H"
        case _:
            model_type = "B"

    autoencoder = load_autoencoder_from_checkpoint(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        random_model=random_model,
        latent_dim=latent_dim,
        device=device
    )

    intermediate_outputs = gather_intermediate_outputs(
        autoencoder=autoencoder,
        data_loader=dataloader,
        only_latent_and_output=only_latent_and_output,
        device=device
    )

    filename = f'intermediate_outputs_{checkpoint_path.name if not random_model else "random_init"}'
    np.savez(output_dir / filename, **intermediate_outputs)


if __name__ == "__main__":
    app()
