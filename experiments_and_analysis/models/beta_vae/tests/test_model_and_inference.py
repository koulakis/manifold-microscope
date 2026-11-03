import numpy as np
import pytest
import torch
from torch import nn

from experiments_and_analysis.models.beta_vae.inference import (
    gather_intermediate_outputs,
    get_dataloader,
)
from experiments_and_analysis.models.beta_vae.model import BetaVAE_B, BetaVAE_H, normal_init
from experiments_and_analysis.models.beta_vae.solver import kl_divergence, reconstruction_loss


@pytest.mark.parametrize("model_class,channels", [(BetaVAE_B, 1), (BetaVAE_H, 3)])
def test_model_forward_preserves_image_shape_and_exposes_latent_statistics(model_class, channels):
    model = model_class(z_dim=4, nc=channels)
    images = torch.zeros(1, channels, 64, 64)

    reconstruction, mu, logvar = model(images)

    assert reconstruction.shape == images.shape
    assert mu.shape == logvar.shape == (1, 4)


def test_training_losses_match_hand_calculated_values():
    images = torch.zeros(2, 1, 2, 2)
    logits = torch.zeros_like(images)
    bernoulli_loss = reconstruction_loss(images, logits, "bernoulli")
    gaussian_loss = reconstruction_loss(images, logits, "gaussian")
    assert bernoulli_loss is not None and gaussian_loss is not None
    assert bernoulli_loss.item() == pytest.approx(4 * np.log(2))
    assert gaussian_loss.item() == pytest.approx(1.0)

    mu = torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]])
    total, per_dimension, mean = kl_divergence(mu, torch.zeros_like(mu))
    torch.testing.assert_close(total, torch.tensor([7.5]))
    torch.testing.assert_close(per_dimension, torch.tensor([2.5, 5.0]))
    torch.testing.assert_close(mean, torch.tensor([3.75]))


def test_gather_intermediate_outputs_keeps_order_and_final_short_batch():
    class TinyAutoencoder(BetaVAE_H):
        z_dim = 1

        def __init__(self):
            nn.Module.__init__(self)
            encoder_projection = nn.Linear(4, 2, bias=False)
            self.encoder = nn.Sequential(nn.Flatten(), encoder_projection)
            decoder_projection = nn.Linear(1, 4, bias=False)
            self.decoder = nn.Sequential(decoder_projection, nn.Unflatten(1, (1, 2, 2)))
            with torch.no_grad():
                encoder_projection.weight.copy_(torch.tensor([[1.0] * 4, [0.0] * 4]))
                decoder_projection.weight.fill_(1.0)

    data = np.arange(20, dtype=np.float32).reshape(5, 2, 2)
    loader = get_dataloader(data, batch_size=2, num_workers=0, device="cpu")

    outputs = gather_intermediate_outputs(
        TinyAutoencoder(), loader, only_latent_and_output=True,
        device="cpu", decoder_sampling=False,
    )

    np.testing.assert_array_equal(outputs["input"], data[:, None])
    np.testing.assert_array_equal(outputs["mu"][:, 0], data.sum(axis=(1, 2)))
    assert outputs["output"].shape == (5, 1, 2, 2)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("layer_class", [nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d])
def test_normal_init_supports_optional_bias(layer_class, bias):
    if layer_class is nn.Linear:
        layer = layer_class(3, 2, bias=bias)
    elif layer_class is nn.Conv2d:
        layer = layer_class(1, 2, 3, bias=bias)
    else:
        layer = layer_class(2)
        if not bias:
            layer.bias = None

    normal_init(layer, mean=0.25, std=0.0)

    expected_weight = 1.0 if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)) else 0.25
    torch.testing.assert_close(layer.weight, torch.full_like(layer.weight, expected_weight))
    if bias:
        assert layer.bias is not None
        torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))
    else:
        assert layer.bias is None
