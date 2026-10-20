"""Focused budget, checkpoint and decoder regressions; not yet executed."""
import numpy as np
import pytest
import torch
from torch import nn

from experiments_and_analysis.models.config import BetaVAEConfig
from experiments_and_analysis.models.beta_vae import solver as solver_module
from experiments_and_analysis.models.beta_vae.inference import get_intermediate_from_autoencoder


class SmallNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.))

    def forward(self, images):
        mu = images.new_zeros((len(images), 1))
        return self.offset.expand_as(images), mu, torch.zeros_like(mu)


@pytest.fixture
def make_solver(tmp_path, monkeypatch):
    monkeypatch.setattr(solver_module, "BetaVAE_H", lambda *args: SmallNetwork())

    def make(*, strict_budget=True, max_iter=1, max_epochs=2):
        config = BetaVAEConfig(ckpt_dir=tmp_path / "checkpoints", output_dir=tmp_path,
                               model="H", objective="H",
                               device="cpu", strict_budget=strict_budget,
                               max_iter=max_iter, max_epochs=max_epochs)
        solver = solver_module.Solver(config)
        solver.data_loader = [torch.zeros(2, 1, 1, 1) for _ in range(3)]
        return solver

    return make


@pytest.mark.parametrize("strict, expected", [(True, 1), (False, 3)])
def test_batch_budget_and_historical_epoch_finish(make_solver, monkeypatch, strict, expected):
    solver = make_solver(strict_budget=strict)
    saved = []
    monkeypatch.setattr(solver, "save_checkpoint", lambda filename=None: saved.append(filename))
    solver.train()
    assert solver.global_iter == expected
    parameter = next(solver.net.parameters())
    assert int(solver.optim.state[parameter]["step"]) == expected
    assert saved[-1] == "last.pth"


def test_strict_epoch_limit(make_solver, monkeypatch):
    solver = make_solver(max_iter=100, max_epochs=1)
    monkeypatch.setattr(solver, "save_checkpoint", lambda filename=None: None)
    solver.train()
    assert solver.global_iter == 3  # Exactly one three-batch epoch.


def test_full_checkpoint_with_numpy_metadata(make_solver):
    solver = make_solver()
    with torch.no_grad():
        solver.net.offset.fill_(0.7)
    torch.save(dict(iter=np.int64(7), win_states=dict(recon=None, kld=None, var=None, mu=None),
                    model_states=dict(net=solver.net.state_dict()),
                    optim_states=dict(optim=solver.optim.state_dict())), solver.ckpt_dir / "saved.pth")
    with torch.no_grad():
        solver.net.offset.zero_()
    solver.load_checkpoint("saved.pth")
    assert solver.global_iter == 7
    torch.testing.assert_close(solver.net.offset, torch.tensor(0.7))
    assert solver.net.offset.device.type == "cpu"


def test_mean_decoder_uses_mu_without_consuming_sampling_rng():
    class IdentityAutoencoder:
        z_dim = 1
        encoder = [nn.Identity()]
        decoder = [nn.Identity()]

    images = torch.tensor([[2., 0.], [2., 0.]])
    before = torch.random.get_rng_state()
    reconstruction, outputs = get_intermediate_from_autoencoder(
        IdentityAutoencoder(), images, True, decoder_sampling=False)
    np.testing.assert_allclose(reconstruction, 1 / (1 + np.exp(-2)), rtol=1e-6)
    np.testing.assert_array_equal(outputs["mu"], [[2.], [2.]])
    assert torch.equal(torch.random.get_rng_state(), before)
