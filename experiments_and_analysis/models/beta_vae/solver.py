"""Training solver for the adapted beta-VAE implementation.

Copied and adapted from https://github.com/1Konny/Beta-VAE. The upstream project is MIT licensed; see this package's
`LICENSE` and the repository's `docs/source/third_party/beta_vae.rst` for copyright and attribution.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from experiments_and_analysis.models.config import BetaVAEConfig

warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import matplotlib

import torch
import torch.optim as optim
import torch.nn.functional as F

from experiments_and_analysis.models.beta_vae.model import BetaVAE_H, BetaVAE_B

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def reconstruction_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    distribution: str
) -> torch.Tensor | None:
    """Compute the reconstruction loss for the selected likelihood.

    Args:
        x: Target images of shape (b c h w); the batch must be nonempty.
        x_recon: Reconstruction logits of shape (b c h w).
        distribution: Reconstruction likelihood, bernoulli or gaussian.
    """
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute total, dimension-wise and mean latent KL divergence.

    Args:
        mu: Latent means of shape (b z), or (b z 1 1).
        logvar: Latent log variances with the same shape as mu.
    """
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class DataGather(object):
    @staticmethod
    def get_empty_data_dict() -> dict[str, list]:
        """Create empty lists for training statistics."""
        return dict(
            iter=[],
            recon_loss=[],
            hausdorff_dists=[],
            avg_dists=[],
            total_kld=[],
            dim_wise_kld=[],
            mean_kld=[],
            mu=[],
            var=[],
            images=[]
        )

    def __init__(self) -> None:
        """Initialize the collected training statistics."""
        self.data = self.get_empty_data_dict()

    def insert(self, **kwargs: object) -> None:
        """Append the supplied training statistics.

        Args:
            **kwargs: Named statistics; tensor values have the shape of their corresponding metric.
        """
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self) -> None:
        """Clear all collected training statistics."""
        self.data = self.get_empty_data_dict()


def plot_training_curve(train_losses: list[float], plots_dir: Path) -> None:
    """Save the reconstruction-loss training curve.

    Args:
        train_losses: Reconstruction losses in training-step order.
        plots_dir: Directory for the saved training curve.
    """
    plt.figure(figsize=(8, 5))
    epochs = len(train_losses)
    plt.plot(range(1, epochs + 1), train_losses, marker='o')
    plt.title("Training Loss Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Binary Cross-entropy Loss")
    plt.grid()
    plt.savefig(plots_dir / "loss_curve.png")
    plt.close()


def visualize_reconstructions(
    output_dir: Path,
    step: int,
    autoencoder: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
    num_images: int = 8
) -> None:
    """Save a small panel of original images and reconstructions.

    Args:
        output_dir: Directory for generated outputs.
        step: Training step used in the output filename.
        autoencoder: Autoencoder whose activations or reconstructions are collected.
        data_loader: Loader yielding image batches of shape (b c h w).
        device: Device used for tensor computations.
        num_images: Maximum number of images to display.
    """
    autoencoder.eval()
    images = next(iter(data_loader))
    if len(images) < num_images:
        num_images = len(images)
    images = images[:num_images].to(device)
    with torch.no_grad():
        reconstructed_images, _, _ = autoencoder(images)
        reconstructed_images = torch.sigmoid(reconstructed_images)

    # Move data to CPU for visualization
    original_images = images.cpu().numpy()
    reconstructed_images = reconstructed_images.cpu().numpy()

    # Plot original and reconstructed images
    fig, axes = plt.subplots(2, num_images, figsize=(12, 4))
    for i in range(num_images):
        # Original images
        axes[0, i].imshow(
            original_images[i].squeeze(),
            cmap='gray',
            vmin=0,
            vmax=1
        )
        axes[0, i].axis('off')
        # Reconstructed images
        axes[1, i].imshow(
            reconstructed_images[i].squeeze(),
            cmap='gray',
            vmin=0,
            vmax=1
        )
        axes[1, i].axis('off')

    axes[0, 0].set_title("Original", fontsize=10)
    axes[1, 0].set_title("Reconstruction", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / f"reconstructions_step_{step:08}.png")
    plt.close(fig)


class Solver(object):
    def _to_device(self, tensor_or_module: torch.Tensor | torch.nn.Module) -> torch.Tensor | torch.nn.Module:
        """Move tensor or module to the configured device.

        Args:
            tensor_or_module: Tensor of shape (...) or module to move to the configured device.
        """
        return tensor_or_module.to(self.device)

    def load_checkpoint(self, filename: str) -> None:
        """Restore solver state when the checkpoint file exists.

        Args:
            filename: Checkpoint filename relative to the checkpoint directory.
        """
        file_path = self.ckpt_dir / filename
        if os.path.isfile(file_path):
            # Full solver checkpoints are trusted local training outputs.
            checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def __init__(
        self,
        args: BetaVAEConfig,
        data_loader: DataLoader | None = None
    ) -> None:
        """Initialize model, optimizer, checkpoint settings and training data.

        Args:
            args: Model, optimizer, device, output and training settings.
            data_loader: Loader yielding image batches of shape (b c h w).
        """
        self.args = args

        self.device = args.device
        self.use_cuda = self.device.startswith("cuda")
        self.max_iter = args.max_iter
        self.max_epochs = args.max_epochs
        self.global_iter = 0
        self.plot_interval = args.plot_interval

        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.loss_threshold = args.loss_threshold

        self.nc = 1

        if args.model == 'H':
            net = BetaVAE_H
            self.decoder_dist = 'gaussian'
        elif args.model == 'B':
            net = BetaVAE_B
            self.decoder_dist = 'bernoulli'
        else:
            raise ValueError(f"Unknown model {args.model}.")

        self.net = self._to_device(net(self.z_dim, self.nc))
        print(f"[DEBUG] Model device: {next(self.net.parameters()).device}")
        self.optim = optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2)
        )

        self.win_recon = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None

        self.ckpt_dir = args.ckpt_dir
        self.ckpt_dir.mkdir(exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = args.output_dir
        self.output_dir.mkdir(exist_ok=True)

        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step

        self.batch_size = args.batch_size

        self.data_loader = data_loader

        self.gather = DataGather()

    def net_mode(self, train: bool) -> None:
        """Set the network training or evaluation mode.

        Args:
            train: Enable training mode when true, or evaluation mode when false.
        """
        if not isinstance(train, bool):
            raise 'Only bool type is supported. True or False'

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(
        self,
        filename: str | None = None,
        silent: bool = True
    ) -> None:
        """Save solver state, loss arrays and reconstruction plots.

        Args:
            filename: Checkpoint filename; None derives a name from recent training losses.
            silent: Suppress the checkpoint confirmation message.
        """
        model_states = {'net': self.net.state_dict()}
        optim_states = {'optim': self.optim.state_dict()}
        win_states = {
            'recon': self.win_recon,
            'kld': self.win_kld,
            'mu': self.win_mu,
            'var': self.win_var
        }
        states = {
            'iter': self.global_iter,
            'win_states': win_states,
            'model_states': model_states,
            'optim_states': optim_states
        }

        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu().numpy()
        avg_dists = torch.stack(self.gather.data['avg_dists']).cpu().numpy()
        hausdorff_dists = torch.stack(self.gather.data['hausdorff_dists']).cpu().numpy()
        file_path = (
            self.ckpt_dir /
            (
                f"iter_{self.global_iter}_rec_loss_{recon_losses[-10:].mean()}"
                f"_haus_dists_{hausdorff_dists[-10:].mean()}_avg_dists_{avg_dists[-10:].mean()}"
                if filename is None else filename
            )
        )
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)

        np.savez(os.path.join(self.ckpt_dir, "train_losses.npz"), recon_losses_train=recon_losses)

        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        visualize_reconstructions(
            plots_dir,
            step=self.global_iter,
            autoencoder=self.net,
            data_loader=self.data_loader,
            device=self.device
        )
        plot_training_curve(
            list(recon_losses),
            Path(plots_dir)
        )

        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def train(self) -> None:
        """Fit the model; strict_budget opts into exact batch and epoch limits.

        Historical default behavior finishes an epoch after a stop condition.
        """
        strict_budget = self.args.strict_budget
        if strict_budget:
            if self.max_iter < 1 or self.max_epochs < 1:
                raise ValueError("Strict training budgets must be positive.")
            if self.global_iter >= self.max_iter:
                print("Training budget already reached by the loaded checkpoint.")
                return
        # print(
        #     f"Train size: {len(self.data_loader.dataset)}, train perc: {self.args.training_ratio}, "
        #     f"dataset: {self.args.dataset}, dim: {self.args.number_of_dims}"
        # )
        # return
        self.net_mode(train=True)
        self.C_max = self._to_device(torch.FloatTensor([self.C_max]))
        out = False

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        epoch = 0
        epoch_save = False
        print(f"Model parameters: {sum([np.prod(p.shape) for p in self.net.parameters()]) / 1_000} thousands(s).")
        losses = []
        avg_distance = None
        patience_exhausted = False
        update_steps = 0
        previous_loss_quantile = float("inf")
        patience_epochs = max(5, int(self.args.patience_percentage_epochs * self.max_epochs))
        patience_violations = 0
        while not out:
            if strict_budget and epoch >= self.max_epochs:
                break
            if epoch % max(1, self.plot_interval - 1) == 0:
                epoch_save = True
            for x in self.data_loader:
                self.global_iter += 1
                update_steps += 1
                if (not self.args.update_pbar_on_epochs) or epoch_save:
                    pbar.update(update_steps)
                    update_steps = 0

                x = self._to_device(x)
                x_recon, mu, logvar = self.net(x)
                # noinspection PyTypeChecker
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                if self.objective == 'H':
                    beta_vae_loss = recon_loss + self.beta*total_kld
                elif self.objective == 'B':
                    # noinspection PyTypeChecker
                    C = torch.clamp(
                        self.C_max / self.C_stop_iter * self.global_iter,
                        min=0,
                        max=self.C_max.data[0]
                    )
                    beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()
                else:
                    raise ValueError(f"Unknown model type in objective: {self.objective}")

                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                self.gather.insert(
                    iter=self.global_iter,
                    recon_loss=recon_loss.data
                )

                x_recon = F.sigmoid(x_recon)
                dist_norm = torch.linalg.norm((x_recon - x).flatten(start_dim=1), axis=1)
                avg_distance = dist_norm.mean()
                hausdorff_distance = dist_norm.max()

                if (self.global_iter % self.gather_step == 0) or epoch_save:
                    self.gather.insert(
                        mu=mu.mean(0).data,
                        var=logvar.exp().mean(0).data,
                        total_kld=total_kld.data,
                        dim_wise_kld=dim_wise_kld.data,
                        mean_kld=mean_kld.data,
                        avg_dists=avg_distance.data,
                        hausdorff_dists=hausdorff_distance.data
                    )

                if (self.global_iter % self.display_step == 0) or epoch_save:
                    pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                        self.global_iter,
                        recon_loss.detach().item(),
                        total_kld.detach().item(),
                        mean_kld.detach().item()
                    )
                    )

                    # var = logvar.exp().mean(0).data
                    # var_str = ''
                    # for j, var_j in enumerate(var):
                    #     var_str += 'var{}:{:.4f} '.format(j+1, var_j)
                    # pbar.write(var_str)

                    if self.objective == 'B':
                        # noinspection PyUnboundLocalVariable
                        pbar.write('C:{:.3f}'.format(C.data[0]))

                if (self.global_iter % 50000 == 0) or epoch_save:
                    self.save_checkpoint()

                if epoch_save:
                    if len(losses) <= patience_epochs:
                        patience_exhausted = False
                    else:
                        small_quantile_last_losses = np.quantile(losses[-patience_epochs:], 0.05)
                        if small_quantile_last_losses > previous_loss_quantile:
                            patience_violations += 1
                        else:
                            patience_violations = 0

                        patience_exhausted = patience_violations >= self.args.patience_num

                        print(
                            f"\nsmall_quantile_last_losses: {small_quantile_last_losses}, "
                            f"mean_previous_loss: {previous_loss_quantile}, "
                            f"patience_violations: {patience_violations}\n"
                        )
                        previous_loss_quantile = min(small_quantile_last_losses, previous_loss_quantile)

                if (
                        (self.global_iter >= self.max_iter)
                        or (epoch > self.max_epochs)
                        or patience_exhausted
                ):
                    out = True
                    if strict_budget:
                        break
                epoch_save = False
            if avg_distance is not None:
                losses.append(avg_distance.detach().cpu().numpy())
            epoch += 1

        self.save_checkpoint(filename='last.pth')

        pbar.write("[Training Finished]")
        pbar.close()
