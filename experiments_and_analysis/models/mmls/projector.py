from __future__ import annotations

import numpy as np
import torch
import pynndescent
from tqdm import tqdm


def _validate_projection_options(projection_method: str, projection_batch_size: int) -> None:
    """Require original or thin_svd projection and a positive query batch size.

    Args:
        projection_method: MMLS projection implementation.
        projection_batch_size: Number of queries per thin-SVD batch.
    """
    if projection_method not in ("original", "thin_svd"):
        raise ValueError(f"Unknown projection method: {projection_method}.")
    if projection_batch_size < 1:
        raise ValueError("projection_batch_size must be positive.")


class ANNMMLSProjector:
    def build_index(self) -> None:
        """Build the approximate nearest-neighbor index."""
        # Build ANN index
        self.ann_index = pynndescent.NNDescent(
            self.X.cpu().numpy(),
            n_neighbors=self.k,
            metric="euclidean",
            n_jobs=-1
        )

    def __init__(
        self,
        X: np.ndarray,
        d: int,
        k: int = 5,
        sigma: float = 1.0,
        device: str = "cuda:0",
        exact_nn_threshold: int | None = 10_000,
        verbose: bool = False,
        projection_method: str = "thin_svd",
        projection_batch_size: int = 32
    ) -> None:
        """Initialize an MMLS projector from training points and valid projection options.

        Args:
            X: Training points of shape (n f); at least k points are required.
            d: Intrinsic manifold dimension.
            k: Number of nearest neighbors.
            sigma: Gaussian kernel width.
            device: Device used for tensor computations.
            exact_nn_threshold: Maximum query count for exact neighbor search; None selects ANN.
            verbose: Show projection progress.
            projection_method: MMLS projection implementation.
            projection_batch_size: Number of queries per thin-SVD batch.
        """
        self.X = torch.from_numpy(X).to(device)
        self.d = d
        self.k = k
        self.sigma = sigma
        self.N, self.D = X.shape
        self.device = device
        self.exact_nn_threshold = exact_nn_threshold
        self.verbose = verbose
        _validate_projection_options(projection_method, projection_batch_size)
        self.projection_method = projection_method
        self.projection_batch_size = projection_batch_size

        if len(X) < k:
            raise ValueError(f"Received less training points: {len(X)} than the number of neighbors: {k}.")

        self.ann_index = None
        # Retain the original eager index behavior. The fast path builds an
        # index only if project() actually selects the same ANN branch.
        if projection_method == "original" and (self.exact_nn_threshold is None or len(X) > self.exact_nn_threshold):
            self.build_index()

    def query_nn(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find neighbor indices and distances for query points.

        Args:
            p: Query points of shape (b f).
        """
        # p: [B,D]
        return self.ann_index.query(p, k=self.k)

    def _project_thin_svd(
        self,
        points: torch.Tensor,
        indices: torch.Tensor,
        distances: torch.Tensor
    ) -> np.ndarray:
        """Project using right singular vectors of sqrt(W) @ centered neighbors.

        A.T @ A equals the original weighted covariance. Its leading right singular vectors span the same tangent space.
        Weights, centering and neighbor selection stay unchanged; roundoff and degenerate eigenspaces can differ from
        the covariance SVD.

        Args:
            points: Query points of shape (b f).
            indices: Neighbor indices of shape (b k).
            distances: Neighbor distances of shape (b k).
        """
        if self.d > min(self.k, self.D):
            raise ValueError("thin_svd requires d <= min(k, ambient dimension).")
        projected = torch.empty_like(points)
        for start in tqdm(range(0, len(points), self.projection_batch_size), disable=not self.verbose):
            stop = start + self.projection_batch_size
            neighbors = self.X[indices[start:stop]]
            weights = torch.exp(-distances[start:stop] / (2 * self.sigma**2))
            mean = (weights.unsqueeze(-1) * neighbors).sum(1) / weights.sum(1, keepdim=True)
            centered = neighbors - mean.unsqueeze(1)
            weighted = weights.sqrt().unsqueeze(-1) * centered
            _, _, vh = torch.linalg.svd(weighted, full_matrices=False)
            basis = vh[:, :self.d].transpose(-1, -2)
            delta = (points[start:stop] - mean).unsqueeze(-1)
            tangent = basis @ (basis.transpose(-1, -2) @ delta)
            projected[start:stop] = mean + tangent.squeeze(-1)
        return projected.detach().cpu().numpy()

    def project(self, p: np.ndarray) -> np.ndarray:
        """Projects B query points p onto the manifold via MMLS.

        Args:
            p: Query points of shape (b f).

        Returns:
            p_proj: [B,D]
        """
        if self.exact_nn_threshold is not None and len(p) <= self.exact_nn_threshold:
            p_tensor = torch.from_numpy(p).to(self.device)

            distances = torch.cdist(p_tensor, self.X)  # [B, N]
            indices_tensor = torch.topk(
                distances,
                self.k,
                largest=False,
                dim=1
            ).indices
            distances_tensor = torch.gather(distances, dim=1, index=indices_tensor)
        else:
            if self.ann_index is None:
                self.build_index()
            indices_array, distances_array = self.query_nn(p)

            distances_tensor = torch.from_numpy(distances_array).to(self.device)
            indices_tensor = torch.from_numpy(indices_array).to(self.device)
            p_tensor = torch.from_numpy(p).to(self.device)

        if self.projection_method == "thin_svd":
            return self._project_thin_svd(p_tensor, indices_tensor, distances_tensor)

        # Historical covariance implementation is deliberately kept unchanged.
        p_proj = torch.zeros_like(p_tensor)
        for i in tqdm(list(range(p_tensor.shape[0])), disable=not self.verbose):
            Xi = self.X[indices_tensor[i]]
            pi = p_tensor[i:i+1]
            wi = torch.exp(-distances_tensor[i] / (2 * self.sigma**2))

            mu = (wi.unsqueeze(1) * Xi).sum(0) / wi.sum()
            Xi0 = Xi - mu
            W = torch.diag(wi)
            cov = Xi0.t() @ W @ Xi0

            U, _, _ = torch.svd(cov)
            Ud = U[:, :self.d]

            vi = (pi - mu).squeeze(0)  # [D]
            proj_vec = Ud @ (Ud.T @ vi)  # [D]
            proj = mu + proj_vec  # [D]
            p_proj[i] = proj  # match [D]

        return p_proj.detach().cpu().numpy()
