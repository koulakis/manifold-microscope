import numpy as np
import torch
import pynndescent
from tqdm import tqdm


class ANNMMLSProjector:
    def __init__(self, X, d, k=5, sigma=1.0, device="cuda:0", exact_nn_threshold=10_000, verbose=False):
        """
        Args:
            - X: torch.Tensor [N, D] manifold samples
            - d: intrinsic manifold dimension
            - k: number of neighbors
            - sigma: Gaussian kernel width
        """
        self.X = torch.from_numpy(X).to(device)
        self.d = d
        self.k = k
        self.sigma = sigma
        self.N, self.D = X.shape
        self.device = device
        self.exact_nn_threshold = exact_nn_threshold
        self.verbose = verbose

        if len(X) < k:
            raise ValueError(f"Received less training points: {len(X)} than the number of neighbors: {k}.")

        self.ann_index = None
        if self.exact_nn_threshold is None or len(X) > self.exact_nn_threshold:
            self.build_index()

    def build_index(self):
        # Build ANN index
        self.ann_index = pynndescent.NNDescent(
            self.X.cpu().numpy(),
            n_neighbors=self.k,
            metric="euclidean",
            n_jobs=-1
        )

    def query_nn(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # p: [B,D]
        return self.ann_index.query(p, k=self.k)

    def project(self, p: np.ndarray):
        """
        Projects B query points p onto the manifold via MMLS.
        Args:
            p: [B,D]
        Returns:
            p_proj: [B,D]
        """
        if self.exact_nn_threshold is not None and len(p) <= self.exact_nn_threshold:
            p = torch.from_numpy(p).to(self.device)

            distances = torch.cdist(p, self.X)  # [B, N]
            inds = torch.topk(distances, self.k, largest=False, dim=1).indices
            Dsq = torch.gather(distances, dim=1, index=inds)
        else:
            if self.ann_index is None:
                self.build_index()
            inds, Dsq = self.query_nn(p)

            Dsq = torch.from_numpy(Dsq).to(self.device)
            inds = torch.from_numpy(inds).to(self.device)
            p = torch.from_numpy(p).to(self.device)

        p_proj = torch.zeros_like(p)
        for i in tqdm(list(range(p.shape[0])), disable=not self.verbose):
            Xi = self.X[inds[i]]
            pi = p[i:i+1]
            wi = torch.exp(-Dsq[i] / (2 * self.sigma**2))

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
