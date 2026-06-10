from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad
import torch
from geomloss import SamplesLoss
from tqdm import tqdm


@dataclass
class MeasuresGlobal:
    """
    The measures below are different global values computed on the manifold. The volume captures the scale of the
    manifold, while all other measures attempt to capture the complexity of the manifold. The toy manifolds all have
    simple definitions, thus all the measures are directly computed with their closed form formulas or with simple
    numerical approximations.

    Attributes:
        volume: The d-volume of the manifold.
        total_principal: The result of integrating the maximum absolute value of the principle curvatures (here it is
            assumed that the toy manifolds are at most 2-dimensional) with the d-volume element.
        max_absolute_principal: The maximum value of any principal curvature over the whole manifold.
        total_gaussian: The d-volume integral of the Gaussian curvature over the whole manifold.
        max_gaussian: The maximum value of the Gaussian curvature over the whole manifold.
        total_mean_curvature: The d-volume integral of the mean curvature over the whole manifold.
        max_mean_curvature: The maximum value of the mean curvature over the whole manifold.
        min_reach: The reach of the manifold.
        average_reach: The d-volume integral of the local reach over the manifold. The local reach on point p is:

            tau_p = inf_{q != p in M} ||q - p||^2 / (2d(q - p, T_p M))

            where the latter distance in the denominator is the distance of another point q to the tangent space at p.
    """
    volume: float
    total_principal: float
    max_absolute_principal: float
    total_gaussian: float
    max_gaussian: float
    total_mean_curvature: float
    max_mean_curvature: float
    min_reach: float
    average_reach: float


@dataclass
class MeasuresLocal:
    """
    Those measures also attempt to capture the complexity of the manifold and are point-wise versions of the global
    measures.
    Attributes:
        volume_element: The volume element of the manifold on each point.
        max_absolute_principal: The maximum absolute value of the principal curvatures on each point.
        gaussian: The Gaussian curvature on each point.
        mean_curvature: The mean curvature on each point.
        reach: The local value of the reach on each point.
    """
    volume_element: np.ndarray
    max_absolute_principal: np.ndarray
    gaussian: np.ndarray
    mean_curvature: np.ndarray
    reach: np.ndarray


class Manifold(ABC):
    @staticmethod
    @abstractmethod
    def normalize_parameters(*args, **kwargs):
        """Given the parameters defining the manifold, it updates them so that they correspond to a manifold of
        d-volume 1.

        *Args: A list of N parameters to be updated.

        *Returns: A list of the N updated parameters.
        """
        pass

    @abstractmethod
    def sample(
        self,
        N: int,
        normalized: bool = False,
        return_unitary_normal: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Sample the manifold, uniformly on its d-volume.

        Args:
            N: The number of points to sample.
            normalized: If true, the manifold is scaled to have d-volume 1 before being sampled.
            return_unitary_normal: If true, it additionally returns on each point the unitary vector orthogonal to the
                tangent space.
        """

    @abstractmethod
    def tesselation(self, N: int, normalized: bool = False) -> np.ndarray:
        """Get samples of the manifold which have local density close to the d-volume element of the manifold,
        i.e. they are as spread as possible.

        Args:
            N: The number of points to sample.
            normalized: If true, the manifold is scaled to have d-volume 1 before being sampled.
        """
        pass

    @abstractmethod
    def measures(self, normalized: bool = False) -> MeasuresGlobal:
        """It computes the global measures of the manifold. For details, look at the 'MeasuresGlobal' class.

        Args:
            normalized: If true, it scales the manifold to have d-volume 1 before computing the measures.

        Returns: An instance of 'MeasuresGlobal' which contains all the measures as attributes.
        """
        pass

    @abstractmethod
    def measures_pointwise(self, points: np.ndarray, normalized: bool = False) -> MeasuresLocal:
        """It computes the local measures of the manifold. For details, look at the 'MeasuresGlobal' class.

        Args:
            points: An array with the points to compute the measures on.
            normalized: If true, it scales the manifold to have d-volume 1 before computing the measures.

        Returns: An instance of 'MeasuresLocal' which contains all the measures as attributes.
        """
        pass

    def __str__(self):
        return self.__class__.__name__

    @property
    @abstractmethod
    def dimension(self):
        """Returns the intrinsic dimension of the manifold."""
        pass


# Lloyd relaxation-based tesselation for surfaces using Sinkhorn loss from GeomLoss
def cvt_surface(
    manifold: Manifold,
    N: int,
    iterations: int = 20,
    n_samples=10_000,
    normalized: bool = False,
    dtype=torch.float32,
    device="cuda:0"
) -> np.ndarray:
    x = torch.tensor(manifold.sample(n_samples, normalized=normalized), dtype=dtype, device=device)
    c = torch.tensor(manifold.sample(N, normalized=normalized), dtype=dtype, requires_grad=True, device=device)

    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.1)

    optimizer = torch.optim.Adam([c], lr=0.1)

    pb = tqdm(range(iterations), total=iterations)
    for _ in pb:
        optimizer.zero_grad()
        loss = loss_fn(c, x)

        pb.set_description(f"Loss: {float(loss.detach().cpu().numpy()):.5f}")

        loss.backward()
        optimizer.step()
    return c.detach().cpu().numpy()


class Circle(Manifold):
    @property
    def dimension(self):
        return 1

    def __init__(self, R=None, dtype=np.float32):
        self.R = self.normalize_parameters() if R is None else R
        self.dtype = dtype

    @staticmethod
    def normalize_parameters(R: Optional[float] = None) -> float:
        return 1 / (2 * np.pi)

    def sample(
        self,
        N: int,
        normalized: bool = False,
        return_unitary_normal: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        R = self.normalize_parameters(self.R) if normalized else self.R
        theta = np.random.uniform(low=0, high=2 * np.pi, size=N).astype(self.dtype)
        samples = R * np.stack([np.cos(theta), np.sin(theta)], axis=1)

        if return_unitary_normal:
            unitary_normal = np.stack([np.cos(theta), np.sin(theta)], axis=1)
            return samples, unitary_normal
        else:
            return samples

    def measures(self, normalized: bool = False) -> MeasuresGlobal:
        R = self.normalize_parameters() if normalized else self.R
        length = 2 * np.pi * R
        curvature = 1 / R
        return MeasuresGlobal(
            volume=length,
            total_principal=length * curvature,
            max_absolute_principal=curvature,
            total_gaussian=0.0,
            max_gaussian=0.0,
            total_mean_curvature=0.0,
            max_mean_curvature=0.0,
            min_reach=R,
            average_reach=R
        )

    def measures_pointwise(self,  points: np.ndarray, normalized: bool = False) -> MeasuresLocal:
        R = self.normalize_parameters() if normalized else self.R

        N = len(points)
        return MeasuresLocal(
            volume_element=np.full(N, R),
            max_absolute_principal=np.full(N, 1 / R),
            gaussian=np.zeros(N),
            mean_curvature=np.zeros(N),
            reach=np.full(N, R)
        )

    def tesselation(self, N, normalized: bool = False) -> np.ndarray:
        R = self.normalize_parameters() if normalized else self.R
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False, dtype=self.dtype)
        return R * np.stack([np.cos(theta), np.sin(theta)], axis=1)


class Moons(Manifold):
    @property
    def dimension(self):
        return 1

    def __init__(self, R=None, dtype=np.float32):
        self.R = self.normalize_parameters() if R is None else R
        self.dtype = dtype

    @staticmethod
    def normalize_parameters(R: Optional[float] = None) -> float:
        return 1 / (2 * np.pi)

    def sample(
        self,
        N: int,
        normalized: bool = False,
        return_unitary_normal: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        R = self.normalize_parameters() if normalized else self.R
        theta = 2 * np.pi * np.random.rand(N).astype(self.dtype)
        points = R * np.stack([np.cos(theta), np.sin(theta)], axis=1)
        upper = points[points[:, 1] >= 0]
        lower = points[points[:, 1] < 0]
        upper_moon = upper
        lower_moon = np.stack([R + lower[:, 0], lower[:, 1] + 0.5 * R], axis=1)
        samples = np.concatenate([upper_moon, lower_moon], axis=0)

        if return_unitary_normal:
            points_normal = np.stack([np.cos(theta), np.sin(theta)], axis=1)
            upper_normal = points_normal[points_normal[:, 1] >= 0]
            lower_normal = points_normal[points_normal[:, 1] < 0]
            unitary_normal = np.concatenate([upper_normal, lower_normal], axis=0)
            return samples, unitary_normal
        else:
            return samples

    @staticmethod
    def reach(points: np.ndarray, R: float) -> np.ndarray:
        """Compute the reach point-wise. Unfortunately, we need to first categorize the points to the two moons of the
        dataset and then compute their reaches. Given the high proximity of the components, the reach equals to the
        distance to the closest point on the opposite component for each point.

        Args:
            points: An array with points of the moons manifolds.
            R: The radius of the two semicircles.
        """
        # First compute which points belong to component A and which on B.
        cA = np.array([0.0, 0.0])
        cB = np.array([R, 0.5 * R])

        distA = np.abs(np.linalg.norm(points - cA, axis=1) - R)
        distB = np.abs(np.linalg.norm(points - cB, axis=1) - R)
        belongs_A = distA <= distB

        # Convert the Euclidean coordinates to polar.
        centers = np.where(belongs_A[:, None], cA, cB)
        dx, dy = (points - centers).T
        theta = np.arctan2(dy, dx)

        thA = theta[belongs_A]
        thB = theta[~belongs_A]

        # Compute the reaches on component A.
        cosA, sinA = np.cos(thA), np.sin(thA)

        dC_A = R * np.sqrt((cosA - 1.0) ** 2 + (sinA - 0.5) ** 2)

        projA = sinA <= 0.5
        dA_proj = np.abs(dC_A - R)

        dA_ep1 = R * np.sqrt(cosA ** 2 + (sinA - 0.5) ** 2)
        dA_ep2 = R * np.sqrt((cosA - 2.0) ** 2 + (sinA - 0.5) ** 2)
        dA_end = np.minimum(dA_ep1, dA_ep2)

        reachA = 0.5 * np.where(projA, dA_proj, dA_end)

        # Compute the reaches on component B.
        cosB, sinB = np.cos(thB), np.sin(thB)

        dC_B = R * np.sqrt((1.0 + cosB) ** 2 + (0.5 + sinB) ** 2)
        projB = (0.5 + sinB) >= 0.0
        dB_proj = np.abs(dC_B - R)

        qx = R * (1.0 + cosB)
        qy = R * (0.5 + sinB)
        dB_ep1 = np.sqrt((qx - R) ** 2 + qy ** 2)
        dB_ep2 = np.sqrt((qx + R) ** 2 + qy ** 2)
        dB_end = np.minimum(dB_ep1, dB_ep2)

        reachB = 0.5 * np.where(projB, dB_proj, dB_end)

        # Combine the reaches.
        reach_vals = np.zeros(points.shape[0])
        reach_vals[belongs_A] = reachA
        reach_vals[~belongs_A] = reachB

        return reach_vals

    @staticmethod
    def average_reach(R, n=10_000):
        """The integral to compute the average reach is a bit challenging, so approximate it.

        Args:
            R: The radius of the semicircles.
            n: Number of theta samples for the integral.
        Returns: The average reach.
        """
        theta = np.linspace(0.0, np.pi, n, endpoint=False) + 0.5 * np.pi / n

        theta = np.asarray(theta)
        dC = R * np.sqrt(9 / 4.0 - 2.0 * np.cos(theta) - np.sin(theta))
        d_ep = R * np.sqrt(1.25 - np.sin(theta))
        proj_mask = np.sin(theta) <= 0.5
        minimal_distance_from_B = np.where(proj_mask, np.abs(dC - R), d_ep)

        return (1.0 / (2.0 * np.pi)) * np.trapz(minimal_distance_from_B, theta)

    def measures(self, normalized: bool = False) -> MeasuresGlobal:
        R = self.normalize_parameters() if normalized else self.R
        arc_len = np.pi * R
        total_length = 2 * arc_len
        curvature = 1 / R

        reach = self.average_reach(R=R)

        return MeasuresGlobal(
            volume=total_length,
            total_principal=total_length * curvature,
            max_absolute_principal=curvature,
            total_gaussian=0.0,
            max_gaussian=0.0,
            total_mean_curvature=0.0,
            max_mean_curvature=0.0,
            min_reach=R / 4,
            average_reach=reach
        )

    def measures_pointwise(self, points: np.ndarray, normalized: bool = False) -> MeasuresLocal:
        R = self.normalize_parameters(self.R) if normalized else self.R
        reach = self.reach(points, R=R)
        N = len(points)
        return MeasuresLocal(
            volume_element=np.full(N, R),
            max_absolute_principal=np.full(N, 1 / R),
            gaussian=np.zeros(N),
            mean_curvature=np.zeros(N),
            reach=reach
        )

    def tesselation(self, N, normalized: bool = False) -> np.ndarray:
        R = self.normalize_parameters(self.R) if normalized else self.R

        # Half points for each moon
        N1 = N // 2
        N2 = N - N1
        theta1 = np.linspace(0, np.pi, N1, endpoint=False, dtype=self.dtype)
        theta2 = np.linspace(np.pi, 2 * np.pi, N2, endpoint=False, dtype=self.dtype)
        upper = R * np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
        lower = R * np.stack([np.cos(theta2), np.sin(theta2)], axis=1)
        lower = np.stack([R - lower[:, 0], lower[:, 1] + 0.5 * R], axis=1)
        return np.concatenate([upper, lower], axis=0)


class Sphere(Manifold):
    @property
    def dimension(self):
        return 2

    def __init__(self, R=None, dtype=np.float32):
        self.R = self.normalize_parameters() if R is None else R
        self.dtype = dtype

    @staticmethod
    def normalize_parameters(R: Optional[float] = None):
        return 1 / np.sqrt(4 * np.pi)

    def sample(
        self,
        N,
        normalized: bool = False,
        return_unitary_normal: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        R = self.normalize_parameters() if normalized else self.R

        vec = np.random.randn(N, 3).astype(self.dtype)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        samples = R * vec

        if return_unitary_normal:
            unitary_normal = vec
            return samples, unitary_normal
        else:
            return samples

    def measures(self, normalized: bool = False) -> MeasuresGlobal:
        R = self.normalize_parameters() if normalized else self.R
        area = 4 * np.pi * R**2
        principal = 1 / R
        gaussian = 1 / R**2
        mean_curvature = 1 / R
        return MeasuresGlobal(
            volume=area,
            total_principal=area * principal,
            max_absolute_principal=principal,
            total_gaussian=area * gaussian,
            max_gaussian=gaussian,
            total_mean_curvature=area * mean_curvature,
            max_mean_curvature=mean_curvature,
            min_reach=R,
            average_reach=R
        )

    def measures_pointwise(self, points: np.ndarray, normalized: bool = False) -> MeasuresLocal:
        R = self.normalize_parameters() if normalized else self.R
        gaussian = 1 / R**2
        mean = 1 / R
        N = len(points)
        return MeasuresLocal(
            volume_element=np.full(N, R**2),
            max_absolute_principal=np.full(N, 1 / R),
            gaussian=np.full(N, gaussian),
            mean_curvature=np.full(N, mean),
            reach=np.full(N, R)
        )

    def tesselation(self, N, normalized: bool = False, iterations: int = 50) -> np.ndarray:
        approx_points = cvt_surface(self, N, iterations=iterations, normalized=normalized)
        R = self.normalize_parameters() if normalized else self.R
        projected_points_on_sphere = R * approx_points / np.linalg.norm(approx_points, axis=-1, keepdims=True)

        return projected_points_on_sphere


class Torus(Manifold):
    @property
    def dimension(self):
        return 2

    def __init__(self, R=1 / np.sqrt(4*np.pi**2*1*0.6), r=0.6 / np.sqrt(4*np.pi**2*1*0.6), dtype=np.float32):
        self.R = R
        self.r = r
        self.dtype = dtype

    @staticmethod
    def normalize_parameters(R: float, r: float) -> tuple[float, float]:
        sqrt_area = np.sqrt(4 * np.pi ** 2 * R * r)

        R = 1 / sqrt_area * R
        r = 1 / sqrt_area * r

        return R, r

    def sample(
        self,
        N,
        normalized: bool = False,
        return_unitary_normal: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        R, r = self.normalize_parameters(self.R, self.r) if normalized else (self.R, self.r)

        u = 2 * np.pi * np.random.rand(N).astype(self.dtype)
        y = np.random.rand(N).astype(self.dtype)
        v = self.inverse_cdf(y, R, r)

        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        samples = np.stack([x, y, z], axis=1)

        if return_unitary_normal:
            unitary_normal = np.stack([np.cos(v)*np.cos(u), np.cos(v)*np.sin(u), np.sin(v)], axis=1)
            return samples, unitary_normal
        else:
            return samples

    @staticmethod
    def inverse_cdf(y_vals, R, r):
        def F(v):
            return (R * v + r * np.sin(v)) / (2 * np.pi * R)

        v_vals = np.zeros_like(y_vals)
        for i, yi in enumerate(y_vals):
            sol = root_scalar(lambda v: F(v) - yi, bracket=[0, 2 * np.pi], method='brentq')
            v_vals[i] = sol.root
        return v_vals

    @staticmethod
    def reach(points: np.ndarray, R: float, r: float) -> np.ndarray:
        """The local reach of the torus is the minimum of the two radii of the principal curvatures
        k1 = 1 / r, k2 = (R + r cos(theta)) / |cos(theta)|. Substituting cos(theta) = (R - d) / r, we get
        the formula used below.

        Args:
            points: An array of points of the torus on which the local reach will be defined.
            R: The major radius of the torus.
            r: The minor radius of the torus.

        Returns: An array with a reach per point.
        """
        d = np.linalg.norm(points[:, :2], axis=1)
        return np.minimum(r, r * d / np.abs(d - R))

    @staticmethod
    def average_reach(R: float, r: float) -> float:

        radii_ratio = R / (2 * r)

        if radii_ratio >= 1:
            return r

        else:
            A = np.sqrt(1 - radii_ratio ** 2)
            return (1 / (4 * np.pi * R)) * (
                    -12 * R * r * np.arccos(radii_ratio)
                    + 4 * np.pi * R * r
                    + 8 * r ** 2 * A
                    + 2 * R ** 2 * np.log((1 + A) / (1 - A))
            )

    def measures(self, normalized: bool = False) -> MeasuresGlobal:
        R, r = self.normalize_parameters(self.R, self.r) if normalized else (self.R, self.r)

        area = 4 * np.pi**2 * R * r
        max_k1 = 1 / r
        max_k2 = 1 / (R - r) if R > r else float('inf')
        max_principal = max(max_k1, max_k2)
        total_principal_result = quad(
            lambda x: np.maximum(np.abs(np.cos(x) / (R + r*np.cos(x))), 1/r)*r*(R + r*np.cos(x)),
            a=0,
            b=2*np.pi,
            limit=100
        )
        total_principal = 2*np.pi * total_principal_result[0]
        total_principal_error = total_principal_result[1]

        int_error_threshold = 1e-6
        if total_principal_error / np.abs(total_principal) > int_error_threshold:
            raise ValueError(
                f"The total principal curvature integral is computed to be {total_principal} with error "
                f"{total_principal_error} above the threshold {int_error_threshold}")

        total_gaussian_abs = 8 * np.pi

        total_mean_result = quad(
            lambda x: np.abs(np.cos(x) / (R + r * np.cos(x)) + 1 / r) / 2 * r * (R + r * np.cos(x)),
            a=0,
            b=2 * np.pi,
            limit=100
        )
        total_mean = 2 * np.pi * total_mean_result[0]
        total_mean_error = total_mean_result[1]

        int_error_threshold = 1e-6
        if total_mean_error / np.abs(total_mean) > int_error_threshold:
            raise ValueError(
                f"The total principal curvature integral is computed to be {total_mean} with error "
                f"{total_mean_error} above the threshold {int_error_threshold}")

        average_reach = self.average_reach(R=R, r=r)

        return MeasuresGlobal(
            volume=area,
            total_principal=total_principal,
            max_absolute_principal=max_principal,
            total_gaussian=total_gaussian_abs,
            max_gaussian=1 / (r * (R - r)) if R > r else float('inf'),
            total_mean_curvature=total_mean,
            max_mean_curvature=(max_k1 + max_k2) / 2,
            min_reach=min(r, R - r),
            average_reach=average_reach
        )

    def measures_pointwise(self, points: np.ndarray, normalized: bool = False) -> MeasuresLocal:
        R, r = self.normalize_parameters(self.R, self.r) if normalized else (self.R, self.r)

        v = np.arccos((np.linalg.norm(points[:, :2], axis=-1) - R) / r)

        volume_element = r*(R + r*np.cos(v))
        k1 = 1 / r
        k2 = np.cos(v) / (R + r * np.cos(v))
        gauss = k1 * k2
        mean = (k1 + k2) / 2
        reach = self.reach(points, R=R, r=r)

        return MeasuresLocal(
            volume_element=volume_element,
            max_absolute_principal=np.maximum(np.abs(k1), np.abs(k2)),
            gaussian=gauss,
            mean_curvature=mean,
            reach=reach
        )

    def tesselation(self, N, iterations: int = 50, normalized: bool = False) -> np.ndarray:
        approx_points = cvt_surface(self, N, iterations=iterations, normalized=normalized)
        # return approx_points
        R, r = self.normalize_parameters(self.R, self.r) if normalized else (self.R, self.r)

        x = np.linalg.norm(approx_points[:, :2], axis=-1) - R

        v = np.arctan2(approx_points[:, 2], x)
        u = np.arctan2(approx_points[:, 1], approx_points[:, 0])

        projected_points_on_torus = np.stack(
            [
                (R + r * np.cos(v)) * np.cos(u),
                (R + r * np.cos(v)) * np.sin(u),
                r * np.sin(v)
            ],
            axis=-1
        )

        return projected_points_on_torus


def plot_manifold(
    measured_manifold: Manifold,
    sampling_type: str = "tesselation",
    N=100,
    normalized=False,
    s=1,
    shrink=0.7,
    figsize=(20, 4)
) -> None:
    import matplotlib.pyplot as plt

    match sampling_type:
        case "tesselation":
            points = measured_manifold.tesselation(N, normalized=normalized)
        case "uniform":
            points = measured_manifold.sample(N, normalized=normalized)
        case _:
            raise ValueError(
                f"Unknown sampling method: {sampling_type}, please select one of: 'tesselation', 'uniform'."
            )
    measures: MeasuresLocal = measured_manifold.measures_pointwise(points=points, normalized=normalized)

    is_3d = points.shape[1] == 3
    fig = plt.figure(figsize=figsize)

    def add_subplot(pos, title, values):
        ax = fig.add_subplot(1, 5, pos, projection='3d' if is_3d else None)
        sc = ax.scatter(*points.T, c=values, cmap='viridis', s=s)
        fig.colorbar(sc, ax=ax, shrink=shrink)
        ax.set_title(title)
        ax.set_aspect('equal')

    add_subplot(pos=1, title="Volume Element", values=measures.volume_element)
    add_subplot(pos=2, title="Max Principal Curvature", values=measures.max_absolute_principal)
    add_subplot(pos=3, title="Gaussian Curvature", values=measures.gaussian)
    add_subplot(pos=4, title="Mean Curvature", values=measures.mean_curvature)
    add_subplot(pos=5, title="Local Reach", values=measures.reach)

    plt.tight_layout()
    plt.show()
