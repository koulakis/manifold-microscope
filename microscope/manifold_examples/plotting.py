from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d


def plot_points(
        pts: np.ndarray,
        curvature: Optional[np.ndarray] = None,
        volume_element: Optional[np.ndarray] = None,
        limit: float = 1,
        s: float = 1,
        cmap: str = "coolwarm",
        alpha: float = 0.5,
        size_mult: float = 4,
        subsample_ratio: Optional[float] = None
) -> None:
    n_plots = 1
    if curvature is not None:
        n_plots += 1
    if volume_element is not None:
        n_plots += 1

    n_points = np.prod(pts.shape[:-1])
    if subsample_ratio is not None:
        idx = np.random.choice(
            n_points,
            size=int(subsample_ratio*n_points),
            replace=False
        )
    else:
        idx = np.arange(n_points)

    pts_flattened = pts.reshape((-1, 3))[idx]
    xs, ys, zs = pts_flattened[:, 0], pts_flattened[:, 1], pts_flattened[:, 2]

    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(size_mult * n_plots, size_mult),
        subplot_kw=dict(projection='3d'),
        squeeze=False
    )
    axes: list[mplot3d.axes3d.Axes3D] = axes[0]
    limits = (-limit, limit)

    ax = axes[0]
    ax.scatter(xs, ys, zs, s=s, alpha=alpha)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_zlim(limits)

    if curvature is not None:
        curvature = curvature.flatten()[idx]
        ax = axes[1]
        plot = ax.scatter(xs, ys, zs, s=s, c=curvature, cmap=cmap, label="curvature", alpha=alpha)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_zlim(limits)
        ax.legend()
        plt.colorbar(plot, ax=ax, fraction=0.03)

    if volume_element is not None:
        volume_element = volume_element.flatten()[idx]
        ax = axes[2]
        plot = ax.scatter(xs, ys, zs, s=s, c=volume_element, cmap=cmap, label="volume element", alpha=alpha)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_zlim(limits)
        ax.legend()
        plt.colorbar(plot, ax=ax, fraction=0.03)

    plt.show()
