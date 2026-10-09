# The Data Manifold under the Microscope

`manifold-microscope` provides toy and image datasets, plus finite-difference estimates of volume, scalar curvature and reach on parameter grids. It accompanies [Marios Koulakis and Constantin Seibold, *The Data Manifold under the Microscope* (ICML 2026)](https://proceedings.mlr.press/v306/koulakis26a.html).

## Install

Python ≥3.10:

```bash
python -m pip install manifold-microscope
```

For the code documented here, install from this repository with `python -m pip install -e .`; published releases may lag behind. Experiment scripts additionally require the `[benchmark]` extra: `python -m pip install -e '.[benchmark]'`.

## Quickstart: scalar curvature on a sphere

```python
import numpy as np
from microscope.computations_grid.curvature import scalar_curvature

radius = 2.0
latitude = np.linspace(-0.7, 0.7, 41)  # Avoid the coordinate singularities at the poles.
longitude = np.linspace(0, 2 * np.pi, 48, endpoint=False)
u, v = np.meshgrid(latitude, longitude, indexing="ij")
points = radius * np.stack(
    [np.cos(u) * np.cos(v), np.cos(u) * np.sin(v), np.sin(u)], axis=-1
)
curvature = scalar_curvature(
    points,
    difference_intervals=[latitude[1] - latitude[0], longitude[1] - longitude[0]],
    cyclic_dimensions=[1],
    patch_sizes=[13, 17],
    device="cpu",
)
print(f"Mean scalar curvature: {curvature.mean():.3f}")
# Approximately 0.5: a sphere has scalar curvature 2 / radius**2.
```

Grid APIs with a `device` argument default to CUDA when available, otherwise CPU. Dense image grids and experiment scripts have additional resource/device requirements.

## Documentation and paper reproduction

- [Usage and numerical conventions](docs/usage.md)
- [Reproduction status, experiment entry points and validation](docs/reproducibility.md)
- [Dataset and geometry notebooks](notebooks/datasets_and_measures)

**Paper reproduction is still being checked.** Recent numerical corrections can change results; the current checkout is not yet a validated reproduction release. Pin the source revision and retain configurations, data and environment details when comparing with the paper.

[BSD-3-Clause license](LICENSE); adapted β-VAE code has a separate [MIT attribution and license](representation_learning/beta_vae/NOTICE.md).
