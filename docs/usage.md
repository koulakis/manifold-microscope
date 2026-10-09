# Usage and numerical conventions

Start with the [CPU quickstart](../README.md#quickstart-scalar-curvature-on-a-sphere). Geometry routines estimate quantities from an ordered parameter grid; they do not accept an unordered point cloud as a substitute.

## Datasets

| Source | Interface |
| --- | --- |
| [Toy manifolds](../microscope/datasets/toy_manifolds.py) | `Circle`, `Moons`, `Sphere`, `Torus`: `sample`, `tesselation`, `measures`, `measures_pointwise` |
| [Custom dSprites](../microscope/datasets/custom_dsprites.py) | `dsprites`, `dsprites_balanced`: generated shapes with scale, rotation and translation axes |
| [Extended COIL-20](../microscope/datasets/coil20.py) | `extended_coil20`: object views, scale and in-plane rotation; requires separately downloaded images |
| [Image loader](../microscope/datasets/generic_dataset_loader.py) | `load_dataset_no_split`: selects dataset/dimension and supplies grid metadata |

Toy measures use analytic formulas and some numerical approximations. Image geometry is estimated from the rendered samples; accuracy depends on resolution, rendering and conditioning. See the [dataset notebooks](../notebooks/datasets_and_measures) for visualizations.

```python
import numpy as np
from microscope.datasets.toy_manifolds import Circle

np.random.seed(0)
manifold = Circle(R=1.0)
points = manifold.sample(100)  # shape (100, 2), uniform in arc length
print(round(manifold.measures().volume, 3))  # 6.283
```

`sample` returns a point cloud. Use an explicit parameter grid, as in the quickstart, for finite differences. `normalized=True` requests unit-volume toy geometry. Sphere/torus `tesselation` currently calls a CUDA-default helper; the circle example works on CPU.

## Grid API

Inputs to NumPy wrappers have shape `(s1, ..., sd, features)` and floating dtype. Flatten image axes into `features`, keeping class and parameter axes distinct. Use equally spaced coordinates; `difference_intervals` contains their spacings, while `range_sizes` contains full parameter-range lengths.

`cyclic_dimensions` lists zero-based **axis indices**, not booleans. Sample a periodic axis without duplicating its endpoint. `get_difference_intervals` in [cyclic_dimensions.py](../microscope/cyclic_dimensions.py) uses `range / n` for periodic axes and `range / (n - 1)` otherwise.

| Operation in `microscope.computations_grid` | Result | Samples removed at each nonperiodic end |
| --- | --- | --- |
| `basic.partial_derivatives_across_all_dims_batched` | NumPy Jacobian, trailing shape `(features, d)` | 1 |
| `volume.volume_element` | NumPy array of `sqrt(det(g))` | 1 |
| `curvature.scalar_curvature` | NumPy scalar-curvature array | 3 |
| `reach.reach_per_point` | NumPy local-reach estimates | 1 |

These wrappers preserve periodic axes and take one `patch_sizes` entry per grid axis. Patch sizes must exceed 2 for derivatives/volume/reach and 6 for curvature; nonperiodic axes also need enough samples for those margins. Smaller patches reduce temporary device memory, but the full input and output grids remain in memory.

Wrappers accept `device="cpu"`, `"cuda:0"` or a `torch.device`; `None` selects CUDA if available, otherwise CPU. Lower-level tensor operations, including `riemannian_metric` and the `*_batch` routines, use the input tensor's device and crop **every** grid axis; periodic padding is the caller's responsibility. NumPy wrappers detach results from autograd. MPS has not been validated.

## Interpreting results

- `scalar_curvature(..., normalize=True)` divides by `d * (d - 1)` and rejects `d < 2`. On a surface, this is Gaussian curvature. This option does not rescale the manifold to unit volume. Singular metrics cannot be inverted.
- `compute_total_volume(element, range_sizes)` is `mean(element) * prod(range_sizes)`. `compute_total_curvature(element, curvature, range_sizes)` integrates their product the same way. Align cropped arrays and choose parameter ranges consistently; these are grid quadratures.
- Reach searches pairs of sampled points using estimated tangents. Its minimum estimates global reach. `subsample_points=k` keeps every kth point along every axis for both query and candidate points, while derivatives use the full grid. Repeated estimates are cropped back to the original interior shape. Subsampling can miss a bottleneck. `return_witnesses=True` additionally returns flat indices into the cropped, subsampled point array, one per sampled query; witnesses are not expanded.
- [Multiclass analysis](../microscope/computations_grid/data_analysis/data_analysis.py) returns an `AnalysisResults` dataclass and requires unique class names. `normalize_for_volume=True` rescales by the combined volume of all classes; `normalize_curvatures` separately controls curvature normalization. Fields keep their `normalized_` names even when scaling is disabled. Volume/reach fields are aligned to the smaller curvature grid. Positive and negative curvature integrals use whole-grid weights; the negative contribution stays signed and an absent sign contributes zero.

Dense grids grow multiplicatively with samples per axis; reach uses an exhaustive pair search on the selected points. Begin with small grids. Grid CPU support does not imply that all training and dataset helpers are CPU-ready.
