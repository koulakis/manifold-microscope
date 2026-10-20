# Usage

Start with the [CPU quickstart](../README.md#quickstart-scalar-curvature-on-a-sphere). Geometry routines estimate quantities from an ordered parameter grid; they do not accept an unordered point cloud as a substitute.

## Datasets

The [toy manifolds](../microscope/datasets/toy_manifolds.py)—`Circle`, `Moons`,
`Sphere`, `Torus`—provide `sample`, `tesselation`, `measures` and
`measures_pointwise`. For example:

```python
import numpy as np
from microscope.datasets.toy_manifolds import Circle

np.random.seed(0)
manifold = Circle(R=1.0)
points = manifold.sample(100)  # shape (100, 2), uniform in arc length
print(round(manifold.measures().volume, 3))  # 6.283
```

`normalized=True` requests unit-volume toy geometry. For sphere/torus
`tesselation`, pass `device="cpu"` to use CPU instead of the CUDA default.

### Image datasets

Custom dSprites is rendered locally and needs no download. For a small loading
example (too coarse for reliable curvature estimates):

```python
from microscope.datasets.custom_dsprites import dsprites

images, labels = dsprites(
    sprites=("square",), n_sizes=3, n_angles=8,
    max_translation=2, translation_stride=2,
    processes=(None, None, None, None), verbose=False,
)
print(images.shape)  # (1, 3, 8, 2, 2, 64, 64)
# Axes: class, scale, rotation, y translation, x translation, height, width.
```

For COIL-20, first download and extract the processed grayscale images from
[Columbia's dataset page](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php).
Set the directory containing all 20 × 72 PNGs, including `obj1__0.png`:

```bash
export COIL20_PATH=/path/to/coil-20-proc
```

```python
from microscope.datasets.coil20 import load_coil20

views, labels = load_coil20(views_stride=4)
print(views.shape)  # (20, 18, 128, 128), float32 intensities divided by 256
```

`extended_coil20` adds scale and in-plane rotation. To load a paper dataset
with grid metadata:

```python
from microscope.datasets.generic_dataset_loader import DatasetName, load_dataset_no_split

(images, labels, patch_sizes, difference_intervals, volume_clip_slices,
 full_clip_slices, range_sizes, cyclic_dimensions) = load_dataset_no_split(
    DatasetName.custom_dsprites_balanced, number_of_dims=2,
)
# images: (classes, scale, rotation, height, width), scaled to [0, 1].
grid = images.reshape(*images.shape[:-2], -1)
# grid[0] is one class, ready for the geometry wrappers below.
```

Balanced dSprites supports dimensions 1–4; `DatasetName.extended_coil20` supports
1–3. These loaders build full image grids before slicing and can need many GiB
of RAM. Clipping slices select the interior after derivative estimation. The
legacy original-dSprites loader needs its external NPZ path configured in
`original_dsprites.py`; paper recipes use generated custom dSprites.

## Grid API

Use floating arrays shaped `(s1, ..., sd, features)` with uniformly spaced
parameter axes. Flatten image axes into `features`, keeping class axes separate.
`difference_intervals` gives coordinate spacings; `range_sizes` gives full domain
lengths. The induced metric must be nonsingular.

`cyclic_dimensions` lists zero-based periodic axis indices. Omit duplicate
endpoints on periodic axes; `get_difference_intervals` computes the spacings.

| Operation in `microscope.computations_grid` | Result | Samples removed at each nonperiodic end |
| --- | --- | --- |
| `basic.partial_derivatives_across_all_dims_batched` | NumPy Jacobian, trailing shape `(features, d)` | 1 |
| `volume.volume_element` | NumPy array of `sqrt(det(g))` | 1 |
| `curvature.scalar_curvature` | NumPy scalar-curvature array | 3 |
| `reach.reach_per_point` | NumPy local-reach estimates | 1 |

Periodic axes are preserved. Supply one patch size per grid axis, greater than
2 for derivatives/volume/reach and 6 for curvature, with enough grid samples for
the stencil. Smaller patches reduce temporary memory; full arrays stay in RAM.

Use `device="cpu"` or `"cuda:0"`; the default selects CUDA when available,
otherwise CPU. Lower-level tensor APIs use the tensor's device and require
manual periodic padding. NumPy wrappers detach from autograd; MPS is unvalidated.

### Compute measures on your own dense dataset

Save your ordered grid as `my_grid.npy`. For images, flatten only the image
axes first. This example assumes `[0, 1] × [0, 2*pi)`, with a periodic second
axis and at least seven samples on the first:

```python
import numpy as np
from microscope.cyclic_dimensions import get_difference_intervals
from microscope.computations_grid.volume import volume_element, compute_total_volume
from microscope.computations_grid.curvature import scalar_curvature, compute_total_curvature
from microscope.computations_grid.reach import reach_per_point

points = np.load("my_grid.npy").astype(np.float64)  # (s1, s2, features)
ranges, cyclic, patches = [1.0, 2 * np.pi], [1], [13, 17]
spacing = get_difference_intervals(points.shape[:-1], ranges, cyclic)
volume = volume_element(points, spacing, cyclic_dimensions=cyclic,
                        patch_sizes=patches, device="cpu")
curvature = scalar_curvature(points, spacing, cyclic_dimensions=cyclic,
                             patch_sizes=patches, device="cpu")
reach = reach_per_point(points, ranges, patches, cyclic,
                        subsample_points=4, device="cpu")
print("Volume:", compute_total_volume(volume, ranges))
print("Sampled reach:", reach.min())
# Volume/reach lose one sample per nonperiodic end; curvature loses three.
aligned_volume = volume[2:-2, :]
print("Total scalar curvature:", compute_total_curvature(
    aligned_volume, curvature, ranges,
))
```

Set ranges and cyclic axes to match your coordinates. Integrals use cropped-grid
means and domain lengths, so boundary handling and resolution affect accuracy.

`scalar_curvature(..., normalize=True)` divides by `d * (d - 1)` (requires
`d >= 2`), giving Gaussian curvature on surfaces. It does not normalize volume.
Reach uses sampled pairs; its minimum estimates global reach. Subsampling speeds
the pair search but can miss bottlenecks. Begin with small grids.

For multiple classes, [compute_measures_multiclass](../microscope/computations_grid/data_analysis/data_analysis.py)
accepts `(classes, s1, ..., sd, features)` and returns arrays, aggregates and
interclass distances. `normalize_for_volume=True` scales the combined volume
to one. Historical signed-curvature aggregates use sign-subset averages, so
they need not sum to total curvature and empty subsets yield NaN.

## Fit models, run inference and analyze representations

The [single-model walkthrough](../experiments_and_analysis/README.md)
shows dataset export, MMLS/beta-VAE fitting, evaluation and layer analysis on
built-in image datasets. A new training dataset requires a compatible loader
and model input; the direct grid API above needs no training. See the
[reproduction guide](reproducibility.md) for paper experiments and the
[dataset notebooks](../notebooks/datasets_and_measures) for visualizations.
