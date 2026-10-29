"""Memory regression: patch views must not be flattened into a full copy."""
import numpy as np
import pytest
import torch

from microscope.patches import extract_patches, stack_patches
from microscope.computations_grid import basic, curvature, curvature_extrinsic, volume


class NoFlatten(np.ndarray):
    def reshape(self, *args, **kwargs):
        raise AssertionError('Do not materialize the overlapping patch grid')


def forbidden_padding(*args, **kwargs):
    raise AssertionError('Wrap curvature boundaries within individual tiles')


@pytest.mark.parametrize('shape,cyclic', [((11, 12), []), ((10, 11), [1]),
                                         ((9, 10, 11), [0, 2]), ((8, 9, 10, 11), [1])])
def test_streamed_results_match_original_patch_assembly(monkeypatch, shape, cyclic):
    dims = len(shape)
    data = np.random.default_rng(42).normal(size=(*shape, dims + 2))
    sizes = [8] * dims
    intervals = [0.2] * dims
    for module, function, kernel, overlap, n_features in [
        (basic, basic.partial_derivatives_across_all_dims_batched,
         lambda t: basic.partial_derivatives_across_all_dims(t, dims, intervals), 2, 2),
        (volume, volume.volume_element, lambda t: volume.volume_element_batch(t, intervals), 2, 0),
        (curvature, curvature.scalar_curvature,
         lambda t: curvature.scalar_curvature_batch(t, intervals), 6, 0),
        (curvature_extrinsic, curvature_extrinsic.scalar_curvature,
         lambda t: curvature_extrinsic.scalar_curvature_batch(t, intervals), 6, 0),
    ]:
        patches = extract_patches(data, sizes, [overlap] * dims, cyclic)
        values = [kernel(torch.from_numpy(patch)).numpy()
                  for patch in patches.reshape(-1, *patches.shape[dims:])]
        assembled = np.stack(values).reshape(*patches.shape[:dims], *values[0].shape)
        reference = stack_patches(assembled, n_features=n_features)
        crop = tuple(slice(0, s if i in cyclic else s - overlap) for i, s in enumerate(shape))
        with monkeypatch.context() as patcher:
            if module in (curvature, curvature_extrinsic):
                patcher.setattr(np, 'pad', forbidden_padding)
            else:
                patcher.setattr(
                    module,
                    'extract_patches',
                    lambda *args, **kwargs: extract_patches(*args, **kwargs).view(NoFlatten)
                )
            result = function(
                data,
                difference_intervals=intervals,
                patch_sizes=sizes,
                cyclic_dimensions=cyclic,
                device='cpu'
            )
        np.testing.assert_array_equal(result, reference[crop])
