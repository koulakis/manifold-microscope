import numpy as np

from microscope.datasets.image_transforms import statistical_scaling


def dsprites_original(
    flat: bool = False,
    scale: bool = False,
    dtype: np.dtype = np.float32
) -> tuple[np.ndarray, np.ndarray]:
    """Load the original dsprites dataset."""
    data_path = (
        "<path to dataset>/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    )

    data = np.load(data_path, encoding="bytes")["imgs"]

    all_ones = np.ones(shape=data.shape[0] // 3, dtype=np.int64)
    target = np.concatenate([0*all_ones, 1*all_ones, 2*all_ones], axis=0)

    if not flat:
        data = data.reshape((3, 6, 40, 32, 32, 64, 64))
        target = target.reshape((3, 6, 40, 32, 32))

    # Scale from [0, 1] to [-1, 1]
    data = (data - data.min()) / (data.max() - data.min())
    data = (2 * data - 1).astype(dtype)

    if scale:
        data = statistical_scaling(data)

    return data, target


def dsprites_original_single_size(
    flat: bool = False,
    scale: bool = False,
    dtype: np.dtype = np.float32
) -> tuple[np.ndarray, np.ndarray]:
    """The original dsprites selecting only the largest sprite size in order to remove one dimension"""
    dsprites = dsprites_original(dtype=dtype)
    dsprites, target = dsprites

    data = (
        dsprites
        .reshape((3, 6, 40, 32, 32, 64, 64))
        [:, 5]
    )
    target = target.reshape((3, 6, 40, 32, 32))[:, 5]

    if flat:
        data = data.reshape((-1, 64, 64))
        target = target.flatten()

    if scale:
        data = statistical_scaling(data)

    return data, target
