from functools import partial
from pathlib import Path
import os

import numpy as np
from PIL import Image

from microscope.datasets.image_transforms import apply_transform, statistical_scaling, shrink_image, rotate_image, \
    place_in_middle_of_background

LABEL_MAPPING = {
    0: "duck",
    1: "wood_triangle",
    2: "car0",
    3: "cat",
    4: "anacin",
    5: "car1",
    6: "wood_square",
    7: "jnj",
    8: "tylenol",
    9: "vaseline",
    10: "wood_semicircle",
    11: "mug",
    12: "pig",
    13: "bolt_thread",
    14: "pot",
    15: "bottle",
    16: "pot2",
    17: "cup",
    18: "car2",
    19: "yoghurt"
}


def load_coil20(views_stride: int = 4) -> tuple[np.ndarray, np.ndarray]:
    path_by_env_var = os.environ.get("COIL20_PATH")
    if path_by_env_var is not None:
        coil_data_path = Path(path_by_env_var)
    else:
        coil_data_path = Path("~/datasets/curvature-landscapes/coil-20-proc").expanduser()
    coil_sorted_paths = sorted(
        coil_data_path.glob("*.png"),
        key=lambda p: (int(p.stem.split('__')[0][3:]), int(p.stem.split('__')[1]))
    )

    # noinspection PyTypeChecker
    x_coil20, y_coil20 = zip(*[
        (np.array(Image.open(path)).flatten(), int(path.stem.split('__')[0][3:]) - 1)
        for path in coil_sorted_paths
    ])

    x_coil20, y_coil20 = np.vstack(x_coil20), np.hstack(y_coil20)

    x_coil20 = x_coil20.reshape((20, 72, 128, 128))
    y_coil20 = y_coil20.reshape((20, 72))

    x_coil20 = x_coil20[:, ::views_stride]
    y_coil20 = y_coil20[:, ::views_stride]

    # Convert gray scale from [0, 256] to [0, 1]
    x_coil20 = x_coil20.astype(np.float32) / 256

    return x_coil20, y_coil20


def extended_coil20(
    image_size: int = 64,
    deformation_transforms_initial_size: int = 182,  # Approximately 128*sqrt(2)
    n_sizes: int = 16,
    n_angles: int = 16,
    min_size_ratio: float = 0.5,
    max_size_ratio: float = 1.0,
    flat: bool = False,
    scale: bool = False,
    buffer_for_measure_estimation: int = 3,
    dtype: np.dtype = np.float32,
    verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Generate and return a version of the dsprites dataset. This contains images of a sprite with the following
    variations:

    - sprite image: Can be a square, triangle or a heart.
    - size: The sprite can have a number of different sizes in a given range of equidistant scales.
    - orientation: The sprite can have a number of different orientations on a finite rotation group.
    - location: The sprite can be in any location of the 2D image, leaving some border space.

    Any possible combination of the above is included in the dataset in order to obtain a full (up to discretization)
    data manifold. The data then has a shape of (sprites sizes orientations y_values x_values image_height image_width).

    Args:
        image_size: The size of the image in pixels. Should be an even number.
        deformation_transforms_initial_size: A size to pad the image before applying rotations.
        n_sizes: The number of different sprite sizes to use.
        n_angles: The number of rotation angles to use.
        min_size_ratio: The minimum size ratio to downscale the sprite.
        max_size_ratio: The maximum size ratio to downscale the sprite.
        flat: If true, the dataset is flattened along on the variation dimensions.
        scale: If true, it centers the data around the mean and divides by the standard deviation.
        buffer_for_measure_estimation: Additional bidirectional margin on the non-cyclic dimensions
            in order to be able to estimate geometric measures which require surrounding points.
        dtype: The type of the data.
        verbose: If true, progress bars will be printed during data generation.

    Returns:
        The given version of the dSprites dataset of shape (n_shapes n_rotations n_sizes n_angles height width). The last
        two dimensions correspond to the image shape and the rest to the transforms of the sprite.
    """
    # Add a buffer for the sizes.
    n_sizes = n_sizes + 2 * buffer_for_measure_estimation

    coil_images, coil_labels = load_coil20()
    coil_images = coil_images.astype(dtype=dtype)

    # Pad the image with -1 to avoid loosing information while rotating.
    background = np.zeros(
        shape=(deformation_transforms_initial_size, deformation_transforms_initial_size),
        dtype=dtype
    )

    coil_images = apply_transform(
        transform_name="zero_pad",
        image_grid=coil_images,
        transform=lambda x, dummy_argument: place_in_middle_of_background(background, x),
        transform_range=("dummy_argument", [0]),
        axis=0,
        verbose=verbose
    )[0]

    # Get all rotations.
    coil_images = apply_transform(
        transform_name="rotate",
        image_grid=coil_images,
        transform=rotate_image,
        transform_range=(
            "angle",
            np.linspace(0, 360, n_angles, endpoint=False, dtype=dtype)
        ),
        axis=-3,
        verbose=verbose
    )

    # Get all sizes. The transform also undoes the original up-scaling.
    coil_images = apply_transform(
        transform_name="resize",
        image_grid=coil_images,
        transform=partial(shrink_image, output_size=image_size),
        transform_range=(
            "size",
            [
                int(image_size * r)
                for r in np.linspace(min_size_ratio, max_size_ratio, n_sizes, dtype=dtype)
            ]
        ),
        axis=-4,
        verbose=verbose
    )

    target = np.stack(n_sizes*[coil_labels], axis=-1)
    target = np.stack(n_angles*[target], axis=-1)

    # Scale images from [0, 1] to [-1, 1].
    coil_images = 2 * coil_images - 1

    if flat:
        coil_images = coil_images.reshape((-1, image_size, image_size))
        target = target.flatten()

    if scale:
        coil_images = statistical_scaling(coil_images)

    return coil_images, target
