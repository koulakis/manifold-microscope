from functools import partial
from typing import Optional

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from microscope.datasets.image_transforms import (
    statistical_scaling,
    place_in_middle_of_background,
    rotate_image,
    shrink_image, translate_image, apply_transform
)


LABEL_MAPPING = {
    0: "square",
    1: "ellipse",
    2: "heart"
}

DEFAULT_BLUR_SIGMA = 0.8


def create_square_sprite(image_size: int, square_size: int, dtype: np.dtype = np.float32) -> np.ndarray:
    """Create a square sprite of a given size in the center of a black image.

    Args:
        image_size: The size of the image containing the square.
        square_size: The size of the square.
        dtype: The data type of the image.

    Returns:
        A numpy array containing the sprite image.
    """
    background = np.zeros((image_size, image_size), dtype=dtype)
    square = np.ones((square_size, square_size), dtype=dtype)

    return place_in_middle_of_background(background, square)


def create_heart_sprite(image_size: int, heart_size: int, dtype: np.dtype = np.float32) -> np.ndarray:
    """Create a heart sprite of a given size in the center of a black image. Using the third equation of this blog:
    https://blogs.lcps.org/academiesonline/2021/02/13/the-equation-of-the-heart

    Args:
        image_size: Size of the square image (image_size x image_size).
        heart_size: Scaling factor for the heart shape.
        dtype: The data type of the image.

    Returns:
        An image with the heart.
    """
    # Create a black image
    image = np.zeros((image_size, image_size), dtype=dtype)

    # Generate heart shape using parametric equations
    t = np.linspace(0, 2 * np.pi, 1000)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

    # Scale and translate heart to the center of the image
    x = (x * heart_size / 32) + image_size // 2
    y: np.ndarray = -(y * heart_size / 32) + image_size // 2  # Invert y-axis to align correctly

    # Convert to integer pixel coordinates
    heart_contour = np.column_stack((x.astype(np.int64), y.astype(np.int64)))

    # Draw the heart on the image
    cv2.fillPoly(image, [heart_contour], [1.0])

    return image


def create_ellipse_sprite(image_size: int, ellipse_size: int, dtype: np.dtype = np.float32) -> np.ndarray:
    """Create an ellipse sprite of a given size in the center of a black image.

    Args:
        image_size: Size of the square image (image_size x image_size).
        ellipse_size: Length of the long axis of the ellipse.
        dtype: The data type of the image.

    Returns:
        An image with the ellipse.
    """
    # Create a black image
    image = np.zeros((image_size, image_size), dtype=dtype)

    # Compute the axis length of the ellipse.
    axis_length = (ellipse_size // 2, ellipse_size // 4)

    # Compute the center of the image
    center = image_size // 2, image_size // 2

    # Draw the ellipse on the image
    cv2.ellipse(
        img=image,
        center=center,
        axes=axis_length,
        angle=0,
        startAngle=0,
        endAngle=360,
        color=[1.0],
        thickness=-1
    )

    return image


def dsprites(
    image_size: int = 64,
    sprite_initial_size: int = 20,
    deformation_transforms_initial_size_factor: int = 4,
    sprites: tuple[str, ...] = ("square", "ellipse", "heart"),
    n_sizes: int = 6,
    n_angles: int = 40,
    min_size_ratio: float = 0.5,
    max_size_ratio: float = 1.0,
    max_translation: int = 16,
    translation_stride: int = 1,
    blur_sigma: Optional[float] = DEFAULT_BLUR_SIGMA,
    flat: bool = False,
    scale: bool = False,
    dtype: np.dtype = np.float32,
    processes: list[Optional[int]] = (None, None, None, None),
    verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Generate and return a version of the dsprites dataset. This contains images of a sprite with the following
    variations:

    - sprite image: Can be a square, ellipse or a heart.
    - size: The sprite can have a number of different sizes in a given range of equidistant scales.
    - orientation: The sprite can have a number of different orientations on a finite rotation group.
    - location: The sprite can be in any location of the 2D image, leaving some border space.

    Any possible combination of the above is included in the dataset in order to obtain a full (up to discretization)
    data manifold. The data then has a shape of (sprites sizes orientations y_values x_values image_height image_width).

    Args:
        image_size: The size of the image in pixels. Should be an even number.
        sprite_initial_size: The sprite size before resizing. Should be an even number.
        deformation_transforms_initial_size_factor: A factor to scale up the image and sprite size before applying the
            deformation transforms in order to minimize information loss.
        sprites: A set of sprites to use, could be any sublist of ["square", "ellipse", "heart"]. If not set, then
            it defaults to the whole list.
        n_sizes: The number of different sprite sizes to use.
        n_angles: The number of rotation angles to use.
        min_size_ratio: The minimum size ratio to downscale the sprite.
        max_size_ratio: The maximum size ratio to downscale the sprite.
        max_translation: The maximum translation of the sprite from the center of the image in pixels. This will be
            applied both to the left and the right.
        blur_sigma: An optional standard deviation of a Gaussian to be used for smoothening the images.
        translation_stride: The stride of the translation. If larger than 1, it will reduce the resolution of the
            manifold, but will reduce the data size. Applied to both the x and y directions.
        flat: If true, the dataset is flattened along on the variation dimensions.
        scale: If true, it centers the data around the mean and divides by the standard deviation.
        dtype: The type of the data.
        processes: A list with the processes used for each data transform.
        verbose: If true, progress bars will be printed during data generation.

    Returns:
        - data: The given version of the dSprites dataset of shape (n_shapes n_sizes n_angles n_dy n_dx height width).
            The last two dimensions correspond to the image shape and the rest to the transforms of the sprite.
        - target: Labels with the class of each example.
    """
    # Image and sprite sizes should be even to simplify splitting them.
    assert image_size % 2 == 0
    assert sprite_initial_size % 2 == 0

    p1, p2, p3, p4 = processes

    # Use upscaled images till the deformation transforms (rotation and scaling) are performed to get smooth shapes.
    upscaled_image_size = deformation_transforms_initial_size_factor * image_size
    upscaled_sprite_size = deformation_transforms_initial_size_factor * sprite_initial_size

    # Define the three sprites.
    sprite_images = []
    if "square" in sprites:
        square = create_square_sprite(image_size=upscaled_image_size, square_size=upscaled_sprite_size, dtype=dtype)
        sprite_images.append(square)
    if "ellipse" in sprites:
        ellipse = create_ellipse_sprite(
            image_size=upscaled_image_size, ellipse_size=upscaled_sprite_size, dtype=dtype
        )
        sprite_images.append(ellipse)
    if "heart" in sprites:
        heart = create_heart_sprite(
            image_size=upscaled_image_size, heart_size=upscaled_sprite_size, dtype=dtype
        )
        sprite_images.append(heart)

    sprite_images = np.stack(sprite_images, axis=0)

    # Get all rotations.
    sprite_images = apply_transform(
        transform_name="rotate",
        image_grid=sprite_images,
        transform=rotate_image,
        transform_range=(
            "angle",
            np.linspace(0, 360, n_angles, endpoint=False, dtype=dtype)
        ),
        axis=-3,
        processes=p1,
        verbose=verbose
    )

    # Get all sizes. The transform also undoes the original up-scaling.
    sprite_images = apply_transform(
        transform_name="resize",
        image_grid=sprite_images,
        transform=partial(shrink_image, output_size=image_size),
        transform_range=(
            "size",
            [
                int(image_size * r)
                for r in np.linspace(min_size_ratio, max_size_ratio, n_sizes, dtype=dtype)
            ]
        ),
        axis=-4,
        processes=p2,
        verbose=verbose
    )

    # Get all y-axis translations.
    sprite_images = apply_transform(
        transform_name="y-translate",
        image_grid=sprite_images,
        transform=partial(translate_image, dx=0.0),
        transform_range=(
            "dy",
            np.arange(-max_translation, max_translation, translation_stride, dtype=dtype)
        ),
        axis=-3,
        processes=p3,
        verbose=verbose
    )

    # Get all x-axis translations.
    sprite_images = apply_transform(
        transform_name="x-translate",
        image_grid=sprite_images,
        transform=partial(translate_image, dy=0.0),
        transform_range=(
            "dx",
            np.arange(-max_translation, max_translation, translation_stride, dtype=dtype)
        ),
        axis=-3,
        processes=p4,
        verbose=verbose
    )

    all_ones = np.ones(sprite_images.shape[1:-2], dtype=np.int64)
    target = np.stack([0*all_ones, 1*all_ones, 2*all_ones], axis=0)

    # Scale the images from [0, 1] to [-1, 1].
    sprite_images = (sprite_images - sprite_images.min()) / (sprite_images.max() - sprite_images.min())
    sprite_images = 2 * sprite_images - 1

    if blur_sigma is not None:
        sprite_images = gaussian_filter(sprite_images, sigma=blur_sigma, axes=[-2, -1])

    if flat:
        sprite_images = sprite_images.reshape((-1, image_size, image_size))
        target = target.flatten()

    if scale:
        sprite_images = statistical_scaling(sprite_images)

    return sprite_images, target


def dsprites_original_remake(
    flat: bool = False,
    scale: bool = False,
    blur_sigma: Optional[float] = DEFAULT_BLUR_SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduces a smoothed version of the original dSprites dataset. The sprite sizes,
    might not be the same, but the sizes of the different dimensions match.

    Args:
        flat: If true, the dataset is flattened along on the variation dimensions.
        scale: If true, it centers the data around the mean and divides by the standard deviation.
        blur_sigma: An optional standard deviation of a Gaussian to be used for smoothening the images.

    Returns:
        - data: The given version of the dSprites dataset of shape (n_shapes n_sizes n_angles n_dy n_dx height width).
            The last two dimensions correspond to the image shape and the rest to the transforms of the sprite.
        - target: Labels with the class of each example.
    """
    return dsprites(
        image_size=64,
        sprite_initial_size=20,
        deformation_transforms_initial_size_factor=4,
        sprites=("square", "ellipse", "heart"),
        n_sizes=6,
        n_angles=40,
        min_size_ratio=0.5,
        max_size_ratio=1.0,
        max_translation=16,
        translation_stride=1,
        blur_sigma=blur_sigma,
        flat=flat,
        scale=scale,
        dtype=np.float32,
        verbose=True
    )


def dsprites_original_remake_single_size(
    flat: bool = False,
    scale: bool = False,
    blur_sigma: Optional[float] = DEFAULT_BLUR_SIGMA
) -> tuple[np.ndarray, np.ndarray]:
    """Again a smoothed version of the original dataset, this time selecting only the
    largest sprite size in order to remove one dimension.

    Args:
        flat: If true, the dataset is flattened along on the variation dimensions.
        scale: If true, it centers the data around the mean and divides by the standard deviation.
        blur_sigma: An optional standard deviation of a Gaussian to be used for smoothening the images.

    Returns:
        - data: The given version of the dSprites dataset of shape (n_shapes n_sizes n_angles n_dy n_dx height width).
            The last two dimensions correspond to the image shape and the rest to the transforms of the sprite.
        - target: Labels with the class of each example.
    """
    data, target = dsprites(
        image_size=64,
        sprite_initial_size=20,
        deformation_transforms_initial_size_factor=4,
        sprites=("square", "ellipse", "heart"),
        n_sizes=2,
        n_angles=40,
        min_size_ratio=0.5,
        max_size_ratio=1.0,
        max_translation=16,
        translation_stride=1,
        flat=False,
        scale=scale,
        blur_sigma=blur_sigma,
        dtype=np.float32,
        verbose=True
    )
    data = data[:, -1]
    target = target[:, -1]

    if flat:
        data = data.reshape((-1, 64, 64))
        target = target.flatten()

    return data, target


def dsprites_balanced(
    flat: bool = False,
    scale: bool = False,
    blur_sigma: Optional[float] = DEFAULT_BLUR_SIGMA,
    buffer_for_measure_estimation: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """A balanced version of the dsprites dataset with equal dimension sizes of 16.

    Args:
        flat: If true, the dataset is flattened along on the variation dimensions.
        scale: If true, it centers the data around the mean and divides by the standard deviation.
        blur_sigma: An optional standard deviation of a Gaussian to be used for smoothening the images.
        buffer_for_measure_estimation: Additional bidirectional margin on the non-cyclic dimensions
            in order to be able to estimate geometric measures which require surrounding points.

    Returns:
        - data: The given version of the dSprites dataset of shape (n_shapes n_sizes n_angles n_dy n_dx height width).
            The last two dimensions correspond to the image shape and the rest to the transforms of the sprite.
        - target: Labels with the class of each example.
    """

    return dsprites(
        image_size=64,
        sprite_initial_size=20,
        deformation_transforms_initial_size_factor=4,
        sprites=("square", "ellipse", "heart"),
        n_sizes=16 + 2*buffer_for_measure_estimation,
        n_angles=16,
        min_size_ratio=0.5,
        max_size_ratio=1.0,
        max_translation=16 + 2*buffer_for_measure_estimation,
        translation_stride=2,
        flat=flat,
        scale=scale,
        blur_sigma=blur_sigma,
        dtype=np.float32,
        verbose=True
    )
