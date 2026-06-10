from functools import partial
from multiprocessing import Pool
from typing import Callable, Optional

import cv2
import numpy as np
from tqdm import tqdm


def statistical_scaling(data: np.ndarray):
    data_mean = data.mean()
    data_std = data.std()

    return (data - data_mean) / data_std


def rotate_image(
        image: np.ndarray,
        angle: float
) -> np.ndarray:
    """Rotate a 2D image given in a numpy array.

    Args:
        image: The image in a numpy array.
        angle: The angle in degrees.

    Returns:
        The rotated image, same size as the input image.
    """
    center = np.array(image.shape) / 2
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    return rotated_image


def place_in_middle_of_background(background: np.ndarray, image: np.ndarray) -> np.ndarray:
    """It positions a foreground image in the middle of a background image. The foreground image will overwrite all
    pixels of the area where it is pasted.

    Args:
        background: The background image.
        image: The foreground image, should be smaller than the background one.

    Returns:
        A copy of the background image with the foreground image pasted in the middle.
    """
    background = background.copy()
    cy, cx = (np.array(background.shape) // 2).astype(np.int64)
    half_h, half_w = (np.array(image.shape) // 2).astype(np.int64)
    h, w = image.shape

    background[
        cy - half_h:cy + h - half_h,
        cx - half_w:cx + w - half_w
    ] = image

    return background


def shrink_image(image: np.ndarray, size: int, output_size: int, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Shrink a given image to a selected size. The new image will have the same size as before and the old shrunk one
    will be placed in the center and padded with zeroes.

    Args:
        image: The image to shrink of shape (height width).
        size: The size of the original image after shrinking.
        output_size: The size of the output image containing the shrunk image.
        interpolation: The interpolation method.

    Returns:
        An image with the shrunk input image in the middle.
    """
    if output_size < size:
        raise ValueError(f"Attempting to shrink image or shape {output_size} to the larger size {size}.")

    background = np.zeros((output_size, output_size), dtype=image.dtype)
    image = cv2.resize(image, dsize=(size, size), interpolation=interpolation)

    return place_in_middle_of_background(background, image)


def translate_image(
        image: np.ndarray,
        dx: int,
        dy: int
) -> np.ndarray:
    """Translate a 2D image given in a numpy array.

    Args:
        image: The image in a numpy array.
        dx: The translation on the x-axis in pixels.
        dy: The translation on the y-axis in pixels.

    Returns:
        The translated image in a numpy array, same size as the input image.
    """
    translation_matrix = np.array([
        [1, 0, dx],
        [0, 1, dy]
    ], dtype=np.float32)

    return cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))


def apply_transform(
    transform_name: str,
    image_grid: np.ndarray,
    transform: Callable[[np.ndarray, ...], np.ndarray],
    transform_range: tuple[str, list | np.ndarray],
    axis: int = 0,
    processes: Optional[int] = None,
    verbose: bool = False
) -> np.ndarray:
    """Apply a transform over all images in a grid and for all given values of the transform.

    Args:
        transform_name: The name of the transform.
        image_grid: A grid of shape (... height width) which contains images.
        transform: A function which transforms a single image.
        transform_range: A tuple with the name of a kwarg of the transform function and a range of values it should
            be computed on.
        axis: The axis along which the transformed versions of the images will be stacked.
        processes: An optional number of processes to parallelize each single transform of the image grid.
        verbose: If true, a progress bar on the number of transforms will be shown.

    Returns:
        A new image grid of shape (n_transforms ... height width).
    """
    original_shape = image_grid.shape

    image_grid = np.reshape(image_grid, (-1, *image_grid.shape[-2:]))

    name, rng = transform_range
    pbar = tqdm(desc=transform_name, total=len(rng), disable=not verbose)

    transformed_images = []
    for val in rng:
        if processes is not None:
            with Pool(processes=processes) as pool:
                transformed = pool.map(
                    partial(transform, **{name: val}),
                    image_grid,
                    chunksize=int(image_grid.shape[0] // (10*processes))
                )

        else:
            transformed = list(map(partial(transform, **{name: val}), image_grid))

        transformed = np.array(transformed)
        transformed = transformed.reshape((*original_shape[:-2], *transformed.shape[-2:]))
        transformed_images.append(transformed)
        pbar.update(1)
    pbar.close()

    return np.stack(transformed_images, axis=axis)
