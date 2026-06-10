import numpy as np


def sphere_volume_element(radius: float, coordinates: np.ndarray) -> float:
    """Compute the volume element of an n-sphere. We simply use the formula

    dV = R^n sin^{n-1}(phi_1) sin^{n-2}(phi_2) ... sin(phi_{n-1})

    Args:
        radius: The radius of the n-sphere.
        coordinates: The (phi_1, phi_2, ..., phi_n) coordinates of the point to compute the element on.

    Returns:
        The volume element value of the point at the given coordinates.
    """
    coordinates = np.array(coordinates)
    n = coordinates.shape[-1]
    components = np.sin(coordinates[..., :-1])
    return radius**n * np.prod(components**np.arange(n-1, 0, -1), axis=-1)


def sphere_scalar_curvature(radius: float, coordinates: np.ndarray, normalized: bool = False) -> np.ndarray:
    """Compute the scalar curvature of an n-sphere. It depends only on the radius and the dimension as seen in:

    R = n (n - 1) / R^2

    Args:
        radius: The radius of the n-sphere.
        coordinates: The (phi_1, phi_2, ..., phi_n n) coordinates of the point to compute the element on.
        normalized: If true, it computes a normalized version of the scalar curvature, like in the definition
            in Do Carmo.

    Returns:
        A tensor with the scalar curvature of the sphere of the given radius repeated to match the input shape.
    """
    if normalized:
        return np.repeat(1/radius**2, np.prod(coordinates.shape[:-1])).reshape(coordinates.shape[:-1])
    else:
        coordinates = np.array(coordinates)
        n = coordinates.shape[-1]
        return np.repeat(n*(n-1)/radius**2, np.prod(coordinates.shape[:-1])).reshape(coordinates.shape[:-1])
