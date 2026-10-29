from collections.abc import Callable

import numpy as np
import sympy as sp
from scipy.integrate import quad

from microscope.manifold_examples.symbolic_computations import ellipsoid_parametrization, hyperboloid_parametrization, \
    ellipsoid_volume_element_formula, hyperboloid_volume_element_formula
from microscope.manifold_examples.utils import parallel_apply, embed_to_ambient_space, random_isometry_transform

DEFAULT_QUAD_KWARGS = {
    "limit": 500,
    "epsrel": 1e-12,
    "epsabs": 1e-12
}

DEFAULT_NEWTON_RAPHSON_KWARGS = {
    "tol": 1e-12,
    "max_iter": 1_000
}


def invert_integral(
    f: Callable[[float], float],
    y: float,
    lower_limit: float,
    t0: float | None = None,
    tol: float = 1e-10,
    max_iter: int = 1_000,
    quad_kwargs: dict | None = None
) -> float:
    """Find an upper bound whose integral matches the target value.

    Args:
        f: Scalar integrand as a function of one coordinate.
        y: Target value of the integral.
        lower_limit: Lower integration bound.
        t0: Optional initial estimate of the upper integration bound.
        tol: Absolute stopping tolerance on the integral residual.
        max_iter: Maximum number of Newton iterations.
        quad_kwargs: Options forwarded to scipy.integrate.quad.

    Returns:
        Estimated upper integration bound, including the final iterate if convergence is not reached.
    """
    quad_args = DEFAULT_QUAD_KWARGS if quad_kwargs is None else quad_kwargs

    # noinspection PyShadowingNames
    def f_int(t: float) -> float:
        """Integrate the density up to the given coordinate.

        Args:
            t: Upper integration bound.

        Returns:
            Integral of f from lower_limit to t.
        """
        return quad(f, lower_limit, t, **quad_args)[0]

    if t0 is None:
        t0 = y / f(1)  # Approximation: assuming g(s) ~ g(1) near the solution
    t = t0
    for _ in range(max_iter):
        ft = f_int(t)
        if abs(ft - y) < tol:
            return t
        gt = f(t)
        t -= (ft - y) / gt
    print(f"Newton-Raphson did not converge after {max_iter} iterations.")
    return t


def uniform_to_parameter_space_dim_2(
    sample: np.ndarray,
    measure_f: Callable[[float, float], float],
    limits: np.ndarray,
    quad_kwargs: dict,
    invert_integral_kwargs: dict
) -> tuple[float, float]:
    """Map a uniform square sample to surface coordinates.

    Args:
        sample: Uniform parameter sample, shape (2).
        measure_f: Volume-density function of two coordinates.
        limits: Coordinate integration bounds, shape (2 2).
        quad_kwargs: Options forwarded to scipy.integrate.quad.
        invert_integral_kwargs: Options forwarded to invert_integral.

    Returns:
        (first_coordinate, second_coordinate), two scalar parameters distributed according to measure_f.
    """
    s1, s2 = sample
    [l1_l, l1_u], [l2_l, l2_u] = limits

    # Invert the second coordinate.
    def inner_int(s: float) -> float:
        """Integrate out the first coordinate at a fixed second coordinate.

        Args:
            s: Coordinate held fixed during the inner integration.

        Returns:
            Marginal density at s after integrating over the first coordinate.
        """
        return quad(lambda t: measure_f(t, s), l1_l, l1_u, **quad_kwargs)[0]

    inner_int_full = quad(lambda s: inner_int(s), l2_l, l2_u, **quad_kwargs)[0]

    inv2 = invert_integral(
        f=inner_int,
        y=inner_int_full * s2,
        lower_limit=l2_l,
        **invert_integral_kwargs
    )

    # Invert the first coordinate.
    inv1 = invert_integral(
        f=lambda t: measure_f(t, inv2),
        y=inner_int(inv2) * s1,
        lower_limit=l1_l,
        **invert_integral_kwargs
    )

    return inv1, inv2


def uniform_to_parameter_space_dim_3(
    sample: np.ndarray,
    measure_f: Callable[[float, float, float], float],
    limits: np.ndarray,
    quad_kwargs: dict,
    invert_integral_kwargs: dict
) -> tuple[float, float, float]:
    """Map a uniform cube sample to manifold coordinates.

    Args:
        sample: Uniform parameter sample, shape (3).
        measure_f: Volume-density function of three coordinates.
        limits: Coordinate integration bounds, shape (3 2).
        quad_kwargs: Options forwarded to scipy.integrate.quad.
        invert_integral_kwargs: Options forwarded to invert_integral.

    Returns:
        (first_coordinate, second_coordinate, third_coordinate), three scalar parameters distributed according to
        measure_f.
    """
    s1, s2, s3 = sample
    [l1_l, l1_u], [l2_l, l2_u], [l3_l, l3_u] = limits

    # Invert the third coordinate.
    def inner_int_2(r: float) -> float:
        """Integrate out the first two coordinates.

        Args:
            r: Third coordinate held fixed during integration.

        Returns:
            Marginal density at r after integrating over the first two coordinates.
        """
        return quad(
            lambda s: quad(lambda t: measure_f(t, s, r), l1_l, l1_u, **quad_kwargs)[0],
            l2_l,
            l2_u,
            **quad_kwargs
        )[0]

    inner_int_2_full = quad(lambda r: inner_int_2(r), l3_l, l3_u, **quad_kwargs)[0]

    inv3 = invert_integral(
        inner_int_2,
        inner_int_2_full * s3,
        lower_limit=l3_l,
        **invert_integral_kwargs
    )

    # Invert the second coordinate.
    def inner_int(s: float) -> float:
        """Integrate out the first coordinate at a fixed second coordinate.

        Args:
            s: Coordinate held fixed during the inner integration.

        Returns:
            Density at (s, inv3) after integrating over the first coordinate.
        """
        return quad(lambda t: measure_f(t, s, inv3), l1_l, l1_u, **quad_kwargs)[0]

    inner_int_full = quad(lambda s: inner_int(s), l2_l, l2_u, **quad_kwargs)[0]

    inv2 = invert_integral(
        inner_int,
        inner_int_full * s2,
        lower_limit=l2_l,
        **invert_integral_kwargs
    )

    # Invert the first coordinate.
    inv1 = invert_integral(
        lambda t: measure_f(t, inv2, inv3),
        inner_int(inv2) * s1,
        lower_limit=l1_l,
        **invert_integral_kwargs
    )

    return inv1, inv2, inv3


def sample_manifold_uniformly(
    n_samples: int,
    measure_f: Callable[..., float],
    limits: np.ndarray,
    parameter_to_ambient_coords_map: Callable[..., np.ndarray],
    ambient_space_dim: int | None = None,
    apply_random_isometry: bool = True,
    processes: int | None = None,
    quad_kwargs: dict | None = None,
    invert_integral_kwargs: dict | None = None,
    seed: int | None = None
) -> tuple[np.ndarray | tuple[np.ndarray, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    # Get optimization configuration.
    """Sample a manifold using inverse volume-weighted coordinate integrals.

    Args:
        n_samples: Number of samples.
        measure_f: Volume-density function of d coordinates.
        limits: Coordinate integration bounds, shape (d 2).
        parameter_to_ambient_coords_map: Map from d coordinate arrays of shape (n) to points of shape (n f).
        ambient_space_dim: Number of embedding coordinates.
        apply_random_isometry: Apply a random rotation and translation.
        processes: Number of worker processes; None uses the available CPU count.
        quad_kwargs: Options forwarded to scipy.integrate.quad.
        invert_integral_kwargs: Options forwarded to invert_integral.
        seed: Optional random seed for the isometry.

    Returns:
        (points, parameters), with shapes (n_samples, f) and (n_samples, d). The coordinate map determines f, extended
        to ambient_space_dim when requested. With apply_random_isometry=True, the points entry is
        (transformed_points, (rotation, translation)), with rotation shape (f f) and translation shape (f).
    """
    if quad_kwargs is None:
        quad_kwargs = DEFAULT_QUAD_KWARGS
    if invert_integral_kwargs is None:
        invert_integral_kwargs = DEFAULT_NEWTON_RAPHSON_KWARGS

    # Select the dimension and the corresponding inverse mapping.
    dim = len(limits)
    if ambient_space_dim is None:
        ambient_space_dim = dim + 1
    if dim == 2:
        reverse_parameter_mapping = uniform_to_parameter_space_dim_2
    elif dim == 3:
        reverse_parameter_mapping = uniform_to_parameter_space_dim_3
    else:
        raise ValueError("Only support uniform sampling on 2 or 3 dimensional manifolds.")

    # Get the uniform samples on the parameter space.
    samples = np.random.uniform(0, 1, size=(n_samples, dim))

    parameters = parallel_apply(
        samples,
        reverse_parameter_mapping,
        element_wise=False,
        processes=processes,
        measure_f=measure_f,
        limits=limits,
        quad_kwargs=quad_kwargs,
        invert_integral_kwargs=invert_integral_kwargs
    )

    # Map to the ambient space coordinates and apply optional random isometry.
    points = parameter_to_ambient_coords_map(*parameters.T)

    if ambient_space_dim > dim + 1:
        points = embed_to_ambient_space(points, ambient_space_dim)

    if apply_random_isometry:
        points = random_isometry_transform(points, limit=1.2*np.abs(points).max(), seed=seed)

    return points, parameters


def sample_ellipsoid_uniformly(
    n_samples: int,
    semi_axes: list[float],
    ambient_space_dim: int | None = None,
    apply_random_isometry: bool = True,
    processes: int | None = None,
    quad_kwargs: dict | None = None,
    invert_integral_kwargs: dict | None = None,
    seed: int | None = None
) -> tuple[np.ndarray | tuple[np.ndarray, tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    # Get the dims and symbols.
    """Sample an ellipsoid uniformly with respect to its volume.

    Args:
        n_samples: Number of samples.
        semi_axes: Principal semi-axis lengths, one per intrinsic dimension plus one.
        ambient_space_dim: Number of embedding coordinates.
        apply_random_isometry: Apply a random rotation and translation.
        processes: Number of worker processes; None uses the available CPU count.
        quad_kwargs: Options forwarded to scipy.integrate.quad.
        invert_integral_kwargs: Options forwarded to invert_integral.
        seed: Optional random seed for the isometry.

    Returns:
        (points, semi_axes, parameters), with semi-axis shape (d + 1) and parameter shape (n_samples, d). Points have
        shape (n_samples, f), where f is ambient_space_dim or d + 1. With apply_random_isometry=True, the points entry
        is (transformed_points, (rotation, translation)), with rotation shape (f f) and translation shape (f).
    """
    semi_axes_array = np.array(semi_axes)
    dim = len(semi_axes_array) - 1
    axes_symbols = sp.symbols("a_1 a_2 a_3 a_4")[:dim+1]
    phi_symbols = sp.symbols("phi_1 phi_2 phi_3")[:dim]
    subs = dict(zip(axes_symbols, semi_axes_array))

    # Define the parameter limits.
    limits = (
        np.array([[0, np.pi], [0, 2*np.pi]])
        if dim == 2
        else np.array([[0, np.pi], [0, np.pi], [0, 2*np.pi]])
    )

    # Define the volume measure.
    element = ellipsoid_volume_element_formula(dim)
    element_np = sp.lambdify(phi_symbols, element.subs(subs))

    # Define the parameter mapping.
    parameter_lambd = sp.lambdify(phi_symbols, ellipsoid_parametrization(dim=dim)[0].subs(subs))

    def parameter_map(*phis: np.ndarray) -> np.ndarray:
        """Map parameter arrays to embedding coordinates.

        Args:
            *phis: Coordinate arrays, each of shape (n).

        Returns:
            Mapped coordinates, normally shape (n, d + 1); singleton axes are squeezed.
        """
        return parameter_lambd(*phis).squeeze().T

    points, parameters = sample_manifold_uniformly(
        n_samples=n_samples,
        measure_f=element_np,
        limits=limits,
        parameter_to_ambient_coords_map=parameter_map,
        ambient_space_dim=ambient_space_dim,
        apply_random_isometry=apply_random_isometry,
        processes=processes,
        quad_kwargs=quad_kwargs,
        invert_integral_kwargs=invert_integral_kwargs,
        seed=seed
    )

    return points, semi_axes_array, parameters


def sample_hyperboloid_uniformly(
    n_samples: int,
    semi_axes: list[float],
    ambient_space_dim: int | None = None,
    apply_random_isometry: bool = True,
    processes: int | None = None,
    quad_kwargs: dict | None = None,
    invert_integral_kwargs: dict | None = None,
    seed: int | None = None
) -> tuple[np.ndarray | tuple[np.ndarray, tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    # Get the dims and symbols.
    """Sample a bounded hyperboloid patch uniformly in volume.

    Args:
        n_samples: Number of samples.
        semi_axes: Principal semi-axis lengths, one per intrinsic dimension plus one.
        ambient_space_dim: Number of embedding coordinates.
        apply_random_isometry: Apply a random rotation and translation.
        processes: Number of worker processes; None uses the available CPU count.
        quad_kwargs: Options forwarded to scipy.integrate.quad.
        invert_integral_kwargs: Options forwarded to invert_integral.
        seed: Optional random seed for the isometry.

    Returns:
        (points, semi_axes, parameters), with semi-axis shape (d + 1) and parameter shape (n_samples, d). Points have
        shape (n_samples, f), where f is ambient_space_dim or d + 1. With apply_random_isometry=True, the points entry
        is (transformed_points, (rotation, translation)), with rotation shape (f f) and translation shape (f).
    """
    semi_axes_array = np.array(semi_axes)
    dim = len(semi_axes_array) - 1
    axes_symbols = sp.symbols("a_1 a_2 a_3 a_4")[:dim + 1]
    phi_symbols = sp.symbols("phi_1 phi_2 phi_3")[:dim]
    subs = dict(zip(axes_symbols, semi_axes_array))

    # Define the parameter limits.
    limits = (
        np.array([[-1, 1], [0, 2*np.pi]])
        if dim == 2
        else np.array([[-1, 1], [-1, 1], [0, 2*np.pi]])
    )

    # Define the volume measure.
    element = hyperboloid_volume_element_formula(dim)
    element_np = sp.lambdify(phi_symbols, element.subs(subs))

    # Define the parameter mapping.
    parameter_lambd = sp.lambdify(phi_symbols, hyperboloid_parametrization(dim=dim)[0].subs(subs))

    def parameter_map(*phis: np.ndarray) -> np.ndarray:
        """Map parameter arrays to embedding coordinates.

        Args:
            *phis: Coordinate arrays, each of shape (n).

        Returns:
            Mapped coordinates, normally shape (n, d + 1); singleton axes are squeezed.
        """
        return parameter_lambd(*phis).squeeze().T

    points, parameters = sample_manifold_uniformly(
        n_samples=n_samples,
        measure_f=element_np,
        limits=limits,
        parameter_to_ambient_coords_map=parameter_map,
        ambient_space_dim=ambient_space_dim,
        apply_random_isometry=apply_random_isometry,
        processes=processes,
        quad_kwargs=quad_kwargs,
        invert_integral_kwargs=invert_integral_kwargs,
        seed=seed
    )

    return points, semi_axes_array, parameters
