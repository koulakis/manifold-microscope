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
        f,
        y,
        lower_limit,
        t0=None,
        tol=1e-10,
        max_iter=1_000,
        quad_kwargs=None
):
    if quad_kwargs is None:
        quad_args = DEFAULT_QUAD_KWARGS

    # noinspection PyShadowingNames
    def f_int(t):
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
        sample,
        measure_f,
        limits,
        quad_kwargs,
        invert_integral_kwargs
):
    s1, s2 = sample
    [l1_l, l1_u], [l2_l, l2_u] = limits

    # Invert the second coordinate.
    def inner_int(s):
        return quad(lambda t: measure_f(t, s), l1_l, l1_u, **quad_kwargs)[0]

    inner_int_full = quad(lambda s: inner_int(s), l2_l, l2_u, **quad_kwargs)[0]

    inv2 = invert_integral(f=inner_int, y=inner_int_full * s2, lower_limit=l2_l, **invert_integral_kwargs)

    # Invert the first coordinate.
    inv1 = invert_integral(
        f=lambda t: measure_f(t, inv2),
        y=inner_int(inv2) * s1,
        lower_limit=l1_l,
        **invert_integral_kwargs
    )

    return inv1, inv2


def uniform_to_parameter_space_dim_3(
        sample,
        measure_f,
        limits,
        quad_kwargs,
        invert_integral_kwargs
):
    s1, s2, s3 = sample
    [l1_l, l1_u], [l2_l, l2_u], [l3_l, l3_u] = limits

    # Invert the third coordinate.
    def inner_int_2(r):
        return quad(
            lambda s: quad(lambda t: measure_f(t, s, r), l1_l, l1_u, **quad_kwargs)[0],
            l2_l,
            l2_u,
            **quad_kwargs
        )[0]

    inner_int_2_full = quad(lambda r: inner_int_2(r), l3_l, l3_u, **quad_kwargs)[0]

    inv3 = invert_integral(inner_int_2, inner_int_2_full * s3, lower_limit=l3_l, **invert_integral_kwargs)

    # Invert the second coordinate.
    def inner_int(s):
        return quad(lambda t: measure_f(t, s, inv3), l1_l, l1_u, **quad_kwargs)[0]

    inner_int_full = quad(lambda s: inner_int(s), l2_l, l2_u, **quad_kwargs)[0]

    inv2 = invert_integral(inner_int, inner_int_full * s2, lower_limit=l2_l, **invert_integral_kwargs)

    # Invert the first coordinate.
    inv1 = invert_integral(
        lambda t: measure_f(t, inv2, inv3),
        inner_int(inv2) * s1,
        lower_limit=l1_l,
        **invert_integral_kwargs
    )

    return inv1, inv2, inv3


def sample_manifold_uniformly(
    n_samples,
    measure_f,
    limits,
    parameter_to_ambient_coords_map,
    ambient_space_dim=None,
    apply_random_isometry=True,
    processes=None,
    quad_kwargs=None,
    invert_integral_kwargs=None,
    seed=None
):
    # Get optimization configuration.
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
        raise ValueError(f"Only support uniform sampling on 2 or 3 dimensional manifolds.")

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
    n_samples,
    semi_axes,
    ambient_space_dim=None,
    apply_random_isometry=True,
    processes=None,
    quad_kwargs=None,
    invert_integral_kwargs=None,
    seed=None
):
    # Get the dims and symbols.
    semi_axes = np.array(semi_axes)
    dim = len(semi_axes) - 1
    axes_symbols = sp.symbols("a_1 a_2 a_3 a_4")[:dim+1]
    phi_symbols = sp.symbols("phi_1 phi_2 phi_3")[:dim]
    subs = dict(zip(axes_symbols, semi_axes))

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

    def parameter_map(*phis):
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

    return points, semi_axes, parameters


def sample_hyperboloid_uniformly(
    n_samples,
    semi_axes,
    ambient_space_dim=None,
    apply_random_isometry=True,
    processes=None,
    quad_kwargs=None,
    invert_integral_kwargs=None,
    seed=None
):
    # Get the dims and symbols.
    semi_axes = np.array(semi_axes)
    dim = len(semi_axes) - 1
    axes_symbols = sp.symbols("a_1 a_2 a_3 a_4")[:dim + 1]
    phi_symbols = sp.symbols("phi_1 phi_2 phi_3")[:dim]
    subs = dict(zip(axes_symbols, semi_axes))

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

    def parameter_map(*phis):
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

    return points, semi_axes, parameters
