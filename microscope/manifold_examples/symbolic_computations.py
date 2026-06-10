import multiprocessing
import time
import signal
from contextlib import contextmanager
from functools import partial

import sympy as sp
from sympy import sin, cos, sinh, cosh, Matrix
import numpy as np
import einops
from joblib import Parallel, delayed

from microscope.manifold_examples.utils import parallel_apply


class TimeIt:
    """Wrapper which can be used to measure python code blocks runtime."""
    def __init__(self, process_name):
        self.process_name = process_name

    def __enter__(self):
        print(f"Computing the {self.process_name}.")
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        time_elapsed = time.time() - self.start
        print(f"{self.process_name} done, time elapsed: {time_elapsed:.3f} seconds.\n")


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds, message=None):
    def signal_handler(signum, frame):
        if message is not None:
            print(message)
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def _time_limited_simplify(formula, limit_seconds: float = 10 * 60, *args, **kwargs):
    """A time limited version of the sympy simplify function. If it does not succeed, it will return
    the original, unsimplified formula.

    Args:
        formula: The formula to simplify.
        limit_seconds: The maximum time for the simplification attempt.

    Returns:
        If the simplification succeeds inside the time limit, then the simplified formula, else
        the original input formula.
    """
    try:
        message = f"Simplification time of {limit_seconds} seconds exceeded. Returning original formula."
        with time_limit(limit_seconds, message=message):
            return sp.simplify(formula, *args, **kwargs)
    except TimeoutException:
        return formula


def _matrix_fast_inverse(matrix):
    """Symbolically computes the inverse of a matrix using Cramer's rule. This proved to be faster in some cases,
    mainly because of ordering of the sympy simplification functions which are slow.

    Args:
        matrix: A sympy matrix.

    Returns:
        A sympy matrix which is the inverse of the input matrix.
    """
    n = matrix.shape[0]

    determinant = matrix.det()

    cofactors_matrix = np.array([
        [
            (-1) ** (i + j) * np.array(sp.det(Matrix(np.delete(np.delete(matrix, i, axis=0), j, axis=1))))
            for i in range(n)
        ]
        for j in range(n)
    ])

    return cofactors_matrix / determinant


def scalar_curvature_formula(u, coordinate_variables: sp.Matrix, normalize: bool = False):
    """Compute a formula for the scalar curvature of a manifold on a patch parametrized by u.

        Args:
            u: A formula with the patch parametrization.
            coordinate_variables: The variables of the parametrization in a sympy array matrix.
            normalize: If true, it computes a normalized version of the scalar curvature, like in the definition
                in Do Carmo.

        Returns:
            A sympy formula for the scalar curvature.
    """
    dim = len(coordinate_variables)

    with TimeIt("Riemannian metric and inverse"):
        u_jacobian = u.jacobian(coordinate_variables)

        g_sp = sp.simplify(u_jacobian.T @ u_jacobian)
        g_inv = parallel_apply(_matrix_fast_inverse(g_sp), sp.simplify, element_wise=True)

    with TimeIt("derivative of the metric"):
        g_deriv = np.stack([np.array(g_sp.diff(p)) for p in coordinate_variables], axis=2)
        g_deriv = parallel_apply(g_deriv, sp.simplify, element_wise=True)

    with TimeIt("Christoffel symbols"):
        chr_first_kind = 1 / 2 * (g_deriv.transpose(1, 2, 0) + g_deriv.transpose(2, 0, 1) - g_deriv)
        chr_first_kind = parallel_apply(chr_first_kind, sp.simplify, element_wise=True)

        chr_second_kind = einops.einsum(g_inv, chr_first_kind, "i r, j k r -> i j k")
        chr_second_kind = parallel_apply(chr_second_kind, sp.simplify, element_wise=True)

    with TimeIt("Christoffel symbols' derivative"):
        chr_first_kind_deriv = np.stack(
            [
                np.vectorize(lambda x: x.diff(p))(chr_first_kind)
                for p in coordinate_variables
            ],
            axis=3
        )
        chr_first_kind_deriv = parallel_apply(
            chr_first_kind_deriv,
            partial(_time_limited_simplify, limit_seconds=10 * 60),
            element_wise=True
        )

    with TimeIt("Riemann curvature tensor of first kind"):
        r_first_kind = (
                einops.rearrange(chr_first_kind_deriv, "j l i k -> i j k l")
                - einops.rearrange(chr_first_kind_deriv, "j k i l -> i j k l")
                + einops.einsum(chr_first_kind, chr_second_kind, "i l r, r j k -> i j k l")
                - einops.einsum(chr_first_kind, chr_second_kind, "i k r, r j l -> i j k l")
        )
        r_first_kind = parallel_apply(
            r_first_kind,
            partial(_time_limited_simplify, limit_seconds=10 * 60),
            element_wise=True
        )

    with TimeIt("Ricci curvature tensor"):
        ricci = einops.einsum(g_inv, r_first_kind, "a b, a i b j -> i j")
        if normalize:
            ricci = 1 / (dim - 1) * ricci

    with TimeIt("scalar curvature"):
        scalar_curvature = einops.einsum(g_inv, ricci, "i j, i j ->")
        if normalize:
            scalar_curvature = 1 / dim * scalar_curvature

    return scalar_curvature, ricci, r_first_kind


def volume_element_formula(u, coordinate_variables: sp.Matrix):
    """Compute a formula for the volume element of a manifold on a patch parametrized by u.

        Args:
            u: A formula with the patch parametrization.
            coordinate_variables: The variables of the parametrization in a sympy array matrix.

        Returns:
            A sympy formula for the volume element.
    """
    with TimeIt("Riemannian metric"):
        u_jacobian = u.jacobian(coordinate_variables)

        g_sp = sp.simplify(u_jacobian.T @ u_jacobian)

    with TimeIt("volume element"):
        volume_element = sp.simplify(sp.sqrt(g_sp.det()))

    return volume_element


def ellipsoid_parametrization(dim: int):
    """Define a spherical coordinates parametrization of an n-ellipsoid.

    Args:
        dim: The dimension of the ellipsoid.

    Returns:
        u: The patch parametrization of the ellipsoid.
        coordinates: A sympy array Matrix with the variables of the parametrization.
    """
    # Define the variables.
    semi_axes = sp.symbols(" ".join([f"a_{i}" for i in range(1, dim + 2)]))
    phis = sp.symbols(" ".join([f"phi_{i}" for i in range(1, dim + 1)]))

    # Define the ellipsoid parametrization.
    coordinates = Matrix(phis)
    u = Matrix([
        a
        * (1 if i == dim else cos(phis[i]))
        * sp.prod([sin(phis[i]) for i in range(i)])
        for i, a in enumerate(semi_axes)
    ])

    return u, coordinates


def hyperboloid_parametrization(dim: int):
    """Define a spherical coordinates parametrization of an n-ellipsoid.

        Args:
            dim: The dimension of the ellipsoid.

        Returns:
            u: The patch parametrization of the ellipsoid.
            coordinates: A sympy array Matrix with the variables of the parametrization.
        """
    # Define the variables.
    semi_axes = sp.symbols(" ".join([f"a_{i}" for i in range(1, dim + 2)]))
    phis = sp.symbols(" ".join([f"phi_{i}" for i in range(1, dim + 1)]))

    # Define the ellipsoid parametrization.
    coordinates = Matrix(phis)
    u = Matrix([
        a
        * (
            1 if i == dim
            else (
                sin(phis[i])
                if i == dim - 1
                else sinh(phis[i])
            ))
        * sp.prod([
            cos(phis[i]) if i == dim - 1 else cosh(phis[i])
            for i in range(i)
        ])
        for i, a in enumerate(semi_axes)
    ])

    return u, coordinates


def ellipsoid_scalar_curvature_formula(dim: int, normalize: bool = False):
    """Compute a formula for the scalar curvature of an n-ellipsoid. In practice this function works up to a
    3-ellipsoid. Above this it becomes too slow and the formula too large. This function is intended to run once
    to generate formulas which will be integrated in the code and used to test scalar curvature approximation algorithms
    on the 2 and 3-ellipsoids.

    Args:
        dim: The dimension of the ellipsoid.
        normalize: If true, it computes a normalized version of the scalar curvature, like in the definition
            in Do Carmo.

    Returns:
        A sympy formula for the scalar curvature of the n-ellipsoid.
    """
    u, coordinates = ellipsoid_parametrization(dim)
    return scalar_curvature_formula(u, coordinates, normalize=normalize)


def hyperboloid_scalar_curvature_formula(dim: int, normalize: bool = False):
    """Compute a formula for the scalar curvature of an n-hyperboloid. In practice this function works up to a
    3-hyperboloid. Above this it becomes too slow and the formula too large. This function is intended to run once
    to generate formulas which will be integrated in the code and used to test scalar curvature approximation algorithms
    on the 2 and 3-hyperboloids.

    Args:
        dim: The dimension of the hyperboloid.
        normalize: If true, it computes a normalized version of the scalar curvature, like in the definition
            in Do Carmo.

    Returns:
        A sympy formula for the scalar curvature of the n-hyperboloid.
    """
    u, coordinates = hyperboloid_parametrization(dim)
    return scalar_curvature_formula(u, coordinates, normalize=normalize)


def ellipsoid_volume_element_formula(dim: int):
    """Compute a formula for the volume element of an n-ellipsoid.

        Args:
            dim: The dimension of the ellipsoid.

        Returns:
            A sympy formula for the scalar curvature of the n-ellipsoid.
        """
    u, coordinates = ellipsoid_parametrization(dim)
    return volume_element_formula(u, coordinates)


def hyperboloid_volume_element_formula(dim: int):
    """Compute a formula for the volume element of an n-hyperboloid_volume_element_formula.

        Args:
            dim: The dimension of the hyperboloid_volume_element_formula.

        Returns:
            A sympy formula for the scalar curvature of the n-hyperboloid_volume_element_formula.
        """
    u, coordinates = hyperboloid_parametrization(dim)
    return volume_element_formula(u, coordinates)


def _prepend_np_to_functions(formula: str, function_names: list[str]) -> str:
    for name in function_names:
        formula = formula.replace(name, f"np.{name}")

    return formula


def formula_to_code(formula, args: list[sp.core.symbol.Symbol], fn_name: str) -> str:
    """Convert a computed formula to a python function to be added to the codebase.

    Args:
        formula: A sympy formula which directly computes a value from some arguments.
        args: A list of arguments for the function.
        fn_name: Name of the function.

    Returns:
        A string with the python code.
    """
    formula_with_np = _prepend_np_to_functions(str(formula), ["sin", "cos", "tan", "sqrt"])

    return (
        f"def {fn_name}({','.join(map(str, args))}):\n"
        + "\treturn "
        + formula_with_np
    )
