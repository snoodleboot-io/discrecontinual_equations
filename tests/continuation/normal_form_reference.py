"""An independent cubic normal form, by Poincare-Dulac in eigen-coordinates.

The library computes codim-2 coefficients by Kuznetsov's projection method: the
homological equation is solved for the center manifold and the normalising
transformation together, with bordered solves and Fredholm conditions on the
multilinear forms ``B`` and ``C``. This reference shares none of that. It diagonalises
the whole linear part, writes the complete field as a polynomial in the eigen-
coordinates, removes *every* non-resonant quadratic monomial in *every* component
(center and stable alike - removing the center-only ones from the stable rows is
what flattens the center manifold), and reads the resonant cubic coefficients off
the transformed field on the critical coordinates.

Both procedures leave the resonant part of the quadratic transformation at zero,
so for the same eigenvectors they define the same normal form, and its resonant
coefficients must agree to rounding. Only semisimple linear parts are handled,
which covers zero-Hopf and Hopf-Hopf; Bogdanov-Takens is checked by its embedded
oracle instead.
"""

import numpy as np

from discrecontinual_equations.continuation.multivariate import Multivariate
from discrecontinual_equations.function.function import Function

_ORDER = 3
_RESONANCE = 1.0e-8


def _field(
    function: Function,
    state: np.ndarray,
    basis: np.ndarray,
    arguments: list[Multivariate],
    inverse: np.ndarray,
) -> list[Multivariate]:
    size = state.size
    point = [
        float(state[i]) + sum(complex(basis[i, k]) * arguments[k] for k in range(size))
        for i in range(size)
    ]
    raw = function.eval(point=point, time=0.0)
    return [
        sum(complex(inverse[j, i]) * raw[i] for i in range(size)) for j in range(size)
    ]


def _degree(series: Multivariate, degree: int) -> dict[tuple[int, ...], complex]:
    return {
        key: complex(value)
        for key, value in series.coefficients.items()
        if sum(key) == degree
    }


def _partial(series: Multivariate, axis: int) -> Multivariate:
    result: dict[tuple[int, ...], complex] = {}
    for key, value in series.coefficients.items():
        if key[axis] == 0:
            continue
        lowered = tuple(k - (1 if i == axis else 0) for i, k in enumerate(key))
        result[lowered] = result.get(lowered, 0.0) + key[axis] * value
    return Multivariate(result, series.order, series.dimension)


def cubic_normal_form(
    function: Function,
    state: np.ndarray,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
) -> list[Multivariate]:
    """The field in normal-form coordinates ``zeta``, exact through degree three.

    ``basis`` holds the eigenvectors as columns, in the order of ``eigenvalues``.
    """
    size = state.size
    inverse = np.linalg.inv(basis)
    variables = [Multivariate.variable(k, 0.0, _ORDER, size) for k in range(size)]
    original = _field(function, state, basis, variables, inverse)
    transformation = []
    for j in range(size):
        removed: dict[tuple[int, ...], complex] = {}
        for key, value in _degree(original[j], 2).items():
            divisor = (
                sum(k * eigenvalues[i] for i, k in enumerate(key)) - eigenvalues[j]
            )
            if abs(divisor) > _RESONANCE:
                removed[key] = value / divisor
        transformation.append(Multivariate(removed, _ORDER, size))
    shifted = [variables[k] + transformation[k] for k in range(size)]
    composed = _field(function, state, basis, shifted, inverse)
    partials = [
        [_partial(transformation[j], k) for k in range(size)] for j in range(size)
    ]
    velocity = composed
    for _ in range(2):
        velocity = [
            composed[j] - sum(partials[j][k] * velocity[k] for k in range(size))
            for j in range(size)
        ]
    return velocity


def coefficient(series: Multivariate, key: tuple[int, ...]) -> complex:
    """One Taylor coefficient of a series (zero if absent)."""
    return complex(series.coefficients.get(key, 0.0))


def _derivative(series: Multivariate, axis: int) -> Multivariate:
    return _partial(series, axis)


def fold_hopf_e(
    coefficients: dict[str, complex],
    omega: float,
    eta: float = 0.0,
) -> dict[str, complex]:
    """Normalise the cubic fold-Hopf form by explicit transformation; return terms.

    Starting from Kuznetsov's (3.4) with the given ``G`` coefficients, apply the
    time rescaling ``dt -> (1 + kappa v) dt`` and the changes
    ``u = v + gamma v^2 + eta |zeta|^2``, ``z = zeta (1 + delta v)``, with
    ``kappa, gamma, delta`` chosen to remove ``v^3``, ``v |zeta|^2`` and
    ``Re zeta |zeta|^2``, and compute the transformed field as a polynomial. The
    real ``v^2 zeta`` coefficient left over is ``e(0)``; it must not depend on the
    free ``eta``. This is a derivation by computation, independent of the closed
    form for ``e(0)`` in the library.
    """
    g200 = coefficients["g200"].real
    g011 = coefficients["g011"].real
    b, c = 0.5 * g200, g011
    cube, cross = coefficients["g300"].real / 6.0, coefficients["g111"].real
    linear = coefficients["g110"]
    square, radial = 0.5 * coefficients["g210"], 0.5 * coefficients["g021"]
    kappa = -cube / b
    shifted = linear + 1j * omega * kappa
    gamma = (2.0 * b * eta + cross + kappa * c + 2.0 * radial.real) / (2.0 * c)
    delta = (radial + eta * shifted) / c
    v, z, zb = (Multivariate.variable(axis, 0.0, _ORDER, 3) for axis in range(3))
    u = v + gamma * v * v + eta * z * zb
    w = z + delta * v * z
    wb = zb + np.conj(delta) * v * zb
    r = w * wb
    rows = [
        b * u * u + c * r + cube * u * u * u + cross * u * r,
        1j * omega * w + linear * u * w + square * u * u * w + radial * w * r,
        -1j * omega * wb
        + np.conj(linear) * u * wb
        + np.conj(square) * u * u * wb
        + np.conj(radial) * wb * r,
    ]
    rows = [row * (1.0 + kappa * v) for row in rows]
    maps, bases = [u, w, wb], [v, z, zb]
    partials = [
        [_derivative(maps[i] - bases[i], a) for a in range(3)] for i in range(3)
    ]
    velocity = rows
    for _ in range(3):
        velocity = [
            rows[i] - sum(partials[i][a] * velocity[a] for a in range(3))
            for i in range(3)
        ]
    return {
        "v3": coefficient(velocity[0], (3, 0, 0)),
        "v|z|^2": coefficient(velocity[0], (1, 1, 1)),
        "z|z|^2": coefficient(velocity[1], (0, 2, 1)),
        "v^2 z": coefficient(velocity[1], (2, 1, 0)),
    }
