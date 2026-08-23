"""Planar Lyapunov focal values by the classical Lyapunov-function recursion.

At a planar Hopf point the field is linearly transformed so its linear part is the
rotation ``(s, t) -> (-t, s)``. A Lyapunov function ``V = (s^2 + t^2)/2 + V_3 +
V_4 + ...`` is built order by order so that its orbital derivative reduces to a sum
of radial terms ``eta_4 (s^2+t^2)^2 + eta_6 (s^2+t^2)^3 + ...``. The residual
coefficients ``eta_4`` and ``eta_6`` are the first and second focal values; their
signs are the first and second Lyapunov quantities (criticality of the Hopf and,
where the first vanishes, of the Bautin point).

The field's full local Taylor expansion is obtained exactly from the
:class:`~.bivariate.Bivariate` automatic-differentiation series, so the only
approximation is the truncation degree, taken high enough for both quantities.
"""

import numpy as np

from discrecontinual_equations.continuation.bivariate import Bivariate
from discrecontinual_equations.function.function import Function

_ORDER = 6
_FIRST_FOCAL_DEGREE = 4
_SECOND_FOCAL_DEGREE = 6


class PlanarLyapunov:
    """First and second Lyapunov quantities at a planar Hopf point."""

    __slots__ = ["_equilibrium", "_function", "_time", "_transform"]

    def __init__(
        self,
        function: Function,
        equilibrium: np.ndarray,
        jacobian: np.ndarray,
        time: float,
    ) -> None:
        self._function = function
        self._equilibrium = equilibrium
        self._time = time
        self._transform = _rotation_transform(jacobian)

    def quantities(self) -> tuple[float, float]:
        """Return the first and second focal values ``(eta_4, eta_6)``."""
        transform, inverse, frequency = self._transform
        first = Bivariate({(1, 0): 1.0}, _ORDER)
        second = Bivariate({(0, 1): 1.0}, _ORDER)
        point = [
            self._equilibrium[i] + transform[i, 0] * first + transform[i, 1] * second
            for i in range(2)
        ]
        raw = self._function.eval(point=point, time=self._time)
        scale = 1.0 / frequency
        field = [
            (inverse[0, 0] * raw[0] + inverse[0, 1] * raw[1]) * scale,
            (inverse[1, 0] * raw[0] + inverse[1, 1] * raw[1]) * scale,
        ]
        return recurse(field)


def _rotation_transform(
    jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    values, vectors = np.linalg.eig(jacobian)
    index = int(np.argmax(values.imag))
    frequency = float(values[index].imag)
    eigenvector = vectors[:, index]
    transform = np.column_stack([eigenvector.real, -eigenvector.imag])
    return transform, np.linalg.inv(transform), frequency


def recurse(field: list[Bivariate]) -> tuple[float, float]:
    lyapunov = Bivariate({(2, 0): 0.5, (0, 2): 0.5}, _ORDER)
    focal: dict[int, float] = {}
    for degree in range(3, _ORDER + 1):
        orbital = lyapunov.derivative(0) * field[0] + lyapunov.derivative(1) * field[1]
        known = orbital.homogeneous(degree)
        correction, coefficient = _solve_order(degree, known)
        lyapunov = lyapunov + correction
        if degree % 2 == 0:
            focal[degree] = coefficient
    return focal[_FIRST_FOCAL_DEGREE], focal[_SECOND_FOCAL_DEGREE]


def _solve_order(
    degree: int,
    known: dict[tuple[int, int], float],
) -> tuple[Bivariate, float]:
    basis = [(a, degree - a) for a in range(degree + 1)]
    index = {monomial: position for position, monomial in enumerate(basis)}
    operator = np.zeros((degree + 1, degree + 1))
    for column, (a, b) in enumerate(basis):
        if b > 0:
            operator[index[(a + 1, b - 1)], column] += b
        if a > 0:
            operator[index[(a - 1, b + 1)], column] -= a
    right = np.array([-known.get(monomial, 0.0) for monomial in basis])
    if degree % 2 == 1:
        solution = np.linalg.solve(operator, right)
        return _to_series(basis, solution), 0.0
    radial = _radial_vector(basis, degree)
    augmented = np.column_stack([operator, -radial])
    result, *_ = np.linalg.lstsq(augmented, right, rcond=None)
    return _to_series(basis, result[:-1]), float(result[-1])


def _radial_vector(basis: list[tuple[int, int]], degree: int) -> np.ndarray:
    half = degree // 2
    expansion = np.zeros(len(basis))
    for k in range(half + 1):
        coefficient = _binomial(half, k)
        expansion[basis.index((2 * k, degree - 2 * k))] = coefficient
    return expansion


def _binomial(top: int, bottom: int) -> float:
    result = 1.0
    for step in range(bottom):
        result = result * (top - step) / (step + 1)
    return result


def _to_series(basis: list[tuple[int, int]], values: np.ndarray) -> Bivariate:
    return Bivariate(
        {monomial: float(value) for monomial, value in zip(basis, values, strict=True)},
        _ORDER,
    )
