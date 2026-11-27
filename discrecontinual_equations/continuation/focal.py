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

The construction is the classical one for a weak focus ``x' = -y + P``,
``y' = x + Q`` (``P, Q`` without constant or linear terms): a local Lyapunov
function ``V`` whose orbital derivative is ``V' = eta_2 r^2 + eta_4 r^4 + ...``,
``r^2 = x^2 + y^2``, with ``eta_(2k)`` the *k*-th focal value; the focus is stable
(unstable) when the first non-zero focal value is negative (positive), and a
center iff all vanish (Euzébio, Llibre and Tonon (2020), Section 2.1, attributing
the construction to Blows and Lloyd). With the linear part exactly a rotation,
``eta_2 = 0``. At each degree ``k`` the homogeneous part ``V_k`` solves a linear
system with the operator ``V_k -> s V_t - t V_s`` (partial derivatives), invertible on
odd degrees; on even degrees its range misses ``(s^2 + t^2)^(k/2)``, and the
coefficient of that missing direction is the focal value (solved here by least
squares on the augmented system).

On the normal form ``s' = -t + a s r^2 + b s r^4``, ``t' = s + a t r^2 + b t r^4``
the recursion returns ``(eta_4, eta_6) = (a, b)`` exactly. The field is scaled by
``1 / omega`` so the rotation has unit frequency; the sign of ``eta_4`` is the sign
of the first Lyapunov coefficient ``l_1`` (Kuznetsov (1998), eq. (3.20)), whose
value depends on normalisation while its sign does not (Kuznetsov's remark after
(3.20)). Only the signs are meant to be compared across methods.

Erratum carried: the source prints the expansion as ``eta_(2k) (x^2 + y^2)^(2k)``;
its own first terms ``eta_2 (x^2 + y^2) + eta_4 (x^2 + y^2)^2`` show the exponent is
``k``, which is what is used here.

Limits: planar systems only (reduce first, as :mod:`~.center_manifold` does); the
field must be polynomial in the :class:`~.bivariate.Bivariate` arithmetic (no
division); ``eta_6`` is meaningful only where ``eta_4 = 0`` (the source: ``eta_(2k)``
is relevant only when the earlier ones vanish), i.e. at a Bautin point; and the
eigen-decomposition assumes
the Jacobian has a complex pair ``+-i omega`` with ``omega > 0``.

References:
- Euzébio, R. D., Llibre, J. and Tonon, D. J. "Lower bounds for the number of
  limit cycles in a generalized Rayleigh-Liénard oscillator", arXiv:2012.13952
  (2020), Section 2.1 (focal values from a Lyapunov function). Blows, T. R. and
  Lloyd, N. G., Proc. Roy. Soc. Edinburgh A 98, 215-239, 1984,
  doi:10.1017/S030821050001341X, is their source; it was not read here.
- Kuznetsov, Yu. A. "Elements of Applied Bifurcation Theory", 2nd ed., Applied
  Mathematical Sciences 112, Springer, 1998, doi:10.1007/b98848, Section 3.5,
  eq. (3.20).
"""

import numpy as np

from discrecontinual_equations.continuation.bivariate import Bivariate
from discrecontinual_equations.function.function import Function

_ORDER = 6
_FIRST_FOCAL_DEGREE = 4
_SECOND_FOCAL_DEGREE = 6


class PlanarLyapunov:
    """First and second Lyapunov quantities at a planar Hopf point.

    Args:
        function: The planar vector field, evaluable on Bivariate series.
        equilibrium: The Hopf equilibrium.
        jacobian: The Jacobian there, with eigenvalues ``+-i omega``.
        time: Time passed to the field.

    Raises:
        numpy.linalg.LinAlgError: If the real and imaginary parts of the critical
            eigenvector are dependent (no complex pair).

    References:
    - Euzébio, Llibre and Tonon (2020), Section 2.1.
    """

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
        """Return the first and second focal values ``(eta_4, eta_6)``.

        Returns:
            ``(eta_4, eta_6)`` of the field transformed to a unit-frequency
            rotation.
        """
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
    """Focal values of a planar field whose linear part is ``(s, t) -> (-t, s)``.

    Args:
        field: The two components as Bivariate series of order at least six.

    Returns:
        ``(eta_4, eta_6)``, the coefficients of ``(s^2 + t^2)^2`` and
        ``(s^2 + t^2)^3`` in the orbital derivative of the constructed ``V``.

    References:
    - Euzébio, Llibre and Tonon (2020), Section 2.1.
    """
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
