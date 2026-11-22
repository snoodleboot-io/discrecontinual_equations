"""Systems the covariant Lyapunov vector tests are checked against.

Kept apart from :mod:`tests.continuation.fields` because nothing else uses them:
a linear flow whose covariant vectors are known in closed form, and the two
planar maps the hyperbolicity literature compares - Henon, which has homoclinic
tangencies, and Lozi, which (off a null set of cusps) has none.
"""

from discrecontinual_equations.function.deterministic import DeterministicFunction


class KuptsovParlitzField(DeterministicFunction):
    """``x' = J x`` with ``J = [[1, -2, 0], [0, -1, 0], [0, 2, -3]]``.

    The worked example of Kuptsov and Parlitz (J. Nonlinear Sci. 22, 2012,
    Sec. V A, Eqs. 69-70): exponents ``1, -1, -3`` and covariant vectors
    ``(1, 0, 0)``, ``(1, 1, 1) / sqrt(3)`` and ``(0, 0, 1)``, the eigenvectors of
    ``J``. None of them is orthogonal to the next, so a Gram-Schmidt basis cannot
    be mistaken for them.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y, z = point[0], point[1], point[2]
        return [x - 2.0 * y, -y, 2.0 * y - 3.0 * z]


class HenonMap(DeterministicFunction):
    """``(x, y) -> (1 - 1.4 x^2 + y, 0.3 x)``: the classical Henon map."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        return [1.0 - 1.4 * x * x + y, 0.3 * x]


class LoziMap(DeterministicFunction):
    """``(x, y) -> (1 - 1.4 |x| + y, 0.3 x)``: the piecewise-linear Lozi map."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        return [1.0 - 1.4 * abs(x) + y, 0.3 * x]


class MatrixField(DeterministicFunction):
    """``x' = A x`` for a matrix ``A`` given at construction."""

    def __init__(self, matrix, variables, parameters, results, time=None) -> None:
        super().__init__(variables, parameters, results, time)
        self._matrix = [list(row) for row in matrix]

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return [
            sum(entry * value for entry, value in zip(row, point, strict=True))
            for row in self._matrix
        ]
