"""The systems the quasipotential is checked against, each with its closed form.

*The double well with a transverse direction.* ``V = x^4/4 - x^2/2 + y^2/2`` has
minima at ``(+-1, 0)`` and a saddle at the origin. With ``f = -grad V`` and
``G = I`` the stationary density is ``exp(-2 V)``, so in the library's
normalisation (``dx = f dt + G dW``, ``D = G G^T``, density ``~ exp(-U)`` at
leading order) the quasipotential of the left well is ``U = 2 (V - V_min)`` on
its whole basin ``x < 0``. Every minimum action path there is a steepest-descent
path run backwards, ``y = y0 (x-path)`` with ``x^2 = 1 / (1 + (1/x0^2 - 1)
(y / y0)^2)``, and the one that leaves the well is the unstable manifold of the
saddle, the segment of the ``x`` axis.

*The rotating Ornstein-Uhlenbeck process* of :mod:`tests.solver.rotating_ornstein_
uhlenbeck`, ``dx = A x dt + G dW``: its density is the Gaussian with the Lyapunov
covariance ``S``, ``A S + S A^T + G G^T = 0``, so ``U = (1/2) x^T S^{-1} x``. The
drift is not a gradient - ``A`` carries a rotation - and ``G`` couples the two
components, so this is the non-gradient, anisotropic check.
"""

import numpy as np

from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.variable import Variable
from tests.solver.rotating_ornstein_uhlenbeck import (
    RotatingOrnsteinUhlenbeck,
    stationary_covariance,
)

WELL = (-1.0, 0.0)
OTHER_WELL = (1.0, 0.0)
SADDLE = (0.0, 0.0)


class _State(Variable, name="QState", abbreviation="q"):
    pass


def _states() -> list[Variable]:
    return [_State(), _State()]


def potential(x, y):
    """``V = x^4/4 - x^2/2 + y^2/2``."""
    return 0.25 * x**4 - 0.5 * x**2 + 0.5 * y**2


class DoubleWellField(DeterministicFunction):
    """``f = -grad V = (x - x^3, -y)``."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        return [x - x**3, -y]


class LinearField(DeterministicFunction):
    """``f = A x`` for the rotating Ornstein-Uhlenbeck drift matrix."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return list(RotatingOrnsteinUhlenbeck.drift_matrix @ np.asarray(point))


class MatrixColumn(DeterministicFunction):
    """Column ``index`` of a constant amplitude ``G``."""

    def __init__(self, matrix: np.ndarray, index: int) -> None:
        super().__init__(_states(), [], _states(), None)
        self._column = [float(value) for value in np.asarray(matrix)[:, index]]

    def eval(
        self,
        point: list[float],  # noqa: ARG002 (constant field)
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return list(self._column)


def constant_noise(matrix) -> NoiseMatrix:
    """A constant ``G`` as one column per driver."""
    matrix = np.asarray(matrix, dtype=float)
    return NoiseMatrix([MatrixColumn(matrix, k) for k in range(matrix.shape[1])])


def double_well() -> DoubleWellField:
    return DoubleWellField(_states(), [], _states(), None)


def rotating_drift() -> LinearField:
    return LinearField(_states(), [], _states(), None)


def rotating_noise_matrix() -> np.ndarray:
    return RotatingOrnsteinUhlenbeck.noise_matrix


def rotating_quasipotential(points: np.ndarray) -> np.ndarray:
    """``(1/2) x^T S^{-1} x`` at each row of ``points``."""
    covariance = stationary_covariance(
        RotatingOrnsteinUhlenbeck.drift_matrix,
        RotatingOrnsteinUhlenbeck.noise_matrix,
    )
    inverse = np.linalg.inv(covariance)
    return 0.5 * np.einsum("ki,ij,kj->k", points, inverse, points)
