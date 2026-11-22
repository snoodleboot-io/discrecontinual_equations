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


# --- Independent references (verification of DEQ-31) -------------------------
#
# None of these reuses the library's own formulas: the Lyapunov covariance below
# is a Kronecker-product solve rather than scipy's Bartels-Stewart, and the
# Maier-Stein barrier was computed outside the library by minimising the
# discretised geometric action with L-BFGS (see ``MAIER_STEIN_BARRIER``).

# Dahiya and Cameron, J. Sci. Comput. 2018 (arXiv 1706.07509), SDE (21) with
# a = 10: dx = (-2x - a y) dt + dw1, dy = (2a x - y) dt + dw2, U = 2x^2 + y^2.
PAPER_ROTATION = 10.0


class PaperLinearField(DeterministicFunction):
    """``b = (-2x - a y, 2a x - y)``, their Eq. (21); ``U = 2 x^2 + y^2`` (22)."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        return [-2.0 * x - PAPER_ROTATION * y, 2.0 * PAPER_ROTATION * x - y]


class PaperCycleField(DeterministicFunction):
    """Their Eq. (24): ``b = (y + x (1 - r^2), -x + y (1 - r^2))``, ``D = I``.

    The unit circle is the cycle and ``U = (1/2) (r^2 - 1)^2`` (their Eq. (25),
    with the correction of its footnote).
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        shrink = 1.0 - x * x - y * y
        return [y + x * shrink, -x + y * shrink]


# f = -D grad V with an anisotropic constant D: the stationary density is
# exp(-2 V) whatever D is, so U = 2 (V - V_min) on the left well's basin. The
# basin is no longer x < 0 - the separatrix is the stable manifold of the saddle,
# tilted by D - but it contains every point with x < -1/2 of the box below
# (integrating the flow from a 301 x 301 grid, the right basin reaches x = -0.46).
ANISOTROPIC_AMPLITUDE = np.array([[1.0, 0.0], [0.4, 0.6]])


class AnisotropicGradientField(DeterministicFunction):
    """``f = -D grad V`` for the double well and ``D = G G^T`` above."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        diffusion = ANISOTROPIC_AMPLITUDE @ ANISOTROPIC_AMPLITUDE.T
        return list(-(diffusion @ np.array([x**3 - x, y])))


# The Maier-Stein model b = (x - x^3 - beta x y^2, -(1 + x^2) y), D = I, at
# beta = 10 > 4: the minimum action path from (-1, 0) to the saddle (0, 0) leaves
# the x axis (max |y| = 0.313) and its action is below the on-axis 1/2.
MAIER_STEIN_BETA = 10.0
# Minimising sum |b_m| |dx| - b_m . dx (midpoint rule) over chains of 200, 400
# and 800 nodes with L-BFGS, re-spaced to equal arc length between rounds:
# 0.340052, 0.340044, 0.340042. The on-axis start converges to 0.500000.
MAIER_STEIN_BARRIER = 0.340042


class MaierSteinField(DeterministicFunction):
    """``b = (x - x^3 - beta x y^2, -(1 + x^2) y)``."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        return [x - x**3 - MAIER_STEIN_BETA * x * y * y, -(1.0 + x * x) * y]


# A non-normal drift (A A^T != A^T A: real eigenvalues -1 and -2 but strongly
# sheared eigenvectors) with a coupled G.
NON_NORMAL_DRIFT = np.array([[-1.0, 4.0], [0.0, -2.0]])
NON_NORMAL_AMPLITUDE = np.array([[1.0, 0.0], [0.7, 0.5]])


class NonNormalField(DeterministicFunction):
    """``f = A x`` for the non-normal ``A`` above."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return list(NON_NORMAL_DRIFT @ np.asarray(point, dtype=float))


def kronecker_covariance(drift: np.ndarray, diffusion: np.ndarray) -> np.ndarray:
    """``S`` with ``A S + S A^T + D = 0``, by ``(I (x) A + A (x) I) vec S = -vec D``."""
    identity = np.eye(drift.shape[0])
    operator = np.kron(identity, drift) + np.kron(drift, identity)
    vector = np.linalg.solve(operator, -diffusion.reshape(-1, order="F"))
    return vector.reshape(drift.shape, order="F")


def system(field_class) -> DeterministicFunction:
    return field_class(_states(), [], _states(), None)


# U from the left well at grid points of the Maier-Stein box, by the same
# independent L-BFGS minimisation as MAIER_STEIN_BARRIER: chains of 100, 200
# and 400 nodes from (-1, 0), the last two extrapolated at second order (the
# 200-to-400 change is 1.4e-5 at (-1.3, 0.4) and at most 5e-6 elsewhere).
MAIER_STEIN_VALUES = {
    (-0.4, 0.3): 0.2587345,
    (-0.7, -0.2): 0.1280867,
    (-1.3, 0.4): 1.0204491,
    (-0.2, 0.5): 0.4192326,
}
