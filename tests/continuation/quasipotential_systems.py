"""The systems the quasipotential is checked against, each with its closed form.

Each field also offers ``eval_batch``, the same arithmetic on columns of
points, which the march uses to evaluate the field at every half-lattice point
in one call (and the tests check it gives the same field as ``eval``).

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

    def eval_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> np.ndarray:
        return np.column_stack(self.eval(list(np.asarray(points).T)))


class LinearField(DeterministicFunction):
    """``f = A x`` for the rotating Ornstein-Uhlenbeck drift matrix."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return list(RotatingOrnsteinUhlenbeck.drift_matrix @ np.asarray(point))

    def eval_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> np.ndarray:
        return np.asarray(points) @ RotatingOrnsteinUhlenbeck.drift_matrix.T


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

    def eval_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> np.ndarray:
        return np.tile(self._column, (np.asarray(points).shape[0], 1))


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

    def eval_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> np.ndarray:
        return np.column_stack(self.eval(list(np.asarray(points).T)))


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

    def eval_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> np.ndarray:
        return np.column_stack(self.eval(list(np.asarray(points).T)))


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

    def eval_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> np.ndarray:
        return np.column_stack(self.eval(list(np.asarray(points).T)))


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

    def eval_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> np.ndarray:
        return np.column_stack(self.eval(list(np.asarray(points).T)))


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

    def eval_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> np.ndarray:
        return np.asarray(points, dtype=float) @ NON_NORMAL_DRIFT.T


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


# --- A rotating well with every cubic term (verification of DEQ-57) ---------
#
# f = -(1/2) grad U + c R grad U, D = I, R the rotation by a right angle: grad U
# . f = -(1/2) |grad U|^2 and (1/2) |grad U|^2 cancel it, so U solves the
# Hamilton-Jacobi equation exactly for any U and c. With U = x^2 + 2 y^2 + U3
# and every coefficient of U3 nonzero, the cubic term at the origin is U3
# itself, and with c = 4 the flow B = J + 2 D M along which paths leave the
# well is far from symmetric.
CUBIC_TERMS = (0.3, -0.2, 0.15, 0.1)
CUBIC_ROTATION = 4.0


class RotatingCubicField(DeterministicFunction):
    """``f = -(1/2) grad U + c R grad U`` for ``U = x^2 + 2 y^2 + U3``."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = float(point[0]), float(point[1])
        a, b, c, d = CUBIC_TERMS
        slope_x = 2.0 * x + 3.0 * a * x * x + 2.0 * b * x * y + c * y * y
        slope_y = 4.0 * y + b * x * x + 2.0 * c * x * y + 3.0 * d * y * y
        return [
            -0.5 * slope_x - CUBIC_ROTATION * slope_y,
            -0.5 * slope_y + CUBIC_ROTATION * slope_x,
        ]


# --- State-dependent noise (DEQ-59) --------------------------------------------
#
# f = -D(x) grad V + c R grad V, with R the rotation by a right angle and D(x) =
# G(x) G(x)^T varying with the state. U = 2 (V - V_min) solves the
# Hamilton-Jacobi equation for any D(x) and any c: with grad U = 2 grad V,
# grad U . f = -2 grad V^T D grad V + 2 c grad V . R grad V and the second term
# vanishes, while (1/2) grad U^T D grad U = 2 grad V^T D grad V. The ticket's
# candidate f = -D grad V + (1/2) div D is the drift whose stationary density
# is exactly exp(-2 V) at unit noise; the (1/2) div D is the order-eps
# correction that the quasipotential does not see, and with it in the drift
# grad U . (1/2) div D is left over and U = 2 V no longer solves the equation.
# G is lower triangular with a positive diagonal, so D is never singular; it
# varies by a factor of about two across the box and is anisotropic
# everywhere. The rotation c = 1 makes the drift non-gradient.
MULTIPLICATIVE_ROTATION = 1.0


def multiplicative_amplitude(x, y):
    """``G = [[1 + y^2 / 5, 0], [3 x / 10, 4 / 5 + x^2 / 10]]``."""
    return np.array([[1.0 + 0.2 * y * y, 0.0], [0.3 * x, 0.8 + 0.1 * x * x]])


class MultiplicativeField(DeterministicFunction):
    """``f = -D(x) grad V + c R grad V`` for the double well's ``V``."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = float(point[0]), float(point[1])
        amplitude = multiplicative_amplitude(x, y)
        slope = np.array([x**3 - x, y])
        turned = np.array([-slope[1], slope[0]])
        return list(
            -(amplitude @ amplitude.T @ slope) + MULTIPLICATIVE_ROTATION * turned,
        )


class MultiplicativeColumn(DeterministicFunction):
    """Column ``index`` of ``G(x)`` above."""

    def __init__(self, index: int) -> None:
        super().__init__(_states(), [], _states(), None)
        self._index = index

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        amplitude = multiplicative_amplitude(float(point[0]), float(point[1]))
        return list(amplitude[:, self._index])


def multiplicative_noise() -> NoiseMatrix:
    return NoiseMatrix([MultiplicativeColumn(index) for index in range(2)])


# --- A fast elliptic cycle (verification of DEQ-57, DEQ-117) ----------------
#
# U = (1/2) (q - 1)^2 with q = x^2 + 2 y^2, and f = -(1/2) D grad U + w R grad q:
# R grad q is perpendicular to grad U, so grad U . f = -(1/2) |grad U|_D^2 and
# U solves the Hamilton-Jacobi equation exactly. The ellipse q = 1 is a stable
# cycle, run round three times faster than it is approached, and the direction
# of R grad q turns across it - which the march's start beside a cycle assumes
# it does not.
ELLIPSE_ROTATION = 3.0
ELLIPSE_AMPLITUDE = np.array([[0.8, 0.0], [0.3, 0.6]])


def ellipse_quasipotential(points: np.ndarray) -> np.ndarray:
    return 0.5 * (points[:, 0] ** 2 + 2.0 * points[:, 1] ** 2 - 1.0) ** 2


def ellipse_states(count: int = 800) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack([np.cos(angle), np.sqrt(0.5) * np.sin(angle)])


class EllipticCycleField(DeterministicFunction):
    """``f = -(1/2) D grad U + w R grad q`` for the ellipse ``q = x^2 + 2 y^2``."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        diffusion = ELLIPSE_AMPLITUDE @ ELLIPSE_AMPLITUDE.T
        excess = x * x + 2.0 * y * y - 1.0
        slope_x, slope_y = 2.0 * x, 4.0 * y
        rise_x, rise_y = excess * slope_x, excess * slope_y
        return [
            -0.5 * (diffusion[0, 0] * rise_x + diffusion[0, 1] * rise_y)
            - ELLIPSE_ROTATION * slope_y,
            -0.5 * (diffusion[1, 0] * rise_x + diffusion[1, 1] * rise_y)
            + ELLIPSE_ROTATION * slope_x,
        ]

    def eval_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> np.ndarray:
        return np.column_stack(self.eval(list(np.asarray(points).T)))
