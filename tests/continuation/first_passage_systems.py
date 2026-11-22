"""The systems first-passage times are checked against, and their closed forms.

*The one-dimensional double well* ``dx = -V'(x) dt + sigma dW``, ``V = x^4/4 -
x^2/2``, started at the left minimum with a reflecting wall at ``a`` and the target
``[b, inf)``: its mean first-passage time is the double integral

    ``tau(x) = (2 / sigma^2) int_x^b dy int_a^y exp(2 (V(y) - V(z)) / sigma^2) dz``

(Berglund, arXiv 1106.5799, Eq. (3.14), with ``eps = sigma^2 / 2`` and the target
on the right). :func:`exact_double_well_time` evaluates it by adaptive quadrature
(scipy's QUADPACK), which shares nothing with the grid.

*The planar well* ``V = x^4/4 - x^2/2 + (1 + k x^2) y^2 / 2 - h x`` with
``f = -grad V + l`` and ``G = sigma I``. Its two terms are the two prefactors the
closed forms are checked on:

* ``k`` makes the transverse curvature differ between the minima (``1 + k``)
  and the saddle (``1``), so the Eyring-Kramers ratio of determinants is not
  the one-dimensional one;
* ``l = gamma y J grad V`` with ``J`` the quarter turn is orthogonal to
  ``grad V`` everywhere, so ``U = 2 V / sigma^2`` is still the quasipotential,
  but ``div l = gamma dV/dx`` is not zero, so ``exp(-U)`` is no longer
  stationary. On the ``x`` axis ``l`` vanishes, so the instanton is the axis
  segment and Bouchet and Reygner's non-Gibbs integral is
  ``int F dt = int gamma (dV/dx) dt = gamma int dx``: ``+gamma`` from the left
  minimum to the saddle, ``-gamma`` from the right one. A transverse field
  that preserves ``exp(-U)`` would hide this term; this one does not.
* ``l = gamma J grad V`` (``rotation=True``) is the divergence-free case: the
  non-Gibbs integral is zero, and the rotation changes only the unstable rate
  at the saddle, to ``sqrt(1 + gamma^2)`` at ``h = 0``.

``h`` tilts the wells, which is the target field of a residence-time sensor.
"""

import math

import numpy as np
from scipy.integrate import quad

from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.variable import Variable


class _Coordinate(Variable, name="FPState", abbreviation="q"):
    pass


def _states(count: int) -> list[Variable]:
    return [_Coordinate() for _ in range(count)]


def double_well_potential(x: float) -> float:
    """``V = x^4/4 - x^2/2``."""
    return 0.25 * x**4 - 0.5 * x**2


def exact_double_well_time(start: float, target: float, wall: float, variance):
    """The double integral above, by nested adaptive quadrature."""

    def inner(y: float) -> float:
        return quad(
            lambda z: math.exp(
                2.0 * (double_well_potential(y) - double_well_potential(z)) / variance,
            ),
            wall,
            y,
            limit=200,
        )[0]

    outer = quad(inner, start, target, limit=200, points=[0.0])[0]
    return 2.0 * outer / variance


class DoubleWellDrift(DeterministicFunction):
    """``f = x - x^3``."""

    def __init__(self) -> None:
        super().__init__(_states(1), [], _states(1), None)

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x = point[0]
        return [x - x**3]


class ConstantColumn(DeterministicFunction):
    """One constant column of ``G``."""

    def __init__(self, column) -> None:
        count = len(column)
        super().__init__(_states(count), [], _states(count), None)
        self._column = [float(value) for value in column]

    def eval(
        self,
        point: list[float],  # noqa: ARG002 (constant field)
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return list(self._column)


def isotropic_noise(sigma: float, dimension: int) -> NoiseMatrix:
    """``G = sigma I`` as one column per driver."""
    return NoiseMatrix(
        [ConstantColumn(sigma * row) for row in np.eye(dimension)],
    )


class PlanarWell:
    """The planar well above: its drift, for the grid and for the ensemble."""

    def __init__(
        self,
        curvature: float = 0.0,
        rotation: float = 0.0,
        tilt: float = 0.0,
        *,
        divergence_free: bool = False,
    ) -> None:
        self.curvature = curvature
        self.rotation = rotation
        self.tilt = tilt
        self.divergence_free = divergence_free

    def gradient(self, x, y):
        """``grad V``, for scalars or arrays."""
        k = self.curvature
        return x**3 - x + k * x * y * y - self.tilt, (1.0 + k * x * x) * y

    def drift(self, x, y):
        """``f = -grad V + l``."""
        vx, vy = self.gradient(x, y)
        weight = self.rotation if self.divergence_free else self.rotation * y
        return -vx - weight * vy, -vy + weight * vx

    def minima(self) -> tuple[float, float]:
        """The two minima on the ``x`` axis, left then right."""
        roots = np.sort(np.roots([1.0, 0.0, -1.0, -self.tilt]).real)
        return float(roots[0]), float(roots[2])

    def saddle(self) -> float:
        """The saddle on the ``x`` axis."""
        roots = np.sort(np.roots([1.0, 0.0, -1.0, -self.tilt]).real)
        return float(roots[1])

    def potential(self, x: float) -> float:
        """``V`` on the ``x`` axis."""
        return 0.25 * x**4 - 0.5 * x**2 - self.tilt * x

    def field(self) -> "PlanarWellField":
        """The drift as a :class:`DeterministicFunction` for the grid."""
        return PlanarWellField(self)

    def equation(self, sigma: float) -> DifferentialEquation:
        """``dx = f dt + sigma dW`` for the ensemble."""
        time = Variable(name="t", discretization=[])
        variables = [
            Variable(name="x", discretization=[]),
            Variable(name="y", discretization=[]),
        ]
        derivative = StochasticPlanarWell(
            variables,
            [],
            [Variable(name=f"d{v.name}/dt", discretization=[]) for v in variables],
            time,
        )
        derivative.well = self
        derivative.sigma = sigma
        return DifferentialEquation(
            variables=variables,
            time=time,
            parameters=[],
            derivative=derivative,
        )


class PlanarWellField(DeterministicFunction):
    """The planar well's drift, one point at a time."""

    def __init__(self, well: PlanarWell) -> None:
        super().__init__(_states(2), [], _states(2), None)
        self._well = well

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return list(self._well.drift(point[0], point[1]))


class StochasticPlanarWell(StochasticFunction):
    """``dx = f dt + sigma dW`` with batched evaluations for the ensemble."""

    well: PlanarWell
    sigma: float

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return list(self.well.drift(point[0], point[1]))

    def diffusion(
        self,
        point: list[float],  # noqa: ARG002 (constant noise)
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return [self.sigma, self.sigma]

    def eval_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> np.ndarray:
        first, second = self.well.drift(points[:, 0], points[:, 1])
        return np.stack([first, second], axis=1)

    def diffusion_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> np.ndarray:
        return np.full_like(points, self.sigma)
