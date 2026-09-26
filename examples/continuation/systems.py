"""The systems the webplot atlas draws, and the two calls that drive them.

Every page in the atlas is one of these fields continued in one parameter. They
lived among the sixty-odd page builders that use them, which made the file hard
to read in either direction: a reader looking for a system had to know which
page mentioned it, and a reader following a page had to scroll past thirty
unrelated fields to reach it.

Each class is a right-hand side and nothing more - normal forms for the
codimension-one and -two bifurcations, a few named systems (van der Pol, a jerk
system, a SNIC circle, the tilted homoclinic), and the drift/diffusion pairs the
stochastic pages need. :func:`equation` wraps one in a
:class:`DifferentialEquation` and :func:`solve` runs a continuation on it.
"""

import math

import numpy as np

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

# Below this radius the SNIC field would divide by zero; the circle is the
# invariant set, so a point at the origin is treated as sitting on it.
_RADIUS_FLOOR = 1.0e-9


class First(Parameter, name="First", abbreviation="p1"):
    pass


class Second(Parameter, name="Second", abbreviation="p2"):
    pass


class Third(Parameter, name="Third", abbreviation="p3"):
    pass


class State(Variable, name="State", abbreviation="s"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class SaddleNode(DeterministicFunction):
    """x' = mu - x^2."""

    def eval(self, point, time=None):  # noqa: ARG002
        return [self.parameters[0].value - point[0] * point[0]]


class Transcritical(DeterministicFunction):
    """x' = mu x - x^2."""

    def eval(self, point, time=None):  # noqa: ARG002
        x = point[0]
        return [self.parameters[0].value * x - x * x]


class Pitchfork(DeterministicFunction):
    """x' = mu x - x^3, a supercritical pitchfork at the origin."""

    def eval(self, point, time=None):  # noqa: ARG002
        x = point[0]
        return [self.parameters[0].value * x - x * x * x]


class Hysteresis(DeterministicFunction):
    """x' = mu + 3x - x^3; an S-shaped branch with two folds."""

    def eval(self, point, time=None):  # noqa: ARG002
        x = point[0]
        return [self.parameters[0].value + 3.0 * x - x * x * x]


class StochasticDrift(DeterministicFunction):
    """Linear drift alpha x for the multiplicative-noise scalar system."""

    def eval(self, point, time=None):  # noqa: ARG002
        return [self.parameters[0].value * point[0]]


class StochasticDiffusion(DeterministicFunction):
    """Multiplicative diffusion sigma x."""

    def eval(self, point, time=None):  # noqa: ARG002
        return [self.parameters[0].value * point[0]]


def _stochastic_drift(alpha):
    return StochasticDrift(
        variables=[State()],
        parameters=[First(value=alpha)],
        results=[State()],
        time=None,
    )


def _stochastic_diffusion(sigma):
    return StochasticDiffusion(
        variables=[State()],
        parameters=[First(value=sigma)],
        results=[State()],
        time=None,
    )


def _homoclinic_tilt():
    yaw = np.array(
        [
            [math.cos(0.6), -math.sin(0.6), 0.0],
            [math.sin(0.6), math.cos(0.6), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    roll = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(0.5), -math.sin(0.5)],
            [0.0, math.sin(0.5), math.cos(0.5)],
        ],
    )
    return yaw @ roll


_HOMOCLINIC_TILT = _homoclinic_tilt()


class TiltedHomoclinic(DeterministicFunction):
    """u'=v, v'=u-u^2, w'=-2w in a rotated frame; saddle eigenvalues {1, -1, -2}."""

    def eval(self, point, time=None):  # noqa: ARG002
        u, v, w = _HOMOCLINIC_TILT.T @ np.array(point)
        base = np.array([v, u - u * u, -2.0 * w])
        return list(_HOMOCLINIC_TILT @ base)


class Jerk(DeterministicFunction):
    """Saddle-focus jerk system x'=y, y'=z, z'=-a z - y + x - x^2."""

    def eval(self, point, time=None):  # noqa: ARG002
        a = self.parameters[0].value
        x, y, z = point[0], point[1], point[2]
        return [y, z, -a * z - y + x - x * x]


class GridSystem(DeterministicFunction):
    """Decoupled quartic gradient x - x^3, y - y^3; nine equilibria."""

    def eval(self, point, time=None):  # noqa: ARG002
        x, y = point[0], point[1]
        return [x - x**3, y - y**3]


class MelnikovSystem(DeterministicFunction):
    """Two-parameter perturbed well x'=y, y'=x-x^2+mu y+nu x y."""

    def eval(self, point, time=None):  # noqa: ARG002
        mu = self.parameters[0].value
        nu = self.parameters[1].value
        x, y = point[0], point[1]
        return [y, x - x * x + mu * y + nu * x * y]


class DampedWell(DeterministicFunction):
    """x' = y, y' = x - x^2 + mu y; homoclinic to the origin at mu = 0."""

    def eval(self, point, time=None):  # noqa: ARG002
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        return [y, x - x * x + mu * y]


class DoubleWell(DeterministicFunction):
    """Double-well x' = y, y' = -x + x^3; saddles at (-1, 0) and (+1, 0)."""

    def eval(self, point, time=None):  # noqa: ARG002
        x, y = point[0], point[1]
        return [y, -x + x**3]


class SpiralSystem(DeterministicFunction):
    """Linear spiral; both Lyapunov exponents equal mu (D-bifurcation at mu = 0)."""

    def eval(self, point, time=None):  # noqa: ARG002
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        return [mu * x - y, x + mu * y]


class SnicSystem(DeterministicFunction):
    """Attracting unit circle with Adler flow; SNIC at mu = 1 (period -> infinity)."""

    def eval(self, point, time=None):  # noqa: ARG002
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        radius = (x * x + y * y) ** 0.5
        if radius < _RADIUS_FLOOR:
            return [0.0, 0.0]
        radial = 1.0 - radius * radius
        angular = mu - y / radius
        return [radial * x - y * angular, radial * y + x * angular]


class FoldCycles(DeterministicFunction):
    """r' = mu r + r^3 - r^5 (Cartesian); two cycles meet at mu = -1/4."""

    def eval(self, point, time=None):  # noqa: ARG002
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        squared = x * x + y * y
        growth = mu + squared - squared * squared
        return [growth * x - y, x + growth * y]


class Hopf(DeterministicFunction):
    """Planar Hopf normal form; a Hopf bifurcation at mu = 0."""

    def eval(self, point, time=None):  # noqa: ARG002
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        radius = x * x + y * y
        return [mu * x - y - x * radius, x + mu * y - y * radius]


class BogdanovTakens(DeterministicFunction):
    """x' = y, y' = b1 + b2 y + x^2 - x y."""

    def eval(self, point, time=None):  # noqa: ARG002
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        x, y = point[0], point[1]
        return [y, b1 + b2 * y + x * x - x * y]


class ZeroHopf(DeterministicFunction):
    """A fold in x coupled to a Hopf in (y, z); zero-Hopf at mu1 = 0."""

    frequency = 1.0

    def eval(self, point, time=None):  # noqa: ARG002
        mu1 = self.parameters[0].value
        mu2 = self.parameters[1].value
        x, y, z = point[0], point[1], point[2]
        radius = y * y + z * z
        return [
            mu1 - x * x,
            mu2 * y - self.frequency * z - y * radius,
            self.frequency * y + mu2 * z - z * radius,
        ]


class HopfHopf(DeterministicFunction):
    """Two decoupled Hopf blocks; a Hopf-Hopf point at mu2 = 0."""

    first_frequency = 1.0
    second_frequency = 2.0

    def eval(self, point, time=None):  # noqa: ARG002
        mu1 = self.parameters[0].value
        mu2 = self.parameters[1].value
        x, y, z, w = point[0], point[1], point[2], point[3]
        first = x * x + y * y
        second = z * z + w * w
        return [
            mu1 * x - self.first_frequency * y - x * first,
            self.first_frequency * x + mu1 * y - y * first,
            mu2 * z - self.second_frequency * w - z * second,
            self.second_frequency * z + mu2 * w - w * second,
        ]


class CuspForm(DeterministicFunction):
    """x' = b1 + b2 x - x^3."""

    def eval(self, point, time=None):  # noqa: ARG002
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        x = point[0]
        return [b1 + b2 * x - x * x * x]


class Bautin(DeterministicFunction):
    """Cubic Bautin form; generalized Hopf at b2 = 0 on the Hopf curve b1 = 0."""

    frequency = 1.0

    def eval(self, point, time=None):  # noqa: ARG002
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        x, y = point[0], point[1]
        radius = x * x + y * y
        return [
            b1 * x - self.frequency * y + b2 * x * radius,
            self.frequency * x + b1 * y + b2 * y * radius,
        ]


class Swallowtail(DeterministicFunction):
    """x' = b1 + b2 x + b3 x^2 - x^4."""

    def eval(self, point, time=None):  # noqa: ARG002
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        b3 = self.parameters[2].value
        x = point[0]
        return [b1 + b2 * x + b3 * x * x - x**4]


class ExtendedBautin(DeterministicFunction):
    """z' = (b1 + i) z + b2 z|z|^2 + b3 z|z|^4 - z|z|^6."""

    def eval(self, point, time=None):  # noqa: ARG002
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        b3 = self.parameters[2].value
        x, y = point[0], point[1]
        radius = x * x + y * y
        coefficient = b1 + b2 * radius + b3 * radius * radius - radius**3
        return [coefficient * x - y, x + coefficient * y]


def _equation(
    function_type: type[DeterministicFunction],
    count: int,
    parameters: list[Parameter],
) -> DifferentialEquation:
    derivative = function_type(
        variables=[State() for _ in range(count)],
        parameters=parameters,
        results=[State() for _ in range(count)],
        time=None,
    )
    return DifferentialEquation(
        variables=[State() for _ in range(count)],
        time=Time(),
        parameters=parameters,
        derivative=derivative,
    )


def _solve(equation: DifferentialEquation, config: ContinuationConfig, seed) -> Branch:
    return ContinuerBuilder.build(config, equation).solve(equation, seed)


class _LorenzDrift(DeterministicFunction):
    """Classic Lorenz system (sigma=10, rho=28, beta=8/3)."""

    def eval(self, point, time=None):  # noqa: ARG002
        x, y, z = point[0], point[1], point[2]
        return [10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z]


class _LorenzNoise(DeterministicFunction):
    """Additive scalar noise: a constant three-vector applied to every component."""

    def eval(self, point, time=None):  # noqa: ARG002
        sigma = self.parameters[0].value
        return [sigma, sigma, sigma]


class VanDerPol(DeterministicFunction):
    """Van der Pol relaxation oscillator x' = y, y' = mu (1 - x^2) y - x."""

    def eval(self, point, time=None):  # noqa: ARG002
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        return [y, mu * (1.0 - x * x) * y - x]


class Saddle(DeterministicFunction):
    """Hamiltonian saddle x' = y, y' = x - x^2; homoclinic loop to the origin."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        return [y, x - x * x]


class NoisyDrift(DeterministicFunction):
    """Pitchfork drift a x - x^3 (First = a)."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return [self.parameters[0].value * point[0] - point[0] ** 3]


class NoisyDiffusion(DeterministicFunction):
    """Multiplicative diffusion s x (Second = s)."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return [self.parameters[1].value * point[0]]


class BogdanovTakensFamily(DeterministicFunction):
    """x' = y, y' = b1 + b2 y + x^2 + x y.

    Continuing in b1 gives a fold at b1 = 0 and a Hopf at b1 = -b2^2 on the same
    branch; as the family parameter b2 decreases to zero the Hopf collides with the
    fold at the codim-2 Bogdanov-Takens point, reorganising the diagram.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        x, y = point[0], point[1]
        return [y, b1 + b2 * y + x * x + x * y]


class BogdanovTakensUnfolding(DeterministicFunction):
    """x' = y, y' = b1 + b2 x + b3 y + b3 x^2 + x y.

    A three-parameter unfolding whose Bogdanov-Takens set is a curve
    (b1 = b3^3, b2 = 2 b3^2); the quadratic coefficient is a = b3, so the curve
    passes through a degenerate Bogdanov-Takens at the origin.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        first = self.parameters[0].value
        second = self.parameters[1].value
        third = self.parameters[2].value
        x, y = point[0], point[1]
        return [y, first + second * x + third * y + third * x * x + x * y]


class BogdanovTakensTypeChange(DeterministicFunction):
    """x' = y, y' = b1 + b2 x + b3 y + x^2 + b3 x y.

    Bogdanov-Takens curve b1 = 1, b2 = 2; here the quadratic coefficient a = 1 while
    the cross coefficient b = b3, so the curve passes through the b = 0 degenerate
    Bogdanov-Takens (the two topological BT types meet) at (1, 2, 0).
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        first = self.parameters[0].value
        second = self.parameters[1].value
        third = self.parameters[2].value
        x, y = point[0], point[1]
        return [y, first + second * x + third * y + x * x + third * x * y]


class BogdanovTakensPlanar(DeterministicFunction):
    """x' = y, y' = b1 + b2 y + x^2 + x y (s = +1), for the organizing centre."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        first = self.parameters[0].value
        second = self.parameters[1].value
        x, y = point[0], point[1]
        return [y, first + second * y + x * x + x * y]
