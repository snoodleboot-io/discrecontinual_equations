"""The systems the continuation tests are checked against.

These are shared: the same saddle carries the manifold tests and the
homoclinic ones, and the same Alpha parameter runs through nearly all of
them. They lived among the tests, which is why the tests could not be split.
"""

import math

import numpy as np

from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

_SADDLE_JACOBIAN = np.array([[0.0, 1.0], [1.0, 0.0]])


class Alpha(Parameter, name="alpha", abbreviation="a"):
    pass


class Sigma(Parameter, name="sigma", abbreviation="s"):
    pass


class State(Variable, name="State", abbreviation="v"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class BetaOne(Parameter, name="beta_one", abbreviation="b1"):
    """First Bogdanov-Takens unfolding parameter."""


class BetaTwo(Parameter, name="beta_two", abbreviation="b2"):
    """Second Bogdanov-Takens unfolding parameter."""


class Saddle(DeterministicFunction):
    """x' = y, y' = x - x^2; homoclinic to the origin."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        return [y, x - x * x]


class BogdanovTakensField(DeterministicFunction):
    """x' = y, y' = b1 + b2 y + x^2 + x y (s = +1)."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        first = self.parameters[0].value
        second = self.parameters[1].value
        x, y = point[0], point[1]
        return [y, first + second * y + x * x + x * y]


class JerkField(DeterministicFunction):
    """Saddle-focus jerk system x'=y, y'=z, z'=-a z - y + x - x^2.

    The origin is a saddle-focus; a homoclinic loop to it exists near a = 0.48,
    where the saddle index is below one and the flow is chaotic (Shilnikov).
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        a = self.parameters[0].value
        x, y, z = point[0], point[1], point[2]
        return [y, z, -a * z - y + x - x * x]


class CubicField(DeterministicFunction):
    """Scalar x^3 - x with roots at -1, 0, +1."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return [point[0] ** 3 - point[0]]


class GridField(DeterministicFunction):
    """Decoupled x - x^3, y - y^3 with nine roots in {-1, 0, 1} squared."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        return [x - x**3, y - y**3]


class MelnikovField(DeterministicFunction):
    """Two-parameter perturbed well x'=y, y'=x-x^2+mu y+nu x y.

    The saddle at the origin has a homoclinic loop whose persistence locus in the
    (mu, nu) plane is the Melnikov line mu I1 + nu I2 = 0.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        nu = self.parameters[1].value
        x, y = point[0], point[1]
        return [y, x - x * x + mu * y + nu * x * y]


class TwoParameterJerkField(DeterministicFunction):
    """Saddle-focus jerk x'=y, y'=z, z'=-a z - b y + x - x^2 with two parameters."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        a = self.parameters[0].value
        b = self.parameters[1].value
        x, y, z = point[0], point[1], point[2]
        return [y, z, -a * z - b * y + x - x * x]


class DampedWellField(DeterministicFunction):
    """x' = y, y' = x - x^2 + mu y; homoclinic to the origin exactly at mu = 0."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        return [y, x - x * x + mu * y]


class HeteroclinicField(DeterministicFunction):
    """Double-well x' = y, y' = -x + x^3; saddles at (-1,0) and (+1,0)."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        return [y, -x + x**3]


class SpiralField(DeterministicFunction):
    """Linear spiral x' = mu x - y, y' = x + mu y; both Lyapunov exponents are mu."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        return [mu * x - y, x + mu * y]


class DiagonalField(DeterministicFunction):
    """Decoupled contraction x' = -0.5 x, y' = -y; spectrum {-0.5, -1}."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return [-0.5 * point[0], -1.0 * point[1]]


class LorenzField(DeterministicFunction):
    """Classic Lorenz system (sigma=10, rho=28, beta=8/3); top exponent ~ 0.906."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y, z = point[0], point[1], point[2]
        return [10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z]


class SnicField(DeterministicFunction):
    """Attracting unit circle with Adler flow: r' = r(1-r^2), theta' = mu - sin(theta).

    For mu > 1 a limit cycle rotates with period 2 pi / sqrt(mu^2 - 1); at mu = 1 a
    saddle-node forms on the circle (SNIC) and the period diverges.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        radius = (x * x + y * y) ** 0.5
        if radius < 1.0e-9:
            return [0.0, 0.0]
        radial = 1.0 - radius * radius
        angular = mu - y / radius
        return [radial * x - y * angular, radial * y + x * angular]


class HopfField(DeterministicFunction):
    """Supercritical Hopf: x' = mu x - y - x r^2, y' = x + mu y - y r^2."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        radius = x * x + y * y
        return [mu * x - y - x * radius, x + mu * y - y * radius]


class VanDerPolField(DeterministicFunction):
    """Van der Pol oscillator x' = y, y' = mu (1 - x^2) y - x (polynomial)."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        return [y, mu * (1.0 - x * x) * y - x]


class FoldOfCyclesField(DeterministicFunction):
    """r' = mu r + r^3 - r^5 in Cartesian form; two cycles collide at mu = -1/4."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        squared = x * x + y * y
        growth = mu + squared - squared * squared
        return [growth * x - y, x + growth * y]


class NeimarkSackerField(DeterministicFunction):
    """Hopf cycle with a transverse rotating block; a torus is born at nu = 0."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        nu = self.parameters[0].value
        x, y, z, w = point[0], point[1], point[2], point[3]
        radius = x * x + y * y
        rotation = 0.3
        return [
            0.5 * x - y - x * radius,
            x + 0.5 * y - y * radius,
            nu * z - rotation * w,
            rotation * z + nu * w,
        ]


class PeriodDoublingField(DeterministicFunction):
    """Transverse block with rotation pi over the period; a flip at alpha = 0."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        alpha = self.parameters[0].value
        x, y, z, w = point[0], point[1], point[2], point[3]
        radius = x * x + y * y
        rotation = 0.5
        return [
            0.5 * x - y - x * radius,
            x + 0.5 * y - y * radius,
            alpha * z - rotation * w,
            rotation * z + alpha * w,
        ]


class Drift(DeterministicFunction):
    """Pitchfork drift a x - x^3."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return [self.parameters[0].value * point[0] - point[0] ** 3]


class Diffusion(DeterministicFunction):
    """Multiplicative diffusion s x."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return [self.parameters[1].value * point[0]]


class ConstantDiffusion(DeterministicFunction):
    """Additive scalar noise: a constant three-vector s applied to every component."""

    def eval(
        self,
        point: list[float],  # noqa: ARG002 (constant field)
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        sigma = self.parameters[0].value
        return [sigma, sigma, sigma]


def _saddle() -> DeterministicFunction:
    return Saddle(
        variables=[State(), State()],
        parameters=[Alpha(value=0.0)],
        results=[State(), State()],
        time=None,
    )


def _scalar(function_type: type[DeterministicFunction], alpha: float, sigma: float):
    return function_type(
        variables=[State()],
        parameters=[Alpha(value=alpha), Sigma(value=sigma)],
        results=[State()],
        time=None,
    )


def _tilt() -> np.ndarray:
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


_TILT = _tilt()


class TiltedHomoclinicField(DeterministicFunction):
    """u'=v, v'=u-u^2, w'=-2w rotated into all three axes by a fixed frame.

    The saddle at the origin has eigenvalues {+1, -1, -2}: a one-dimensional
    unstable direction and a two-dimensional stable eigenspace. The homoclinic loop
    is the rotated image of u = 1.5 sech^2(t/2).
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        u, v, w = _TILT.T @ np.array(point)
        base = np.array([v, u - u * u, -2.0 * w])
        return list(_TILT @ base)


def _integrate(
    function: DeterministicFunction,
    state: np.ndarray,
    step: float,
    steps: int,
) -> np.ndarray:
    current = np.array(state, dtype=float)
    for _ in range(steps):
        k1 = np.array(function.eval(point=list(current), time=None))
        k2 = np.array(function.eval(point=list(current + 0.5 * step * k1), time=None))
        k3 = np.array(function.eval(point=list(current + 0.5 * step * k2), time=None))
        k4 = np.array(function.eval(point=list(current + step * k3), time=None))
        current = current + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return current


class Kappa(Parameter, name="kappa", abbreviation="k"):
    """How fast the noise amplitude grows with the radius."""


class DiagonalGrowthField(DeterministicFunction):
    """x' = a1 x, y' = a2 y; the eigenvalues are the parameters themselves."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        first = self.parameters[0].value
        second = self.parameters[1].value
        return [first * point[0], second * point[1]]


class IsotropicNoiseColumn(DeterministicFunction):
    """One column of the isotropic amplitude ``s sqrt(1 + k r^2) I``.

    Each column drives a different component from an independent Brownian motion, so
    ``D = G G^T = s^2 (1 + k r^2) I`` has full rank and the planar density exists. At
    ``k = 0`` the noise is additive, and the exact density of the noisy Hopf system is
    ``exp((mu r^2 - r^4 / 2) / s^2)``; at ``k > 0`` it is multiplicative but stays
    bounded away from zero, which keeps the stationary equation elliptic while still
    shifting the phenomenological threshold to ``mu = k s^2`` (Ito).
    """

    def __init__(
        self,
        index: int,
        variables: list[Variable],
        parameters: list[Parameter],
        results: list[Variable],
        time: Variable | None = None,
    ) -> None:
        super().__init__(variables, parameters, results, time)
        self._index = index

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        sigma = self.parameters[1].value
        kappa = self.parameters[2].value
        radius = sum(value * value for value in point)
        amplitude = sigma * math.sqrt(1.0 + kappa * radius)
        return [amplitude if axis == self._index else 0.0 for axis in range(len(point))]


class ConstantNoiseColumn(DeterministicFunction):
    """A constant noise column, one parameter per component.

    Two of these with overlapping components give a diffusion matrix with off-diagonal
    entries, which is the only way to exercise the cross-diffusion part of the
    Fokker-Planck discretisation.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return [parameter.value for parameter in self.parameters[: len(point)]]


class DiagonalMultiplicativeColumn(DeterministicFunction):
    """Column ``index`` of ``diag(s_i x_i)``: independent noise on each component.

    The components decouple into geometric Brownian motions, so the Lyapunov spectrum
    is ``{a_i - s_i^2 / 2}`` (Ito) exactly, which is the analytic oracle for a genuine
    matrix amplitude above one dimension.
    """

    def __init__(
        self,
        index: int,
        variables: list[Variable],
        parameters: list[Parameter],
        results: list[Variable],
        time: Variable | None = None,
    ) -> None:
        super().__init__(variables, parameters, results, time)
        self._index = index

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        sigma = self.parameters[self._index].value
        return [
            sigma * point[axis] if axis == self._index else 0.0
            for axis in range(len(point))
        ]


class ConformalColumn(DeterministicFunction):
    """Column ``index`` of ``s [[x, -y], [y, x]]``: the conformal noise ``s z dB``.

    Two independent Brownian motions multiplying ``z = x + i y`` by a complex
    increment. ``D = G G^T = s^2 r^2 I`` is isotropic and vanishes only at the
    origin, which stays invariant, so the planar density of a radial drift is
    the one-dimensional closed form in the radius and the origin's top exponent
    is exact: for the Hopf drift ``p = C r^(2 mu / s^2 - 2) exp(-r^2 / s^2)``,
    cratering at ``mu = s^2``, and the exponent is ``mu`` itself. The
    Stratonovich correction vanishes identically, since the first column's
    Jacobian is ``s I`` and the second's a right-angle rotation.
    """

    def __init__(
        self,
        index: int,
        variables: list[Variable],
        parameters: list[Parameter],
        results: list[Variable],
        time: Variable | None = None,
    ) -> None:
        super().__init__(variables, parameters, results, time)
        self._index = index

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        sigma = self.parameters[1].value
        x, y = point[0], point[1]
        if self._index == 0:
            return [sigma * x, sigma * y]
        return [-sigma * y, sigma * x]


class ScaledStateColumn(DeterministicFunction):
    """The single column ``s x``: one Brownian motion scaling the whole state.

    For a linear drift this is the one planar case whose top exponent is known in
    closed form: ``x = exp(s W - s^2 t / 2) y`` with ``y' = A y``, so the exponent is
    ``max Re eig(A) - s^2 / 2`` (Ito) and ``max Re eig(A)`` (Stratonovich).
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        sigma = self.parameters[1].value
        return [sigma * value for value in point]
