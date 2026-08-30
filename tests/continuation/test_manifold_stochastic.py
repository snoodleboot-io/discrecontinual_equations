"""Manifolds, connecting orbits, and stochastic bifurcation, against oracles.

The Hamiltonian saddle ``x' = y, y' = x - x^2`` has an explicit homoclinic loop
``x(t) = 3/2 sech^2(t/2)`` to the origin, which is also its unstable manifold; both
the manifold chart and the homoclinic solver are checked against it. The pitchfork
with multiplicative noise ``dx = (a x - x^3) dt + s x dW`` (Stratonovich) has a
phenomenological bifurcation at ``a = s^2 / 2`` and a dynamical bifurcation at
``a = 0``; the stationary density and the Lyapunov exponent are checked against
both thresholds.
"""

import itertools
import math
from unittest import TestCase

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.connecting_orbit import (
    HeteroclinicOrbit,
    HomoclinicOrbit,
    MeshSpec,
    Terminus,
    _projection_conditions,
    _split_eigenspaces,
)
from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CycleSeed,
)
from discrecontinual_equations.continuation.deflation import DeflatedSolver
from discrecontinual_equations.continuation.homoclinic_curve import HomoclinicCurve
from discrecontinual_equations.continuation.homoclinic_shooting import (
    Departure,
    HomoclinicShooting,
    ReturnSettings,
)
from discrecontinual_equations.continuation.lyapunov import (
    LyapunovSettings,
    StochasticLyapunovSettings,
    StochasticSystem,
    lyapunov_spectrum,
    stochastic_lyapunov,
)
from discrecontinual_equations.continuation.manifold import (
    StableManifold,
    TaylorManifold,
    UnstableManifold,
)
from discrecontinual_equations.continuation.periodic_orbit import (
    AdaptivePeriodicOrbit,
    AnalyticPeriodicOrbit,
    HermiteSimpsonOrbit,
    PeriodicOrbit,
    RobustPeriodicOrbit,
)
from discrecontinual_equations.continuation.region_analysis import (
    _jacobian,
    classify_equilibrium,
    describe_region,
    find_equilibria,
)
from discrecontinual_equations.continuation.shilnikov import classify_saddle_focus
from discrecontinual_equations.continuation.snic import characterize_snic
from discrecontinual_equations.continuation.stochastic import (
    DensityModes,
    Ito,
    LyapunovExponent,
    StationaryDensity,
    Stratonovich,
)
from discrecontinual_equations.continuation.symmetry import (
    cyclic_action,
    equivariance_defect,
    fourier_reduce,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.systems.coupled_ring import (
    FerroelectricRing,
    FluxgateRing,
)
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


class Saddle(DeterministicFunction):
    """x' = y, y' = x - x^2; homoclinic to the origin."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        return [y, x - x * x]


class BetaOne(Parameter, name="beta_one", abbreviation="b1"):
    """First Bogdanov-Takens unfolding parameter."""


class BetaTwo(Parameter, name="beta_two", abbreviation="b2"):
    """Second Bogdanov-Takens unfolding parameter."""


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


class TestInvariantManifold(TestCase):
    def test_unstable_manifold_is_flow_invariant(self):
        chart = TaylorManifold(order=6).compute(
            _saddle(),
            np.zeros(2),
            _SADDLE_JACOBIAN,
            UnstableManifold(),
        )
        start = chart.point([0.05])
        flowed = chart.flow_image([0.05], 0.5)
        integrated = _integrate(_saddle(), start, 0.001, 500)
        assert np.linalg.norm(integrated - flowed) < 1.0e-8

    def test_unstable_manifold_lies_on_homoclinic_level_set(self):
        chart = TaylorManifold(order=6).compute(
            _saddle(),
            np.zeros(2),
            _SADDLE_JACOBIAN,
            UnstableManifold(),
        )
        for theta in np.linspace(0.0, 0.5, 6):
            x, y = chart.point([float(theta)])
            energy = 0.5 * y * y - 0.5 * x * x + x**3 / 3.0
            assert abs(energy) < 1.0e-6

    def test_stable_manifold_has_negative_eigenvalue(self):
        chart = TaylorManifold(order=4).compute(
            _saddle(),
            np.zeros(2),
            _SADDLE_JACOBIAN,
            StableManifold(),
        )
        assert chart.eigenvalues[0] < 0.0


class TestHomoclinicOrbit(TestCase):
    def test_converges_to_exact_homoclinic(self):
        half, intervals = 10.0, 80
        mesh = MeshSpec(half, intervals, phase_index=1, phase_value=0.0)
        times = np.linspace(-half, half, intervals + 1)
        seed = np.zeros((intervals + 1, 2))
        for k, t in enumerate(times):
            seed[k, 0] = 1.2 / math.cosh(0.45 * t) ** 2
            seed[k, 1] = -0.9 * math.tanh(0.45 * t) / math.cosh(0.45 * t) ** 2
        solution = HomoclinicOrbit(
            _saddle(),
            np.zeros(2),
            _SADDLE_JACOBIAN,
            mesh,
        ).solve(
            seed,
        )
        exact = 1.5 / np.cosh(times / 2.0) ** 2
        assert np.max(np.abs(solution.component(0) - exact)) < 5.0e-2
        assert abs(solution.component(0)[intervals // 2] - 1.5) < 5.0e-2


class TestHomoclinicCurve(TestCase):
    """The BT homoclinic curve obeys b1 = -(49/25) b2^2 (the 5/7 asymptotic)."""

    def test_bogdanov_takens_homoclinic_asymptotic(self):
        parameters = [BetaOne(value=0.0), BetaTwo(value=0.0)]
        equation = DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=BogdanovTakensField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )
        epsilon, nodes, span = 0.25, 101, 5.0
        half = span * math.sqrt(2.0) / epsilon
        times = np.linspace(-half, half, nodes)
        scaled = epsilon * times / math.sqrt(2.0)
        profile = 1.0 / np.cosh(scaled) ** 2
        orbit = np.column_stack(
            [
                epsilon**2 * (1.0 - 3.0 * profile),
                epsilon**3 * 3.0 * math.sqrt(2.0) * profile * np.tanh(scaled),
            ],
        )
        mesh = MeshSpec(half, nodes - 1, phase_index=1, phase_value=0.0)
        curve = HomoclinicCurve(equation, 0, 1, mesh)
        sweep = (5.0 / 7.0) * epsilon**2
        point = curve.solve_point(
            sweep,
            orbit,
            np.array([epsilon**2, 0.0]),
            -(epsilon**4),
        )
        assert point is not None
        oracle = -(49.0 / 25.0) * sweep**2
        assert point.continuation < 0.0
        assert abs(point.continuation / oracle - 1.0) < 1.5e-2


class TestRegionAnalysis(TestCase):
    """Region labels from equilibria, stability, and Poincare-Bendixson cycles."""

    def _hopf(self, mu: float) -> DifferentialEquation:
        parameters = [Alpha(value=mu)]
        return DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=HopfField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )

    def test_supercritical_hopf_reports_limit_cycle(self):
        label = describe_region(
            self._hopf(0.25).derivative,
            [np.zeros(2)],
            cycle_start=np.array([0.05, 0.0]),
        )
        assert label == "unstable focus + limit cycle"

    def test_stable_focus_has_no_cycle(self):
        label = describe_region(
            self._hopf(-0.25).derivative,
            [np.zeros(2)],
            cycle_start=np.array([0.5, 0.0]),
        )
        assert label == "stable focus"

    def _bt(self, first: float, second: float) -> DifferentialEquation:
        parameters = [BetaOne(value=first), BetaTwo(value=second)]
        return DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=BogdanovTakensField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )

    def test_bogdanov_takens_regions(self):
        seeds = [np.array([0.4, 0.0]), np.array([-0.4, 0.0])]
        top = describe_region(self._bt(0.05, 0.16).derivative, seeds)
        above = describe_region(self._bt(-0.012, 0.16).derivative, seeds)
        below = describe_region(self._bt(-0.05, 0.16).derivative, seeds)
        assert top == "no equilibria"
        assert above == "saddle + unstable focus"
        assert below == "saddle + stable focus"


class TestPeriodicOrbit(TestCase):
    """Limit cycle and Floquet multipliers against the Hopf normal form.

    The supercritical Hopf has an exact cycle of radius sqrt(mu), period 2 pi, a
    trivial Floquet multiplier +1, and a nontrivial multiplier exp(-4 pi mu).
    """

    def _hopf(self, mu: float) -> DifferentialEquation:
        parameters = [Alpha(value=mu)]
        return DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=HopfField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )

    def test_cycle_radius_period_and_floquet(self):
        mu, intervals = 0.25, 80
        nodes = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [
                math.sqrt(mu) * np.cos(2.0 * math.pi * nodes),
                math.sqrt(mu) * np.sin(2.0 * math.pi * nodes),
            ],
        )
        orbit = PeriodicOrbit(
            self._hopf(mu).derivative,
            intervals,
            phase_index=1,
            phase_value=0.0,
        )
        solution = orbit.solve(seed, 6.0)
        assert solution is not None
        radius = float(np.mean(np.linalg.norm(solution.states, axis=1)))
        assert abs(radius - math.sqrt(mu)) < 1.0e-3
        assert abs(solution.period - 2.0 * math.pi) < 1.0e-2
        multipliers = sorted(orbit.floquet_multipliers(solution).real)
        assert abs(multipliers[1] - 1.0) < 1.0e-2
        assert abs(multipliers[0] - math.exp(-4.0 * math.pi * mu)) < 5.0e-3


class TestCycleBifurcations(TestCase):
    """Fold of cycles, Neimark-Sacker, and period-doubling against oracles."""

    def _nodes(self, intervals: int) -> np.ndarray:
        return np.linspace(0.0, 1.0, intervals + 1)

    def _circle(self, radius: float, intervals: int, extra: int) -> np.ndarray:
        nodes = self._nodes(intervals)
        columns = [
            radius * np.cos(2.0 * math.pi * nodes),
            radius * np.sin(2.0 * math.pi * nodes),
        ]
        columns += [np.zeros(intervals + 1) for _ in range(extra)]
        return np.column_stack(columns)

    def _equation(self, field_type: type, dimension: int, value: float):
        parameters = [Alpha(value=value)]
        return DifferentialEquation(
            variables=[State() for _ in range(dimension)],
            time=Time(),
            parameters=parameters,
            derivative=field_type(
                variables=[State() for _ in range(dimension)],
                parameters=parameters,
                results=[State() for _ in range(dimension)],
                time=None,
            ),
        )

    def test_fold_of_cycles(self):
        intervals = 60
        mu0 = -0.1
        radius = math.sqrt((1.0 + math.sqrt(1.0 + 4.0 * mu0)) / 2.0)
        seed = CycleSeed(self._circle(radius, intervals, 0), 2.0 * math.pi, mu0)
        continuation = CycleContinuation(
            self._equation(FoldOfCyclesField, 2, mu0),
            0,
            intervals,
            phase_index=1,
            phase_value=0.0,
        )
        _points, bifurcations = continuation.trace(seed, 0.03, 120, direction=-1.0)
        folds = [b for b in bifurcations if b.kind == "fold_of_cycles"]
        assert folds
        assert any(abs(b.parameter + 0.25) < 5.0e-3 for b in folds)

    def test_neimark_sacker(self):
        intervals = 60
        seed = CycleSeed(
            self._circle(math.sqrt(0.5), intervals, 2),
            2.0 * math.pi,
            -0.1,
        )
        continuation = CycleContinuation(
            self._equation(NeimarkSackerField, 4, -0.1),
            0,
            intervals,
            phase_index=1,
            phase_value=0.0,
        )
        _points, bifurcations = continuation.trace(seed, 0.02, 12)
        torus = [b for b in bifurcations if b.kind == "neimark_sacker"]
        assert torus
        assert any(abs(b.parameter) < 5.0e-3 for b in torus)

    def test_period_doubling(self):
        intervals = 60
        seed = CycleSeed(
            self._circle(math.sqrt(0.5), intervals, 2),
            2.0 * math.pi,
            -0.1,
        )
        continuation = CycleContinuation(
            self._equation(PeriodDoublingField, 4, -0.1),
            0,
            intervals,
            phase_index=1,
            phase_value=0.0,
        )
        _points, bifurcations = continuation.trace(seed, 0.02, 12)
        flips = [b for b in bifurcations if b.kind == "period_doubling"]
        assert flips
        assert any(abs(b.parameter) < 5.0e-3 for b in flips)


class TestLyapunovSpectrum(TestCase):
    """Numerical Lyapunov exponents and the dynamical (D) bifurcation."""

    def _spiral(self, mu: float) -> DifferentialEquation:
        parameters = [Alpha(value=mu)]
        return DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=SpiralField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )

    def test_top_exponent_tracks_parameter(self):
        settings = LyapunovSettings(dt=0.01, horizon=6000, transient=100)
        for mu in (-0.2, 0.0, 0.1):
            spectrum = lyapunov_spectrum(
                self._spiral(mu).derivative,
                np.array([1.0, 0.0]),
                settings=settings,
            )
            assert abs(spectrum.top - mu) < 5.0e-3

    def test_dynamical_bifurcation_sign_change(self):
        settings = LyapunovSettings(dt=0.01, horizon=6000, transient=100)
        below = lyapunov_spectrum(
            self._spiral(-0.15).derivative,
            np.array([1.0, 0.0]),
            settings=settings,
        )
        above = lyapunov_spectrum(
            self._spiral(0.1).derivative,
            np.array([1.0, 0.0]),
            settings=settings,
        )
        assert below.top < 0.0 < above.top

    def test_diagonal_spectrum_is_exact(self):
        equation = DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=[],
            derivative=DiagonalField(
                variables=[State(), State()],
                parameters=[],
                results=[State(), State()],
                time=None,
            ),
        )
        spectrum = lyapunov_spectrum(
            equation.derivative,
            np.array([1.0, 1.0]),
            settings=LyapunovSettings(dt=0.01, horizon=20000, transient=50),
        )
        assert abs(spectrum.exponents[0] + 0.5) < 1.0e-2
        assert abs(spectrum.exponents[1] + 1.0) < 1.0e-2

    def test_lorenz_top_exponent(self):
        equation = DifferentialEquation(
            variables=[State(), State(), State()],
            time=Time(),
            parameters=[],
            derivative=LorenzField(
                variables=[State(), State(), State()],
                parameters=[],
                results=[State(), State(), State()],
                time=None,
            ),
        )
        spectrum = lyapunov_spectrum(
            equation.derivative,
            np.array([1.0, 1.0, 1.0]),
            count=1,
            settings=LyapunovSettings(dt=0.005, horizon=30000, transient=3000),
        )
        assert abs(spectrum.top - 0.906) < 5.0e-2


class TestAnalyticPeriodicOrbit(TestCase):
    """Exact, analytically-assembled BVP Jacobian via automatic differentiation."""

    def _field(self, field_type, mu: float) -> object:
        return field_type(
            variables=[State(), State()],
            parameters=[Alpha(value=mu)],
            results=[State(), State()],
            time=None,
        )

    def test_analytic_jacobian_matches_finite_difference(self):
        mu, intervals, dimension = 0.5, 12, 2
        solver = AnalyticPeriodicOrbit(
            self._field(HopfField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        )
        nodes = intervals + 1
        grid = np.linspace(0.0, 1.0, nodes)
        states = np.column_stack(
            [
                math.sqrt(mu) * np.cos(2.0 * math.pi * grid),
                math.sqrt(mu) * np.sin(2.0 * math.pi * grid),
            ],
        )
        unknowns = np.concatenate([states.flatten(), [6.3]])
        analytic = solver._analytic_jacobian(unknowns, nodes, dimension)  # noqa: SLF001
        base = solver._residual(unknowns, nodes, dimension)  # noqa: SLF001
        finite = np.zeros((base.size, unknowns.size))
        step = 1.0e-7
        for j in range(unknowns.size):
            shifted = unknowns.copy()
            shifted[j] += step
            finite[:, j] = (
                solver._residual(shifted, nodes, dimension) - base  # noqa: SLF001
            ) / step
        assert np.max(np.abs(analytic - finite)) < 1.0e-5

    def test_matches_base_solver_on_hopf(self):
        mu, intervals = 0.5, 60
        grid = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [
                math.sqrt(mu) * np.cos(2.0 * math.pi * grid),
                math.sqrt(mu) * np.sin(2.0 * math.pi * grid),
            ],
        )
        analytic = AnalyticPeriodicOrbit(
            self._field(HopfField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed.copy(), 6.3)
        base = PeriodicOrbit(
            self._field(HopfField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed.copy(), 6.3)
        assert analytic is not None
        assert abs(analytic.period - base.period) < 1.0e-7

    def test_solves_polynomial_van_der_pol(self):
        mu, intervals = 1.0, 200
        grid = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [2.0 * np.cos(2.0 * math.pi * grid), -2.0 * np.sin(2.0 * math.pi * grid)],
        )
        analytic = AnalyticPeriodicOrbit(
            self._field(VanDerPolField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed.copy(), 6.6)
        base = PeriodicOrbit(
            self._field(VanDerPolField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed.copy(), 6.6)
        assert analytic is not None
        assert abs(analytic.period - base.period) < 1.0e-6


class TestAdaptivePeriodicOrbit(TestCase):
    """Mesh adaptation for stiff relaxation oscillations, vs an integrated oracle."""

    def _field(self, field_type, mu: float) -> object:
        return field_type(
            variables=[State(), State()],
            parameters=[Alpha(value=mu)],
            results=[State(), State()],
            time=None,
        )

    def _van_der_pol_oracle(self, mu: float, intervals: int):
        def flow(_t, point):
            x, y = point
            return [y, mu * (1.0 - x * x) * y - x]

        settled = solve_ivp(
            flow,
            [0.0, 100.0 * mu],
            [2.0, 0.0],
            method="Radau",
            rtol=1.0e-9,
            atol=1.0e-11,
        ).y[:, -1]

        def section(_t, point):
            return point[1]

        section.direction = 1.0
        cycle = solve_ivp(
            flow,
            [0.0, 20.0 * mu],
            settled,
            method="Radau",
            rtol=1.0e-10,
            atol=1.0e-12,
            events=section,
            dense_output=True,
        )
        crossings = cycle.t_events[0]
        period = crossings[1] - crossings[0]
        times = crossings[0] + np.linspace(0.0, period, intervals + 1)
        return period, cycle.sol(times).T

    def test_beats_uniform_on_stiff_cycle(self):
        mu, intervals = 10.0, 200
        period, seed = self._van_der_pol_oracle(mu, intervals)
        uniform = AnalyticPeriodicOrbit(
            self._field(VanDerPolField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed.copy(), period)
        adaptive = AdaptivePeriodicOrbit(
            self._field(VanDerPolField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed.copy(), period)
        assert uniform is not None
        assert adaptive is not None
        uniform_error = abs(uniform.period - period) / period
        adaptive_error = abs(adaptive.period - period) / period
        assert uniform_error > 0.03
        assert adaptive_error < 5.0e-3
        assert adaptive_error < 0.1 * uniform_error

    def test_no_regression_on_smooth_cycle(self):
        mu, intervals = 0.5, 60
        grid = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [
                math.sqrt(mu) * np.cos(2.0 * math.pi * grid),
                math.sqrt(mu) * np.sin(2.0 * math.pi * grid),
            ],
        )
        adaptive = AdaptivePeriodicOrbit(
            self._field(HopfField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed, 6.3)
        assert adaptive is not None
        assert abs(adaptive.period - 2.0 * math.pi) / (2.0 * math.pi) < 1.0e-2

    def test_error_estimate_flags_an_under_resolved_cycle(self):
        """The estimate separates a resolved cycle from an under-resolved one.

        A converged solve only proves the discrete system was satisfied; nothing in
        the residual gate knows whether the mesh resolves the jump layers, so too
        coarse a mesh converges confidently to the wrong cycle. Refining is what
        catches it: an under-resolved solution either fails to seed the finer mesh
        at all, or shifts a long way when it does.
        """
        mu, coarse, fine = 8.0, 50, 200

        def estimate(intervals: int) -> tuple[float | None, float]:
            period, seed = self._van_der_pol_oracle(mu, intervals)
            orbit = AdaptivePeriodicOrbit(
                self._field(VanDerPolField, mu),
                intervals,
                phase_index=1,
                phase_value=0.0,
            )
            solution = orbit.solve(seed.copy(), period)
            if solution is None:
                return None, 1.0
            true_error = abs(solution.period - period) / period
            return orbit.estimate_period_error(
                solution.states,
                solution.period,
            ), true_error

        coarse_estimate, coarse_error = estimate(coarse)
        # Either it cannot be refined at all, or refining moves it a long way.
        # Both are the detector doing its job; which one occurs is not the contract.
        assert coarse_estimate is None or coarse_estimate > 1.0e-2

        fine_estimate, fine_error = estimate(fine)
        assert fine_estimate is not None
        assert fine_estimate < 1.0e-2
        # It must not flatter the true error - conservative is the useful direction.
        assert fine_estimate >= fine_error / 10.0
        # And it must actually discriminate.
        assert fine_error < coarse_error

    def test_continuation_reaches_extreme_stiffness(self):
        """Continuation reaches mu = 16, which needs 400 intervals, not 200.

        The jump layers of a van der Pol relaxation oscillation narrow as mu grows,
        and at mu = 16 a 200-interval mesh cannot resolve them however the nodes are
        redistributed: the continuation stalls at mu ~ 11.7 and reports it by
        returning None. At 400 intervals it reaches the target. See FUTURE_WORK.md
        section 1b.
        """
        intervals, target = 400, 16.0
        period, seed = self._van_der_pol_oracle(target, intervals)
        cold = AdaptivePeriodicOrbit(
            self._field(VanDerPolField, target),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed.copy(), period)
        cold_failed = cold is None or abs(cold.period - period) / period > 0.1
        assert cold_failed
        start_period, start_seed = self._van_der_pol_oracle(6.0, intervals)
        continued = AdaptivePeriodicOrbit(
            self._field(VanDerPolField, 6.0),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).continue_to(0, target, start_seed.copy(), start_period)
        assert continued is not None
        assert abs(continued.period - period) / period < 1.0e-2


class TestHermiteSimpsonOrbit(TestCase):
    """Fourth-order collocation for limit cycles, checked by convergence order."""

    def _hopf(self, mu: float) -> DifferentialEquation:
        parameters = [Alpha(value=mu)]
        return DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=HopfField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )

    def _period_error(self, solver_type, intervals: int, mu: float) -> float:
        nodes = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [
                math.sqrt(mu) * np.cos(2.0 * math.pi * nodes),
                math.sqrt(mu) * np.sin(2.0 * math.pi * nodes),
            ],
        )
        solution = solver_type(
            self._hopf(mu).derivative,
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed, 6.0)
        return abs(solution.period - 2.0 * math.pi)

    def test_fourth_order_convergence(self):
        mu = 0.5
        errors = [self._period_error(HermiteSimpsonOrbit, n, mu) for n in (20, 40, 80)]
        orders = [
            math.log(errors[k] / errors[k + 1]) / math.log(2.0)
            for k in range(len(errors) - 1)
        ]
        for order in orders:
            assert order > 3.5

    def test_more_accurate_than_trapezoidal(self):
        mu, intervals = 0.5, 40
        hermite = self._period_error(HermiteSimpsonOrbit, intervals, mu)
        trapezoidal = self._period_error(PeriodicOrbit, intervals, mu)
        assert hermite < 1.0e-4
        assert hermite < 0.01 * trapezoidal


class TestRobustPeriodicOrbit(TestCase):
    """Newton robustness against collapse to the trivial (zero-period) solution."""

    def _hopf(self, mu: float) -> DifferentialEquation:
        parameters = [Alpha(value=mu)]
        return DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=HopfField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )

    def _snic(self, mu: float) -> DifferentialEquation:
        parameters = [Alpha(value=mu)]
        return DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=SnicField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )

    def test_matches_plain_solver_on_regular_cycle(self):
        mu, intervals = 0.25, 80
        nodes = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [
                np.sqrt(mu) * np.cos(2.0 * math.pi * nodes),
                np.sqrt(mu) * np.sin(2.0 * math.pi * nodes),
            ],
        )
        solution = RobustPeriodicOrbit(
            self._hopf(mu).derivative,
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed, 6.0)
        assert solution is not None
        assert abs(solution.period - 2.0 * math.pi) < 1.0e-2
        radius = float(np.mean(np.linalg.norm(solution.states, axis=1)))
        assert abs(radius - math.sqrt(mu)) < 1.0e-3

    def test_rescues_collapse_near_snic(self):
        intervals = 120
        nodes = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [np.cos(2.0 * math.pi * nodes), np.sin(2.0 * math.pi * nodes)],
        )
        plain_failures = 0
        for mu in (1.06, 1.08, 1.10, 1.12):
            oracle = 2.0 * math.pi / math.sqrt(mu * mu - 1.0)
            plain = PeriodicOrbit(
                self._snic(mu).derivative,
                intervals,
                phase_index=1,
                phase_value=0.0,
            ).solve(seed.copy(), oracle * 1.05)
            robust = RobustPeriodicOrbit(
                self._snic(mu).derivative,
                intervals,
                phase_index=1,
                phase_value=0.0,
            ).solve(seed.copy(), oracle * 1.05)
            assert robust is not None
            assert abs(robust.period - oracle) / oracle < 1.0e-3
            if plain is None or abs(plain.period - oracle) / oracle > 0.5:
                plain_failures += 1
        assert plain_failures >= 1


class TestSnic(TestCase):
    """Saddle-node on an invariant circle: period law, detector, and structure."""

    def _snic(self, mu: float) -> DifferentialEquation:
        parameters = [Alpha(value=mu)]
        return DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=SnicField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )

    def test_period_matches_oracle(self):
        mu, intervals = 1.3, 200
        nodes = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [np.cos(2.0 * math.pi * nodes), np.sin(2.0 * math.pi * nodes)],
        )
        oracle = 2.0 * math.pi / math.sqrt(mu * mu - 1.0)
        orbit = PeriodicOrbit(
            self._snic(mu).derivative,
            intervals,
            phase_index=1,
            phase_value=0.0,
        )
        solution = orbit.solve(seed, oracle * 1.02)
        assert solution is not None
        assert abs(solution.period - oracle) / oracle < 1.0e-3

    def test_detector_identifies_inverse_square_root(self):
        mus = [1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.15, 1.20]
        periods = [2.0 * math.pi / math.sqrt(m * m - 1.0) for m in mus]
        result = characterize_snic(mus, periods)
        assert result.is_inverse_square_root
        assert abs(result.critical_parameter - 1.0) < 1.0e-2
        assert abs(result.exponent + 0.5) < 0.08

    def test_detector_rejects_logarithmic_divergence(self):
        mus = [1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.15, 1.20]
        periods = [-3.0 * math.log(m - 1.0) for m in mus]
        assert not characterize_snic(mus, periods).is_inverse_square_root

    def test_saddle_node_lies_on_the_circle(self):
        seeds = [
            np.array([0.6, 0.8]),
            np.array([-0.6, 0.8]),
            np.array([0.6, -0.8]),
        ]
        equilibria = find_equilibria(self._snic(0.8).derivative, seeds)
        assert len(equilibria) == 2
        labels = sorted(
            classify_equilibrium(_jacobian(self._snic(0.8).derivative, point))
            for point in equilibria
        )
        assert labels == ["saddle", "stable node"]
        for point in equilibria:
            assert abs(float(np.hypot(point[0], point[1])) - 1.0) < 1.0e-6


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


class TestShilnikovContinuation(TestCase):
    """Two-parameter continuation of a saddle-focus homoclinic through a Belyakov point.

    There is no closed-form oracle for a saddle-focus homoclinic locus, so this is
    validated by convergent structural evidence: the located curve is smooth and
    monotone, the equilibrium is a saddle-focus along it, and the saddle index
    crosses one (the Belyakov transition from Shilnikov chaos to a tame homoclinic).
    """

    def _jerk(self, a: float, b: float) -> TwoParameterJerkField:
        return TwoParameterJerkField(
            variables=[State(), State(), State()],
            parameters=[Alpha(value=a), BetaOne(value=b)],
            results=[State(), State(), State()],
            time=None,
        )

    def _jacobian(self, a: float, b: float) -> np.ndarray:
        return np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -b, -a]])

    def test_traces_saddle_focus_homoclinic_through_belyakov(self):
        shooter = HomoclinicShooting(
            self._jerk(0.5, 0.9),
            0,
            Departure(np.zeros(3), np.array([1.0, 0.0, 0.0])),
            ReturnSettings(dt=0.002, horizon=100.0, departure_radius=0.6),
        )
        seed = shooter.locate(0.45, 0.58)
        curve = shooter.trace(1, [0.90, 0.85, 0.80, 0.75, 0.70], seed, 0.07)
        controls = [b for b, _ in curve]
        located = [a for _, a in curve]
        assert controls == [0.90, 0.85, 0.80, 0.75, 0.70]
        # smooth, monotone locus: a increases as b decreases
        for earlier, later in itertools.pairwise(located):
            assert later > earlier
        # every point is a genuine saddle-focus
        for b, a in curve:
            assert classify_saddle_focus(self._jacobian(a, b)) is not None
        # the saddle index crosses one along the curve: Shilnikov -> tame (Belyakov)
        top = classify_saddle_focus(self._jacobian(located[0], controls[0]))
        bottom = classify_saddle_focus(self._jacobian(located[-1], controls[-1]))
        assert top.saddle_index < 1.0
        assert bottom.saddle_index > 1.0


class TestShilnikovHomoclinic(TestCase):
    """The full pipeline: locate a saddle-focus homoclinic and confirm its chaos."""

    def _jerk(self, a: float) -> JerkField:
        return JerkField(
            variables=[State(), State(), State()],
            parameters=[Alpha(value=a)],
            results=[State(), State(), State()],
            time=None,
        )

    def _jacobian(self, a: float) -> np.ndarray:
        return np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -1.0, -a]])

    def test_locates_saddle_focus_homoclinic(self):
        departure = Departure(np.zeros(3), np.array([1.0, 0.0, 0.0]))
        shooter = HomoclinicShooting(
            self._jerk(0.48),
            0,
            departure,
            ReturnSettings(dt=0.002, horizon=100.0, departure_radius=0.6),
        )
        located = shooter.locate(0.46, 0.50)
        assert 0.46 < located < 0.50
        focus = classify_saddle_focus(self._jacobian(located))
        assert focus is not None
        assert focus.satisfies_shilnikov_criterion

    def test_chaos_tracks_the_saddle_index(self):
        settings = LyapunovSettings(dt=0.005, horizon=18000, transient=5000)
        chaotic = lyapunov_spectrum(
            self._jerk(0.50),
            np.array([0.1, 0.0, 0.0]),
            count=1,
            settings=settings,
        ).top
        tame = lyapunov_spectrum(
            self._jerk(0.70),
            np.array([0.1, 0.0, 0.0]),
            count=1,
            settings=settings,
        ).top
        assert chaotic > 0.05
        assert tame < 0.02
        assert classify_saddle_focus(self._jacobian(0.50)).saddle_index < 1.0
        assert classify_saddle_focus(self._jacobian(0.70)).saddle_index > 1.0


class TestHomoclinicShooting(TestCase):
    """Locate a homoclinic orbit by shooting, and seed the boundary-value problem."""

    def _field(self, mu: float) -> DampedWellField:
        return DampedWellField(
            variables=[State(), State()],
            parameters=[Alpha(value=mu)],
            results=[State(), State()],
            time=None,
        )

    def _shooter(self) -> HomoclinicShooting:
        departure = Departure(np.array([0.05, 0.05]), np.array([1.0, 1.0]))
        return HomoclinicShooting(self._field(0.0), 0, departure)

    def test_return_gap_changes_sign_at_homoclinic(self):
        shooter = self._shooter()
        assert shooter.gap(-0.03) > 0.0
        assert shooter.gap(0.03) < 0.0

    def test_locates_homoclinic_parameter(self):
        assert abs(self._shooter().locate(-0.05, 0.05)) < 1.0e-3

    def test_traces_homoclinic_curve_along_melnikov_line(self):
        grid = np.linspace(-40.0, 40.0, 200001)
        loop_x = 1.5 / np.cosh(grid / 2.0) ** 2
        loop_y = -1.5 / np.cosh(grid / 2.0) ** 2 * np.tanh(grid / 2.0)
        first = float(np.trapezoid(loop_y * loop_y, grid))
        second = float(np.trapezoid(loop_x * loop_y * loop_y, grid))
        slope = -second / first
        field = MelnikovField(
            variables=[State(), State()],
            parameters=[Alpha(value=0.0), BetaOne(value=0.0)],
            results=[State(), State()],
            time=None,
        )
        shooter = HomoclinicShooting(
            field,
            0,
            Departure(np.array([0.05, 0.05]), np.array([1.0, 1.0])),
            ReturnSettings(dt=0.002, horizon=120.0, departure_radius=0.6),
        )
        curve = shooter.trace(1, [0.0, 0.05, 0.10, 0.15, 0.20], 0.0, 0.06)
        for control, located in curve:
            assert abs(located - slope * control) < 5.0e-3

    def test_shooting_seed_refines_to_exact_loop(self):
        mu = self._shooter().locate(-0.05, 0.05)
        field = self._field(mu)
        half_length, intervals = 12.0, 240
        times = np.linspace(-half_length, half_length, intervals + 1)
        exact = np.column_stack(
            [
                1.5 / np.cosh(times / 2.0) ** 2,
                -1.5 / np.cosh(times / 2.0) ** 2 * np.tanh(times / 2.0),
            ],
        )
        seed = np.column_stack(
            [1.5 / np.cosh(times / 1.8) ** 2, np.zeros_like(times)],
        )
        jacobian = np.array([[0.0, 1.0], [1.0, mu]])
        solution = HomoclinicOrbit(
            field,
            np.zeros(2),
            jacobian,
            MeshSpec(
                half_length=half_length,
                intervals=intervals,
                phase_index=0,
                phase_value=1.5,
            ),
        ).solve(seed)
        assert np.max(np.linalg.norm(solution.states - exact, axis=1)) < 2.0e-2


class TestThreeDimensionalConnection(TestCase):
    """Complex-eigenspace projection and a three-dimensional homoclinic orbit."""

    def test_complex_projection_constrains_the_spiral(self):
        jacobian = np.array(
            [[-0.3, -2.0, 0.0], [2.0, -0.3, 0.0], [0.0, 0.0, 1.0]],
        )
        values, left, stable, _ = _split_eigenspaces(jacobian)
        in_unstable = _projection_conditions(
            left,
            values,
            stable,
            np.array([0.0, 0.0, 1.0]),
        )
        assert len(in_unstable) == 2
        assert max(abs(value) for value in in_unstable) < 1.0e-9
        mixed = _projection_conditions(
            left,
            values,
            stable,
            np.array([0.5, 0.2, 1.0]),
        )
        assert max(abs(value) for value in mixed) > 1.0e-2

    def test_three_dimensional_homoclinic_matches_exact(self):
        half_length, intervals = 12.0, 240
        times = np.linspace(-half_length, half_length, intervals + 1)
        base = np.column_stack(
            [
                1.5 / np.cosh(times / 2.0) ** 2,
                -1.5 / np.cosh(times / 2.0) ** 2 * np.tanh(times / 2.0),
                np.zeros_like(times),
            ],
        )
        exact = base @ _TILT.T
        seed = (
            np.column_stack(
                [
                    1.2 / np.cosh(times / 2.5) ** 2,
                    np.zeros_like(times),
                    np.zeros_like(times),
                ],
            )
            @ _TILT.T
        )
        jacobian = (
            _TILT
            @ np.array(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -2.0]],
            )
            @ _TILT.T
        )
        field = TiltedHomoclinicField(
            variables=[State(), State(), State()],
            parameters=[],
            results=[State(), State(), State()],
            time=None,
        )
        orbit = HomoclinicOrbit(
            field,
            np.zeros(3),
            jacobian,
            MeshSpec(
                half_length=half_length,
                intervals=intervals,
                phase_index=0,
                phase_value=float(exact[intervals // 2, 0]),
            ),
        )
        solution = orbit.solve(seed)
        assert np.max(np.linalg.norm(solution.states - exact, axis=1)) < 1.0e-2


class TestHeteroclinicOrbit(TestCase):
    """Saddle-to-saddle connection against the exact double-well oracle."""

    def _field(self) -> HeteroclinicField:
        return HeteroclinicField(
            variables=[State(), State()],
            parameters=[],
            results=[State(), State()],
            time=None,
        )

    def _jacobian(self, x: float) -> np.ndarray:
        return np.array([[0.0, 1.0], [-1.0 + 3.0 * x * x, 0.0]])

    def test_converges_to_exact_tanh_connection(self):
        half_length, intervals = 8.0, 200
        times = np.linspace(-half_length, half_length, intervals + 1)
        exact = np.column_stack(
            [
                np.tanh(times / np.sqrt(2.0)),
                (1.0 / np.sqrt(2.0)) / np.cosh(times / np.sqrt(2.0)) ** 2,
            ],
        )
        seed = np.column_stack([np.tanh(times / 2.0), np.zeros_like(times)])
        source = Terminus(np.array([-1.0, 0.0]), self._jacobian(-1.0))
        target = Terminus(np.array([1.0, 0.0]), self._jacobian(1.0))
        orbit = HeteroclinicOrbit(
            self._field(),
            source,
            target,
            MeshSpec(half_length=half_length, intervals=intervals),
        )
        solution = orbit.solve(seed)
        error = np.max(np.linalg.norm(solution.states - exact, axis=1))
        assert error < 1.0e-3

    def test_lands_on_both_saddles(self):
        half_length, intervals = 8.0, 200
        times = np.linspace(-half_length, half_length, intervals + 1)
        seed = np.column_stack([np.tanh(times / 2.0), np.zeros_like(times)])
        source = Terminus(np.array([-1.0, 0.0]), self._jacobian(-1.0))
        target = Terminus(np.array([1.0, 0.0]), self._jacobian(1.0))
        solution = HeteroclinicOrbit(
            self._field(),
            source,
            target,
            MeshSpec(half_length=half_length, intervals=intervals),
        ).solve(seed)
        assert np.linalg.norm(solution.states[0] - np.array([-1.0, 0.0])) < 1.0e-3
        assert np.linalg.norm(solution.states[-1] - np.array([1.0, 0.0])) < 1.0e-3


class TestDeflatedSolver(TestCase):
    """Deflated Newton finds distinct roots; matrix-free matches the direct solve."""

    def _cubic(self) -> CubicField:
        return CubicField(
            variables=[State()],
            parameters=[],
            results=[State()],
            time=None,
        )

    def _grid(self) -> GridField:
        return GridField(
            variables=[State(), State()],
            parameters=[],
            results=[State(), State()],
            time=None,
        )

    def test_finds_all_cubic_roots_from_one_seed(self):
        roots = DeflatedSolver(self._cubic()).find([np.array([0.5])])
        recovered = sorted(round(float(r[0]), 4) for r in roots)
        assert recovered == [-1.0, 0.0, 1.0]

    def test_finds_all_nine_grid_roots(self):
        seeds = [
            np.array([0.3, 0.4]),
            np.array([-0.6, 0.2]),
            np.array([0.2, -0.7]),
        ]
        roots = DeflatedSolver(self._grid()).find(seeds)
        assert len(roots) == 9
        field = self._grid()
        for root in roots:
            residual = np.linalg.norm(np.array(field.eval(point=list(root), time=None)))
            assert residual < 1.0e-6

    def test_matrix_free_matches_direct(self):
        seeds = [
            np.array([0.3, 0.4]),
            np.array([-0.6, 0.2]),
            np.array([0.2, -0.7]),
        ]
        direct = DeflatedSolver(self._grid()).find(seeds)
        krylov = DeflatedSolver(self._grid(), matrix_free=True).find(seeds)
        assert len(krylov) == len(direct) == 9

        def as_set(roots: list[np.ndarray]) -> set[tuple[float, float]]:
            return {(round(float(r[0])), round(float(r[1]))) for r in roots}

        assert as_set(krylov) == as_set(direct)


class TestSymmetryReduction(TestCase):
    """Cyclic-symmetry (Fourier) reduction of the ring linearisation."""

    def _fluxgate(self, size: int, coupling: float, gain: float) -> FluxgateRing:
        parameters = [Alpha(value=coupling), BetaOne(value=gain), BetaTwo(value=0.0)]
        return FluxgateRing(
            variables=[State() for _ in range(size)],
            parameters=parameters,
            results=[State() for _ in range(size)],
            time=None,
        )

    def _ferroelectric(
        self,
        size: int,
        coupling: float,
        gain: float,
    ) -> FerroelectricRing:
        parameters = [Alpha(value=coupling), BetaOne(value=gain), BetaTwo(value=0.0)]
        return FerroelectricRing(
            variables=[State() for _ in range(size)],
            parameters=parameters,
            results=[State() for _ in range(size)],
            time=None,
        )

    def test_ring_is_cyclically_equivariant(self):
        field = self._fluxgate(5, 0.4, 0.3)
        point = np.array([0.1, -0.2, 0.3, 0.05, -0.15])
        assert equivariance_defect(field, cyclic_action(5), point) < 1.0e-9

    def test_fluxgate_breaks_symmetry_steadily(self):
        coupling, gain, size = 0.4, 0.3, 5
        reduction = fourier_reduce(
            _jacobian(self._fluxgate(size, coupling, gain), np.zeros(size)),
        )
        assert reduction.is_circulant
        modes = np.exp(2j * np.pi * np.arange(size) / size)
        oracle = (gain - 1.0) + coupling * modes
        assert np.allclose(
            np.sort(reduction.eigenvalues.real),
            np.sort(oracle.real),
            atol=1.0e-5,
        )
        assert reduction.critical_mode == 0
        assert not reduction.is_oscillatory

    def test_ferroelectric_hopf_is_a_complex_mode(self):
        gain, size = -0.3, 3
        coupling = -2.0 * gain  # Hopf at lambda = -2a
        reduction = fourier_reduce(
            _jacobian(self._ferroelectric(size, coupling, gain), np.zeros(size)),
        )
        assert reduction.is_circulant
        assert reduction.is_oscillatory
        assert abs(reduction.critical_eigenvalue.real) < 1.0e-3


class TestSaddleFocus(TestCase):
    """Saddle-focus detection and the Shilnikov saddle index."""

    def _block(self, spiral_real: float, frequency: float, real: float) -> np.ndarray:
        return np.array(
            [
                [spiral_real, -frequency, 0.0],
                [frequency, spiral_real, 0.0],
                [0.0, 0.0, real],
            ],
        )

    def test_saddle_index_matches_prescribed(self):
        chaotic = classify_saddle_focus(self._block(-0.5, 3.0, 1.0))
        assert chaotic is not None
        assert abs(chaotic.saddle_index - 0.5) < 1.0e-9
        assert chaotic.satisfies_shilnikov_criterion
        tame = classify_saddle_focus(self._block(-2.0, 3.0, 1.0))
        assert tame is not None
        assert abs(tame.saddle_index - 2.0) < 1.0e-9
        assert not tame.satisfies_shilnikov_criterion

    def test_saddle_index_is_similarity_invariant(self):
        generator = np.random.default_rng(0)
        change = generator.standard_normal((3, 3))
        transformed = change @ self._block(-0.5, 3.0, 1.0) @ np.linalg.inv(change)
        result = classify_saddle_focus(transformed)
        assert result is not None
        assert abs(result.saddle_index - 0.5) < 1.0e-8

    def test_rejects_real_saddle(self):
        assert classify_saddle_focus(np.diag([1.0, -1.0, -2.0])) is None

    def test_lorenz_equilibria_are_shilnikov_saddle_foci(self):
        field = LorenzField(
            variables=[State(), State(), State()],
            parameters=[],
            results=[State(), State(), State()],
            time=None,
        )
        seeds = [np.array([8.5, 8.5, 27.0]), np.array([-8.5, -8.5, 27.0])]
        equilibria = find_equilibria(field, seeds)
        assert len(equilibria) == 2
        for point in equilibria:
            focus = classify_saddle_focus(_jacobian(field, point))
            assert focus is not None
            assert focus.satisfies_shilnikov_criterion


class TestStochasticLyapunov(TestCase):
    """Simulated Lyapunov exponent of a noisy flow and the stochastic D-bifurcation."""

    def _system(self, convention, alpha: float, sigma: float) -> StochasticSystem:
        return StochasticSystem(
            _scalar(Drift, alpha, sigma),
            _scalar(Diffusion, alpha, sigma),
            convention,
        )

    def _top(self, convention, alpha: float, sigma: float) -> float:
        settings = StochasticLyapunovSettings(
            dt=0.005,
            horizon=80000,
            transient=0,
            seed=7,
        )
        return stochastic_lyapunov(
            self._system(convention, alpha, sigma),
            np.array([0.0]),
            settings=settings,
        ).top

    def test_convention_shift_is_half_diffusion_squared(self):
        # Stratonovich minus Ito equals s^2 / 2 exactly (shared Brownian path cancels).
        alpha, sigma = 0.3, 0.6
        difference = self._top(Stratonovich(), alpha, sigma) - self._top(
            Ito(),
            alpha,
            sigma,
        )
        assert abs(difference - 0.5 * sigma * sigma) < 2.0e-2

    def test_ito_matches_closed_form(self):
        alpha, sigma = 0.3, 0.6
        oracle = LyapunovExponent(
            _scalar(Drift, alpha, sigma),
            _scalar(Diffusion, alpha, sigma),
            Ito(),
        ).at(0.0)
        assert abs(self._top(Ito(), alpha, sigma) - oracle) < 5.0e-2

    def test_noise_shifts_dynamical_bifurcation(self):
        # Deterministically neutral (alpha = 0) but stabilised by noise; unstable above.
        sigma = 0.6
        assert self._top(Ito(), 0.0, sigma) < 0.0
        assert self._top(Ito(), 2.0 * 0.5 * sigma * sigma + 0.18, sigma) > 0.0

    def test_zero_noise_recovers_drift(self):
        assert abs(self._top(Ito(), 0.2, 0.0) - 0.2) < 1.0e-2

    def test_spectrum_on_noisy_lorenz_attractor(self):
        # Full spectrum on a bounded chaotic attractor under additive noise. The sum
        # equals the exact phase-space divergence -(10 + 1 + 8/3); the top stays
        # positive (chaos persists) and the middle exponent is the neutral flow.
        divergence = -(10.0 + 1.0 + 8.0 / 3.0)
        settings = StochasticLyapunovSettings(
            dt=0.005,
            horizon=25000,
            transient=5000,
            seed=7,
        )
        for sigma in (0.0, 0.3):
            system = StochasticSystem(
                LorenzField(
                    variables=[State(), State(), State()],
                    parameters=[Alpha(value=sigma)],
                    results=[State(), State(), State()],
                    time=None,
                ),
                ConstantDiffusion(
                    variables=[State(), State(), State()],
                    parameters=[Alpha(value=sigma)],
                    results=[State(), State(), State()],
                    time=None,
                ),
                Ito(),
            )
            spectrum = stochastic_lyapunov(
                system,
                np.array([1.0, 1.0, 1.0]),
                count=3,
                settings=settings,
            ).exponents
            assert spectrum[0] > 0.6
            assert abs(spectrum[1]) < 0.1
            assert abs(float(spectrum.sum()) - divergence) < 0.5


class TestStochasticBifurcation(TestCase):
    def test_phenomenological_threshold(self):
        sigma = 0.6
        grid = np.linspace(1.0e-3, 3.0, 3000)
        drift = _scalar(Drift, 0.0, sigma)
        diffusion = _scalar(Diffusion, 0.0, sigma)
        below = self._dominant_mode(drift, diffusion, grid, 0.10)
        above = self._dominant_mode(drift, diffusion, grid, 0.30)
        assert below < 0.05
        assert above > 0.05
        threshold = self._mode_threshold(drift, diffusion, grid)
        assert abs(threshold - sigma * sigma / 2.0) < 1.0e-2

    def test_dynamical_threshold(self):
        sigma = 0.6
        drift = _scalar(Drift, 0.0, sigma)
        diffusion = _scalar(Diffusion, 0.0, sigma)
        exponent = LyapunovExponent(drift, diffusion, Stratonovich())
        for alpha in (-0.2, 0.0, 0.25):
            drift.parameters[0].value = alpha
            assert abs(exponent.at(0.0) - alpha) < 1.0e-4

    def _dominant_mode(self, drift, diffusion, grid, alpha):
        drift.parameters[0].value = alpha
        xs, density = StationaryDensity(
            drift,
            diffusion,
            Stratonovich(),
            grid,
        ).evaluate()
        return DensityModes(xs, density).dominant_mode()

    def _mode_threshold(self, drift, diffusion, grid):
        low, high = 0.0, 0.5
        for _ in range(30):
            mid = 0.5 * (low + high)
            if self._dominant_mode(drift, diffusion, grid, mid) > 0.02:
                high = mid
            else:
                low = mid
        return 0.5 * (low + high)


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
