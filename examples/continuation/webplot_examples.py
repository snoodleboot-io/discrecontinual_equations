"""Generate self-contained D3 HTML plots for continuation results.

Run with ``python -m examples.continuation.webplot_examples [output_dir]``. It
continues equilibrium branches and their codim-1, codim-2, and codim-3
bifurcations, renders each with the D3 renderer using the system's own parameter
notation, and writes an atlas linking them.
"""

import math
import sys
from pathlib import Path

import numpy as np
from sweet_tea.registry import Registry

import discrecontinual_equations
from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.codim2_builder import (
    Codim2Driver,
    CurveSeed,
)
from discrecontinual_equations.continuation.codim2_config import Codim2Config
from discrecontinual_equations.continuation.codim3_builder import (
    Codim3Driver,
    Codim3Seed,
)
from discrecontinual_equations.continuation.codim3_config import Codim3Config
from discrecontinual_equations.continuation.connecting_orbit import (
    HeteroclinicOrbit,
    HomoclinicOrbit,
    MeshSpec,
    Terminus,
)
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CycleSeed,
)
from discrecontinual_equations.continuation.deflation import DeflatedSolver
from discrecontinual_equations.continuation.exploration import (
    BifurcationExplorer,
    ExplorationConfig,
)
from discrecontinual_equations.continuation.family_builder import FamilyDriver
from discrecontinual_equations.continuation.family_config import FamilyConfig
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
    HermiteSimpsonOrbit,
    PeriodicOrbit,
)
from discrecontinual_equations.continuation.region_analysis import describe_region
from discrecontinual_equations.continuation.stochastic import (
    Ito,
    LyapunovExponent,
    StationaryDensity,
    Stratonovich,
)
from discrecontinual_equations.continuation.symmetry import fourier_reduce
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.systems import (
    FerroelectricRing,
    FluxgateRing,
)
from discrecontinual_equations.variable import Variable
from discrecontinual_equations.webplot.renderer import D3Renderer
from discrecontinual_equations.webplot.report import AtlasEntry, PlotReport
from discrecontinual_equations.webplot.scene import Scene
from discrecontinual_equations.webplot.scene_builder import (
    bifurcation_scene,
    codim2_scene,
    codim3_scene,
    curve_family_scene,
    curve_scene,
    exploration_diagram,
    exploration_skeleton,
    exploration_tree,
    family_scene,
    surface_scene,
    with_equation,
    with_regions,
)
from discrecontinual_equations.webplot.stage_renderer import StageRenderer

try:  # python -m examples.continuation.webplot_examples
    from examples.continuation import stage_bogdanov_takens as bt
    from examples.continuation import stage_ferroelectric_ring as ring
except ImportError:  # run as a script path: only this directory is on sys.path
    import stage_bogdanov_takens as bt
    import stage_ferroelectric_ring as ring

MU = "\u03bc"
ALPHA = "\u03b1"
LAMBDA = "\u03bb"
_FOLD_WINDOW = 0.02
_RADIUS_FLOOR = 1.0e-9
_LOOP_ESCAPE = 50.0
_LOOP_DEPARTURE = 0.6
_LOOP_RETURN = 0.25
_IMAG_TOLERANCE = 1.0e-9
_MODE_IMAG = 1.0e-6
_PERIOD_TOLERANCE = 1.0e-2
BETA1, BETA2, BETA3 = "\u03b2\u2081", "\u03b2\u2082", "\u03b2\u2083"
NORM = "\u2016x\u2016"
_IMAGINARY_TOLERANCE = 1.0e-9


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


def _saddle_node() -> Scene:
    equation = _equation(SaddleNode, 1, [First(value=0.0)])
    config = ContinuationConfig(
        detectors=["fold"],
        initial_parameter=1.2,
        direction=-1,
        measure="component",
        parameter_lower_bound=-1.0,
        parameter_upper_bound=1.4,
    )
    branch = _solve(equation, config, [1.2**0.5])
    return bifurcation_scene(
        branch,
        "Saddle-node (fold)",
        "x' = " + MU + " - x^2. Stable and unstable branches meet at the fold.",
        MU,
        "x",
    )


def _transcritical() -> Scene:
    equation = _equation(Transcritical, 1, [First(value=0.0)])
    config = ContinuationConfig(
        detectors=["branch_point"],
        initial_parameter=1.0,
        direction=-1,
        measure="component",
        parameter_lower_bound=-1.2,
        parameter_upper_bound=1.2,
    )
    branch = _solve(equation, config, [1.0])
    return bifurcation_scene(
        branch,
        "Transcritical",
        "x' = " + MU + " x - x^2. Two branches cross and exchange stability.",
        MU,
        "x",
    )


def _pitchfork() -> Scene:
    equation = _equation(Pitchfork, 1, [First(value=0.0)])
    config = ContinuationConfig(
        detectors=["branch_point"],
        initial_parameter=-1.2,
        direction=1,
        measure="component",
        parameter_lower_bound=-1.2,
        parameter_upper_bound=1.2,
    )
    branch = _solve(equation, config, [0.0])
    return bifurcation_scene(
        branch,
        "Pitchfork",
        "x' = " + MU + " x - x^3. A symmetric pair splits off the trivial branch; "
        "the quadratic coefficient vanishes so it is not a transcritical crossing.",
        MU,
        "x",
    )


def _hysteresis() -> Scene:
    equation = _equation(Hysteresis, 1, [First(value=0.0)])
    config = ContinuationConfig(
        detectors=["fold"],
        initial_parameter=3.0,
        direction=-1,
        measure="component",
        parameter_lower_bound=-3.2,
        parameter_upper_bound=3.2,
    )
    branch = _solve(equation, config, [2.1038])
    return bifurcation_scene(
        branch,
        "Hysteresis (two folds)",
        "x' = " + MU + " + 3x - x^3. An S-shaped branch with two folds.",
        MU,
        "x",
    )


def _collocation_convergence() -> Scene:
    mu = 0.5

    def period_error(solver_type, intervals: int) -> float:
        field = _equation(Hopf, 2, [First(value=mu)]).derivative
        nodes = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [
                math.sqrt(mu) * np.cos(2.0 * np.pi * nodes),
                math.sqrt(mu) * np.sin(2.0 * np.pi * nodes),
            ],
        )
        solution = solver_type(
            field,
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed, 6.0)
        return abs(solution.period - 2.0 * np.pi)

    counts = [20, 40, 80, 160]
    lines = []
    for solver_type, kind, label in (
        (PeriodicOrbit, "unstable", "trapezoidal (order 2)"),
        (HermiteSimpsonOrbit, "stable", "Hermite-Simpson (order 4)"),
    ):
        points = [
            (math.log10(n), math.log10(period_error(solver_type, n))) for n in counts
        ]
        lines.append((kind, label, points, None))
    return curve_scene(
        lines,
        [],
        ("log10 intervals", "log10 period error"),
        "Collocation order: trapezoidal vs Hermite-Simpson",
        "Period error of the exact Hopf cycle (T = 2 pi) against node count, on log "
        "axes. The slopes are -2 and -4: Hermite-Simpson gains two orders, reaching "
        "the same accuracy with far fewer nodes.",
    )


def _deflated_roots() -> Scene:
    field = _equation(GridSystem, 2, []).derivative
    seeds = [
        np.array([0.3, 0.4]),
        np.array([-0.6, 0.2]),
        np.array([0.2, -0.7]),
    ]
    roots = DeflatedSolver(field).find(seeds)
    markers = [
        (
            "equilibrium",
            "roots found",
            [(float(r[0]), float(r[1])) for r in roots],
            None,
        ),
        ("saddle", "seeds", [(float(s[0]), float(s[1])) for s in seeds], None),
    ]
    return curve_scene(
        [],
        markers,
        ("x", "y"),
        "Deflated Newton: distinct roots from a few seeds",
        "Deflation repels Newton from roots already found, so a handful of seeds "
        "uncover all nine equilibria of the decoupled quartic gradient - "
        "disconnected branches an ordinary solver would miss.",
    )


def _homoclinic_curve_shooting() -> Scene:
    field = _equation(
        MelnikovSystem,
        2,
        [First(value=0.0), Second(value=0.0)],
    ).derivative
    shooter = HomoclinicShooting(
        field,
        0,
        Departure(np.array([0.05, 0.05]), np.array([1.0, 1.0])),
        ReturnSettings(dt=0.003, horizon=70.0, departure_radius=0.6),
    )
    upward = shooter.trace(1, [0.05 * k for k in range(3)], 0.0, 0.06)
    downward = shooter.trace(1, [-0.05, -0.10], 0.0, 0.06)
    points = sorted(
        [(nu, mu) for nu, mu in downward] + [(nu, mu) for nu, mu in upward],
    )
    lines = [("stable", "homoclinic curve", points, None)]
    markers = [("bifurcation", "unperturbed loop", [(0.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        ("nu", MU),
        "Homoclinic curve traced by shooting",
        "The locus of saddle homoclinic orbits in the two-parameter plane, each "
        "point located by shooting the unstable manifold. It follows the Melnikov "
        "line mu I1 + nu I2 = 0 through the unperturbed loop at the origin.",
    )


def _symmetry_modes() -> Scene:
    gain, size = -0.3, 3

    def jacobian(coupling: float) -> np.ndarray:
        ring = FerroelectricRing(
            variables=[State() for _ in range(size)],
            parameters=[
                First(value=coupling),
                Second(value=gain),
                Third(value=0.0),
            ],
            results=[State() for _ in range(size)],
            time=None,
        )
        base = np.array(ring.eval(point=[0.0] * size, time=None))
        matrix = np.zeros((size, size))
        for j in range(size):
            shifted = [0.0] * size
            shifted[j] = 1.0e-7
            matrix[:, j] = (
                np.array(ring.eval(point=shifted, time=None)) - base
            ) / 1.0e-7
        return matrix

    steady: list[tuple[float, float]] = []
    oscillatory: list[tuple[float, float]] = []
    for step in range(29):
        coupling = -0.6 + 0.05 * step
        reduction = fourier_reduce(jacobian(coupling))
        for value in reduction.eigenvalues:
            if abs(value.imag) < _MODE_IMAG:
                steady.append((coupling, float(value.real)))
            else:
                oscillatory.append((coupling, float(value.real)))
                break
    lines = [
        ("stable", "steady mode (k = 0)", steady, None),
        ("unstable", "wave mode (k = 1, 2)", oscillatory, None),
    ]
    markers = [
        ("bifurcation", "symmetry breaking", [(gain, 0.0)], None),
        ("bifurcation", "Hopf (rotating wave)", [(-2.0 * gain, 0.0)], None),
    ]
    return curve_scene(
        lines,
        markers,
        (LAMBDA, "mode eigenvalue real part"),
        "Symmetry reduction of a ring: Fourier-mode stability",
        "The circulant linearisation of the three-cell ferroelectric ring, "
        "block-diagonalised by the Fourier basis. The real k=0 mode gives steady "
        "symmetry breaking at lambda = a; the complex k=1,2 modes give a rotating "
        "wave through a Hopf bifurcation at lambda = -2a.",
    )


def _shilnikov_loop() -> Scene:
    field = _equation(Jerk, 3, [First(value=0.48)]).derivative
    jacobian = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -1.0, -0.48]])
    values, vectors = np.linalg.eig(jacobian)
    unstable = vectors[:, int(np.argmax(values.real))].real
    unstable = unstable / np.linalg.norm(unstable)
    if unstable[0] < 0.0:
        unstable = -unstable
    state = 1.0e-6 * unstable
    dt = 0.005
    path: list[tuple[float, float]] = [(float(state[0]), float(state[2]))]
    left = False
    for _ in range(40000):
        values_k1 = np.array(field.eval(point=list(state), time=None))
        mid = state + 0.5 * dt * values_k1
        values_k2 = np.array(field.eval(point=list(mid), time=None))
        mid2 = state + 0.5 * dt * values_k2
        values_k3 = np.array(field.eval(point=list(mid2), time=None))
        end = state + dt * values_k3
        values_k4 = np.array(field.eval(point=list(end), time=None))
        state = state + dt / 6.0 * (
            values_k1 + 2.0 * values_k2 + 2.0 * values_k3 + values_k4
        )
        distance = float(np.linalg.norm(state))
        if distance > _LOOP_ESCAPE:
            break
        if distance > _LOOP_DEPARTURE:
            left = True
        path.append((float(state[0]), float(state[2])))
        if left and distance < _LOOP_RETURN:
            break
    lines = [("unstable", "unstable manifold", path, None)]
    markers = [("saddle", "saddle-focus", [(0.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        ("x", "z"),
        "Shilnikov saddle-focus homoclinic loop",
        "The one-dimensional unstable manifold of the saddle-focus, located by "
        "shooting at a = 0.48, leaves along a line and spirals back toward the "
        "origin along the two-dimensional stable manifold. Its saddle index is below "
        "one, so nearby dynamics are chaotic.",
    )


def _homoclinic_return() -> Scene:
    field = _equation(DampedWell, 2, [First(value=0.0)]).derivative
    departure = Departure(np.array([0.05, 0.05]), np.array([1.0, 1.0]))
    shooter = HomoclinicShooting(field, 0, departure)
    points: list[tuple[float, float]] = []
    for step in range(11):
        mu = -0.05 + 0.01 * step
        gap = shooter.gap(mu)
        if not math.isnan(gap):  # skip NaN (diverged before returning)
            points.append((mu, gap))
    lines = [("stable", "return gap", points, None)]
    markers = [("bifurcation", "homoclinic at mu = 0", [(0.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        (MU, "return gap"),
        "Locating a homoclinic orbit by shooting",
        "The unstable manifold is integrated and its signed miss on return to the "
        "saddle is recorded. The gap crosses zero at mu = 0, where the homoclinic "
        "orbit exists; a bracketing search converges to it.",
    )


def _three_dimensional_homoclinic() -> Scene:
    tilt = _homoclinic_tilt()
    half_length, intervals = 12.0, 240
    times = np.linspace(-half_length, half_length, intervals + 1)
    base = np.column_stack(
        [
            1.5 / np.cosh(times / 2.0) ** 2,
            -1.5 / np.cosh(times / 2.0) ** 2 * np.tanh(times / 2.0),
            np.zeros_like(times),
        ],
    )
    exact = base @ tilt.T
    seed = (
        np.column_stack(
            [
                1.2 / np.cosh(times / 2.5) ** 2,
                np.zeros_like(times),
                np.zeros_like(times),
            ],
        )
        @ tilt.T
    )
    jacobian = (
        tilt @ np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -2.0]]) @ tilt.T
    )
    field = _equation(TiltedHomoclinic, 3, []).derivative
    mesh = MeshSpec(
        half_length=half_length,
        intervals=intervals,
        phase_index=0,
        phase_value=float(exact[intervals // 2, 0]),
    )
    solution = HomoclinicOrbit(field, np.zeros(3), jacobian, mesh).solve(seed)
    path = list(zip(solution.component(0), solution.component(2), strict=True))
    lines = [("unstable", "homoclinic loop", path, None)]
    markers = [("saddle", "saddle", [(0.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        ("x", "z"),
        "Three-dimensional homoclinic orbit",
        "A homoclinic loop to a saddle with a two-dimensional stable eigenspace, "
        "solved by the projection boundary-value problem and shown projected onto the "
        "x-z plane. The same machinery projects onto complex (spiral) eigenspaces.",
    )


def _heteroclinic_cycle() -> Scene:
    field = _equation(DoubleWell, 2, []).derivative
    half_length, intervals = 8.0, 200
    times = np.linspace(-half_length, half_length, intervals + 1)
    mesh = MeshSpec(half_length=half_length, intervals=intervals)

    def jacobian(x: float) -> np.ndarray:
        return np.array([[0.0, 1.0], [-1.0 + 3.0 * x * x, 0.0]])

    left = Terminus(np.array([-1.0, 0.0]), jacobian(-1.0))
    right = Terminus(np.array([1.0, 0.0]), jacobian(1.0))
    upper = HeteroclinicOrbit(field, left, right, mesh).solve(
        np.column_stack([np.tanh(times / 2.0), np.zeros_like(times)]),
    )
    lower = HeteroclinicOrbit(field, right, left, mesh).solve(
        np.column_stack([-np.tanh(times / 2.0), np.zeros_like(times)]),
    )
    lines = [
        (
            "unstable",
            "connection -1 to +1",
            list(zip(upper.component(0), upper.component(1), strict=True)),
            None,
        ),
        (
            "stable",
            "connection +1 to -1",
            list(zip(lower.component(0), lower.component(1), strict=True)),
            None,
        ),
    ]
    markers = [("saddle", "saddles", [(-1.0, 0.0), (1.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        ("x", "y"),
        "Heteroclinic cycle between two saddles",
        "Two saddle-to-saddle connections, each solved by a projection "
        "boundary-value problem, close into the double-well separatrix loop joining "
        "the saddles at (-1, 0) and (+1, 0).",
    )


def _saddle_focus_onset() -> Scene:
    sigma, beta = 10.0, 8.0 / 3.0
    stable: list[tuple[float, float]] = []
    focus: list[tuple[float, float]] = []
    for step in range(28):
        rho = 20.0 + 0.5 * step
        coordinate = math.sqrt(beta * (rho - 1.0))
        jacobian = np.array(
            [
                [-sigma, sigma, 0.0],
                [rho - (rho - 1.0), -1.0, -coordinate],
                [coordinate, coordinate, -beta],
            ],
        )
        eigenvalues = np.linalg.eigvals(jacobian)
        spiral = max(
            value.real for value in eigenvalues if abs(value.imag) > _IMAG_TOLERANCE
        )
        (focus if spiral > 0.0 else stable).append((rho, spiral))
    lines = [
        ("stable", "stable focus (Re < 0)", stable, None),
        ("unstable", "saddle-focus (Re > 0)", focus, None),
    ]
    hopf = sigma * (sigma + beta + 3.0) / (sigma - beta - 1.0)
    markers = [("bifurcation", "Hopf: saddle-focus onset", [(hopf, 0.0)], None)]
    scene = curve_scene(
        lines,
        markers,
        ("rho", "spiral eigenvalue real part"),
        "Shilnikov saddle-focus onset in the Lorenz equilibria",
        "Real part of the complex eigenvalue pair at the Lorenz C-plus equilibrium. "
        "It crosses zero at the subcritical Hopf point rho = 24.74, beyond which the "
        "equilibrium is a saddle-focus whose homoclinic loops drive chaos.",
    )
    return with_regions(
        scene,
        [("saddle-focus: chaos region", 25.5, focus[-1][1])],
    )


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


_VDP_STEP = 0.001
_VDP_SETTLE = 50.0


def _van_der_pol_cycle(mu: float, intervals: int):
    def rhs(state: np.ndarray) -> np.ndarray:
        x, y = state
        return np.array([y, mu * (1.0 - x * x) * y - x])

    def step(state: np.ndarray) -> np.ndarray:
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * _VDP_STEP * k1)
        k3 = rhs(state + 0.5 * _VDP_STEP * k2)
        k4 = rhs(state + _VDP_STEP * k3)
        return state + _VDP_STEP / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    state = np.array([2.0, 0.0])
    for _ in range(int(_VDP_SETTLE / _VDP_STEP)):
        state = step(state)
    while True:
        nxt = step(state)
        if state[1] < 0.0 <= nxt[1]:
            state = nxt
            break
        state = nxt
    times = [0.0]
    trajectory = [state.copy()]
    current = state.copy()
    clock = 0.0
    while True:
        current = step(current)
        clock += _VDP_STEP
        trajectory.append(current.copy())
        times.append(clock)
        if trajectory[-2][1] < 0.0 <= current[1] and clock > 1.0:
            break
    trajectory = np.array(trajectory)
    times = np.array(times)
    period = times[-1]
    sample = np.linspace(0.0, period, intervals + 1)
    seed = np.column_stack(
        [
            np.interp(sample, times, trajectory[:, 0]),
            np.interp(sample, times, trajectory[:, 1]),
        ],
    )
    return period, seed, trajectory


def _adaptive_mesh() -> Scene:
    mu, intervals = 5.0, 120
    field = _equation(VanDerPol, 2, [First(value=mu)]).derivative
    period, seed, trajectory = _van_der_pol_cycle(mu, intervals)
    solution = AdaptivePeriodicOrbit(
        field,
        intervals,
        phase_index=1,
        phase_value=0.0,
    ).solve(seed, period)
    cycle = [(float(p[0]), float(p[1])) for p in trajectory[::20]]
    nodes = [(float(s[0]), float(s[1])) for s in solution.states]
    lines = [("stable", "limit cycle", cycle, None)]
    markers = [("equilibrium", "adapted nodes", nodes, None)]
    return curve_scene(
        lines,
        markers,
        ("x", "y"),
        "Adaptive mesh on a van der Pol relaxation oscillation",
        "The limit cycle of van der Pol at mu = 5, with the solver's adapted nodes. "
        "Equidistributing a curvature monitor concentrates nodes on the fast jumps "
        "and spares the slow branches - the mesh a uniform time grid cannot supply.",
    )


def _stochastic_attractor() -> Scene:
    def top(sigma: float) -> float:
        drift = _LorenzDrift(
            variables=[State(), State(), State()],
            parameters=[],
            results=[State(), State(), State()],
            time=None,
        )
        noise = _LorenzNoise(
            variables=[State(), State(), State()],
            parameters=[First(value=sigma)],
            results=[State(), State(), State()],
            time=None,
        )
        settings = StochasticLyapunovSettings(
            dt=0.005,
            horizon=12000,
            transient=4000,
            seed=7,
        )
        system = StochasticSystem(drift, noise, Ito())
        return stochastic_lyapunov(
            system,
            np.array([1.0, 1.0, 1.0]),
            count=1,
            settings=settings,
        ).top

    points = [(0.5 * k, top(0.5 * k)) for k in range(5)]
    lines = [("unstable", "top Lyapunov exponent", points, None)]
    markers = [("bifurcation", "chaos threshold", [(0.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        ("noise amplitude", "top Lyapunov exponent"),
        "Chaos persists under noise: Lorenz top Lyapunov vs noise",
        "Top Lyapunov exponent of the Lorenz attractor under additive noise, from "
        "the stochastic Benettin estimator. It stays near the deterministic value "
        "and well above zero, so the bounded chaotic attractor survives the noise.",
    )


def _stochastic_bifurcation() -> Scene:
    settings = StochasticLyapunovSettings(dt=0.005, horizon=80000, transient=0, seed=7)
    curves: list[tuple[str, str, list[tuple[float, float]], str | None]] = []
    for sigma, kind, label in (
        (0.0, "stable", "no noise (s = 0)"),
        (0.6, "unstable", "noise s = 0.6"),
    ):
        points: list[tuple[float, float]] = []
        for step in range(13):
            alpha = -0.1 + 0.03 * step
            system = StochasticSystem(
                _stochastic_drift(alpha),
                _stochastic_diffusion(sigma),
                Ito(),
            )
            top = stochastic_lyapunov(system, np.array([0.0]), settings=settings).top
            points.append((alpha, top))
        curves.append((kind, label, points, None))
    markers = [("bifurcation", "shifted threshold", [(0.18, 0.0)], None)]
    scene = curve_scene(
        curves,
        markers,
        (ALPHA, "top Lyapunov exponent"),
        "Stochastic (D) bifurcation: noise shifts the threshold",
        "The top Lyapunov exponent of the trivial solution, simulated by the "
        "stochastic Benettin method. Without noise it crosses zero at alpha = 0; "
        "multiplicative noise shifts the crossing to alpha = s^2 / 2 = 0.18.",
    )
    return with_regions(
        scene,
        [("noise stabilises: alpha in (0, s^2/2)", 0.09, -0.04)],
    )


def _dynamical_bifurcation() -> Scene:
    equation = _equation(SpiralSystem, 2, [First(value=-0.3)])
    field = equation.derivative
    settings = LyapunovSettings(dt=0.01, horizon=6000, transient=100)
    curve: list[tuple[float, float]] = []
    for step in range(19):
        mu = -0.3 + 0.03 * step
        field.parameters[0].value = mu
        spectrum = lyapunov_spectrum(field, np.array([1.0, 0.0]), settings=settings)
        curve.append((mu, spectrum.top))
    lines = [("stable", "top Lyapunov exponent", curve, None)]
    markers = [("bifurcation", "D-bifurcation", [(0.0, 0.0)], None)]
    scene = curve_scene(
        lines,
        markers,
        (MU, "Lyapunov exponent"),
        "Dynamical (D) bifurcation: top Lyapunov exponent",
        "The top Lyapunov exponent is computed numerically from the linearised flow "
        "by the Benettin QR method; it passes through zero at mu = 0, where the "
        "reference motion loses stability.",
    )
    return with_regions(
        scene,
        [
            ("stable: exponent < 0", -0.18, curve[2][1]),
            ("unstable: exponent > 0", 0.16, curve[-3][1]),
        ],
    )


def _snic_period_divergence() -> Scene:
    equation = _equation(SnicSystem, 2, [First(value=1.3)])
    field = equation.derivative
    intervals = 200
    grid = np.linspace(0.0, 1.0, intervals + 1)
    seed = np.column_stack([np.cos(2.0 * math.pi * grid), np.sin(2.0 * math.pi * grid)])
    curve: list[tuple[float, float]] = []
    for step in range(18):
        mu = 1.3 + 0.1 * step
        field.parameters[0].value = mu
        oracle = 2.0 * math.pi / math.sqrt(mu * mu - 1.0)
        solution = PeriodicOrbit(
            field,
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed, oracle * 1.02)
        if (
            solution is not None
            and abs(solution.period - oracle) / oracle < _PERIOD_TOLERANCE
        ):
            curve.append((mu, solution.period))
    lines = [("stable", "limit cycle period", curve, None)]
    scene = curve_scene(
        lines,
        [],
        (MU, "period"),
        "SNIC: infinite-period bifurcation",
        "Each point is a limit cycle solved by the periodic boundary-value problem; "
        "the period diverges as mu approaches the SNIC at mu = 1, where a saddle-node "
        "forms on the invariant circle.",
    )
    return with_regions(
        scene,
        [("SNIC: period to infinity at mu=1", 1.35, curve[0][1])],
    )


def _fold_of_cycles_branch() -> Scene:
    equation = _equation(FoldCycles, 2, [First(value=-0.1)])
    intervals = 50
    grid = np.linspace(0.0, 1.0, intervals + 1)
    radius = math.sqrt((1.0 + math.sqrt(1.0 + 4.0 * -0.1)) / 2.0)
    seed = CycleSeed(
        np.column_stack(
            [
                radius * np.cos(2.0 * math.pi * grid),
                radius * np.sin(2.0 * math.pi * grid),
            ],
        ),
        2.0 * math.pi,
        -0.1,
    )
    continuation = CycleContinuation(
        equation,
        0,
        intervals,
        phase_index=1,
        phase_value=0.0,
    )
    points, bifurcations = continuation.trace(seed, 0.03, 130, direction=-1.0)
    stable: list[tuple[float, float]] = []
    unstable: list[tuple[float, float]] = []
    for point in points:
        order = np.argsort(np.abs(point.multipliers - 1.0))
        nontrivial = point.multipliers[order[1:]]
        dominant = float(np.max(np.abs(nontrivial)))
        target = stable if dominant < 1.0 else unstable
        target.append((point.parameter, point.amplitude))
    lines = [
        ("stable", "stable cycle", stable, None),
        ("unstable", "unstable cycle", unstable, None),
    ]
    fold = next((b for b in bifurcations if b.kind == "fold_of_cycles"), None)
    markers = []
    if fold is not None:
        amplitude = min(
            (
                p.amplitude
                for p in points
                if abs(p.parameter - fold.parameter) < _FOLD_WINDOW
            ),
            default=radius,
        )
        markers = [("fold", "fold of cycles", [(fold.parameter, amplitude)], None)]
    return curve_scene(
        lines,
        markers,
        (MU, "amplitude"),
        "Fold of cycles",
        "Two limit cycles - one stable, one unstable - collide and annihilate as a "
        "Floquet multiplier passes +1; the branch turns at mu = -1/4.",
    )


def _limit_cycle_branch() -> Scene:
    equation = _equation(Hopf, 2, [First(value=0.0)])
    field = equation.derivative
    intervals = 60
    nodes = np.linspace(0.0, 1.0, intervals + 1)
    cycle: list[tuple[float, float]] = [(0.0, 0.0)]
    seed = None
    period = 2.0 * math.pi
    for step in range(1, 26):
        mu = 0.024 * step
        field.parameters[0].value = mu
        if seed is None:
            seed = np.column_stack(
                [
                    math.sqrt(mu) * np.cos(2.0 * math.pi * nodes),
                    math.sqrt(mu) * np.sin(2.0 * math.pi * nodes),
                ],
            )
        orbit = PeriodicOrbit(field, intervals, phase_index=1, phase_value=0.0)
        solution = orbit.solve(seed, period)
        if solution is None:
            continue
        amplitude = float(np.max(np.linalg.norm(solution.states, axis=1)))
        cycle.append((mu, amplitude))
        seed = solution.states
        period = solution.period
    stable = [(mu / 100.0 - 0.6, 0.0) for mu in range(61)]
    unstable = [(0.6 * mu / 60.0, 0.0) for mu in range(61)]
    lines = [
        ("stable", "stable equilibrium", stable, None),
        ("unstable", "unstable equilibrium", unstable, None),
        ("stable", "limit cycle", cycle, None),
    ]
    markers = [("hopf", "Hopf", [(0.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        (MU, "amplitude"),
        "Supercritical Hopf: limit-cycle branch",
        "The periodic orbit born at the Hopf grows as amplitude = sqrt(mu); each "
        "point is a limit cycle solved by the periodic boundary-value problem.",
    )


def _hopf() -> Scene:
    equation = _equation(Hopf, 2, [First(value=0.0)])
    config = ContinuationConfig(
        detectors=["hopf"],
        initial_parameter=0.6,
        direction=-1,
        measure="norm",
        parameter_lower_bound=-0.6,
        parameter_upper_bound=0.7,
    )
    branch = _solve(equation, config, [0.0, 0.0])
    return bifurcation_scene(
        branch,
        "Hopf bifurcation",
        "Planar Hopf normal form; a periodic orbit is born at " + MU + " = 0.",
        MU,
        NORM,
    )


def _bogdanov_takens() -> Scene:
    equation = _equation(BogdanovTakens, 2, [First(value=-0.25), Second(value=-0.5)])
    seed = CurveSeed(state=[-0.5, 0.0], parameter_a=-0.25, frequency=1.0)
    config = Codim2Config(
        continuation_parameter_index=0,
        second_parameter_index=1,
        curve="hopf",
        codim2_detectors=["bogdanov_takens"],
        initial_parameter=-0.5,
        direction=1,
        parameter_lower_bound=-0.6,
        parameter_upper_bound=0.05,
    )
    result = Codim2Driver.run(config, equation, seed)
    return codim2_scene(
        result,
        "Hopf curve to Bogdanov-Takens",
        "Along the Hopf curve the frequency falls to zero at the codim-2 point.",
        BETA2,
        BETA1,
    )


def _zero_hopf() -> Scene:
    equation = _equation(ZeroHopf, 3, [First(value=0.5), Second(value=0.0)])
    seed = CurveSeed(state=[0.5**0.5, 0.0, 0.0], parameter_a=0.0, frequency=1.0)
    config = Codim2Config(
        continuation_parameter_index=1,
        second_parameter_index=0,
        curve="hopf",
        codim2_detectors=["zero_hopf"],
        initial_parameter=0.5,
        direction=-1,
        parameter_lower_bound=-0.1,
        parameter_upper_bound=0.6,
    )
    result = Codim2Driver.run(config, equation, seed)
    return codim2_scene(
        result,
        "Hopf curve to zero-Hopf",
        "The Hopf curve meets the fold in x, giving a zero-Hopf point.",
        MU + "\u2081",
        MU + "\u2082",
    )


def _hopf_hopf() -> Scene:
    equation = _equation(HopfHopf, 4, [First(value=0.0), Second(value=0.5)])
    seed = CurveSeed(state=[0.0, 0.0, 0.0, 0.0], parameter_a=0.0, frequency=1.0)
    config = Codim2Config(
        continuation_parameter_index=0,
        second_parameter_index=1,
        curve="hopf",
        codim2_detectors=["hopf_hopf"],
        initial_parameter=0.5,
        direction=-1,
        parameter_lower_bound=-0.3,
        parameter_upper_bound=0.6,
    )
    result = Codim2Driver.run(config, equation, seed)
    return codim2_scene(
        result,
        "Hopf curve to Hopf-Hopf",
        "Continuing the first Hopf curve, the second pair reaches the axis.",
        MU + "\u2082",
        MU + "\u2081",
    )


def _cusp() -> Scene:
    equation = _equation(CuspForm, 1, [First(value=-0.25), Second(value=0.75)])
    seed = CurveSeed(state=[0.5], parameter_a=0.75)
    config = Codim2Config(
        continuation_parameter_index=1,
        second_parameter_index=0,
        curve="fold",
        codim2_detectors=["cusp"],
        initial_parameter=-0.25,
        direction=1,
        parameter_lower_bound=-0.35,
        parameter_upper_bound=0.35,
    )
    result = Codim2Driver.run(config, equation, seed)
    return codim2_scene(
        result,
        "Fold curve to cusp",
        "The fold curve turns through the cusp where the coefficient a vanishes.",
        BETA1,
        BETA2,
    )


def _generalized_hopf() -> Scene:
    equation = _equation(Bautin, 2, [First(value=0.0), Second(value=0.5)])
    seed = CurveSeed(state=[0.0, 0.0], parameter_a=0.0, frequency=1.0)
    config = Codim2Config(
        continuation_parameter_index=0,
        second_parameter_index=1,
        curve="hopf",
        codim2_detectors=["generalized_hopf"],
        initial_parameter=0.5,
        direction=-1,
        parameter_lower_bound=-0.6,
        parameter_upper_bound=0.6,
    )
    result = Codim2Driver.run(config, equation, seed)
    return codim2_scene(
        result,
        "Hopf curve to generalized Hopf",
        "Along the Hopf curve the first Lyapunov coefficient vanishes (Bautin).",
        BETA2,
        BETA1,
    )


def _swallowtail() -> Scene:
    equation = _equation(
        Swallowtail,
        1,
        [First(value=0.1875), Second(value=-1.0), Third(value=1.5)],
    )
    seed = Codim3Seed(state=[0.5], parameter_a=0.1875, parameter_b=1.5)
    config = Codim3Config(
        continuation_parameter_index=0,
        second_parameter_index=2,
        third_parameter_index=1,
        codim3_curve="cusp",
        codim3_detectors=["swallowtail"],
        initial_parameter=-1.0,
        direction=1,
        parameter_lower_bound=-1.2,
        parameter_upper_bound=1.2,
        initial_step=0.01,
        maximum_step=0.02,
        maximum_points=500,
    )
    result = Codim3Driver.run(config, equation, seed)
    return codim3_scene(
        result,
        (1, 2),
        "Cusp curve to swallowtail",
        "Cusp curve projected onto its plane; the swallowtail sits at the origin.",
        (BETA1, BETA2, BETA3),
    )


def _degenerate_bautin() -> Scene:
    equation = _equation(
        ExtendedBautin,
        2,
        [First(value=0.0), Second(value=0.0), Third(value=0.6)],
    )
    seed = Codim3Seed(
        state=[0.0, 0.0],
        parameter_a=0.0,
        parameter_b=0.0,
        frequency=1.0,
    )
    config = Codim3Config(
        continuation_parameter_index=0,
        second_parameter_index=1,
        third_parameter_index=2,
        codim3_curve="generalized_hopf",
        codim3_detectors=["degenerate_bautin"],
        initial_parameter=0.6,
        direction=-1,
        parameter_lower_bound=-0.7,
        parameter_upper_bound=0.7,
        initial_step=0.02,
        maximum_step=0.05,
        maximum_points=400,
    )
    result = Codim3Driver.run(config, equation, seed)
    return codim3_scene(
        result,
        (2, 1),
        "Generalized-Hopf curve to degenerate Bautin",
        "The second Lyapunov quantity vanishes along the generalized-Hopf curve.",
        (BETA1, BETA2, BETA3),
    )


def _cusp_family() -> Scene:
    equation = _equation(CuspForm, 1, [First(value=0.0), Second(value=0.0)])
    config = FamilyConfig(
        continuation_parameter_index=0,
        family_parameter_index=1,
        family_values=[-0.5, -0.25, 0.0, 0.5, 1.0, 1.5],
        initial_parameter=-3.0,
        direction=1,
        measure="component",
        parameter_lower_bound=-3.2,
        parameter_upper_bound=3.2,
        initial_step=0.01,
        maximum_step=0.05,
        maximum_points=1200,
        detectors=["fold"],
    )
    family = FamilyDriver.run(config, equation, [-(3.0 ** (1 / 3))])
    return family_scene(
        family,
        (BETA1, "x", BETA2),
        "Cusp: a diagram family",
        "As " + BETA2 + " passes the cusp, a fold pair is born in the diagram.",
    )


def _fluxgate_family() -> Scene:
    equation = _equation(
        FluxgateRing,
        3,
        [First(value=0.0), Second(value=3.0), Third(value=0.0)],
    )
    config = FamilyConfig(
        continuation_parameter_index=0,
        family_parameter_index=1,
        family_values=[2.0, 2.5, 3.0, 3.5, 4.0],
        initial_parameter=0.0,
        direction=-1,
        measure="norm",
        parameter_lower_bound=-5.0,
        parameter_upper_bound=0.2,
        initial_step=0.01,
        maximum_step=0.03,
        maximum_points=2000,
        detectors=["branch_point", "fold", "hopf"],
    )
    family = FamilyDriver.run(config, equation, [0.0, 0.0, 0.0])
    return family_scene(
        family,
        ("\u03bb", NORM, "c"),
        "Coupled-core fluxgate: symmetry breaking vs gain",
        "N = 3 ring; the symmetry-breaking point tracks " + "\u03bb = 1 - c.",
    )


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


_SADDLE_JACOBIAN = np.array([[0.0, 1.0], [1.0, 0.0]])
_PHASE_WINDOW = 2.2


def _saddle() -> DeterministicFunction:
    return _equation(Saddle, 2, [First(value=0.0)]).derivative


def _chart_branches(selection, colour):
    chart = TaylorManifold(order=7).compute(
        _saddle(),
        np.zeros(2),
        _SADDLE_JACOBIAN,
        selection,
    )
    lines = []
    for sign in (1.0, -1.0):
        points = []
        for theta in np.linspace(0.0, 0.9 * sign, 60):
            state = chart.point([float(theta)])
            if abs(state[0]) < _PHASE_WINDOW and abs(state[1]) < _PHASE_WINDOW:
                points.append((float(state[0]), float(state[1])))
        kind = "manifold_unstable" if colour == "unstable" else "manifold_stable"
        lines.append((kind, "", points, None))
    return lines


def _manifolds() -> Scene:
    lines = _chart_branches(UnstableManifold(), "unstable")
    lines += _chart_branches(StableManifold(), "stable")
    markers = [("equilibrium", "saddle", [(0.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        ("x", "y"),
        "Stable and unstable manifolds of a saddle",
        "Taylor charts of W^u (red) and W^s (green); their union is the homoclinic.",
    )


def _homoclinic_orbit():
    half, intervals = 13.0, 200
    times = np.linspace(-half, half, intervals + 1)
    seed = np.zeros((intervals + 1, 2))
    for k, t in enumerate(times):
        seed[k, 0] = 1.2 / math.cosh(0.45 * t) ** 2
        seed[k, 1] = -0.9 * math.tanh(0.45 * t) / math.cosh(0.45 * t) ** 2
    mesh = MeshSpec(half, intervals, phase_index=1, phase_value=0.0)
    return HomoclinicOrbit(_saddle(), np.zeros(2), _SADDLE_JACOBIAN, mesh).solve(seed)


def _homoclinic_loop(orbit) -> Scene:
    loop = [(float(x), float(y)) for x, y in orbit.states]
    lines = [("orbit", "homoclinic orbit", loop, None)]
    lines += _chart_branches(UnstableManifold(), "unstable")
    markers = [("equilibrium", "saddle", [(0.0, 0.0)], None)]
    return curve_scene(
        lines,
        markers,
        ("x", "y"),
        "Homoclinic orbit (phase plane)",
        "The connecting orbit leaves and returns to the saddle along its manifolds.",
    )


def _homoclinic_pulse(orbit) -> Scene:
    pulse = [
        (float(t), float(x))
        for t, x in zip(orbit.times, orbit.component(0), strict=True)
    ]
    return curve_scene(
        [("orbit", "x(t)", pulse, None)],
        [],
        ("t", "x"),
        "Homoclinic orbit (time series)",
        "The pulse matches the exact 3/2 sech^2(t/2) to trapezoidal order.",
    )


def _density_family() -> Scene:
    sigma = 0.6
    equation = _equation(NoisyDrift, 1, [First(value=0.0), Second(value=sigma)])
    drift = equation.derivative
    diffusion = _equation(NoisyDiffusion, 1, [First(value=0.0), Second(value=sigma)])
    diffusion_function = diffusion.derivative
    diffusion_function.parameters[1].value = sigma
    grid = np.linspace(1.0e-3, 2.6, 1600)
    values = [0.0, 0.1, 0.18, 0.3, 0.5, 0.7]
    curves = []
    for alpha in values:
        drift.parameters[0].value = alpha
        _grid, density = StationaryDensity(
            drift,
            diffusion_function,
            Stratonovich(),
            grid,
        ).evaluate()
        peak = float(np.max(density))
        curves.append(
            [(float(x), float(d / peak)) for x, d in zip(grid, density, strict=True)],
        )
    return curve_family_scene(
        curves,
        values,
        ("x", "stationary density (scaled)"),
        "\u03b1",
        (
            "Stochastic P-bifurcation: stationary density",
            "The density mode leaves the origin at \u03b1 = \u03c3\u00b2/2 = 0.18.",
        ),
    )


def _lyapunov() -> Scene:
    sigma = 0.6
    drift = _equation(NoisyDrift, 1, [First(value=0.0), Second(value=sigma)]).derivative
    diffusion = _equation(
        NoisyDiffusion,
        1,
        [First(value=0.0), Second(value=sigma)],
    ).derivative
    diffusion.parameters[1].value = sigma
    exponent = LyapunovExponent(drift, diffusion, Stratonovich())
    curve = []
    for alpha in np.linspace(-0.4, 0.4, 41):
        drift.parameters[0].value = float(alpha)
        curve.append((float(alpha), float(exponent.at(0.0))))
    markers = [("zero_crossing", "D-bifurcation (\u03bb = 0)", [(0.0, 0.0)], None)]
    return curve_scene(
        [("curve", "Lyapunov exponent", curve, None)],
        markers,
        ("\u03b1", "top Lyapunov exponent \u03bb"),
        "Stochastic D-bifurcation: Lyapunov exponent",
        "The linear top Lyapunov exponent changes sign at \u03b1 = 0.",
    )


def _cusp_surface() -> Scene:
    equation = _equation(CuspForm, 1, [First(value=0.0), Second(value=0.0)])
    values = [round(float(v), 3) for v in np.linspace(-0.3, 1.5, 16)]
    config = FamilyConfig(
        continuation_parameter_index=0,
        family_parameter_index=1,
        family_values=values,
        initial_parameter=-3.0,
        direction=1,
        measure="component",
        parameter_lower_bound=-3.2,
        parameter_upper_bound=3.2,
        initial_step=0.01,
        maximum_step=0.05,
        maximum_points=1500,
        detectors=["fold"],
    )
    family = FamilyDriver.run(config, equation, [-(3.0 ** (1 / 3))])
    return surface_scene(
        family,
        (BETA1, BETA2, "x"),
        (
            "Cusp: equilibrium surface with continued fold curve",
            "The fold locus (pink) is the semicubical cusp curve over the plane.",
        ),
    )


def _fluxgate_surface() -> Scene:
    equation = _equation(
        FluxgateRing,
        3,
        [First(value=0.0), Second(value=3.0), Third(value=0.0)],
    )
    values = [round(float(v), 3) for v in np.linspace(2.0, 4.0, 12)]
    config = FamilyConfig(
        continuation_parameter_index=0,
        family_parameter_index=1,
        family_values=values,
        initial_parameter=0.0,
        direction=-1,
        measure="norm",
        parameter_lower_bound=-5.0,
        parameter_upper_bound=0.2,
        initial_step=0.01,
        maximum_step=0.03,
        maximum_points=2000,
        detectors=["branch_point"],
    )
    family = FamilyDriver.run(config, equation, [0.0, 0.0, 0.0])
    return surface_scene(
        family,
        ("\u03bb", "c", NORM),
        (
            "Coupled-core fluxgate: equilibrium surface",
            "The symmetry-breaking locus (green) tracks \u03bb = 1 - c over the sheet.",
        ),
    )


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


def _bogdanov_takens_config(values: list[float]) -> FamilyConfig:
    return FamilyConfig(
        continuation_parameter_index=0,
        family_parameter_index=1,
        family_values=values,
        initial_parameter=-1.5,
        direction=1,
        measure="component",
        parameter_lower_bound=-1.7,
        parameter_upper_bound=0.05,
        initial_step=0.005,
        maximum_step=0.02,
        maximum_points=3000,
        detectors=["fold", "hopf"],
    )


def _bogdanov_takens_family() -> Scene:
    equation = _equation(BogdanovTakensFamily, 2, [First(value=0.0), Second(value=0.5)])
    values = [0.2, 0.4, 0.6, 0.8, 1.0]
    family = FamilyDriver.run(
        _bogdanov_takens_config(values),
        equation,
        [-(1.5**0.5), 0.0],
    )
    return family_scene(
        family,
        (BETA1, "x", BETA2),
        "Fold and Hopf in a family, meeting at Bogdanov-Takens",
        "Each slice has a fold and a Hopf; they merge at the BT point as b2\u21920.",
    )


def _bogdanov_takens_surface() -> Scene:
    equation = _equation(
        BogdanovTakensFamily,
        2,
        [First(value=0.0), Second(value=0.15)],
    )
    values = [round(float(v), 3) for v in np.linspace(0.15, 1.1, 12)]
    family = FamilyDriver.run(
        _bogdanov_takens_config(values),
        equation,
        [-(1.5**0.5), 0.0],
    )
    return surface_scene(
        family,
        (BETA1, BETA2, "x"),
        (
            "Bogdanov-Takens: equilibrium surface with two continued loci",
            "Fold curve (pink) and Hopf curve (amber) meet at the codim-2 point.",
        ),
    )


def _efield_operating_map() -> Scene:
    equation = _equation(
        FerroelectricRing,
        3,
        [First(value=0.0), Second(value=1.0), Third(value=0.0)],
    )
    values = [round(float(v), 3) for v in np.linspace(0.4, 1.4, 11)]
    config = FamilyConfig(
        continuation_parameter_index=0,
        family_parameter_index=1,
        family_values=values,
        initial_parameter=1.6,
        direction=-1,
        measure="norm",
        parameter_lower_bound=-3.2,
        parameter_upper_bound=1.7,
        initial_step=0.01,
        maximum_step=0.03,
        maximum_points=2500,
        detectors=["branch_point", "hopf"],
    )
    family = FamilyDriver.run(config, equation, [0.0, 0.0, 0.0])
    breaking, oscillation = [], []
    for item in family.slices:
        for point in item.branch.special_points:
            if point.kind == "branch_point":
                breaking.append((point.parameter, item.value))
            elif point.kind == "hopf":
                oscillation.append((point.parameter, item.value))
    lines = [
        ("branch_point", "symmetry breaking (\u03bb=a)", breaking, "#a3e635"),
        ("hopf", "oscillation onset (\u03bb=-2a)", oscillation, "#fbbf24"),
    ]
    return curve_scene(
        lines,
        [],
        ("\u03bb", "a"),
        "Electric-field sensor: operating map",
        "Ferroelectric ring: oscillations for \u03bb<-2a, broken symmetry \u03bb>a.",
    )


def _efield_detection() -> Scene:
    equation = _equation(
        FerroelectricRing,
        3,
        [First(value=-0.5), Second(value=1.0), Third(value=0.0)],
    )
    values = [round(float(v), 3) for v in np.linspace(-0.6, 0.7, 12)]
    config = FamilyConfig(
        continuation_parameter_index=2,
        family_parameter_index=0,
        family_values=values,
        initial_parameter=-1.3,
        direction=1,
        measure="component",
        parameter_lower_bound=-1.4,
        parameter_upper_bound=1.4,
        initial_step=0.005,
        maximum_step=0.02,
        maximum_points=3000,
        detectors=["fold"],
    )
    seed = _newton_symmetric(1.0 - (-0.5), -1.3)
    family = FamilyDriver.run(config, equation, [seed, seed, seed])
    return surface_scene(
        family,
        ("\u03b5", "\u03bb", "x"),
        (
            "Electric-field sensor: detection surface",
            "Target field \u03b5 sweeps the well; coupling \u03bb tunes the folds.",
        ),
    )


def _fluxgate_bias_surface() -> Scene:
    equation = _equation(
        FluxgateRing,
        3,
        [First(value=0.0), Second(value=2.0), Third(value=0.0)],
    )
    values = [round(float(v), 3) for v in np.linspace(0.8, 3.0, 12)]
    config = FamilyConfig(
        continuation_parameter_index=2,
        family_parameter_index=1,
        family_values=values,
        initial_parameter=-1.2,
        direction=1,
        measure="component",
        parameter_lower_bound=-1.4,
        parameter_upper_bound=1.4,
        initial_step=0.005,
        maximum_step=0.02,
        maximum_points=3000,
        detectors=["fold"],
    )
    seed = -0.9
    family = FamilyDriver.run(config, equation, [seed, seed, seed])
    return surface_scene(
        family,
        ("\u03b5", "c", "x"),
        (
            "Fluxgate: bias-field surface over gain",
            "Hysteresis in the bias \u03b5 emerges once the gain c exceeds one.",
        ),
    )


def _newton_symmetric(effective_gain: float, field: float) -> float:
    x = field
    for _ in range(80):
        residual = effective_gain * x - x**3 + field
        derivative = effective_gain - 3.0 * x * x
        x -= residual / derivative
    return x


def _exploration_diagram() -> Scene:
    equation = _equation(
        Swallowtail,
        1,
        [First(value=0.1875), Second(value=-1.0), Third(value=1.0)],
    )
    config = ExplorationConfig(
        parameters=[0, 1, 2],
        ranges=[(-1.4, 1.4), (-1.4, 1.4), (-1.4, 1.4)],
        initial_parameter=0.1875,
        initial_step=0.01,
        maximum_step=0.02,
        maximum_points=800,
    )
    root = BifurcationExplorer.explore(config, equation, [0.5])
    return exploration_diagram(
        root,
        (BETA1, BETA2),
        (
            "Automatic exploration: two-parameter diagram",
            "Fold curve continued from a detected fold; its two cusps are marked.",
        ),
    )


def _exploration_skeleton() -> Scene:
    equation = _equation(
        Swallowtail,
        1,
        [First(value=0.1875), Second(value=-1.0), Third(value=1.0)],
    )
    config = ExplorationConfig(
        parameters=[0, 1, 2],
        ranges=[(-1.4, 1.4), (-1.4, 1.4), (-1.4, 1.4)],
        initial_parameter=0.1875,
        initial_step=0.01,
        maximum_step=0.02,
        maximum_points=800,
    )
    root = BifurcationExplorer.explore(config, equation, [0.5])
    return exploration_skeleton(
        root,
        (BETA1, BETA2, "b\u2083"),
        (
            "Automatic exploration: 3-parameter skeleton",
            "The cusp curve descends through b\u2083 to the swallowtail, rotatable.",
        ),
    )


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


def _bogdanov_takens_type_curve() -> Scene:
    equation = _equation(
        BogdanovTakensTypeChange,
        2,
        [First(value=0.6), Second(value=2.0), Third(value=0.4)],
    )
    config = ExplorationConfig(
        parameters=[0, 1, 2],
        ranges=[(0.2, 1.6), (1.2, 2.8), (-0.7, 0.7)],
        initial_parameter=0.6,
        initial_step=0.01,
        maximum_step=0.03,
        maximum_points=800,
    )
    roots = [
        value.real
        for value in np.roots([1.0, 2.0, 0.6])
        if abs(value.imag) < _IMAGINARY_TOLERANCE
    ]
    root = BifurcationExplorer.explore(config, equation, [roots[0], 0.0])
    return exploration_skeleton(
        root,
        (BETA1, BETA2, "b\u2083"),
        (
            "Bogdanov-Takens type change (b = 0)",
            "The BT curve's cross coefficient vanishes at the marked point.",
        ),
    )


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


def _bogdanov_takens_portrait() -> Scene:
    parameters = [First(value=0.0), Second(value=0.0)]
    equation = DifferentialEquation(
        variables=[State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=BogdanovTakensPlanar(
            variables=[State(), State()],
            parameters=parameters,
            results=[State(), State()],
            time=None,
        ),
    )
    homoclinic: list[tuple[float, float]] = [(0.0, 0.0)]
    for step in range(1, 13):
        epsilon = 0.12 + 0.02 * step
        half = 5.0 * math.sqrt(2.0) / epsilon
        nodes = 121
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
        sweep = (5.0 / 7.0) * epsilon**2
        point = HomoclinicCurve(equation, 0, 1, mesh).solve_point(
            sweep,
            orbit,
            np.array([epsilon**2, 0.0]),
            -(epsilon**4),
        )
        if point is not None:
            homoclinic.append((point.sweep, point.continuation))
    span = homoclinic[-1][0]
    grid = [span * k / 40.0 for k in range(41)]
    fold = [(value, 0.0) for value in grid]
    hopf = [(value, -(value**2)) for value in grid]
    lines = [
        ("fold", "fold (saddle-node)", fold, None),
        ("hopf", "Hopf", hopf, None),
        ("bogdanov_takens", "homoclinic", homoclinic, None),
    ]
    markers = [("bogdanov_takens", "Bogdanov-Takens", [(0.0, 0.0)], None)]
    scene = curve_scene(
        lines,
        markers,
        (BETA2, BETA1),
        "Bogdanov-Takens organizing centre",
        "Three curves emanate from the codim-2 point: fold, Hopf, and the homoclinic "
        "curve, tangent to the Hopf with the 5/7 slope ratio.",
    )
    seeds = [np.array([0.3, 0.0]), np.array([-0.3, 0.0])]
    samples = [
        (0.040, 0.006, None),
        (0.078, -0.0028, np.array([-0.03, 0.0])),
        (0.050, -0.012, np.array([-0.09, 0.0])),
    ]
    regions = []
    for sweep, first, cycle_start in samples:
        parameters[0].value = first
        parameters[1].value = sweep
        label = describe_region(equation.derivative, seeds, cycle_start=cycle_start)
        regions.append((label, sweep, first))
    return with_regions(scene, regions)


def _bogdanov_takens_curve() -> Scene:
    equation = _equation(
        BogdanovTakensUnfolding,
        2,
        [First(value=-0.4), Second(value=0.32), Third(value=0.4)],
    )
    config = ExplorationConfig(
        parameters=[0, 1, 2],
        ranges=[(-0.8, 0.3), (-0.2, 0.9), (-0.6, 0.7)],
        initial_parameter=-0.4,
        initial_step=0.01,
        maximum_step=0.03,
        maximum_points=700,
    )
    roots = [
        value.real
        for value in np.roots([0.4, 0.32, -0.4])
        if abs(value.imag) < _IMAGINARY_TOLERANCE
    ]
    root = BifurcationExplorer.explore(config, equation, [roots[0], 0.0])
    return exploration_skeleton(
        root,
        (BETA1, BETA2, "b\u2083"),
        (
            "Bogdanov-Takens curve with a degenerate BT",
            "The BT curve's quadratic coefficient vanishes at the marked point.",
        ),
    )


def _sevencell_exploration() -> Scene:
    count = 7
    equation = _equation(
        FerroelectricRing,
        count,
        [First(value=0.0), Second(value=1.0), Third(value=0.0)],
    )
    config = ExplorationConfig(
        parameters=[0, 1, 2],
        ranges=[(-3.0, 3.0), (0.4, 1.8), (-1.0, 1.0)],
        initial_parameter=0.0,
        initial_step=0.03,
        maximum_step=0.06,
        maximum_points=150,
    )
    root = BifurcationExplorer.explore(config, equation, [0.0] * count)
    return exploration_diagram(
        root,
        ("\u03bb", "a"),
        (
            "Seven-cell ring: high-dimensional exploration",
            "Codim-2 Hopf curves continued from a seven-dimensional equilibrium.",
        ),
    )


def _sevencell_ring_map() -> Scene:
    count = 7
    equation = _equation(
        FerroelectricRing,
        count,
        [First(value=0.0), Second(value=1.0), Third(value=0.0)],
    )
    values = [round(0.4 + 0.1 * i, 3) for i in range(13)]
    branch: list[tuple[float, float]] = []
    hopf: dict[int, list[tuple[float, float]]] = {}
    for direction in (1, -1):
        config = FamilyConfig(
            continuation_parameter_index=0,
            family_parameter_index=1,
            family_values=values,
            initial_parameter=0.0,
            direction=direction,
            measure="norm",
            parameter_lower_bound=-5.5,
            parameter_upper_bound=3.6,
            initial_step=0.02,
            maximum_step=0.05,
            maximum_points=2000,
            detectors=["branch_point", "hopf"],
        )
        family = FamilyDriver.run(config, equation, [0.0] * count)
        for item in family.slices:
            for point in item.branch.special_points:
                if point.kind == "branch_point":
                    branch.append((point.parameter, item.value))
                else:
                    mode = _ring_mode(point.parameter / item.value, count)
                    hopf.setdefault(mode, []).append((point.parameter, item.value))
    palette = {1: "#fbbf24", 2: "#fb923c", 3: "#f472b6"}
    lines = [
        (
            "branch_point",
            "symmetry breaking (\u03bb=a)",
            sorted(branch, key=lambda pair: pair[1]),
            "#a3e635",
        ),
    ]
    lines.extend(
        (
            f"hopf{mode}",
            f"Hopf k={mode} (\u03bb=a/cos 2\u03c0k/7)",
            sorted(hopf[mode], key=lambda pair: pair[1]),
            palette.get(mode, "#fbbf24"),
        )
        for mode in sorted(hopf)
    )
    return curve_scene(
        lines,
        [],
        ("\u03bb", "a"),
        "Seven-cell ferroelectric ring: operating map",
        "A seven-dimensional sensor array; three Hopf lines from the ring spectrum.",
    )


def _ring_mode(ratio: float, count: int) -> int:
    modes = {
        mode: 1.0 / math.cos(2.0 * math.pi * mode / count)
        for mode in range(1, count // 2 + 1)
    }
    return min(modes, key=lambda mode: abs(ratio - modes[mode]))


def _exploration_tree() -> Scene:
    equation = _equation(
        Swallowtail,
        1,
        [First(value=0.1875), Second(value=-1.0), Third(value=1.0)],
    )
    config = ExplorationConfig(
        parameters=[0, 1, 2],
        ranges=[(-1.4, 1.4), (-1.4, 1.4), (-1.4, 1.4)],
        initial_parameter=0.1875,
        initial_step=0.01,
        maximum_step=0.02,
        maximum_points=800,
    )
    root = BifurcationExplorer.explore(config, equation, [0.5])
    return exploration_tree(
        root,
        (
            "Automatic exploration: escalation tree",
            "Each continuation and the degeneracies it found; edges show escalation.",
        ),
    )


_FERROELECTRIC = r"\dot{x}_i = a\,x_i - x_i^3 - \lambda\,x_{i+1} + \varepsilon"
_FLUXGATE = r"\dot{x}_i = -x_i + \tanh(g\,x_i + c\,x_{i+1} + h)"
_CUSP_EQN = r"\dot{x} = \beta_1 + \beta_2 x - x^3"
_SWALLOWTAIL_EQN = r"\dot{x} = \beta_1 + \beta_2 x + \beta_3 x^2 - x^4"
_BT_FAMILY_EQN = r"\dot{x} = y,\quad \dot{y} = \beta_1 + \beta_2 y + x^2 + x y"

_EQUATIONS = {
    "saddle_node.html": r"\dot{x} = \lambda - x^2",
    "transcritical.html": r"\dot{x} = \lambda x - x^2",
    "pitchfork.html": r"\dot{x} = \lambda x - x^3",
    "hysteresis.html": r"\dot{x} = \lambda + 3x - x^3",
    "hopf.html": r"\dot{x} = \mu x - y - x(x^2+y^2),\quad "
    r"\dot{y} = x + \mu y - y(x^2+y^2)",
    "limit_cycle_branch.html": r"\dot{r} = \mu r - r^3,\quad r_{\max} = \sqrt{\mu}",
    "fold_of_cycles_branch.html": r"\dot{r} = \mu r + r^3 - r^5",
    "dynamical_bifurcation.html": r"\dot{x} = \mu x - y,\quad \dot{y} = x + \mu y",
    "collocation_convergence.html": r"\dot{x}=\mu x - y - x r^2,\ "
    r"\dot{y}=x+\mu y - y r^2",
    "deflated_roots.html": r"\dot{x}=x-x^3,\quad \dot{y}=y-y^3",
    "homoclinic_curve_shooting.html": r"\dot{x}=y,\ \dot{y}=x-x^2+\mu y+\nu x y",
    "symmetry_modes.html": r"\dot{x}_i = a x_i - x_i^3 - \lambda x_{i+1}",
    "shilnikov_loop.html": r"\dot{x}=y,\ \dot{y}=z,\ \dot{z}=-a z - y + x - x^2",
    "homoclinic_return.html": r"\dot{x} = y,\quad \dot{y} = x - x^2 + \mu y",
    "three_dimensional_homoclinic.html": r"\dot{u}=v,\ \dot{v}=u-u^2,\ \dot{w}=-2w",
    "heteroclinic_cycle.html": r"\dot{x} = y,\quad \dot{y} = -x + x^3",
    "saddle_focus_onset.html": r"\dot{x}=\sigma(y-x),\ "
    r"\dot{y}=x(\rho-z)-y,\ \dot{z}=xy-\beta z",
    "adaptive_mesh.html": r"\dot{x}=y,\quad \dot{y}=\mu(1-x^2)y - x",
    "stochastic_attractor.html": r"dx = f_{\mathrm{Lorenz}}(x)\,dt + \sigma\,dW",
    "stochastic_bifurcation.html": r"dx = \alpha x\,dt + \sigma x\,dW",
    "snic_period_divergence.html": r"\dot{r} = r(1-r^2),\quad "
    r"\dot{\theta} = \mu - \sin\theta",
    "bogdanov_takens.html": r"\dot{x} = y,\quad \dot{y} = \beta_1 + \beta_2 y "
    r"+ x^2 - x y",
    "bogdanov_takens_portrait.html": r"\dot{x} = y,\quad \dot{y} = \beta_1 "
    r"+ \beta_2 y + x^2 + x y",
    "zero_hopf.html": r"\dot{x} = \mu_1 - x^2,\quad \dot{r} = \mu_2 r - r^3",
    "hopf_hopf.html": r"\dot{r}_1 = \mu_1 r_1 - r_1^3,\quad "
    r"\dot{r}_2 = \mu_2 r_2 - r_2^3",
    "cusp.html": _CUSP_EQN,
    "generalized_hopf.html": r"\dot{r} = \beta_1 r + \beta_2 r^3",
    "swallowtail.html": _SWALLOWTAIL_EQN,
    "degenerate_bautin.html": r"\dot{r} = \beta_1 r + \beta_2 r^3 + \beta_3 r^5",
    "cusp_family.html": _CUSP_EQN,
    "fluxgate_family.html": _FLUXGATE,
    "cusp_surface.html": _CUSP_EQN,
    "fluxgate_surface.html": _FLUXGATE,
    "bogdanov_takens_family.html": _BT_FAMILY_EQN,
    "bogdanov_takens_surface.html": _BT_FAMILY_EQN,
    "efield_operating_map.html": _FERROELECTRIC,
    "efield_detection.html": _FERROELECTRIC,
    "fluxgate_bias_surface.html": _FLUXGATE,
    "exploration_diagram.html": _SWALLOWTAIL_EQN,
    "exploration_skeleton.html": _SWALLOWTAIL_EQN,
    "exploration_tree.html": _SWALLOWTAIL_EQN,
    "bogdanov_takens_curve.html": r"\dot{x} = y,\quad \dot{y} = \beta_1 "
    r"+ \beta_2 x + \beta_3 y + \beta_3 x^2 + x y",
    "bogdanov_takens_type_curve.html": r"\dot{x} = y,\quad \dot{y} = \beta_1 "
    r"+ \beta_2 x + \beta_3 y + x^2 + \beta_3 x y",
    "sevencell_ring_map.html": _FERROELECTRIC,
    "sevencell_exploration.html": _FERROELECTRIC,
}


def _catalogue() -> list[tuple[str, Scene]]:
    entries = _raw_catalogue()
    return [
        (name, with_equation(scene, _EQUATIONS[name]) if name in _EQUATIONS else scene)
        for name, scene in entries
    ]


def _raw_catalogue() -> list[tuple[str, Scene]]:
    return [
        ("saddle_node.html", _saddle_node()),
        ("transcritical.html", _transcritical()),
        ("pitchfork.html", _pitchfork()),
        ("hysteresis.html", _hysteresis()),
        ("hopf.html", _hopf()),
        ("limit_cycle_branch.html", _limit_cycle_branch()),
        ("fold_of_cycles_branch.html", _fold_of_cycles_branch()),
        ("dynamical_bifurcation.html", _dynamical_bifurcation()),
        ("collocation_convergence.html", _collocation_convergence()),
        ("deflated_roots.html", _deflated_roots()),
        ("homoclinic_curve_shooting.html", _homoclinic_curve_shooting()),
        ("symmetry_modes.html", _symmetry_modes()),
        ("shilnikov_loop.html", _shilnikov_loop()),
        ("homoclinic_return.html", _homoclinic_return()),
        ("three_dimensional_homoclinic.html", _three_dimensional_homoclinic()),
        ("heteroclinic_cycle.html", _heteroclinic_cycle()),
        ("saddle_focus_onset.html", _saddle_focus_onset()),
        ("stochastic_bifurcation.html", _stochastic_bifurcation()),
        ("adaptive_mesh.html", _adaptive_mesh()),
        ("stochastic_attractor.html", _stochastic_attractor()),
        ("snic_period_divergence.html", _snic_period_divergence()),
        ("bogdanov_takens.html", _bogdanov_takens()),
        ("bogdanov_takens_portrait.html", _bogdanov_takens_portrait()),
        ("zero_hopf.html", _zero_hopf()),
        ("hopf_hopf.html", _hopf_hopf()),
        ("cusp.html", _cusp()),
        ("generalized_hopf.html", _generalized_hopf()),
        ("swallowtail.html", _swallowtail()),
        ("degenerate_bautin.html", _degenerate_bautin()),
        ("cusp_family.html", _cusp_family()),
        ("fluxgate_family.html", _fluxgate_family()),
        ("cusp_surface.html", _cusp_surface()),
        ("fluxgate_surface.html", _fluxgate_surface()),
        ("bogdanov_takens_family.html", _bogdanov_takens_family()),
        ("bogdanov_takens_surface.html", _bogdanov_takens_surface()),
        ("efield_operating_map.html", _efield_operating_map()),
        ("efield_detection.html", _efield_detection()),
        ("fluxgate_bias_surface.html", _fluxgate_bias_surface()),
        ("exploration_diagram.html", _exploration_diagram()),
        ("exploration_skeleton.html", _exploration_skeleton()),
        ("exploration_tree.html", _exploration_tree()),
        ("bogdanov_takens_curve.html", _bogdanov_takens_curve()),
        ("bogdanov_takens_type_curve.html", _bogdanov_takens_type_curve()),
        ("sevencell_ring_map.html", _sevencell_ring_map()),
        ("sevencell_exploration.html", _sevencell_exploration()),
        ("manifolds.html", _manifolds()),
        *_connecting_orbit_entries(),
        ("stochastic_density.html", _density_family()),
        ("stochastic_lyapunov.html", _lyapunov()),
    ]


def _connecting_orbit_entries() -> list[tuple[str, Scene]]:
    orbit = _homoclinic_orbit()
    return [
        ("homoclinic_loop.html", _homoclinic_loop(orbit)),
        ("homoclinic_pulse.html", _homoclinic_pulse(orbit)),
    ]


def main(output_dir: str = "plots") -> None:
    """Render every scene to ``output_dir`` and write the atlas."""
    Registry.fill_registry(
        path=str(Path(discrecontinual_equations.__file__).parent),
        module="discrecontinual_equations",
        exclude=["*.tests", "*.examples", "*.plot"],
    )
    report = PlotReport(D3Renderer(), output_dir)
    entries: list[AtlasEntry] = []
    for filename, scene in _catalogue():
        report.write(scene, filename)
        entries.append(AtlasEntry(filename, scene.title, scene.subtitle))
    entries.extend(_stage_entries(output_dir))
    report.write_atlas(entries)


def _stage_entries(output_dir: str) -> list[AtlasEntry]:
    """Write the playable pages with their own renderer; they share the atlas."""
    stage = PlotReport(StageRenderer(), output_dir)
    stage.write(bt.bogdanov_takens_stage(), bt.STAGE_FILENAME)
    stage.write(ring.ferroelectric_ring_stage(), ring.STAGE_FILENAME)
    return [bt.STAGE_ENTRY, ring.STAGE_ENTRY]


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots")
