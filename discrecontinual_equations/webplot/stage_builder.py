"""Build a :class:`~.stage.StageScene` from continuation results.

The equilibrium branch and the cycle branch are the library's own output; this
module only re-shapes them by frame and adds what the flow needs to be visible:
the vector field sampled on the lattice, and the saddle manifolds, which start
on the Taylor chart of the manifold and are then carried by the flow until they
leave the box. Nothing here is specific to a system - any planar field with one
continuation parameter can be staged.
"""

from collections.abc import Callable, Sequence
from itertools import pairwise

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.continuation_point import ContinuationPoint
from discrecontinual_equations.continuation.cycle_continuation import CyclePoint
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.manifold import (
    ManifoldSelection,
    StableManifold,
    TaylorManifold,
    UnstableManifold,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.webplot.stage import (
    Box,
    BranchPoint,
    Cycle,
    CycleBranch,
    Equilibrium,
    Frame,
    Lattice,
    Manifold,
    Pairs,
    SpecialPoint,
    StageScene,
    StageSystem,
    Timeline,
)

_DEFAULT_MANIFOLD_TIME = 60.0
_DEFAULT_MANIFOLD_ORDER = 5
# How far along the chart the manifold is started. Far enough off the
# equilibrium that the flow moves it in reasonable time, close enough that the
# truncated chart is still accurate there.
_CHART_OFFSET = 1.0e-3
_MANIFOLD_SAMPLES = 360
# The manifold is followed a little past the box so it leaves the frame cleanly
# rather than stopping on the edge.
_BOX_MARGIN = 0.2
_HALF = 0.5
_TOLERANCE = 1.0e-12
# Two crossings closer than this are the same equilibrium seen from both sides
# of a branch point that sits exactly on the frame's parameter.
_COINCIDENT = 1.0e-9
_AUTODIFF = AutomaticDifferentiation()


class Continued:
    """The continuation results to stage: the equation, its branch, its cycles.

    ``terminus`` names what the cycle branch runs into where it ends, if known.
    """

    __slots__ = ["branch", "cycles", "equation", "parameter_index", "terminus"]

    def __init__(
        self,
        equation: DifferentialEquation,
        parameter_index: int,
        branch: Branch,
        cycles: Sequence[CyclePoint] = (),
        terminus: str | None = None,
    ) -> None:
        self.equation = equation
        self.parameter_index = parameter_index
        self.branch = branch
        self.cycles = cycles
        self.terminus = terminus


class Film:
    """How the results are shot: the frames, the lattice, and the narration.

    ``frames`` are the parameter values sampled; the browser interpolates the
    field between them, so they need to be fine enough that the flow does not
    jump. ``describe`` may name each frame's regime.
    """

    __slots__ = ["describe", "frames", "lattice", "system"]

    def __init__(
        self,
        frames: Sequence[float],
        lattice: Lattice,
        system: StageSystem,
        describe: Callable[[Frame], str | None] | None = None,
    ) -> None:
        self.frames = frames
        self.lattice = lattice
        self.system = system
        self.describe = describe


class StageSettings:
    """How far and how accurately the saddle manifolds are followed."""

    __slots__ = ["manifold_order", "manifold_time"]

    def __init__(
        self,
        manifold_time: float = _DEFAULT_MANIFOLD_TIME,
        manifold_order: int = _DEFAULT_MANIFOLD_ORDER,
    ) -> None:
        self.manifold_time = manifold_time
        self.manifold_order = manifold_order


def stage_scene(
    continued: Continued,
    film: Film,
    settings: StageSettings | None = None,
) -> StageScene:
    """Stage ``continued`` over the frames of ``film``."""
    settings = settings or StageSettings()
    function = continued.equation.derivative
    parameter = function.parameters[continued.parameter_index]
    spacing = _spacing(film.frames)
    cycles = CycleBranch(
        [_cycle(point) for point in continued.cycles], continued.terminus,
    )
    frames: list[Frame] = []
    for value in film.frames:
        parameter.value = float(value)
        equilibria = frame_equilibria(continued.branch, float(value))
        manifolds: list[Manifold] = []
        for item in equilibria:
            if item.stability == "saddle":
                manifolds.extend(
                    saddle_manifolds(
                        function,
                        np.array([item.x, item.y]),
                        film.lattice.box,
                        settings,
                    ),
                )
        frame = Frame(
            float(value),
            sample_field(function, film.lattice),
            equilibria,
            manifolds,
            nearest_cycle(cycles.cycles, float(value), spacing),
        )
        if film.describe is not None:
            frame.label = film.describe(frame)
        frames.append(frame)
    timeline = Timeline(
        [_branch_point(point) for point in continued.branch.points],
        [_special_point(point) for point in continued.branch.special_points],
    )
    return StageScene(film.system, film.lattice, timeline, cycles, frames)


def sample_field(function, lattice: Lattice) -> list[float]:
    """The field on the lattice's ``grid x grid`` nodes, row-major ``u, v``."""
    (x0, x1), (y0, y1) = lattice.box
    field: list[float] = []
    for y in np.linspace(y0, y1, lattice.grid):
        for x in np.linspace(x0, x1, lattice.grid):
            u, v = function.eval(point=[float(x), float(y)], time=None)
            field.extend([float(u), float(v)])
    return field


def frame_equilibria(branch: Branch, value: float) -> list[Equilibrium]:
    """Every equilibrium the branch passes through at parameter ``value``.

    A branch that rounds a fold crosses a parameter value more than once, on
    different arcs, so the branch is walked pairwise and each crossing is taken:
    the state is interpolated between the two points that straddle ``value``,
    and the stability and eigenvalues come from the nearer of them, since those
    are measurements rather than quantities to average.
    """
    found: list[Equilibrium] = []
    for a, b in pairwise(branch.points):
        if (a.parameter - value) * (b.parameter - value) > 0:
            continue
        if abs(b.parameter - a.parameter) < _TOLERANCE:
            continue
        fraction = (value - a.parameter) / (b.parameter - a.parameter)
        nearer = a if fraction < _HALF else b
        x = a.state[0] + fraction * (b.state[0] - a.state[0])
        y = a.state[1] + fraction * (b.state[1] - a.state[1])
        if any(abs(item.x - x) + abs(item.y - y) < _COINCIDENT for item in found):
            continue
        found.append(Equilibrium(x, y, nearer.stability, list(nearer.eigenvalues)))
    return found


def saddle_manifolds(
    function,
    equilibrium: np.ndarray,
    box: Box,
    settings: StageSettings | None = None,
) -> list[Manifold]:
    """Both branches of the stable and unstable manifolds of a planar saddle.

    Each branch starts on the manifold's Taylor chart a small distance from the
    equilibrium, so the start is on the manifold to the chart's order rather than
    on its tangent line, and is then carried by the flow - forward for the
    unstable manifold, backward for the stable - until it leaves the box.
    """
    settings = settings or StageSettings()
    jacobian = _AUTODIFF.jacobian(function, equilibrium, 0.0)
    manifolds: list[Manifold] = []
    for kind, selection, forward in (
        ("unstable", UnstableManifold(), True),
        ("stable", StableManifold(), False),
    ):
        chart = _Chart(function, equilibrium, jacobian, selection, settings)
        for sign in (1.0, -1.0):
            points = _flow(function, chart.start(sign), box, settings, forward)
            manifolds.append(Manifold(kind, points))
    return manifolds


def nearest_cycle(cycles: list[Cycle], value: float, spacing: float) -> int | None:
    """Index of the cycle at ``value``, or ``None`` if none lies within a frame."""
    if not cycles:
        return None
    index = min(range(len(cycles)), key=lambda i: abs(cycles[i].parameter - value))
    if abs(cycles[index].parameter - value) <= _HALF * spacing:
        return index
    return None


class _Chart:
    """Where a manifold branch starts: on its Taylor chart, or its tangent."""

    __slots__ = ["_direction", "_equilibrium", "_point"]

    def __init__(
        self,
        function,
        equilibrium: np.ndarray,
        jacobian: np.ndarray,
        selection: ManifoldSelection,
        settings: StageSettings,
    ) -> None:
        self._equilibrium = equilibrium
        self._direction = _eigenvector(jacobian, selection)
        try:
            chart = TaylorManifold(order=settings.manifold_order).compute(
                function,
                equilibrium,
                jacobian,
                selection,
            )
            self._point = chart.point
        except (ValueError, TypeError, np.linalg.LinAlgError):
            # A chart the parameterisation cannot build (a resonance, say) falls
            # back to the eigenvector: exact to first order, which is all the
            # offset needs.
            self._point = None

    def start(self, sign: float) -> np.ndarray:
        if self._point is not None:
            return np.asarray(self._point([sign * _CHART_OFFSET]), dtype=float)
        return self._equilibrium + sign * _CHART_OFFSET * self._direction


def _eigenvector(jacobian: np.ndarray, selection: ManifoldSelection) -> np.ndarray:
    values, vectors = np.linalg.eig(jacobian)
    unstable = isinstance(selection, UnstableManifold)
    wanted = values.real > 0 if unstable else values.real < 0
    index = int(np.argmax(np.where(wanted, np.abs(values.real), -np.inf)))
    vector = np.real(vectors[:, index])
    return vector / np.linalg.norm(vector)


def _flow(
    function,
    start: np.ndarray,
    box: Box,
    settings: StageSettings,
    forward: bool,  # noqa: FBT001 (a direction, not a mode switch)
) -> Pairs:
    (x0, x1), (y0, y1) = box
    sense = 1.0 if forward else -1.0

    def rhs(_t, state):
        u, v = function.eval(point=[float(state[0]), float(state[1])], time=None)
        return [sense * u, sense * v]

    def leaving(_t, state):
        return min(
            state[0] - (x0 - _BOX_MARGIN),
            (x1 + _BOX_MARGIN) - state[0],
            state[1] - (y0 - _BOX_MARGIN),
            (y1 + _BOX_MARGIN) - state[1],
        )

    leaving.terminal = True
    solution = solve_ivp(
        rhs,
        [0.0, settings.manifold_time],
        start,
        rtol=1.0e-8,
        atol=1.0e-10,
        events=leaving,
        max_step=0.05,
        dense_output=True,
    )
    times = np.linspace(0.0, solution.t[-1], _MANIFOLD_SAMPLES)
    return [(float(x), float(y)) for x, y in solution.sol(times).T]


def _spacing(frames: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in frames)
    gaps = [b - a for a, b in pairwise(ordered) if b > a]
    return min(gaps) if gaps else 1.0


def _cycle(point: CyclePoint) -> Cycle:
    return Cycle(
        float(point.parameter),
        float(point.solution.period),
        float(point.amplitude),
        [(float(m.real), float(m.imag)) for m in point.multipliers],
        [(float(x), float(y)) for x, y in point.solution.states],
    )


def _branch_point(point: ContinuationPoint) -> BranchPoint:
    return BranchPoint(
        float(point.parameter),
        float(point.state[0]),
        point.stability,
        point.kind,
        list(point.eigenvalues),
    )


def _special_point(point: ContinuationPoint) -> SpecialPoint:
    return SpecialPoint(float(point.parameter), float(point.state[0]), point.kind)
