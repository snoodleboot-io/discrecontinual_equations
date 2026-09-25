"""Build a :class:`~.stage.StageScene` from continuation results.

The equilibrium branches and the cycle branch are the library's own output;
this module only re-shapes them by frame and adds what the flow needs to be
visible. For a planar system that is the vector field sampled on the lattice
and the saddle manifolds, which start on the Taylor chart of the manifold and
are then carried by the flow until they leave the box. For a higher-dimensional
system seen through a :class:`~.stage.View`, the equilibria and cycles are
projected onto the plane, the manifolds are left out (a saddle's manifolds are
no longer curves), and the view's polynomial field is checked against the
system before it is trusted. Nothing here is specific to a system.
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
    View,
    evaluate_terms,
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
# Two traced cycles whose amplitudes agree this closely, relative to the larger
# of them, are the same orbit sampled twice rather than two coexisting cycles.
_DISTINCT_ORBIT = 0.05
# Two branches detecting one bifurcation land on it to detector accuracy, not to
# machine precision.
_SAME_EVENT = 1.0e-3
_BRANCH_POINTS = ("branch_point", "pitchfork", "transcritical")
# How many random states a view's polynomial field is checked at, and how
# closely it must agree with the system it claims to be.
_FIELD_CHECKS = 8
_FIELD_AGREEMENT = 1.0e-9
_AUTODIFF = AutomaticDifferentiation()
# Step for the finite-difference Jacobian a non-analytic field falls back to.
_STEP = 1.0e-7


class Continued:
    """The continuation results to stage: the equation, its branches, cycles.

    ``branches`` may be one branch or several - a trivial and a nontrivial
    equilibrium branch, say - each drawn as its own arc. ``terminus`` names
    what the cycle branch runs into where it ends, if known.
    """

    __slots__ = ["branches", "cycles", "equation", "parameter_index", "terminus"]

    def __init__(
        self,
        equation: DifferentialEquation,
        parameter_index: int,
        branches: Branch | Sequence[Branch],
        cycles: Sequence[CyclePoint] = (),
        terminus: str | None = None,
    ) -> None:
        self.equation = equation
        self.parameter_index = parameter_index
        self.branches = [branches] if isinstance(branches, Branch) else list(branches)
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
    view = film.lattice.view
    if view is not None:
        check_view(view, function, parameter, float(film.frames[0]))
    spacing = _spacing(film.frames)
    cycles = CycleBranch(
        [_cycle(point, view) for point in continued.cycles],
        continued.terminus,
    )
    frames: list[Frame] = []
    for value in film.frames:
        parameter.value = float(value)
        equilibria: list[Equilibrium] = []
        for branch in continued.branches:
            for item in frame_equilibria(branch, float(value), view):
                # Branches meet at branch points; the shared equilibrium is one.
                if not any(_coincide(item, seen) for seen in equilibria):
                    equilibria.append(item)
        manifolds: list[Manifold] = []
        if view is None:
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
            [] if view is not None else sample_field(function, film.lattice),
            equilibria,
            manifolds,
            cycles_at(cycles.cycles, float(value), spacing),
        )
        if film.describe is not None:
            frame.label = film.describe(frame)
        frames.append(frame)
    timeline = Timeline(
        [
            [_branch_point(point, view) for point in branch.points]
            for branch in continued.branches
        ],
        _specials(continued.branches, view),
    )
    return StageScene(film.system, film.lattice, timeline, cycles, frames)


def _specials(branches: Sequence[Branch], view: View | None) -> list[SpecialPoint]:
    """Every detected bifurcation once, even where two branches detect it.

    Branches meet at branch points and each reports the same point.
    """
    found: list[SpecialPoint] = []
    for branch in branches:
        for point in branch.special_points:
            special = _special_point(point, view)
            if not any(_same_event(s, special) for s in found):
                found.append(special)
    # A branch that folds through a pitchfork's vertex trips the fold detector
    # at the pitchfork itself; that fold is the pitchfork, not a second event.
    crossings = [s for s in found if s.kind in _BRANCH_POINTS]
    return [
        s
        for s in found
        if not (s.kind == "fold" and any(_same_place(s, c) for c in crossings))
    ]


def _same_place(a: SpecialPoint, b: SpecialPoint) -> bool:
    return abs(a.parameter - b.parameter) < _SAME_EVENT and abs(a.x - b.x) < _SAME_EVENT


def _same_event(a: SpecialPoint, b: SpecialPoint) -> bool:
    return a.kind == b.kind and _same_place(a, b)


def check_view(view: View, function, parameter, value: float) -> None:
    """Refuse a view whose polynomial field is not the system it stands for.

    The page integrates the terms, not the system, so a mistyped coefficient
    would put particles through a flow the skeleton was not computed in. The
    terms are evaluated at random states inside the view's bounds and compared
    with ``function.eval`` at the first frame's parameter.
    """
    generator = np.random.default_rng(0)
    saved = parameter.value
    parameter.value = value
    try:
        for _ in range(_FIELD_CHECKS):
            state = [float(generator.uniform(lo, hi)) for lo, hi in view.bounds]
            expected = np.array(function.eval(point=state, time=None), dtype=float)
            claimed = np.array(evaluate_terms(view.field, state, value))
            scale = 1.0 + float(np.max(np.abs(expected)))
            if float(np.max(np.abs(claimed - expected))) > _FIELD_AGREEMENT * scale:
                message = (
                    f"the view's polynomial field disagrees with the system at "
                    f"{state}: terms give {claimed.tolist()}, the system gives "
                    f"{expected.tolist()}"
                )
                raise ValueError(message)
    finally:
        parameter.value = saved


def sample_field(function, lattice: Lattice) -> list[float]:
    """The field on the lattice's ``grid x grid`` nodes, row-major ``u, v``."""
    (x0, x1), (y0, y1) = lattice.box
    field: list[float] = []
    for y in np.linspace(y0, y1, lattice.grid):
        for x in np.linspace(x0, x1, lattice.grid):
            u, v = function.eval(point=[float(x), float(y)], time=None)
            field.extend([float(u), float(v)])
    return field


def frame_equilibria(
    branch: Branch,
    value: float,
    view: View | None = None,
) -> list[Equilibrium]:
    """Every equilibrium the branch passes through at parameter ``value``.

    A branch that rounds a fold crosses a parameter value more than once, on
    different arcs, so the branch is walked pairwise and each crossing is taken:
    the state is interpolated between the two points that straddle ``value``,
    and the stability and eigenvalues come from the nearer of them, since those
    are measurements rather than quantities to average. With a ``view`` the
    interpolated state is projected onto the plane.
    """
    found: list[Equilibrium] = []
    for a, b in pairwise(branch.points):
        if (a.parameter - value) * (b.parameter - value) > 0:
            continue
        if abs(b.parameter - a.parameter) < _TOLERANCE:
            continue
        fraction = (value - a.parameter) / (b.parameter - a.parameter)
        nearer = a if fraction < _HALF else b
        state = [
            float(p) + fraction * (float(q) - float(p))
            for p, q in zip(a.state, b.state, strict=True)
        ]
        x, y = _plane(state, view)
        item = Equilibrium(x, y, nearer.stability, list(nearer.eigenvalues))
        if not any(_coincide(item, seen) for seen in found):
            found.append(item)
    return found


def _coincide(a: Equilibrium, b: Equilibrium) -> bool:
    return abs(a.x - b.x) + abs(a.y - b.y) < _COINCIDENT


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
    jacobian = _jacobian(function, equilibrium)
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


def _jacobian(function, point: np.ndarray) -> np.ndarray:
    """The field's Jacobian at ``point``, analytically where that is possible.

    Automatic differentiation evaluates the field on Taylor jets, and a field
    written with ``math`` or a fractional power - a radius, say - rejects them.
    That is not an error: the cycle and connecting-orbit solvers fall back to
    finite differences in the same situation, and so does this, so a
    transcendental field can still be staged.
    """
    try:
        return _AUTODIFF.jacobian(function, point, 0.0)
    except (TypeError, ValueError, AttributeError):
        base = np.array(function.eval(point=list(point), time=None), dtype=float)
        columns = []
        for index in range(point.size):
            shifted = np.asarray(point, dtype=float).copy()
            shifted[index] += _STEP
            moved = np.array(
                function.eval(point=list(shifted), time=None),
                dtype=float,
            )
            columns.append((moved - base) / _STEP)
        return np.column_stack(columns)


def sheets(cycles: list[Cycle]) -> list[list[int]]:
    """The cycle branch split into runs that are monotone in the parameter.

    A branch that folds visits the same parameter on each of its sheets, and a
    fold is exactly where the traced parameter turns around. Splitting there is
    what separates two genuinely distinct orbits from one orbit sampled twice,
    which no test on the parameter alone can tell apart: both look like two
    points at one value. The turning point belongs to the sheets on either
    side of it, since at a fold the two orbits really do coincide.
    """
    if not cycles:
        return []
    runs: list[list[int]] = [[0]]
    sense = 0.0
    for index in range(1, len(cycles)):
        step = cycles[index].parameter - cycles[index - 1].parameter
        if sense and step * sense < 0.0:
            runs.append([index - 1])
            sense = 0.0
        runs[-1].append(index)
        if step:
            sense = step
    return runs


def cycles_at(cycles: list[Cycle], value: float, spacing: float) -> list[int]:
    """Every distinct cycle at ``value``, smallest orbit first.

    A folded cycle branch carries two cycles at the same parameter - a stable
    and an unstable one - and taking only the nearest of them would hide half
    the picture and flicker between the two as the parameter moves. So each
    sheet of the branch contributes the one point of its own that sits nearest
    ``value``, and a sheet contributes nothing where it does not reach.

    Asking each sheet for its nearest point, rather than taking every point
    inside a window, is what makes this independent of how densely the branch
    happened to be traced. A window wide enough to catch a coarsely stepped
    sheet took two neighbours from a finely stepped one and drew a second
    circle that was not there; a window narrow enough to avoid that lost whole
    sheets. The sheet's own range is the honest question, and the half-frame
    of slack past its ends only lets a sheet still draw on the frame just
    beyond its last traced point.

    At a fold the two sheets share their turning point and report the same
    orbit twice, so an amplitude test still has the last word. Its tolerance
    is a fraction of the whole branch's amplitude range rather than of the
    amplitudes being compared, so it does not tighten as the orbits shrink.
    """
    if not cycles:
        return []
    amplitudes = [cycle.amplitude for cycle in cycles]
    tolerance = _DISTINCT_ORBIT * (max(amplitudes) - min(amplitudes))
    slack = _HALF * spacing
    nearest: list[int] = []
    for run in sheets(cycles):
        low = min(cycles[index].parameter for index in run)
        high = max(cycles[index].parameter for index in run)
        if low - slack <= value <= high + slack:
            nearest.append(
                min(run, key=lambda index: abs(cycles[index].parameter - value)),
            )
    taken: list[int] = []
    for index in sorted(nearest, key=lambda i: abs(cycles[i].parameter - value)):
        amplitude = cycles[index].amplitude
        if all(abs(cycles[other].amplitude - amplitude) > tolerance for other in taken):
            taken.append(index)
    # Smallest first, so the order a frame reports its cycles in does not flip
    # with which traced point happened to sit nearest the frame's parameter.
    return sorted(taken, key=lambda index: cycles[index].amplitude)


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


def _plane(state: Sequence[float], view: View | None) -> tuple[float, float]:
    if view is None:
        return float(state[0]), float(state[1])
    return view.project(state)


def _spacing(frames: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in frames)
    gaps = [b - a for a, b in pairwise(ordered) if b > a]
    return min(gaps) if gaps else 1.0


def _cycle(point: CyclePoint, view: View | None) -> Cycle:
    cycle = Cycle(
        float(point.parameter),
        float(point.solution.period),
        float(point.amplitude),
        [(float(m.real), float(m.imag)) for m in point.multipliers],
        [_plane(state, view) for state in point.solution.states],
    )
    cycle.error = float(point.floquet_error)
    return cycle


def _branch_point(point: ContinuationPoint, view: View | None) -> BranchPoint:
    return BranchPoint(
        float(point.parameter),
        _plane(point.state, view)[0],
        point.stability,
        point.kind,
        list(point.eigenvalues),
    )


def _special_point(point: ContinuationPoint, view: View | None) -> SpecialPoint:
    return SpecialPoint(
        float(point.parameter),
        _plane(point.state, view)[0],
        point.kind,
    )
