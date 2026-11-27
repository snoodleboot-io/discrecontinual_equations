"""The van der Pol canard explosion as a film.

``x' = y - x^3/3 + x``, ``y' = eps (a - x)`` at ``eps = 0.05`` - van der Pol
with constant forcing ``a`` on the fast time (Desroches et al., SIAM Review 54,
2012, eq. 2.7). The equilibrium sits at ``x = a`` on the cubic critical manifold
and loses stability at a Hopf at ``a = 1``, where it crosses the fold. Just below,
the cycle born there grows from Hopf size to a full relaxation oscillation, and it
does so inside an exponentially small window of ``a``: every cycle of the
explosion has ``a = 0.9934909325...`` to ten figures. In between it follows the
*repelling* slow manifold, which no forward integration can do - a canard.

Because the explosion lives in a window of about ``1e-9``, the film does not
scrub ``a`` itself. It scrubs ``s`` with ``a = a_c + eta sinh(s)``, ``eta =
1e-10``: logarithmic in ``|a - a_c|`` away from the canard, linear through it,
so the Hopf, the explosion and the relaxation each get room. ``a_c`` is the
maximal canard, computed before anything else as the parameter where the
attracting and repelling slow manifolds - each a family of boundary-value orbit
segments seeded by Fenichel's expansion - meet on the section ``x = 1`` through
the fold. On every frame both manifolds are drawn (continued in ``a`` from the
frame before) with the cycle, and the narration says how far apart they pass.
"""

import math
import sys

import numpy as np
from scipy import integrate

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CyclePoint,
    CycleSeed,
)
from discrecontinual_equations.continuation.slow_manifold import (
    CanardSearch,
    FastSlowSystem,
    FenichelExpansion,
    OrbitSegment,
    SlowManifold,
    SlowManifoldError,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.webplot.report import AtlasEntry
from discrecontinual_equations.webplot.stage import (
    Frame,
    Lattice,
    Manifold,
    SpecialPoint,
    StageScene,
    StageSystem,
)
from discrecontinual_equations.webplot.stage_builder import Continued, Film, stage_scene

try:  # python -m examples.continuation.<module>
    from examples.continuation.stage_support import (
        BranchLimits,
        Mu,
        continue_branch,
        equation,
        publish,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from stage_support import BranchLimits, Mu, continue_branch, equation, publish

EPSILON = 0.05
# a = a_c + ETA sinh(s). The explosion runs over |a - a_c| from about 1e-6 to
# 1e-11, which this puts between s = 12 and s = -7.
ETA = 1.0e-10
S_LO, S_HI, FRAMES = -18.0, 20.5, 97
BOX = ((-2.5, 2.5), (-1.0, 1.0))
# The cycle branch: seeded from the stable cycle integrated at SEED_A (for
# SETTLE time units, the approach being slow this near the Hopf), on a
# uniform mesh of INTERVALS under Hermite-Simpson, and traced towards smaller a
# in arclength steps of ARCLENGTH (the norm is over every node, so a step moves
# each node by about ARCLENGTH / sqrt(INTERVALS)).
SEED_A = 0.999
INTERVALS = 200
ARCLENGTH = 1.0
SETTLE = 20000.0
MAX_STEPS = 400
# Where the slow manifolds start, on the order-ORDER Fenichel expansion: the
# attracting one on the outer branch at x = 2, the repelling one in the middle
# at x = 0, and the section both are brought to, through the fold.
ORDER = 3
ATTRACTING_X = 2.0
REPELLING_X = 0.0
FOLD = 1.0
# How many points of each manifold the page is given.
MANIFOLD_POINTS = 160
# The cycle's leftmost x: past -1 it has jumped to the left branch (a head).
TOP_FOLD = -1.0
RELAXED = -1.7
# Krupa and Szmolyan's series through eps^4, for the narration.
SERIES = (1.0, -1.0 / 8.0, -3.0 / 32.0, -173.0 / 1024.0, -7593.0 / 16384.0)
HOPF = 1.0


class VanDerPolFastSlow(FastSlowSystem):
    """``x' = y - x^3/3 + x``, ``y' = eps (a - x)``, for the slow manifolds."""

    def fast(self, x, y, epsilon, parameter):  # noqa: ARG002 (protocol signature)
        return y - x**3 * (1.0 / 3.0) + x

    def slow(self, x, y, epsilon, parameter):  # noqa: ARG002 (protocol signature)
        return parameter - x


class Centre(Parameter, name="Centre", abbreviation="a_c"):
    """The canard's ``a``, held fixed: the origin of the stretched axis."""


class Stretched(DeterministicFunction):
    """Van der Pol at ``a = a_c + eta sinh(s)``; ``s`` is the film's parameter."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        a = forcing(self.parameters[0].value, self.parameters[1].value)
        x, y = point[0], point[1]
        return [y - x * x * x / 3.0 + x, EPSILON * (a - x)]


class Forced(DeterministicFunction):
    """Van der Pol with the forcing ``a`` itself as the parameter."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        a = self.parameters[0].value
        x, y = point[0], point[1]
        return [y - x * x * x / 3.0 + x, EPSILON * (a - x)]


def forcing(stretched: float, centre: float) -> float:
    """``a`` on the film's axis ``s``."""
    return centre + ETA * math.sinh(stretched)


def stretched_axis(a: float, centre: float) -> float:
    """``s`` for a forcing ``a``: the inverse of :func:`forcing`."""
    return math.asinh((a - centre) / ETA)


def series(epsilon: float) -> float:
    """The canard's ``a`` from the asymptotic series, through ``eps^4``."""
    return sum(c * epsilon**k for k, c in enumerate(SERIES))


def critical(x: float) -> float:
    """The critical manifold ``y = x^3/3 - x``."""
    return x**3 / 3.0 - x


class SlowManifolds:
    """Both slow manifolds, continued in ``a`` from frame to frame."""

    __slots__ = ["attracting", "repelling", "search", "segments"]

    def __init__(self, system: VanDerPolFastSlow, parameter: float) -> None:
        expansion = FenichelExpansion(system, ORDER)
        start_a = expansion.at(critical(ATTRACTING_X), ATTRACTING_X, parameter)
        start_r = expansion.at(critical(REPELLING_X), REPELLING_X, parameter)
        self.attracting = SlowManifold(system, start_a)
        self.repelling = SlowManifold(system, start_r)
        self.search = CanardSearch(self.attracting, self.repelling, FOLD)
        self.segments: dict[str, OrbitSegment | None] = {
            "attracting": None,
            "repelling": None,
        }

    def at(self, a: float) -> tuple[OrbitSegment | None, OrbitSegment | None]:
        """Both manifolds up to the fold section at ``a``, or ``None`` where not."""
        return (
            self._continue("attracting", self.attracting, a),
            self._continue("repelling", self.repelling, a),
        )

    def _continue(self, name: str, manifold: SlowManifold, a: float):
        previous = self.segments[name]
        try:
            if previous is None:
                segment = manifold.sweep(a, FOLD)[-1]
            else:
                segment = manifold.follow(previous, [a], FOLD)[-1]
        except SlowManifoldError:
            try:
                segment = manifold.sweep(a, FOLD)[-1]
            except SlowManifoldError:
                segment = None
        self.segments[name] = segment
        return segment


def locate_canard() -> tuple[float, SlowManifolds]:
    """The maximal canard's ``a`` at ``EPSILON``, and the manifolds that found it."""
    system = VanDerPolFastSlow(EPSILON)
    manifolds = SlowManifolds(system, series(EPSILON))
    canard = manifolds.search.locate(0.993, 0.994)
    return canard.parameter, manifolds


def cycle_branch() -> list[CyclePoint]:
    """The cycle branch in ``a``, from near the Hopf through the explosion."""
    eq = equation(Forced, [Mu(value=SEED_A)])
    continuation = CycleContinuation(
        eq,
        0,
        INTERVALS,
        phase_index=1,
        phase_value=critical(FOLD),
    )
    seed = _integrated_seed(SEED_A)
    points, _ = continuation.trace(seed, ARCLENGTH, MAX_STEPS, direction=-1.0)
    return points


def _integrated_seed(a: float) -> CycleSeed:
    """The stable cycle at ``a``, integrated onto and sampled over one period."""

    def rhs(_t, state):
        return [state[1] - critical(state[0]), EPSILON * (a - state[0])]

    def upward(_t, state):
        return state[1] - critical(FOLD)

    upward.direction = 1.0
    settled = integrate.solve_ivp(
        rhs, (0.0, SETTLE), [a + 0.01, critical(FOLD)], rtol=1e-11
    )
    orbit = integrate.solve_ivp(
        rhs,
        (0.0, 400.0),
        settled.y[:, -1],
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-14,
        events=upward,
        dense_output=True,
    )
    start, finish = orbit.t_events[0][-2], orbit.t_events[0][-1]
    states = orbit.sol(np.linspace(start, finish, INTERVALS + 1)).T
    return CycleSeed(states, finish - start, a)


def liouville(point: CyclePoint) -> tuple[np.ndarray, float]:
    """The cycle's multipliers by Liouville's formula, and their error.

    For a planar flow the nontrivial multiplier is ``exp(int_0^T div f dt)``
    exactly, and here ``div f = 1 - x^2``. Through the explosion the monodromy
    that CycleContinuation integrates is a product of factors as large as
    ``e^40`` and as small as ``e^-80``, and the multipliers it returns are
    meaningless (it reports errors above one there, and the page would draw
    those cycles as unstable); the trace needs no product at all. It is taken
    by the trapezoidal rule on the cycle's own nodes, and its error estimated
    against Simpson's rule on the same nodes.
    """
    x = point.solution.states[:, 0]
    times = point.mesh * point.solution.period
    divergence = 1.0 - x * x
    trapezoid = float(np.trapezoid(divergence, times))
    simpson = float(integrate.simpson(divergence, x=times))
    multiplier = math.exp(simpson)
    return np.array([1.0, multiplier]), multiplier * abs(simpson - trapezoid)


def on_axis(points: list[CyclePoint], centre: float) -> list[CyclePoint]:
    """The cycles re-labelled by ``s``, kept while ``a`` keeps falling.

    The film's axis is ``s``, so each cycle is re-issued at its ``s``, with its
    multipliers from :func:`liouville`; and the run is cut where ``a`` stops
    decreasing or leaves the filmed range, so a rounding-level wobble in the
    flat middle of the explosion cannot split the branch into spurious sheets.
    """
    kept: list[CyclePoint] = []
    for point in points:
        s = stretched_axis(point.parameter, centre)
        if s < S_LO:
            break
        if kept and point.parameter >= forcing(kept[-1].parameter, centre):
            continue
        multipliers, error = liouville(point)
        kept.append(CyclePoint(s, point.solution, multipliers, point.mesh, error))
    return [point for point in kept if point.parameter <= S_HI]


def _thin(segment: OrbitSegment) -> list[tuple[float, float]]:
    picks = np.unique(
        np.linspace(0, len(segment.mesh) - 1, MANIFOLD_POINTS).astype(int),
    )
    return [(float(segment.states[i, 0]), float(segment.states[i, 1])) for i in picks]


def leftmost(point: CyclePoint) -> float:
    """The cycle's smallest ``x``: how far along ``S_r`` it gets before it jumps."""
    return float(np.min(point.solution.states[:, 0]))


def regime(frame: Frame, cycles: list[CyclePoint], gap: float | None) -> str:
    """Name the frame's regime, with how far the slow manifolds miss each other."""
    distance = forcing(frame.parameter, 0.0)
    head = f"a - a_c = {distance:+.2e}"
    miss = "" if gap is None else f"; S_a and S_r miss by {gap:+.1e} at x = 1"
    if frame.parameter > stretched_axis(HOPF, 0.0) or not frame.cycles:
        return f"{head}: stable focus past the Hopf{miss}"
    reach = min(leftmost(cycles[index]) for index in frame.cycles)
    if reach > FOLD - 0.5:
        story = "small cycle born at the Hopf"
    elif reach > TOP_FOLD:
        story = "canard without head: along S_r, then back to the right"
    elif reach > RELAXED:
        story = "canard with head: along S_r, then a jump to the left branch"
    else:
        story = "relaxation oscillation"
    return f"{head}: {story}{miss}"


def canard_stage() -> StageScene:
    """Build the stage: the canard located, the branches, the manifolds."""
    centre, manifolds = locate_canard()
    eq = equation(Stretched, [Mu(value=S_LO), Centre(value=centre)])
    origin = continue_branch(
        eq,
        [forcing(S_LO, centre), critical(forcing(S_LO, centre))],
        (S_LO, S_HI),
        ["hopf"],
        BranchLimits(step=0.1, maximum_points=2000),
    )
    cycles = on_axis(cycle_branch(), centre)
    system = StageSystem(
        "Van der Pol canard explosion",
        "Constant forcing a moves the equilibrium along the cubic critical "
        "manifold; at a = 1 it crosses the fold and a cycle is born. Within "
        "1e-9 of the maximal canard a_c that cycle explodes from Hopf size to a "
        "relaxation oscillation by following the repelling slow manifold S_r. "
        "The axis is s with a = a_c + 1e-10 sinh(s): logarithmic in |a - a_c| "
        "away from the canard, linear through it.",
        "s",
        equation=(
            r"\dot x = y - x^3/3 + x,\quad \dot y = \varepsilon\,(a - x),\quad "
            r"\varepsilon = 0.05,\quad a = a_c + 10^{-10}\sinh s"
        ),
        note=_note(centre),
    )
    scene = stage_scene(
        Continued(eq, 0, origin, cycles),
        Film(
            np.linspace(S_LO, S_HI, FRAMES),
            Lattice(BOX, 36, "x", "y"),
            system,
        ),
    )
    for frame in scene.frames:
        attracting, repelling = manifolds.at(forcing(frame.parameter, centre))
        gap = None
        if attracting is not None:
            frame.manifolds.append(Manifold("stable-slow", _thin(attracting)))
        if repelling is not None:
            frame.manifolds.append(Manifold("unstable-slow", _thin(repelling)))
        if attracting is not None and repelling is not None:
            gap = manifolds.search.gap(attracting, repelling)
        frame.label = regime(frame, cycles, gap)
    for point in scene.timeline.special:
        if point.kind == "hopf":
            point.label = "Hopf (a = 1)"
    scene.timeline.special.append(
        SpecialPoint(0.0, centre, "canard", "Maximal canard (BVP)"),
    )
    return scene


def _note(centre: float) -> str:
    estimate = series(EPSILON)
    return (
        f"a_c = {centre:.12f} is where the attracting slow manifold (from x = 2, "
        "forward) and the repelling one (from x = 0, backward) cross x = 1 at "
        "the same height: both are families of boundary-value orbit segments "
        "(Desroches et al. 2012, problem 8.1) started on Fenichel's expansion to "
        "eps^3, and the gap between them is zeroed by secant. Desroches et al. "
        "report 0.993491 for every orbit of the explosion at this eps. "
        "Krupa and Szmolyan's series 1 - eps/8 - 3 eps^2/32 - 173 eps^3/1024 - "
        f"7593 eps^4/16384 gives {estimate:.10f}, {estimate - centre:+.1e} away: "
        "the series is asymptotic and its error at eps = 0.05 is a thousand "
        "times the width of the explosion, which is why the film's origin is "
        "the computed canard and not the formula. The cycles are "
        "CycleContinuation on 200 Hermite-Simpson intervals, traced in a and "
        "placed on the s axis. Their multipliers are Liouville's, exp of the "
        "integral of 1 - x^2 over the period: through the explosion the "
        "monodromy is a product of factors from e^40 to e^-80 and the one "
        "integrated along the orbit is meaningless there, while the trace is "
        "exact. Every cycle is stable, the canards with head by up to e^-100."
    )


STAGE_FILENAME = "canard_explosion_stage.html"
STAGE_ENTRY = AtlasEntry(
    STAGE_FILENAME,
    "Van der Pol canard explosion",
    "The cycle grows from Hopf size to relaxation within 1e-9 of the maximal "
    "canard, following the repelling slow manifold; both slow manifolds drawn "
    "from boundary-value families.",
    kind="stage",
)


def main(output_dir: str = "plots") -> None:
    """Write the stage page (and a one-card atlas) to ``output_dir``."""
    path = publish(canard_stage(), STAGE_FILENAME, STAGE_ENTRY, output_dir)
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots")
