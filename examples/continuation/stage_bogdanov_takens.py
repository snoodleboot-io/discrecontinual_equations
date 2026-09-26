"""A Bogdanov-Takens slice as a film.

Fix b2 and scrub b1 through the whole story: a stable focus, an unstable cycle
born at a subcritical Hopf, the cycle swallowed by the saddle at a homoclinic
loop, then the fold where both equilibria vanish. The equilibrium branch and the
cycle branch are the library's own continuation output; :func:`stage_scene`
re-shapes them by frame and adds the sampled field and the saddle manifolds.

Run directly to write ``plots/bogdanov_takens_stage.html``; the webplot atlas
example includes it as its "play" card.
"""

import math
import sys

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CyclePoint,
    CycleSeed,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.webplot.report import AtlasEntry
from discrecontinual_equations.webplot.stage import (
    Frame,
    Lattice,
    StageScene,
    StageSystem,
)
from discrecontinual_equations.webplot.stage_builder import Continued, Film, stage_scene

try:  # python -m examples.continuation.stage_bogdanov_takens
    from examples.continuation.stage_support import (
        BranchLimits,
        continue_branch,
        equation,
        keep_cycles,
        publish,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from stage_support import (
        BranchLimits,
        continue_branch,
        equation,
        keep_cycles,
        publish,
    )

B2 = 0.5
P_LO, P_HI, FRAMES = -0.6, 0.1, 57
BOX = ((-1.25, 0.9), (-0.85, 0.85))
# 80 nodes build in minutes and carry a Floquet error the page reports beside
# the multipliers (9% where the orbit hugs the saddle; 160 would give 2.5%, 320
# 0.6%). CycleContinuation's dense finite-difference Jacobian (DEQ-15) makes the
# finer meshes hour-long builds until it is replaced.
INTERVALS = 80
GRID = 36
# The cycle continuation is seeded this far below the Hopf, where the cycle is
# large enough to be found by integration but still far from the homoclinic.
SEED_OFFSET = 0.05
_MIN_AMPLITUDE = 1.0e-3
_PERIOD_RANGE = (2.0, 400.0)
# The fold sits at b1 = 0; frames this close to it are called the fold.
_FOLD_TOLERANCE = 1.0e-9
# The branch this film wants is tighter than the shared default: it stops close
# to its own span and takes a finer step, which is what its own hand-rolled
# continuation used to do.
LIMITS = BranchLimits(step=0.015, margin=0.02, maximum_points=600)


class First(Parameter, name="First", abbreviation="p1"):
    pass


class Second(Parameter, name="Second", abbreviation="p2"):
    pass


class BogdanovTakensFamily(DeterministicFunction):
    """x' = y, y' = b1 + b2 y + x^2 + x y."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        x, y = point[0], point[1]
        return [y, b1 + b2 * y + x * x + x * y]


def slice_equation(b1: float) -> DifferentialEquation:
    """The slice at ``b1``, with b2 held fixed."""
    return equation(BogdanovTakensFamily, [First(value=b1), Second(value=B2)])


def _field(eq: DifferentialEquation, b1: float, x: float, y: float) -> np.ndarray:
    eq.derivative.parameters[0].value = b1
    return np.array(eq.derivative.eval(point=[x, y], time=None), dtype=float)


def _integrated_cycle(eq: DifferentialEquation, b1: float) -> tuple[np.ndarray, float]:
    """One period of the unstable cycle at ``b1``, found in reversed time.

    An unstable cycle is an attractor of the time-reversed flow, so integrating
    backwards from near the focus lands on it; one period is then cut at the
    ``y = 0`` section and reversed back into forward-time order.
    """
    focus = -math.sqrt(-b1)

    def reversed_flow(_t, state):
        return -_field(eq, b1, state[0], state[1])

    settled = solve_ivp(
        reversed_flow,
        [0.0, 400.0],
        [focus + 0.05, 0.0],
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.05,
    ).y[:, -1]

    def section(_t, state):
        return state[1]

    section.direction = -1.0  # a forward-time upward crossing, seen backwards
    run = solve_ivp(
        reversed_flow,
        [0.0, 200.0],
        settled,
        rtol=1.0e-10,
        atol=1.0e-12,
        events=section,
        dense_output=True,
        max_step=0.05,
    )
    crossings = run.t_events[0]
    period = float(crossings[1] - crossings[0])
    times = crossings[0] + np.linspace(0.0, period, INTERVALS + 1)
    return run.sol(times).T[::-1], period


def _trace_from(b1: float, direction: float) -> list[CyclePoint]:
    eq = slice_equation(b1)
    states, period = _integrated_cycle(eq, b1)
    continuation = CycleContinuation(eq, 0, INTERVALS, phase_index=1, phase_value=0.0)
    points, _ = continuation.trace(CycleSeed(states, period, b1), 0.02, 400, direction)
    return keep_cycles(points, _MIN_AMPLITUDE, _PERIOD_RANGE)


def cycle_branch(hopf: float) -> list[CyclePoint]:
    """The unstable cycle branch from the Hopf down to where the mesh gives out."""
    seed = hopf - SEED_OFFSET
    downward = _trace_from(seed, -1.0)
    upward = _trace_from(seed, +1.0)
    return sorted(upward[1:] + downward, key=lambda point: point.parameter)


def describe(frame: Frame) -> str | None:
    """Name the regime in this slice's own terms."""
    focus = next((e for e in frame.equilibria if e.stability != "saddle"), None)
    if not frame.equilibria:
        return "No equilibria — everything escapes"
    if frame.parameter >= -_FOLD_TOLERANCE:
        return "Fold — focus and saddle annihilate"
    if focus is not None and focus.stability != "stable":
        return "Unstable focus + saddle — no attractor"
    if frame.cycles:
        return "Unstable cycle fences the stable focus"
    return "Stable focus, basin bounded by the saddle's stable manifold"


def bogdanov_takens_stage() -> StageScene:
    """Build the stage: equilibrium branch, cycle branch, and 57 frames."""
    eq = slice_equation(P_LO)
    branch = continue_branch(
        eq,
        [-math.sqrt(-P_LO), 0.0],
        (P_LO, P_HI),
        ["fold", "hopf"],
        LIMITS,
    )
    hopf = next(
        (p.parameter for p in branch.special_points if p.kind == "hopf"),
        -(B2 * B2),
    )
    cycles = cycle_branch(hopf)
    system = StageSystem(
        "Bogdanov\u2013Takens, one slice",
        "Scrub \u03b2\u2081 and watch the mechanism: a stable focus, an unstable "
        "cycle born at a subcritical Hopf, the cycle swallowed by the saddle at a "
        "homoclinic loop, then the fold where both equilibria vanish. Every curve "
        "and every eigenvalue is the library's own continuation output; the flow "
        "is real.",
        "\u03b2\u2081",
        equation=(
            r"\dot x = y,\quad \dot y = \beta_1 + \beta_2 y + x^2 + x y,"
            r"\quad \beta_2 = " + f"{B2:g}"
        ),
        note=(
            "Equilibrium branch with eigenvalues from ContinuerBuilder; cycle "
            "branch with Floquet multipliers from CycleContinuation, seeded from a "
            "reversed-time integration because the cycle is unstable; saddle "
            "manifolds started on their Taylor charts and carried by the flow; "
            "field sampled on a 36\u00d736 grid per frame. The cycle continuation "
            "stops where the fixed 80-node mesh can no longer resolve the orbit "
            "hugging the saddle; the manifolds carry the homoclinic the rest of "
            "the way. The Floquet readout carries its own error: one multiplier "
            "is exactly 1 along the orbit, and how far the computed one sits from "
            "1 is how far all of them are off."
        ),
    )
    scene = stage_scene(
        Continued(eq, 0, branch, cycles, terminus="homoclinic"),
        Film(np.linspace(P_LO, P_HI, FRAMES), Lattice(BOX, GRID), system, describe),
    )
    for point in scene.timeline.special:
        if point.kind == "hopf":
            point.label = "Hopf (subcritical)"
    return scene


STAGE_FILENAME = "bogdanov_takens_stage.html"
STAGE_ENTRY = AtlasEntry(
    STAGE_FILENAME,
    "Bogdanov\u2013Takens, one slice",
    "Particles through the real field, eigenvalues on a spectral clock, the "
    "bifurcation diagram as a timeline you scrub.",
    kind="stage",
)


def main(output_dir: str = "plots") -> None:
    """Write the stage page (and a one-card atlas) to ``output_dir``."""
    scene = bogdanov_takens_stage()
    print(f"wrote {publish(scene, STAGE_FILENAME, STAGE_ENTRY, output_dir)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots")
