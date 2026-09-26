"""A three-cell ferroelectric ring as a film: the rotating wave and its onset.

The Aven-Palacios-In-Longhini electric-field sensor: three Landau double-well
cells coupled forward in a ring, ``x_i' = a x_i - x_i^3 - lambda x_{i+1}``. Fix
``a = 1`` and no target field, and scrub the coupling ``lambda``. Weakly coupled,
every cell sits in the same well and the ring rests at ``+s(1, 1, 1)`` or its
mirror; above a critical coupling the cells hand their state around the ring as
a rotating wave, born with an infinite period. The symmetric states then lose
stability through a Hopf bifurcation and merge with the trivial state at
``lambda = a``.

The ring is three-dimensional, so the stage sees it through a view: the plane of
cells one and two, with particles integrated in all three cells from the ring's
polynomial terms and drawn projected. The equilibrium branches and the cycle
branch are the library's own continuation output.
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
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.systems import FerroelectricRing
from discrecontinual_equations.webplot.report import AtlasEntry
from discrecontinual_equations.webplot.stage import (
    Frame,
    Lattice,
    Projection,
    StageScene,
    StageSystem,
    Term,
    View,
)
from discrecontinual_equations.webplot.stage_builder import Continued, Film, stage_scene

try:  # python -m examples.continuation.stage_ferroelectric_ring
    from examples.continuation.stage_support import (
        BranchLimits,
        continue_branch,
        ensure_registry,
        equation,
        keep_cycles,
        publish,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from stage_support import (
        BranchLimits,
        continue_branch,
        ensure_registry,
        equation,
        keep_cycles,
        publish,
    )

CELLS = 3
GAIN = 1.0
P_LO, P_HI, FRAMES = -0.6, 1.3, 77
BOX = ((-1.7, 1.7), (-1.7, 1.7))
# The ring is unidirectionally coupled, so a rotating wave passes through the
# cells in order and which pair is on screen decides what the phase lag looks
# like. The ring's cyclic symmetry makes the three planes congruent - same
# extent, same enclosed area - but not identical: the wave runs x1 -> x2 -> x3,
# so x1 against x2 and x2 against x3 are traversed one way and x1 against x3
# the other. Turning the loop the other way round is the coupling's direction,
# visible. The same box serves all three, since the cells share their range.
PLANES = (
    Projection(
        "x\u2081 - x\u2082",
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ("x\u2081", "x\u2082"),
        BOX,
    ),
    Projection(
        "x\u2082 - x\u2083",
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ("x\u2082", "x\u2083"),
        BOX,
    ),
    Projection(
        "x\u2081 - x\u2083",
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ("x\u2081", "x\u2083"),
        BOX,
    ),
)
# CycleContinuation still builds a dense finite-difference Jacobian (DEQ-15), so
# the trace costs ~1 s/step at 80 nodes and ~8 s/step at 160; 80 keeps the
# Floquet error near 3%, which the page reports beside the multipliers.
INTERVALS = 80
ARCLENGTH = 0.04
STEPS_DOWN, STEPS_UP = 700, 60
# The rotating wave is found by integration here, well above its onset, and
# continued both ways from there.
SEED_COUPLING = 1.2
_MIN_AMPLITUDE = 1.0e-3
_PERIOD_RANGE = (1.0, 400.0)
# The wave onsets at lambda ~ 0.34 with an infinite period (measured by
# integration; DEQ-14). A fixed mesh cannot follow the branch there, so frames
# between the onset and where the trace ends know the wave exists without
# drawing it.
ONSET = 0.34
# A state whose first two cells agree this closely lies on the symmetric axis.
_SYMMETRIC_TOLERANCE = 1.0e-6
# The ring's branch stops close to its own span, as its own hand-rolled
# continuation did.
LIMITS = BranchLimits(step=0.02, margin=0.02, maximum_points=800)


class Coupling(Parameter, name="Coupling", abbreviation="lam"):
    pass


class Gain(Parameter, name="Gain", abbreviation="a"):
    pass


class Field(Parameter, name="Field", abbreviation="eps"):
    pass


def ring_equation(coupling: float) -> DifferentialEquation:
    """The ring at this coupling, with the gain fixed and no applied field."""
    return equation(
        FerroelectricRing,
        [Coupling(value=coupling), Gain(value=GAIN), Field(value=0.0)],
        count=CELLS,
    )


def ring_terms() -> list[list[Term]]:
    """The ring's right-hand side as polynomial terms, cell by cell."""
    terms = []
    for i in range(CELLS):
        own = [0] * CELLS
        own[i] = 1
        cube = [0] * CELLS
        cube[i] = 3
        neighbour = [0] * CELLS
        neighbour[(i + 1) % CELLS] = 1
        terms.append([Term(GAIN, own), Term(-1.0, cube), Term(-1.0, neighbour, 1)])
    return terms


def _flow(eq: DifferentialEquation, coupling: float):
    eq.derivative.parameters[0].value = coupling

    def rhs(_t, state):
        return eq.derivative.eval(point=list(state), time=None)

    return rhs


def _integrated_wave(eq: DifferentialEquation, coupling: float):
    """One period of the rotating wave at ``coupling``, cut at cell one's zero."""
    rhs = _flow(eq, coupling)
    settled = solve_ivp(
        rhs,
        [0.0, 300.0],
        [0.9, -0.2, -0.9],
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.05,
    ).y[:, -1]

    def section(_t, state):
        return state[0]

    section.direction = 1.0
    run = solve_ivp(
        rhs,
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
    return run.sol(times).T, period


def _trace_from(coupling: float, direction: float) -> list[CyclePoint]:
    eq = ring_equation(coupling)
    states, period = _integrated_wave(eq, coupling)
    continuation = CycleContinuation(eq, 0, INTERVALS, phase_index=0, phase_value=0.0)
    points, _ = continuation.trace(
        CycleSeed(states, period, coupling),
        ARCLENGTH,
        STEPS_DOWN if direction < 0 else STEPS_UP,
        direction,
    )
    return keep_cycles(points, _MIN_AMPLITUDE, _PERIOD_RANGE)


def wave_branch() -> list[CyclePoint]:
    """The rotating wave, from where the mesh gives out near onset up to P_HI."""
    downward = _trace_from(SEED_COUPLING, -1.0)
    upward = _trace_from(SEED_COUPLING, +1.0)
    return sorted(upward[1:] + downward, key=lambda point: point.parameter)


def _branch(seed: list[float], detectors: list[str]):
    return continue_branch(ring_equation(P_LO), seed, (P_LO, P_HI), detectors, LIMITS)


def describe(frame: Frame) -> str | None:
    """Name the regime in the ring's own terms."""
    symmetric = [
        e
        for e in frame.equilibria
        if abs(e.x - e.y) < _SYMMETRIC_TOLERANCE and e.x != 0
    ]
    stable_symmetric = any(e.stability == "stable" for e in symmetric)
    if frame.cycles and stable_symmetric:
        return "Rotating wave coexists with the symmetric states"
    if frame.cycles:
        return "Rotating wave: cells hand their state around the ring"
    if frame.parameter > ONSET and stable_symmetric:
        return "Rotating wave (period beyond the mesh) with the symmetric states"
    if frame.parameter > ONSET:
        return "Rotating wave (period beyond the mesh); symmetric states unstable"
    if stable_symmetric:
        return "Every cell in the same well: two symmetric states"
    return "Symmetric states unstable; trivial state ahead"


def ferroelectric_ring_stage() -> StageScene:
    """Build the stage: the symmetric branch, the trivial branch, the wave."""
    ensure_registry()
    s = math.sqrt(GAIN - P_LO)
    # The symmetric branch folds through the pitchfork vertex at lambda = a and
    # comes back as its mirror, so one continuation covers both +s and -s.
    symmetric = _branch([s] * CELLS, ["fold", "hopf", "branch_point"])
    trivial = _branch([0.0] * CELLS, ["hopf", "branch_point"])
    waves = wave_branch()
    system = StageSystem(
        "Ferroelectric ring, three cells",
        "Three Landau double-well cells coupled forward in a ring - the "
        "electric-field sensor. Scrub the coupling: the cells rest together in "
        "one well, then above a critical coupling hand their state around the "
        "ring as a rotating wave born with an infinite period; the symmetric "
        "states lose stability at a Hopf and merge with the trivial state at "
        "λ = a. Seen in the plane of cells one and two; the particles are "
        "integrated in all three cells.",
        "λ",
        equation=r"\dot x_i = a\,x_i - x_i^3 - \lambda\,x_{i+1},\quad a = 1",
        note=(
            "Symmetric branches ±s(1, 1, 1), the trivial branch, and the "
            "rotating wave from CycleContinuation, seeded by integration at "
            "λ = 1.2 and continued down toward its onset until the mesh gives "
            "out as the period diverges. Particles are integrated in three "
            "dimensions from the ring's polynomial terms and drawn projected onto "
            "(x₁, x₂); the saddle manifolds are surfaces there and are not "
            "drawn. Every equilibrium of the ring lies on the branches shown "
            "except the six mixed states near λ = 0, which the flow reveals "
            "as slow corners. The plane picker swaps which pair of cells is on "
            "screen: the three are congruent by the ring's symmetry, but a "
            "rotating wave turns one way in x₁-x₂ and x₂-x₃ and the other way "
            "in x₁-x₃, which is the coupling's direction made visible."
        ),
    )
    lattice = Lattice(
        BOX,
        36,
        "x₁",
        "x₂",
        view=View(
            PLANES,
            [(-1.7, 1.7)] * CELLS,
            ring_terms(),
        ),
    )
    scene = stage_scene(
        Continued(
            ring_equation(P_LO),
            0,
            [symmetric, trivial],
            waves,
            terminus="infinite period",
        ),
        Film(np.linspace(P_LO, P_HI, FRAMES), lattice, system, describe),
    )
    for point in scene.timeline.special:
        if point.kind == "branch_point":
            point.label = "symmetry breaking (λ = a)"
    return scene


STAGE_FILENAME = "ferroelectric_ring_stage.html"
STAGE_ENTRY = AtlasEntry(
    STAGE_FILENAME,
    "Ferroelectric ring, three cells",
    "The rotating wave of the electric-field sensor, born with an infinite "
    "period; particles integrated in all three cells and drawn projected.",
    kind="stage",
)


def main(output_dir: str = "plots") -> None:
    """Write the stage page (and a one-card atlas) to ``output_dir``."""
    scene = ferroelectric_ring_stage()
    print(f"wrote {publish(scene, STAGE_FILENAME, STAGE_ENTRY, output_dir)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots")
