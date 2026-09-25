"""Three cycles at once: the extended Bautin form as a film.

``r' = r (b1 + b2 r^2 + b3 r^4 - r^6)``, drawn in the plane with
``theta' = 1``. The cycles are the positive roots of a cubic in ``u = r^2``,
so with ``b2`` and ``b3`` held at -2.9 and 3.3 the count is whatever that
cubic allows: one small cycle just past the Hopf at ``b1 = 0``, then **three
at once** between two folds of cycles, then one large one again. Nested
attracting and repelling cycles, alternating, with two basins between them.

This is the system that says most about the instrument: at ``b1 = 0.5`` there
are three coexisting cycles at radii 0.48, 1.07 and 1.39, every one of them
drawn, and all six of their Floquet multipliers on the clock at once.
"""

import math
import sys

import numpy as np

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CyclePoint,
)
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

try:  # python -m examples.continuation.stage_extended_bautin
    from examples.continuation.stage_support import (
        Mu,
        circle_seed,
        continue_branch,
        equation,
        keep_cycles,
        publish,
        within,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from stage_support import (
        Mu,
        circle_seed,
        continue_branch,
        equation,
        keep_cycles,
        publish,
        within,
    )

SECOND, THIRD = -2.9, 3.3
P_LO, P_HI, FRAMES = -0.15, 1.05, 85
BOX = ((-1.7, 1.7), (-1.7, 1.7))
INTERVALS = 80
HOPF = 0.0
# The two folds of cycles, where the cubic in r^2 has a double root.
LOWER_FOLD, UPPER_FOLD = 0.288, 0.768
# Seeded above the upper fold, where only the outermost cycle exists, so one
# continuation can fold twice on its way down and trace all three sheets. A
# coarser step than 0.1 rounds the folds too fast and jumps between sheets.
SEED_B1 = 0.95
# A root this close to the real axis, and this far above zero, is a real cycle.
_REAL = 1.0e-9


class Second(Parameter, name="Second", abbreviation="b2"):
    """The quintic coefficient, held fixed."""


class Third(Parameter, name="Third", abbreviation="b3"):
    """The septic coefficient, held fixed."""


class ExtendedBautin(DeterministicFunction):
    """r' = r (b1 + b2 r^2 + b3 r^4 - r^6) in Cartesian form."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        b3 = self.parameters[2].value
        x, y = point[0], point[1]
        radius = x * x + y * y
        growth = b1 + b2 * radius + b3 * radius * radius - radius**3
        return [growth * x - y, x + growth * y]


def cycle_radii(b1: float) -> list[float]:
    """Every cycle radius at ``b1``: the positive roots of the cubic in r^2."""
    roots = np.roots([1.0, -THIRD, -SECOND, -b1])
    return sorted(
        float(root.real) ** 0.5
        for root in roots
        if abs(root.imag) < _REAL and root.real > _REAL
    )


def parameters(b1: float) -> list[Parameter]:
    """The three coefficients, with ``b1`` the one being continued."""
    return [Mu(value=b1), Second(value=SECOND), Third(value=THIRD)]


def cycle_branch() -> list[CyclePoint]:
    """All three sheets: one continuation that folds twice on its way down."""
    eq = equation(ExtendedBautin, parameters(SEED_B1))
    radius = cycle_radii(SEED_B1)[-1]
    seed = circle_seed(radius, 2.0 * math.pi, SEED_B1, INTERVALS)
    continuation = CycleContinuation(eq, 0, INTERVALS, phase_index=1, phase_value=0.0)
    downward, _ = continuation.trace(seed, 0.1, 140, direction=-1.0)
    upward, _ = continuation.trace(seed, 0.1, 25, direction=1.0)
    ordered = list(reversed(keep_cycles(upward))) + keep_cycles(downward)[1:]
    return within(ordered, (P_LO, P_HI))


def describe(frame: Frame) -> str | None:
    """Name the regime by how many cycles are standing."""
    count = len(frame.cycles)
    if frame.parameter < HOPF and not count:
        return "Stable origin alone"
    if count > 1:
        return f"{count} nested cycles, attracting and repelling in turn"
    if count == 1:
        return "One attracting cycle around an unstable origin"
    return "No cycle resolved at this mesh"


def extended_bautin_stage() -> StageScene:
    """Build the stage: the origin's branch and all three cycle sheets."""
    eq = equation(ExtendedBautin, parameters(P_LO))
    origin = continue_branch(eq, [0.0, 0.0], (P_LO, P_HI), ["hopf"])
    system = StageSystem(
        "Extended Bautin: three cycles at once",
        "The cycles are the positive roots of a cubic in r^2, so their number "
        "is whatever that cubic allows. Past the Hopf at b1 = 0 there is one "
        "small cycle; at b1 = 0.288 a fold of cycles brings two more into "
        "being, and for a while there are three nested orbits, attracting and "
        "repelling in turn with a basin between each pair; at b1 = 0.768 a "
        "second fold takes two of them away again.",
        "β₁",
        equation=(
            r"\dot r = r\,(\beta_1 + \beta_2 r^2 + \beta_3 r^4 - r^6),"
            r"\quad \beta_2 = -2.9,\ \beta_3 = 3.3"
        ),
        note=(
            "The origin's branch and its Hopf come from ContinuerBuilder. All "
            "three cycle sheets come from one CycleContinuation, seeded on the "
            "outermost cycle at b1 = 0.95 and continued down: the branch folds "
            "twice on the way and returns along each sheet in turn, so a single "
            "trace carries every cycle. Each frame draws every cycle standing "
            "at its parameter and puts all their multipliers on the clock."
        ),
    )
    scene = stage_scene(
        Continued(eq, 0, origin, cycle_branch(), terminus="fold of cycles"),
        Film(
            np.linspace(P_LO, P_HI, FRAMES),
            Lattice(BOX, 36, "x", "y"),
            system,
            describe,
        ),
    )
    for point in scene.timeline.special:
        if point.kind == "hopf":
            point.label = "Hopf (supercritical)"
    return scene


STAGE_FILENAME = "extended_bautin_stage.html"
STAGE_ENTRY = AtlasEntry(
    STAGE_FILENAME,
    "Extended Bautin: three cycles at once",
    "Nested attracting and repelling cycles, brought in and taken away by two "
    "folds of cycles, with every multiplier on the clock.",
    kind="stage",
)


def main(output_dir: str = "plots") -> None:
    """Write the stage page (and a one-card atlas) to ``output_dir``."""
    scene = extended_bautin_stage()
    print(f"wrote {publish(scene, STAGE_FILENAME, STAGE_ENTRY, output_dir)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots")
