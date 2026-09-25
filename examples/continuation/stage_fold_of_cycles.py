"""A fold of cycles as a film: two cycles meet and annihilate.

``r' = (mu + r^2 - r^4) r``, drawn in the plane with ``theta' = 1``. Below
``mu = -1/4`` there is nothing but the stable origin. At ``mu = -1/4`` two
cycles appear at once out of clear air - a fold of cycles - and separate: the
outer one attracting, the inner one repelling. Between there and ``mu = 0``
the system is bistable, the inner cycle the fence between the origin's basin
and the outer cycle's. At ``mu = 0`` the inner cycle shrinks into the origin
and takes its stability with it, a subcritical Hopf run backwards.

Scrub ``mu`` and watch the pair born, separate, and one of them swallowed.
The clock shows the other half: at the fold the two cycles' Floquet
multipliers meet at ``+1``, which is what a fold of cycles *is*.
"""

import math
import sys

import numpy as np

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CyclePoint,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.webplot.report import AtlasEntry
from discrecontinual_equations.webplot.stage import (
    Frame,
    Lattice,
    StageScene,
    StageSystem,
)
from discrecontinual_equations.webplot.stage_builder import Continued, Film, stage_scene

try:  # python -m examples.continuation.<module>
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

P_LO, P_HI, FRAMES = -0.35, 0.3, 79
BOX = ((-1.45, 1.45), (-1.45, 1.45))
INTERVALS = 80
FOLD = -0.25
HOPF = 0.0
# Seeded on the outer cycle partway between the fold and the Hopf, then
# continued down: the branch rounds the fold and returns as the inner cycle.
SEED_MU = -0.1


class FoldCycles(DeterministicFunction):
    """r' = (mu + r^2 - r^4) r in Cartesian form; two cycles meet at mu = -1/4."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        squared = x * x + y * y
        growth = mu + squared - squared * squared
        return [growth * x - y, x + growth * y]


def outer_radius(mu: float) -> float:
    """The attracting cycle's radius, the larger root of r^4 - r^2 - mu."""
    return math.sqrt((1.0 + math.sqrt(1.0 + 4.0 * mu)) / 2.0)


def cycle_branch() -> list[CyclePoint]:
    """Both cycles: one continuation, because it rounds the fold between them."""
    eq = equation(FoldCycles, [Mu(value=SEED_MU)])
    seed = circle_seed(outer_radius(SEED_MU), 2.0 * math.pi, SEED_MU, INTERVALS)
    continuation = CycleContinuation(eq, 0, INTERVALS, phase_index=1, phase_value=0.0)
    downward, _ = continuation.trace(seed, 0.02, 400, direction=-1.0)
    upward, _ = continuation.trace(seed, 0.02, 200, direction=1.0)
    # The two traces share the seed and run opposite ways, so the branch is
    # continuous only if one is reversed: outer cycle in from the right, round
    # the fold, back out as the inner one. Concatenated the other way round the
    # timeline would join the inner cycle to the outer across the diagram.
    ordered = list(reversed(keep_cycles(upward))) + keep_cycles(downward)[1:]
    return within(ordered, (P_LO, P_HI))


def describe(frame: Frame) -> str | None:
    """Name the regime in the pair's own terms."""
    if frame.parameter < FOLD:
        return "Stable origin alone: no cycles yet"
    if frame.parameter < HOPF:
        return "Bistable: the inner cycle fences the origin from the outer one"
    return "One attracting cycle; the origin gave up its stability at the Hopf"


def fold_of_cycles_stage() -> StageScene:
    """Build the stage: the origin's branch and the pair of cycles."""
    eq = equation(FoldCycles, [Mu(value=P_LO)])
    origin = continue_branch(eq, [0.0, 0.0], (P_LO, P_HI), ["hopf"])
    cycles = cycle_branch()
    system = StageSystem(
        "Fold of cycles",
        "Below mu = -1/4 there is only the stable origin. At mu = -1/4 two "
        "cycles appear at once out of clear air and separate, the outer "
        "attracting and the inner repelling, and between there and mu = 0 the "
        "system is bistable with the inner cycle as the fence. At mu = 0 the "
        "inner cycle shrinks into the origin and takes its stability with it.",
        "μ",
        equation=r"\dot r = (\mu + r^2 - r^4)\,r,\quad \dot\theta = 1",
        note=(
            "The origin's branch and its Hopf come from ContinuerBuilder; both "
            "cycles come from one CycleContinuation, seeded on the outer cycle "
            "at mu = -0.1 and continued down, because the branch rounds the "
            "fold at mu = -1/4 and returns as the inner cycle. At that fold the "
            "two cycles' nontrivial Floquet multipliers meet at 1, which is "
            "what a fold of cycles is; the readout carries the error on those "
            "multipliers beside them."
        ),
    )
    scene = stage_scene(
        Continued(eq, 0, origin, cycles, terminus="fold of cycles"),
        Film(
            np.linspace(P_LO, P_HI, FRAMES),
            Lattice(BOX, 36, "x", "y"),
            system,
            describe,
        ),
    )
    for point in scene.timeline.special:
        if point.kind == "hopf":
            point.label = "Hopf (subcritical)"
    return scene


STAGE_FILENAME = "fold_of_cycles_stage.html"
STAGE_ENTRY = AtlasEntry(
    STAGE_FILENAME,
    "Fold of cycles",
    "Two cycles born at once out of clear air, one attracting and one "
    "repelling, with their Floquet multipliers meeting at 1.",
    kind="stage",
)


def main(output_dir: str = "plots") -> None:
    """Write the stage page (and a one-card atlas) to ``output_dir``."""
    scene = fold_of_cycles_stage()
    print(f"wrote {publish(scene, STAGE_FILENAME, STAGE_ENTRY, output_dir)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots")
