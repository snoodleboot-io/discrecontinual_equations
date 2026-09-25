"""Hopf-Hopf as a film: one block turning while the other is damped.

Two Hopf blocks that do not talk to each other, at frequencies 1 and 2:

    x' = mu1 x - y - x s,    y' = x + mu1 y - y s,      s = x^2 + y^2
    z' = mu2 z - 2w - z t,   w' = 2z + mu2 w - w t,     t = z^2 + w^2

with ``mu2`` held at -0.4 so the second block is damped throughout and the
codimension-two point at ``mu1 = mu2 = 0`` is approached along one axis. The
origin is the only equilibrium, and its four eigenvalues are two pairs:
``mu1 +/- i`` and ``mu2 +/- 2i``. Only the first pair moves, and when it
crosses at ``mu1 = 0`` a cycle opens in the first block while the second
keeps pulling everything onto it.

The clock is the point of this one: four eigenvalues at once, one pair
walking across the Hopf line while the other sits still on the left.
"""

import math
import sys

import numpy as np

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CyclePoint,
    CycleSeed,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.webplot.report import AtlasEntry
from discrecontinual_equations.webplot.stage import (
    Frame,
    Lattice,
    StageScene,
    StageSystem,
    Term,
    View,
)
from discrecontinual_equations.webplot.stage_builder import Continued, Film, stage_scene

try:  # python -m examples.continuation.stage_hopf_hopf
    from examples.continuation.stage_support import (
        Mu,
        continue_branch,
        equation,
        keep_cycles,
        publish,
        within,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from stage_support import (
        Mu,
        continue_branch,
        equation,
        keep_cycles,
        publish,
        within,
    )

FIRST_FREQUENCY, SECOND_FREQUENCY = 1.0, 2.0
SECOND = -0.4
P_LO, P_HI, FRAMES = -0.5, 1.2, 81
BOX = ((-1.3, 1.3), (-1.3, 1.3))
INTERVALS = 80
HOPF = 0.0
SEED_MU1 = 0.6


class Second(Parameter, name="Second", abbreviation="mu2"):
    """The damped block's parameter, held fixed and negative."""


class HopfHopf(DeterministicFunction):
    """Two decoupled Hopf blocks; a Hopf-Hopf point at mu1 = mu2 = 0."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        mu1 = self.parameters[0].value
        mu2 = self.parameters[1].value
        x, y, z, w = point[0], point[1], point[2], point[3]
        first = x * x + y * y
        second = z * z + w * w
        return [
            mu1 * x - FIRST_FREQUENCY * y - x * first,
            FIRST_FREQUENCY * x + mu1 * y - y * first,
            mu2 * z - SECOND_FREQUENCY * w - z * second,
            SECOND_FREQUENCY * z + mu2 * w - w * second,
        ]


def terms() -> list[list[Term]]:
    """The field as polynomial terms, exact in the continued parameter."""
    return [
        [
            Term(1.0, [1, 0, 0, 0], 1),
            Term(-FIRST_FREQUENCY, [0, 1, 0, 0]),
            Term(-1.0, [3, 0, 0, 0]),
            Term(-1.0, [1, 2, 0, 0]),
        ],
        [
            Term(FIRST_FREQUENCY, [1, 0, 0, 0]),
            Term(1.0, [0, 1, 0, 0], 1),
            Term(-1.0, [2, 1, 0, 0]),
            Term(-1.0, [0, 3, 0, 0]),
        ],
        [
            Term(SECOND, [0, 0, 1, 0]),
            Term(-SECOND_FREQUENCY, [0, 0, 0, 1]),
            Term(-1.0, [0, 0, 3, 0]),
            Term(-1.0, [0, 0, 1, 2]),
        ],
        [
            Term(SECOND_FREQUENCY, [0, 0, 1, 0]),
            Term(SECOND, [0, 0, 0, 1]),
            Term(-1.0, [0, 0, 2, 1]),
            Term(-1.0, [0, 0, 0, 3]),
        ],
    ]


def parameters(mu1: float) -> list[Parameter]:
    """The two parameters, with ``mu1`` the one being continued."""
    return [Mu(value=mu1), Second(value=SECOND)]


def cycle_branch() -> list[CyclePoint]:
    """The first block's cycle, at radius sqrt(mu1) with the rest at rest."""
    eq = equation(HopfHopf, parameters(SEED_MU1), count=4)
    grid = np.linspace(0.0, 1.0, INTERVALS + 1)
    angle = 2.0 * math.pi * grid
    radius = math.sqrt(SEED_MU1)
    states = np.column_stack(
        [
            radius * np.cos(angle),
            radius * np.sin(angle),
            np.zeros_like(grid),
            np.zeros_like(grid),
        ],
    )
    seed = CycleSeed(states, 2.0 * math.pi / FIRST_FREQUENCY, SEED_MU1)
    continuation = CycleContinuation(eq, 0, INTERVALS, phase_index=1, phase_value=0.0)
    downward, _ = continuation.trace(seed, 0.2, 60, direction=-1.0)
    upward, _ = continuation.trace(seed, 0.2, 40, direction=1.0)
    ordered = list(reversed(keep_cycles(upward))) + keep_cycles(downward)[1:]
    return within(ordered, (P_LO, P_HI))


def describe(frame: Frame) -> str | None:
    """Name the regime by which pair has crossed."""
    if frame.parameter < HOPF:
        return "Both pairs left of the line: everything falls to the origin"
    if frame.cycles:
        return "First block turning, second still pulling everything onto it"
    return "First pair has crossed; cycle not yet resolved at this mesh"


def hopf_hopf_stage() -> StageScene:
    """Build the stage: the origin's branch and the first block's cycle."""
    eq = equation(HopfHopf, parameters(P_LO), count=4)
    origin = continue_branch(eq, [0.0, 0.0, 0.0, 0.0], (P_LO, P_HI), ["hopf"])
    system = StageSystem(
        "Hopf-Hopf",
        "Two Hopf blocks that do not talk to each other, at frequencies 1 and "
        "2, with mu2 held negative so the second is damped throughout. The "
        "origin is the only equilibrium and its four eigenvalues are two "
        "pairs; only the first moves, and when it crosses at mu1 = 0 a cycle "
        "opens in the first block while the second keeps pulling everything "
        "onto it. The clock is the point of this one: four eigenvalues at "
        "once, one pair walking across the line while the other sits still.",
        "μ₁",
        equation=(
            r"\dot x = \mu_1 x - y - x s,\quad \dot y = x + \mu_1 y - y s,\quad "
            r"\dot z = \mu_2 z - 2w - z t,\quad \dot w = 2z + \mu_2 w - w t"
        ),
        note=(
            "The origin's branch and its Hopf come from ContinuerBuilder; the "
            "cycle from CycleContinuation, seeded at radius sqrt(mu1) in the "
            "first block with the second at rest. Seen in the plane of the "
            "first block, with the particles integrated in all four "
            "coordinates - which is why a trail can enter the drawn cycle: it "
            "is above or below the plane in the damped block, on its way down."
        ),
    )
    return stage_scene(
        Continued(eq, 0, origin, cycle_branch()),
        Film(
            np.linspace(P_LO, P_HI, FRAMES),
            Lattice(
                BOX,
                36,
                "x",
                "y",
                view=View(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                    [(-1.3, 1.3)] * 2 + [(-0.8, 0.8)] * 2,
                    terms(),
                ),
            ),
            system,
            describe,
        ),
    )


STAGE_FILENAME = "hopf_hopf_stage.html"
STAGE_ENTRY = AtlasEntry(
    STAGE_FILENAME,
    "Hopf-Hopf",
    "Four eigenvalues on the clock at once: one pair crossing the Hopf line "
    "while the other sits damped on the left.",
    kind="stage",
)


def main(output_dir: str = "plots") -> None:
    """Write the stage page (and a one-card atlas) to ``output_dir``."""
    scene = hopf_hopf_stage()
    print(f"wrote {publish(scene, STAGE_FILENAME, STAGE_ENTRY, output_dir)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots")
