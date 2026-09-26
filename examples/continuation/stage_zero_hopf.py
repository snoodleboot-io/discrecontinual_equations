"""Zero-Hopf as a film: a fold that carries a circle with it.

``x' = mu1 - x^2`` coupled to a Hopf block ``y' = mu2 y - z - y r^2``,
``z' = y + mu2 z - z r^2`` with ``r^2 = y^2 + z^2``. The two halves do not
talk to each other, which is what makes the codimension-two point legible: at
``mu1 = 0`` the ``x`` half folds, two equilibria appearing out of nothing at
``x = +/- sqrt(mu1)``, and because ``mu2`` is held positive each of them
carries a cycle of radius ``sqrt(mu2)`` in the plane transverse to it. So the
fold does not produce two points, it produces two *circles*.

The system is three-dimensional, so the stage sees it through a view: the
plane of ``x`` and ``y``, with particles integrated in all three coordinates
from the polynomial terms and drawn projected.
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
    Projection,
    StageScene,
    StageSystem,
    Term,
    View,
)
from discrecontinual_equations.webplot.stage_builder import Continued, Film, stage_scene

try:  # python -m examples.continuation.stage_zero_hopf
    from examples.continuation.stage_support import (
        BranchLimits,
        Mu,
        continue_branch,
        equation,
        keep_cycles,
        publish,
        within,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from stage_support import (
        BranchLimits,
        Mu,
        continue_branch,
        equation,
        keep_cycles,
        publish,
        within,
    )

FREQUENCY = 1.0
SECOND = 0.5
P_LO, P_HI, FRAMES = -0.25, 0.6, 79
BOX = ((-1.0, 1.0), (-1.1, 1.1))
# The fold lives in x and the circle in (y, z), so the two halves of the
# codimension-two point are in different planes: (x, y) cuts across both and
# shows the circles edge-on, while (y, z) is the plane they are actually round
# in, with the two of them superimposed.
CIRCLE_BOX = ((-1.1, 1.1), (-1.1, 1.1))
PLANES = (
    Projection("x - y", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], ("x", "y"), BOX),
    Projection(
        "y - z (the circles)",
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ("y", "z"),
        CIRCLE_BOX,
    ),
    Projection(
        "x - z",
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ("x", "z"),
        ((-1.0, 1.0), (-1.1, 1.1)),
    ),
)
INTERVALS = 80
FOLD = 0.0
SEED_MU1 = 0.4


class Second(Parameter, name="Second", abbreviation="mu2"):
    """The Hopf block's parameter, held fixed and positive."""


class ZeroHopf(DeterministicFunction):
    """A fold in x carrying a Hopf circle in (y, z); zero-Hopf at mu1 = 0."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        mu1 = self.parameters[0].value
        mu2 = self.parameters[1].value
        x, y, z = point[0], point[1], point[2]
        radius = y * y + z * z
        return [
            mu1 - x * x,
            mu2 * y - FREQUENCY * z - y * radius,
            FREQUENCY * y + mu2 * z - z * radius,
        ]


def terms() -> list[list[Term]]:
    """The field as polynomial terms, exact in the continued parameter."""
    return [
        [Term(1.0, [0, 0, 0], 1), Term(-1.0, [2, 0, 0])],
        [
            Term(SECOND, [0, 1, 0]),
            Term(-FREQUENCY, [0, 0, 1]),
            Term(-1.0, [0, 3, 0]),
            Term(-1.0, [0, 1, 2]),
        ],
        [
            Term(FREQUENCY, [0, 1, 0]),
            Term(SECOND, [0, 0, 1]),
            Term(-1.0, [0, 2, 1]),
            Term(-1.0, [0, 0, 3]),
        ],
    ]


def parameters(mu1: float) -> list[Parameter]:
    """The two parameters, with ``mu1`` the one being continued."""
    return [Mu(value=mu1), Second(value=SECOND)]


def cycle_seed(mu1: float, sign: float) -> CycleSeed:
    """The circle of radius sqrt(mu2) riding on one of the two equilibria."""
    grid = np.linspace(0.0, 1.0, INTERVALS + 1)
    angle = 2.0 * math.pi * grid
    radius = math.sqrt(SECOND)
    states = np.column_stack(
        [
            np.full_like(grid, sign * math.sqrt(mu1)),
            radius * np.cos(angle),
            radius * np.sin(angle),
        ],
    )
    return CycleSeed(states, 2.0 * math.pi / FREQUENCY, mu1)


def cycle_branch() -> list[CyclePoint]:
    """Both circles: one rides each arm of the fold, so each is traced."""
    traced: list[CyclePoint] = []
    for sign in (1.0, -1.0):
        eq = equation(ZeroHopf, parameters(SEED_MU1), count=3)
        continuation = CycleContinuation(
            eq,
            0,
            INTERVALS,
            phase_index=2,
            phase_value=0.0,
        )
        downward, _ = continuation.trace(cycle_seed(SEED_MU1, sign), 0.1, 60, -1.0)
        upward, _ = continuation.trace(cycle_seed(SEED_MU1, sign), 0.1, 30, 1.0)
        arm = list(reversed(keep_cycles(upward))) + keep_cycles(downward)[1:]
        traced.extend(within(arm, (P_LO, P_HI)))
    return traced


def describe(frame: Frame) -> str | None:
    """Name the regime in the coupled system's own terms."""
    if frame.parameter < FOLD:
        return "Nothing stands: the x half has no equilibrium"
    if len(frame.cycles) > 1:
        return "Two equilibria, and a circle riding on each of them"
    if frame.cycles:
        return "Two equilibria; one circle resolved at this mesh"
    return "Two equilibria out of the fold, circles not yet resolved"


def zero_hopf_stage() -> StageScene:
    """Build the stage: the folded x branch and the circle on each arm."""
    eq = equation(ZeroHopf, parameters(P_HI), count=3)
    # The equilibria exist only for mu1 >= 0, so the branch is entered from
    # the high end and descends; it rounds its own fold at mu1 = 0 and returns
    # as the other arm, so one continuation carries both.
    arms = continue_branch(
        eq,
        [math.sqrt(P_HI), 0.0, 0.0],
        (P_HI, P_LO),
        ["fold"],
        BranchLimits(step=0.01),
    )
    system = StageSystem(
        "Zero-Hopf",
        "A fold in x and a Hopf in (y, z) that do not talk to each other, "
        "which is what makes the codimension-two point legible. At mu1 = 0 the "
        "x half folds and two equilibria appear out of nothing; because mu2 is "
        "held positive, each of them carries a cycle of radius sqrt(mu2) in "
        "the plane transverse to it. The fold does not produce two points, it "
        "produces two circles. Seen in the plane of x and y, with the "
        "particles integrated in all three coordinates.",
        "μ₁",
        equation=(
            r"\dot x = \mu_1 - x^2,\quad "
            r"\dot y = \mu_2 y - z - y r^2,\quad \dot z = y + \mu_2 z - z r^2"
        ),
        note=(
            "The folded equilibrium branch comes from ContinuerBuilder - one "
            "continuation, because it rounds the fold and returns as the other "
            "arm. Each circle comes from its own CycleContinuation, seeded "
            "analytically at radius sqrt(mu2) on the arm it rides. The saddle's "
            "one-dimensional stable manifold is drawn, the line along which "
            "the flow falls onto it; its unstable manifold is a surface, and a "
            "surface is not drawn."
        ),
    )
    return stage_scene(
        Continued(eq, 0, arms, cycle_branch(), terminus="fold"),
        Film(
            np.linspace(P_LO, P_HI, FRAMES),
            Lattice(
                BOX,
                36,
                "x",
                "y",
                view=View(
                    PLANES,
                    [(-1.0, 1.0), (-1.1, 1.1), (-1.1, 1.1)],
                    terms(),
                ),
            ),
            system,
            describe,
        ),
    )


STAGE_FILENAME = "zero_hopf_stage.html"
STAGE_ENTRY = AtlasEntry(
    STAGE_FILENAME,
    "Zero-Hopf",
    "A fold that produces not two points but two circles, seen in projection "
    "with the particles integrated in all three coordinates.",
    kind="stage",
)


def main(output_dir: str = "plots") -> None:
    """Write the stage page (and a one-card atlas) to ``output_dir``."""
    scene = zero_hopf_stage()
    print(f"wrote {publish(scene, STAGE_FILENAME, STAGE_ENTRY, output_dir)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots")
