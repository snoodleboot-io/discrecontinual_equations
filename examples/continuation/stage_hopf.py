"""The Hopf bifurcation as a film: a pair crosses and a cycle appears.

The planar normal form ``x' = mu x - y - x r^2``, ``y' = x + mu y - y r^2``.
The origin is the only equilibrium at every ``mu``, and its eigenvalues are
``mu +/- i``: a stable focus below zero, unstable above. They cross the
imaginary axis together at ``mu = 0`` and a cycle of radius ``sqrt(mu)``
appears, attracting, growing as the square root - the supercritical Hopf in
its plainest form, and the thing every other Hopf in the library is a
disguised version of.

Watch the two eigenvalues walk across the Hopf line on the clock while the
cycle opens out on the stage.
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

try:  # python -m examples.continuation.stage_hopf
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

P_LO, P_HI, FRAMES = -0.5, 1.5, 81
BOX = ((-1.5, 1.5), (-1.5, 1.5))
INTERVALS = 80
HOPF = 0.0
# Seeded where the exact cycle is comfortably round, and continued both ways.
SEED_MU = 0.6


class Hopf(DeterministicFunction):
    """x' = mu x - y - x r^2, y' = x + mu y - y r^2; a Hopf at mu = 0."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        radius = x * x + y * y
        return [mu * x - y - x * radius, x + mu * y - y * radius]


def cycle_branch() -> list[CyclePoint]:
    """The cycle, which the normal form puts at radius sqrt(mu)."""
    eq = equation(Hopf, [Mu(value=SEED_MU)])
    seed = circle_seed(math.sqrt(SEED_MU), 2.0 * math.pi, SEED_MU, INTERVALS)
    continuation = CycleContinuation(eq, 0, INTERVALS, phase_index=1, phase_value=0.0)
    downward, _ = continuation.trace(seed, 0.2, 50, direction=-1.0)
    upward, _ = continuation.trace(seed, 0.2, 35, direction=1.0)
    ordered = list(reversed(keep_cycles(upward))) + keep_cycles(downward)[1:]
    return within(ordered, (P_LO, P_HI))


def describe(frame: Frame) -> str | None:
    """Name the regime in the normal form's own terms."""
    if frame.parameter < HOPF:
        return "Stable focus: the pair sits left of the Hopf line"
    if not frame.cycles:
        return "Unstable focus, cycle not yet resolved at this mesh"
    return "Attracting cycle of radius sqrt(mu), around an unstable focus"


def hopf_stage() -> StageScene:
    """Build the stage: the origin's branch and the cycle born from it."""
    eq = equation(Hopf, [Mu(value=P_LO)])
    origin = continue_branch(eq, [0.0, 0.0], (P_LO, P_HI), ["hopf"])
    system = StageSystem(
        "Hopf bifurcation",
        "The origin is the only equilibrium at every mu, with eigenvalues "
        "mu +/- i: a stable focus below zero, unstable above. They cross the "
        "imaginary axis together at mu = 0 and an attracting cycle of radius "
        "sqrt(mu) opens out of the point - the supercritical Hopf in its "
        "plainest form, and what every other Hopf in the library is a "
        "disguised version of.",
        "μ",
        equation=r"\dot x = \mu x - y - x r^2,\quad \dot y = x + \mu y - y r^2",
        note=(
            "The origin's branch and its Hopf come from ContinuerBuilder; the "
            "cycle from CycleContinuation, seeded at mu = 0.6 as the circle of "
            "radius sqrt(mu) the normal form predicts and continued both ways. "
            "The cycle is exactly circular here, so the Floquet readout stays "
            "near its best: what it reports is close to the mesh's own floor "
            "rather than the orbit's difficulty."
        ),
    )
    return stage_scene(
        Continued(eq, 0, origin, cycle_branch()),
        Film(
            np.linspace(P_LO, P_HI, FRAMES),
            Lattice(BOX, 36, "x", "y"),
            system,
            describe,
        ),
    )


STAGE_FILENAME = "hopf_stage.html"
STAGE_ENTRY = AtlasEntry(
    STAGE_FILENAME,
    "Hopf bifurcation",
    "The canonical codimension-one: an eigenvalue pair crosses the imaginary "
    "axis and a cycle of radius sqrt(mu) opens out of the point.",
    kind="stage",
)


def main(output_dir: str = "plots") -> None:
    """Write the stage page (and a one-card atlas) to ``output_dir``."""
    print(f"wrote {publish(hopf_stage(), STAGE_FILENAME, STAGE_ENTRY, output_dir)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots")
