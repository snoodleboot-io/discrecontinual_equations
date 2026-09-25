"""Van der Pol as a film: a cycle born at a Hopf, growing into a relaxation.

``x' = y, y' = mu (1 - x^2) y - x``. The origin is the only equilibrium at
every ``mu``: a stable focus while ``mu < 0``, unstable after. At ``mu = 0``
its eigenvalues cross the imaginary axis together and a cycle is born
supercritically, small and nearly round. As ``mu`` grows the cycle stops being
round: the flow races across the fast jumps and crawls along the slow
branches, and by ``mu = 3`` it is the relaxation oscillation whose narrow
layers the adaptive mesh exists to resolve.

Scrub ``mu`` and watch the eigenvalue pair cross the Hopf line on the clock as
the cycle appears, then watch the cycle square off.
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

P_LO, P_HI, FRAMES = -0.6, 3.0, 73
BOX = ((-3.0, 3.0), (-5.0, 5.0))
INTERVALS = 80
# The cycle is seeded just past the Hopf, where it is still nearly the circle
# of radius two that the normal form predicts, and continued up from there.
SEED_MU = 0.08
SEED_RADIUS = 2.0
HOPF = 0.0
# Past this the jumps outrun the slow branches and the cycle stops being round.
RELAXATION = 1.5


class VanDerPol(DeterministicFunction):
    """x' = y, y' = mu (1 - x^2) y - x."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        return [y, mu * (1.0 - x * x) * y - x]


def cycle_branch() -> list[CyclePoint]:
    """The limit cycle, from just past the Hopf up to the stiff end."""
    eq = equation(VanDerPol, [Mu(value=SEED_MU)])
    seed = circle_seed(SEED_RADIUS, 2.0 * math.pi, SEED_MU, INTERVALS)
    continuation = CycleContinuation(eq, 0, INTERVALS, phase_index=1, phase_value=0.0)
    points, _ = continuation.trace(seed, 0.04, 400, direction=1.0)
    return within(keep_cycles(points), (P_LO, P_HI))


def describe(frame: Frame) -> str | None:
    """Name the regime in van der Pol's own terms."""
    if frame.parameter < HOPF:
        return "Stable focus: every trajectory spirals in"
    if not frame.cycles:
        return "Unstable focus, cycle not yet resolved at this mesh"
    if frame.parameter > RELAXATION:
        return "Relaxation oscillation: fast jumps, slow branches"
    return "Limit cycle born at the Hopf, still nearly round"


def van_der_pol_stage() -> StageScene:
    """Build the stage: the origin's branch and the cycle born from it."""
    eq = equation(VanDerPol, [Mu(value=P_LO)])
    origin = continue_branch(eq, [0.0, 0.0], (P_LO, P_HI), ["hopf"])
    cycles = cycle_branch()
    system = StageSystem(
        "Van der Pol",
        "The origin is the only equilibrium at every mu: a stable focus below "
        "zero, unstable above. At mu = 0 its eigenvalue pair crosses together "
        "and a cycle is born, small and nearly round. As mu grows the cycle "
        "squares off into a relaxation oscillation - fast jumps across, slow "
        "crawls along the branches - and those narrow layers are what the "
        "adaptive mesh exists to resolve.",
        "μ",
        equation=r"\dot x = y,\quad \dot y = \mu\,(1 - x^2)\,y - x",
        note=(
            "The origin's branch and its Hopf come from ContinuerBuilder; the "
            "cycle from CycleContinuation, seeded just past the Hopf as the "
            "circle of radius two the normal form predicts. The Floquet "
            "readout carries its own error - one multiplier is exactly 1 along "
            "the orbit, so how far the computed one sits from 1 is how far all "
            "of them are off, and it grows with mu as the fixed mesh loses the "
            "jump layers."
        ),
    )
    scene = stage_scene(
        Continued(eq, 0, origin, cycles),
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


STAGE_FILENAME = "van_der_pol_stage.html"
STAGE_ENTRY = AtlasEntry(
    STAGE_FILENAME,
    "Van der Pol",
    "A cycle born at a Hopf and squaring off into a relaxation oscillation, "
    "with the eigenvalue pair crossing on the spectral clock.",
    kind="stage",
)


def main(output_dir: str = "plots") -> None:
    """Write the stage page (and a one-card atlas) to ``output_dir``."""
    path = publish(van_der_pol_stage(), STAGE_FILENAME, STAGE_ENTRY, output_dir)
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots")
