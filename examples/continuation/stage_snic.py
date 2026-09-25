"""SNIC as a film: a cycle born with an infinite period.

An attracting unit circle carrying an Adler flow, ``theta' = mu - sin(theta)``.
While ``mu < 1`` the circle holds two equilibria - a node where the flow is
swept in and a saddle where it is held back - and every trajectory ends at the
node. As ``mu`` rises the two slide toward each other, and at ``mu = 1`` they
collide *on the invariant circle*: a saddle-node on an invariant circle. What
is released is not an ordinary cycle. The collision leaves a bottleneck where
the flow still nearly stops, so the first cycle takes forever and the period
comes down from infinity as ``mu`` rises, exactly as ``2 pi / sqrt(mu^2 - 1)``.

Scrub ``mu`` and watch the two equilibria meet, vanish, and leave a circulation
that is at first too slow to see.
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

P_LO, P_HI, FRAMES = 0.2, 2.4, 89
BOX = ((-1.7, 1.7), (-1.7, 1.7))
INTERVALS = 80
SNIC = 1.0
# The cycle is seeded where its period is comfortably finite and continued down
# toward the collision, where the period runs away and the mesh gives out.
SEED_MU = 1.6
# Below this radius the angular velocity is undefined, so the field is held at
# rest there rather than dividing by a vanishing radius.
RADIUS_FLOOR = 1.0e-9


class SnicSystem(DeterministicFunction):
    """Attracting unit circle with an Adler flow; SNIC at mu = 1."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        # Not math.sqrt: the analytic Jacobian evaluates this field on
        # Taylor jets, which the math module rejects but ** accepts.
        radius = (x * x + y * y) ** 0.5
        if radius < RADIUS_FLOOR:
            return [0.0, 0.0]
        radial = 1.0 - radius * radius
        angular = mu - y / radius
        return [radial * x - y * angular, radial * y + x * angular]


def exact_period(mu: float) -> float:
    """The circulation period once the equilibria are gone."""
    return 2.0 * math.pi / math.sqrt(mu * mu - 1.0)


def cycle_branch() -> list[CyclePoint]:
    """The circulation, traced down toward the collision and up away from it."""
    eq = equation(SnicSystem, [Mu(value=SEED_MU)])
    seed = circle_seed(1.0, exact_period(SEED_MU), SEED_MU, INTERVALS)
    continuation = CycleContinuation(eq, 0, INTERVALS, phase_index=1, phase_value=0.0)
    downward, _ = continuation.trace(seed, 0.02, 320, direction=-1.0)
    upward, _ = continuation.trace(seed, 0.04, 120, direction=1.0)
    traced = keep_cycles(downward) + keep_cycles(upward)[1:]
    return within(sorted(traced, key=lambda point: point.parameter), (P_LO, P_HI))


def describe(frame: Frame) -> str | None:
    """Name the regime in the circle's own terms."""
    if frame.parameter < SNIC:
        return "Node and saddle on the circle: the flow ends at the node"
    if not frame.cycles:
        return "Circulating, but the period is beyond this mesh"
    return "Circulating: the period came down from infinity at the collision"


def snic_stage() -> StageScene:
    """Build the stage: the pair on the circle, their collision, the cycle."""
    start = math.sqrt(1.0 - P_LO * P_LO)
    eq = equation(SnicSystem, [Mu(value=P_LO)])
    # The branch rounds its own fold at mu = 1 and returns as the saddle, so
    # one continuation carries both equilibria.
    pair = continue_branch(eq, [start, P_LO], (P_LO, P_HI), ["fold"], step=0.01)
    cycles = cycle_branch()
    system = StageSystem(
        "Saddle-node on an invariant circle",
        "An attracting circle carrying the flow theta' = mu - sin(theta). "
        "Below mu = 1 the circle holds a node and a saddle and everything ends "
        "at the node. They slide together and collide on the circle itself, "
        "and what is released circulates through the bottleneck they left - so "
        "the period comes down from infinity rather than starting finite, as a "
        "cycle born at a Hopf would.",
        "μ",
        equation=(r"\dot r = (1 - r^2)\,r,\quad \dot\theta = \mu - \sin\theta"),
        note=(
            "The equilibrium pair and their fold come from ContinuerBuilder - "
            "one continuation, because the branch rounds the fold at mu = 1 and "
            "returns as the saddle. The cycle comes from CycleContinuation, "
            "seeded at mu = 1.6 as the unit circle with the exact period "
            "2 pi / sqrt(mu^2 - 1) and continued both ways; it stops short of "
            "the collision, where the period runs away and no fixed mesh can "
            "follow it."
        ),
    )
    scene = stage_scene(
        Continued(eq, 0, pair, cycles, terminus="infinite period"),
        Film(
            np.linspace(P_LO, P_HI, FRAMES),
            Lattice(BOX, 36, "x", "y"),
            system,
            describe,
        ),
    )
    for point in scene.timeline.special:
        if point.kind == "fold":
            point.label = "SNIC (mu = 1)"
    return scene


STAGE_FILENAME = "snic_stage.html"
STAGE_ENTRY = AtlasEntry(
    STAGE_FILENAME,
    "Saddle-node on an invariant circle",
    "A node and a saddle collide on the circle and leave a circulation whose "
    "period comes down from infinity.",
    kind="stage",
)


def main(output_dir: str = "plots") -> None:
    """Write the stage page (and a one-card atlas) to ``output_dir``."""
    print(f"wrote {publish(snic_stage(), STAGE_FILENAME, STAGE_ENTRY, output_dir)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots")
