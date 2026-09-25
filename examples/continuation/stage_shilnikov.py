"""The Shilnikov jerk system as a film: a saddle-focus turning.

``x' = y, y' = z, z' = -a z - y + x - x^2``. Two equilibria stand at every
``a``: the origin and ``x = 1``. What the film is about is the *kind* of
equilibrium, which the clock shows and the stage cannot. At the origin one
eigenvalue is real and positive and the other two are a complex pair with
negative real part - a saddle-focus, the configuration Shilnikov's theorem
needs, where an orbit leaving along the real direction can spiral back in and
close, and the chaos near such a loop is what makes the system worth having.

As ``a`` grows the damping tightens: the pair's real part moves, the saddle
index crosses one, and the tame side of Shilnikov's condition takes over.
Three-dimensional, so the stage sees it through a view of the ``(x, y)``
plane with particles integrated in all three coordinates.
"""

import sys

import numpy as np

from discrecontinual_equations.function.deterministic import DeterministicFunction
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

try:  # python -m examples.continuation.stage_shilnikov
    from examples.continuation.stage_support import (
        Mu,
        continue_branch,
        equation,
        publish,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from stage_support import Mu, continue_branch, equation, publish

P_LO, P_HI, FRAMES = 0.25, 1.3, 75
BOX = ((-0.6, 1.6), (-0.9, 0.9))
# An eigenvalue with an imaginary part this big is a genuine spiral.
_COMPLEX = 1.0e-9


class Jerk(DeterministicFunction):
    """x' = y, y' = z, z' = -a z - y + x - x^2; a saddle-focus at the origin."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        a = self.parameters[0].value
        x, y, z = point[0], point[1], point[2]
        return [y, z, -a * z - y + x - x * x]


def terms() -> list[list[Term]]:
    """The field as polynomial terms, exact in the continued parameter."""
    return [
        [Term(1.0, [0, 1, 0])],
        [Term(1.0, [0, 0, 1])],
        [
            Term(-1.0, [0, 0, 1], 1),
            Term(-1.0, [0, 1, 0]),
            Term(1.0, [1, 0, 0]),
            Term(-1.0, [2, 0, 0]),
        ],
    ]


def saddle_index(equilibrium) -> float | None:
    """|Re of the spiral| over the escaping rate, at a saddle-focus.

    Shilnikov's theorem bites below one: the orbit returns along the spiral
    more slowly than it leaves along the real direction, and the loop has
    infinitely many periodic orbits near it. Above one the loop is tame.
    """
    spiral = [re for re, im in equilibrium.eigenvalues if abs(im) > _COMPLEX]
    escaping = [
        re for re, im in equilibrium.eigenvalues if abs(im) <= _COMPLEX and re > 0
    ]
    if not spiral or not escaping:
        return None
    return abs(spiral[0]) / escaping[0]


def describe(frame: Frame) -> str | None:
    """Name the two equilibria by what Shilnikov's condition makes of them."""
    indices = [
        index
        for index in (saddle_index(e) for e in frame.equilibria)
        if index is not None
    ]
    if not indices:
        return "No saddle-focus resolved at this damping"
    smallest = min(indices)
    if smallest < 1.0:
        return f"Saddle index {smallest:.2f} < 1: a loop here would be wild"
    return f"Saddle index {smallest:.2f} > 1: a loop here would be tame"


def shilnikov_stage() -> StageScene:
    """Build the stage: both equilibrium branches, seen in the (x, y) plane."""
    eq = equation(Jerk, [Mu(value=P_LO)], count=3)
    origin = continue_branch(eq, [0.0, 0.0, 0.0], (P_LO, P_HI), ["hopf", "fold"])
    upper = continue_branch(eq, [1.0, 0.0, 0.0], (P_LO, P_HI), ["hopf", "fold"])
    system = StageSystem(
        "Shilnikov jerk system",
        "Two equilibria stand at every damping a: the origin and x = 1. What "
        "matters is the kind of equilibrium, which the clock shows and the "
        "stage cannot. At the origin one eigenvalue is real and positive while "
        "the other two are a complex pair with negative real part - a "
        "saddle-focus, the configuration Shilnikov's theorem needs, where an "
        "orbit leaving along the real direction can spiral back in and close. "
        "As a grows the damping tightens and the pair moves.",
        "a",
        equation=r"\dot x = y,\quad \dot y = z,\quad \dot z = -a z - y + x - x^2",
        note=(
            "Both equilibrium branches come from ContinuerBuilder, each "
            "continued from its own seed and drawn as its own arc. There is no "
            "cycle branch: the orbits near a Shilnikov loop are what "
            "HomoclinicShooting traces, and the tangle around the primary loop "
            "is where that locator is unreliable - see future-work section 2. "
            "The saddle manifolds are surfaces in three dimensions and are not "
            "drawn; the particles, integrated in all three coordinates from the "
            "polynomial terms, are what shows the spiral."
        ),
    )
    return stage_scene(
        Continued(eq, 0, [origin, upper]),
        Film(
            np.linspace(P_LO, P_HI, FRAMES),
            Lattice(
                BOX,
                36,
                "x",
                "y",
                view=View(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [(-0.6, 1.6), (-0.9, 0.9), (-0.9, 0.9)],
                    terms(),
                ),
            ),
            system,
            describe,
        ),
    )


STAGE_FILENAME = "shilnikov_stage.html"
STAGE_ENTRY = AtlasEntry(
    STAGE_FILENAME,
    "Shilnikov jerk system",
    "The saddle-focus whose spiral return makes Shilnikov's theorem bite, "
    "with its complex pair moving on the clock as the damping grows.",
    kind="stage",
)


def main(output_dir: str = "plots") -> None:
    """Write the stage page (and a one-card atlas) to ``output_dir``."""
    scene = shilnikov_stage()
    print(f"wrote {publish(scene, STAGE_FILENAME, STAGE_ENTRY, output_dir)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots")
