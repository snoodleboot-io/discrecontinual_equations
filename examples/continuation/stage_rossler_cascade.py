"""Rossler's period-doubling cascade as a film, and Feigenbaum's delta from it.

``x' = -y - z, y' = x + a y, z' = b + z (x - c)`` with ``a = b = 0.2``,
continued in ``c``. The cycle born at small ``c`` loses its stability at
``c = 2.8324`` when a Floquet multiplier passes ``-1``, and a cycle of twice the
period takes over; that one doubles at ``c = 3.8374``, its successor at
``4.1242``, and so on, the doublings crowding together until the attractor is
chaotic. The gaps between them shrink by a ratio that tends to Feigenbaum's
``delta = 4.6692...``, the same number the logistic map gives: the universality
that made the constant famous.

Every branch comes from :class:`PeriodDoublingCascade`, which locates each
doubling on its branch, switches onto the doubled cycle along the flip
multiplier's eigenvector, and traces that one on a mesh of twice the intervals.
The film scrubs ``c`` through the first three doublings; the clock shows the
multiplier walking out along the negative axis to ``-1`` at each one. On the
cycle born there it starts again from ``+1`` - a multiplier of ``-1`` once round
the old cycle is ``+1`` twice round it - meets the small multiplier and turns
into a complex pair, and comes out on the negative axis to walk to ``-1``
again. A second page draws the branches and gives the ratios measured over six
doublings.
"""

import sys
from functools import cache

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.cycle_continuation import (
    CyclePoint,
    CycleSeed,
)
from discrecontinual_equations.continuation.period_doubling import (
    CascadeLevel,
    PeriodDoublingCascade,
    feigenbaum_ratios,
    flip_multiplier,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.webplot.renderer import D3Renderer
from discrecontinual_equations.webplot.report import AtlasEntry, PlotReport
from discrecontinual_equations.webplot.scene import Scene
from discrecontinual_equations.webplot.scene_builder import curve_scene, with_equation
from discrecontinual_equations.webplot.stage import (
    Frame,
    Lattice,
    Projection,
    StageScene,
    StageSystem,
    Term,
    View,
)
from discrecontinual_equations.webplot.stage_builder import (
    Continued,
    Film,
    StageSettings,
    stage_scene,
)

try:  # python -m examples.continuation.stage_rossler_cascade
    from examples.continuation.stage_support import (
        Mu,
        continue_branch,
        equation,
        publish,
    )
except ImportError:  # run as a script path: only this directory is on sys.path
    from stage_support import Mu, continue_branch, equation, publish

A = B = 0.2
FEIGENBAUM_DELTA = 4.669201609102990671853
# The film: the first three doublings, from the period-1 cycle to period 8.
P_LO, P_HI, FRAMES = 2.7, 4.16, 74
FILM_LEVELS = 4
# The table: six doublings, five ratios' worth of gaps.
TABLE_LEVELS = 6
SEED_C = 2.7
INTERVALS = 20
ARCLENGTH = 1.0
POINTS_PER_GAP = 8
BOX = ((-9.0, 11.0), (-11.0, 8.0))
PLANES = (
    Projection("x - y", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], ("x", "y"), BOX),
    Projection(
        "x - z",
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ("x", "z"),
        ((-9.0, 11.0), (-1.0, 25.0)),
    ),
    Projection(
        "y - z",
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ("y", "z"),
        ((-11.0, 8.0), (-1.0, 25.0)),
    ),
)
_SETTLE = 400.0
MANIFOLD_TIME = 6.0
MANIFOLD_FAN = 4


class Rossler(DeterministicFunction):
    """x' = -y - z, y' = x + a y, z' = b + z (x - c), with a = b = 0.2."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        c = self.parameters[0].value
        x, y, z = point[0], point[1], point[2]
        return [-y - z, x + A * y, B + z * (x - c)]


def terms() -> list[list[Term]]:
    """The field as polynomial terms, exact in the continued parameter."""
    return [
        [Term(-1.0, [0, 1, 0]), Term(-1.0, [0, 0, 1])],
        [Term(1.0, [1, 0, 0]), Term(A, [0, 1, 0])],
        [Term(B, [0, 0, 0]), Term(1.0, [1, 0, 1]), Term(-1.0, [0, 0, 1], 1)],
    ]


def attractor_seed(c: float, intervals: int) -> CycleSeed:
    """One turn of the attracting cycle at ``c``, from the section y = 0 upward."""

    def rhs(_time: float, u: np.ndarray) -> list[float]:
        return [-u[1] - u[2], u[0] + A * u[1], B + u[2] * (u[0] - c)]

    def section(_time: float, u: np.ndarray) -> float:
        return float(u[1])

    section.direction = 1.0  # type: ignore[attr-defined]
    settled = solve_ivp(
        rhs,
        (0.0, _SETTLE),
        [1.0, 1.0, 0.0],
        rtol=1e-10,
        atol=1e-12,
        events=section,
    )
    start, end = settled.t_events[0][-2], settled.t_events[0][-1]
    turn = solve_ivp(
        rhs,
        (0.0, end - start),
        settled.y_events[0][-2],
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
    )
    times = np.linspace(0.0, end - start, intervals + 1)
    return CycleSeed(turn.sol(times).T, float(end - start), c)


@cache
def cascade() -> tuple[CascadeLevel, ...]:
    """The cascade from the period-1 cycle at ``c = 2.7``, six branches deep.

    Cached: the film and the table are drawn from the one trace.
    """
    eq = equation(Rossler, [Mu(value=SEED_C)], count=3)
    follower = PeriodDoublingCascade(eq, 0, INTERVALS, phase_index=1, phase_value=0.0)
    seed = attractor_seed(SEED_C, INTERVALS)
    return tuple(
        follower.follow(seed, ARCLENGTH, TABLE_LEVELS, points_per_gap=POINTS_PER_GAP),
    )


def describe(frame: Frame) -> str | None:
    """Name the period of the attracting cycle, in turns of the first."""
    doublings = [level.doubling for level in cascade() if level.doubling is not None]
    crossed = sum(1 for doubling in doublings if frame.parameter > doubling)
    return f"Attracting cycle of period {2**crossed} (in turns of the first)"


def rossler_cascade_stage() -> StageScene:
    """Build the stage: the inner equilibrium and the first four cycle branches."""
    film = cascade()[:FILM_LEVELS]
    cycles: list[CyclePoint] = [point for level in film for point in level.points]
    eq = equation(Rossler, [Mu(value=P_LO)], count=3)
    inner = continue_branch(
        eq,
        _equilibrium(P_LO, inner=True),
        (P_LO, P_HI),
        ["hopf", "fold"],
    )
    system = StageSystem(
        "Rossler period-doubling cascade",
        "The Rossler system at a = b = 0.2, continued in c. The cycle born at "
        "small c loses its stability when a Floquet multiplier passes -1, and "
        "a cycle of twice the period takes over; that one doubles in turn, and "
        "the doublings crowd together, their gaps shrinking by a ratio that "
        "tends to Feigenbaum's delta = 4.669. Scrub c through the first three "
        "and watch the multiplier reach -1 on the clock at each.",
        "c",
        equation=(
            r"\dot x = -y - z,\quad \dot y = x + a y,\quad "
            r"\dot z = b + z (x - c),\quad a = b = 0.2"
        ),
        note=_film_note(film),
    )
    return stage_scene(
        Continued(eq, 0, inner, cycles, terminus="period-doubling cascade"),
        Film(
            np.linspace(P_LO, P_HI, FRAMES),
            Lattice(
                BOX,
                36,
                "x",
                "y",
                view=View(PLANES, [(-9.0, 11.0), (-11.0, 8.0), (-1.0, 25.0)], terms()),
            ),
            system,
            describe,
        ),
        # The inner equilibrium's unstable surface spirals out onto the
        # attractor, and at the default sixty time units and twelve curves it
        # took the page to 24 MB and ten units and six to 13 MB; six and four
        # still show a turn of the funnel.
        StageSettings(manifold_time=MANIFOLD_TIME, manifold_fan=MANIFOLD_FAN),
    )


def _equilibrium(c: float, *, inner: bool) -> list[float]:
    """The inner (near the origin) or outer equilibrium at ``c``."""
    root = np.sqrt(c * c - 4.0 * A * B)
    z = (c - root) / (2.0 * A) if inner else (c + root) / (2.0 * A)
    return [A * z, -z, z]


def _film_note(levels: list[CascadeLevel]) -> str:
    located = ", ".join(
        f"period {level.multiple} at c = {level.doubling:.6f}"
        for level in levels
        if level.doubling is not None
    )
    return (
        "Every cycle comes from PeriodDoublingCascade on Gauss-Legendre "
        f"collocation, {INTERVALS} mesh intervals per turn - the mesh doubles "
        "with the period at each switch. Doublings located (flip multiplier "
        f"through -1, by inverse cubic interpolation): {located}. The inner "
        "equilibrium's branch comes from ContinuerBuilder; it is a saddle-focus "
        "throughout and the cycles wind round it."
    )


def cascade_page() -> Scene:
    """The cascade page: every branch traced, with its ratios."""
    return with_equation(
        cascade_scene(list(cascade())),
        r"\dot x = -y - z,\ \dot y = x + 0.2 y,\ \dot z = 0.2 + z (x - c)",
    )


def cascade_scene(levels: list[CascadeLevel]) -> Scene:
    """The branches' flip multipliers against c, with the doublings marked."""
    lines = []
    for level in levels:
        flips = [
            (point.parameter, flip_multiplier(point))
            for point in level.points
            if np.isfinite(flip_multiplier(point))
        ]
        lines.append(("stable", f"period {level.multiple}", flips, None))
    doublings = [level.doubling for level in levels if level.doubling is not None]
    markers = [("hopf", "period-doubling", [(c, -1.0) for c in doublings], None)]
    ratios = feigenbaum_ratios(doublings)
    table = "; ".join(f"delta_{k + 1} = {ratio:.4f}" for k, ratio in enumerate(ratios))
    return curve_scene(
        lines,
        markers,
        ("c", "flip multiplier"),
        "Rossler cascade: Feigenbaum's delta from six doublings",
        f"Each branch's real negative Floquet multiplier against c; it reaches -1 "
        f"at each doubling. Doublings at c = "
        f"{', '.join(f'{c:.7f}' for c in doublings)}. Ratios of successive gaps: "
        f"{table}; Feigenbaum's delta = 4.669202.",
    )


CASCADE_FILENAME = "rossler_cascade.html"
STAGE_FILENAME = "rossler_cascade_stage.html"
STAGE_ENTRY = AtlasEntry(
    STAGE_FILENAME,
    "Rossler period-doubling cascade",
    "A cycle doubling its period three times as c grows, the Floquet "
    "multiplier reaching -1 on the clock at each doubling.",
    kind="stage",
)


def main(output_dir: str = "plots") -> None:
    """Write the stage page, the cascade page, and a one-card atlas."""
    scene = rossler_cascade_stage()
    print(f"wrote {publish(scene, STAGE_FILENAME, STAGE_ENTRY, output_dir)}")
    PlotReport(D3Renderer(), output_dir).write(cascade_page(), CASCADE_FILENAME)
    print(f"wrote {output_dir}/{CASCADE_FILENAME}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plots")
