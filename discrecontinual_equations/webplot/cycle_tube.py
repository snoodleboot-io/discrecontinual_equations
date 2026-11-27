"""The manifolds of a cycle on the stage: a tube of fibres from its chart.

A cycle's stable or unstable manifold is a surface - a tube round a saddle
cycle in three dimensions, a band either side of a repelling cycle in the
plane - and the stage draws a surface as a fan of curves marked ``-surface``.
Here the fan is the chart's own fibres: for each of a ring of phases the curve
``sigma -> P(theta, sigma)`` of the cycle's Fourier-Taylor chart
(:class:`~discrecontinual_equations.continuation.cycle_manifold.CycleManifold`)
out to the radius where the chart is still accurate, then carried on by the
flow - forward on an unstable manifold, backward on a stable one - until it
leaves the box, the globalisation the equilibrium manifolds get from the
Taylor chart. Each fibre starts on the cycle itself, so the tube is anchored
to the orbit it belongs to.
"""

from collections.abc import Sequence

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.cycle_manifold import (
    CycleManifold,
    CycleManifoldChart,
)
from discrecontinual_equations.continuation.manifold import (
    StableManifold,
    UnstableManifold,
)
from discrecontinual_equations.continuation.periodic_orbit import (
    PeriodicOrbitSolution,
)
from discrecontinual_equations.function.function import Function
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.webplot.stage import Manifold, StageScene

# A multiplier this far outside the unit circle marks an unstable cycle; the
# page's own test uses the same margin.
_UNSTABLE_MULTIPLIER = 1.02
_BOX_MARGIN = 0.2
# A chart whose series ends early (a polynomial manifold) is accurate out to
# infinity; its fibres hand over to the flow at this fraction of the box instead.
_REACH_FRACTION = 0.05


class TubeSettings:
    """How a cycle's tube is computed and drawn.

    ``order`` and ``harmonics`` are the chart's; ``tolerance`` is how accurate
    the chart must be where its fibres hand over to the flow; ``fibres`` is the
    number of phases round the cycle, each drawn on both sides; ``time`` bounds
    the flow and ``samples`` is the points per fibre.
    """

    __slots__ = ["fibres", "harmonics", "order", "samples", "time", "tolerance"]

    def __init__(  # noqa: PLR0913 (independent drawing knobs)
        self,
        order: int = 8,
        harmonics: int = 32,
        tolerance: float = 1.0e-8,
        fibres: int = 8,
        time: float = 20.0,
        samples: int = 90,
    ) -> None:
        self.order = order
        self.harmonics = harmonics
        self.tolerance = tolerance
        self.fibres = fibres
        self.time = time
        self.samples = samples


def tube(
    function: Function,
    chart: CycleManifoldChart,
    bounds: Sequence[tuple[float, float]],
    settings: TubeSettings | None = None,
) -> list[Manifold]:
    """The chart's fibres, globalised by the flow until they leave ``bounds``."""
    settings = settings or TubeSettings()
    unstable = chart.exponent > 0.0
    kind = "unstable-surface" if unstable else "stable-surface"
    (x_lo, x_hi), (y_lo, y_hi) = bounds[0], bounds[1]
    size = float(np.hypot(x_hi - x_lo, y_hi - y_lo))
    reach = min(chart.radius(settings.tolerance), _REACH_FRACTION * size)
    local = np.linspace(0.0, 1.0, max(2, settings.samples // 6))
    fibres: list[Manifold] = []
    for theta in np.linspace(0.0, chart.period, settings.fibres, endpoint=False):
        for side in (1.0, -1.0):
            head = [chart.point(float(theta), side * reach * s) for s in local]
            tail = _flow(function, head[-1], bounds, settings, forward=unstable)
            states = [list(map(float, state)) for state in head] + tail[1:]
            fibre = Manifold(kind, [(state[0], state[1]) for state in states])
            fibre.curve = states
            fibres.append(fibre)
    return fibres


def cycle_tube(
    function: Function,
    solution: PeriodicOrbitSolution,
    bounds: Sequence[tuple[float, float]],
    unstable: bool,  # noqa: FBT001 (which manifold, not a mode switch)
    settings: TubeSettings | None = None,
) -> list[Manifold]:
    """The tube of one manifold of the cycle ``solution``, as stage fibres."""
    settings = settings or TubeSettings()
    selection = UnstableManifold() if unstable else StableManifold()
    chart = CycleManifold(order=settings.order, harmonics=settings.harmonics).compute(
        function,
        solution,
        selection,
    )
    return tube(function, chart, bounds, settings)


def add_cycle_tubes(
    scene: StageScene,
    function: Function,
    parameter: Parameter,
    settings: TubeSettings | None = None,
) -> int:
    """Add the tube of every unstable cycle to each frame it appears in.

    The cycle is refined at the frame's own parameter (the branch point nearest
    it is only the seed), and its unstable manifold - the one a repelling or
    saddle cycle sheds orbits along - is drawn. A cycle the chart cannot
    resolve, at a fold where its exponent vanishes, say, is skipped. Returns
    the number of tubes drawn.
    """
    settings = settings or TubeSettings()
    bounds = list(scene.lattice.box)
    drawn = 0
    for frame in scene.frames:
        for index in frame.cycles:
            cycle = scene.cycles.cycles[index]
            sizes = [abs(complex(*pair)) for pair in cycle.multipliers]
            if max(sizes) < _UNSTABLE_MULTIPLIER:
                continue
            states = np.array(cycle.orbit or cycle.states, dtype=float)
            parameter.value = frame.parameter
            try:
                fibres = cycle_tube(
                    function,
                    PeriodicOrbitSolution(states, cycle.period),
                    bounds,
                    unstable=True,
                    settings=settings,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            frame.manifolds.extend(fibres)
            drawn += 1
    return drawn


def _flow(
    function: Function,
    start: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    settings: TubeSettings,
    forward: bool,  # noqa: FBT001 (a direction, not a mode switch)
) -> list[list[float]]:
    sense = 1.0 if forward else -1.0

    def rhs(_t: float, state: np.ndarray) -> list[float]:
        values = function.eval(point=[float(v) for v in state], time=None)
        return [sense * float(v) for v in values]

    def leaving(_t: float, state: np.ndarray) -> float:
        return min(
            margin
            for value, (lo, hi) in zip(state, bounds, strict=False)
            for margin in (value - (lo - _BOX_MARGIN), (hi + _BOX_MARGIN) - value)
        )

    leaving.terminal = True  # type: ignore[attr-defined]
    solution = solve_ivp(
        rhs,
        [0.0, settings.time],
        np.asarray(start, dtype=float),
        rtol=1.0e-8,
        atol=1.0e-10,
        events=leaving,
        max_step=0.05,
        dense_output=True,
    )
    times = np.linspace(0.0, solution.t[-1], settings.samples)
    return [[float(v) for v in state] for state in solution.sol(times).T]
