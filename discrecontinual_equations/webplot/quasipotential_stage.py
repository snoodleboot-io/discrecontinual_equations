"""Give a stochastic stage the quasipotential as an optional panel.

The density panel shows where the mass sits; it cannot show how a path gets
anywhere. The quasipotential ``U`` of :mod:`...continuation.quasipotential`
does: noise of size ``eps`` reaches a state with probability ``exp(-U / eps)``,
and the minimum action path to it is the route it most probably takes. This
module adds both to a scene that :func:`~.stochastic_stage_builder.stochastic_stage`
has already made stochastic, as a field on its own panel beside the density
and as curves drawn over it, one per chosen starting state.

It is a separate step rather than another argument to ``stochastic_stage``
because it is optional and costs one ordered-line-integral march per frame,
and a film that does not want it should neither pay for it nor carry a key for
it in its payload.
"""

from collections.abc import Callable, Sequence

import numpy as np

from discrecontinual_equations.continuation.fokker_planck import DensityGrid
from discrecontinual_equations.continuation.quasipotential import (
    CycleAttractor,
    EquilibriumAttractor,
    GridQuasipotential,
    OrderedLineIntegral,
)
from discrecontinual_equations.webplot.stage import (
    Box,
    QuasipotentialFrame,
    QuasipotentialPanel,
    StageScene,
)
from discrecontinual_equations.webplot.stochastic_stage_builder import density_window

Attractor = EquilibriumAttractor | CycleAttractor
QuasipotentialProblem = tuple[OrderedLineIntegral, Attractor]
_DECIMALS = 2
QuasipotentialFamily = Callable[[float, DensityGrid], QuasipotentialProblem | None]


class QuasipotentialSettings:
    """What the panel shows.

    ``family`` gives the march and the attractor at a parameter value on a grid,
    or ``None`` where there is no attractor to measure ``U`` from. ``starts`` are
    the states the minimum action paths are traced back from; a start where ``U``
    is not finite is skipped. ``cap`` is the largest ``U`` the colour ramp
    resolves: the field grows without bound towards a point where the noise
    vanishes, and an uncapped ramp would spend all its colour there.
    """

    __slots__ = ["cap", "family", "grid", "starts"]

    def __init__(
        self,
        family: QuasipotentialFamily,
        grid: DensityGrid,
        starts: Sequence[Sequence[float]] = (),
        cap: float = 4.0,
    ) -> None:
        self.family = family
        self.grid = grid
        self.starts = [np.asarray(start, dtype=float) for start in starts]
        self.cap = float(cap)


def frame_quasipotential(
    field: GridQuasipotential,
    window: Box,
    cap: float,
    paths: Sequence[np.ndarray],
) -> QuasipotentialFrame:
    """A computed field as one frame draws it: windowed, capped, its paths."""
    xs, ys, _extent = density_window(field.grid, window)
    block = np.minimum(field.values[xs, ys], cap)
    # Rows of constant y, x running fastest: the layout the density uses. Two
    # decimals is a two-hundredth of the default cap, finer than the ramp's 256
    # steps resolve, and halves what the field adds to the page.
    values = [round(float(v), _DECIMALS) for v in block.T.ravel()]
    curves = [[(float(x), float(y)) for x, y in path] for path in paths]
    return QuasipotentialFrame(values, cap, curves)


def with_quasipotential(
    scene: StageScene,
    settings: QuasipotentialSettings,
) -> StageScene:
    """Add the quasipotential panel to a stochastic ``scene``, one field per frame."""
    if scene.stochastic is None:
        message = "the quasipotential panel belongs on a stochastic stage"
        raise ValueError(message)
    window = scene.lattice.box
    xs, ys, extent = density_window(settings.grid, window)
    frames: list[QuasipotentialFrame | None] = []
    for frame in scene.frames:
        problem = settings.family(frame.parameter, settings.grid)
        if problem is None:
            frames.append(None)
            continue
        solver, attractor = problem
        field = solver.solve(attractor)
        paths = [
            solver.minimum_action_path(field, start, end=attractor.end()).states
            for start in settings.starts
            if np.isfinite(field.at(start))
        ]
        frames.append(frame_quasipotential(field, window, settings.cap, paths))
    scene.stochastic.quasipotential = QuasipotentialPanel(
        extent,
        (xs.stop - xs.start, ys.stop - ys.start),
        frames,
    )
    return scene
