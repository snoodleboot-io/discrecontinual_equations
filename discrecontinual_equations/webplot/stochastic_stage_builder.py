"""Make a staged system stochastic: paths, a density per frame, an exponent.

A stochastic film is a deterministic stage with three things added after
:func:`~.stage_builder.stage_scene` has built it. The particles become sample
paths of ``dx = f dt + G dW``, which the page integrates itself from the
drift terms the lattice's view already carries and the noise terms added here.
Each frame gets the stationary density at its parameter, solved on a grid by
:class:`~...continuation.fokker_planck.StationaryFokkerPlanck`, with the
interior maxima that say whether its peak has left the reference state. And
the spectral clock is replaced by the top Lyapunov exponent across the
parameter, since for a stochastic system that one real number, not a spectrum,
is what the dynamical bifurcation changes the sign of.

The two bifurcations are then *located* rather than read off a picture, with
the same :class:`~...continuation.stochastic_threshold.ParameterThreshold`
machinery the library uses to answer the question without a film, and put on
the timeline as two special points at their own parameter values. That the
two values differ is the whole content of keeping the phenomenological and
dynamical bifurcations as separate objects, and it is what the timeline
exists to show. Nothing here is specific to a system; the film supplies the
density family, the exponent family, and the closed forms it wants drawn
beside them.

One thing learned building the first film is encoded in :class:`Thresholds`
taking its own grid. A noise that vanishes at the reference state makes the
diffusion singular there, and the finite-volume solution overshoots in the few
cells around that point by a fixed factor in cells, not in length: at the four
cells nearest the origin by half again, one cell out by seven percent, four
cells out by under one percent, whatever the spacing. A probe one cell from the
centre therefore carries a constant bias no refinement removes, while a probe
several cells out carries the probe's own ``O(r^2)`` bias instead, which does
shrink with the spacing. Locating on a finer grid than the frames are drawn on,
and several cells out, is what makes the located threshold agree with the
closed form to a few percent rather than thirty.
"""

import math
from collections.abc import Callable, Sequence

import numpy as np

from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    GridDensity,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.noise import (
    NoiseMatrix,
    field_at,
    state_drift_shift,
    stratonovich_factor,
)
from discrecontinual_equations.continuation.root_finder import ScalarRootFinder
from discrecontinual_equations.continuation.stochastic import NoiseConvention
from discrecontinual_equations.continuation.stochastic_threshold import (
    DensityProbe,
    DynamicalThreshold,
    PhenomenologicalThreshold,
)
from discrecontinual_equations.function.function import Function
from discrecontinual_equations.webplot.stage import (
    Box,
    Density,
    DensityWindow,
    ExponentCurve,
    Frame,
    Pairs,
    PathModel,
    SpecialPoint,
    StageScene,
    StochasticStage,
    Term,
    View,
    evaluate_terms,
)
from discrecontinual_equations.webplot.stage_builder import Continued

P_BIFURCATION = "p_bifurcation"
D_BIFURCATION = "d_bifurcation"
# How many random states the noise terms are checked at, and how closely they
# must agree with the matrix they claim to be; the same bar check_view sets.
_CHECKS = 8
_AGREEMENT = 1.0e-9
# A maximum closer to the reference state than this many cells is the peak
# still sitting on it, not a ring around it.
_AT_CENTRE_CELLS = 1.5
_DEFAULT_SEED = 0
_DEFAULT_STEP = 0.02

DensityFamily = Callable[[float, DensityGrid], StationaryFokkerPlanck | None]


class NoiseModel:
    """The amplitude ``G``, both as the page integrates it and as the library does.

    ``terms`` is one :class:`Term` list per component per driver, what the page
    reads; ``matrix`` is the same amplitude as a
    :class:`~...continuation.noise.NoiseMatrix`, what the density and exponent
    were computed from. Carrying both is what lets the builder check that the
    paths on the stage are paths of the same equation the panels describe.
    ``convention`` is the reading the equation is written in. ``seed`` and
    ``step`` are the page's generator seed and Euler-Maruyama step.
    """

    __slots__ = ["convention", "matrix", "seed", "step", "terms"]

    def __init__(
        self,
        terms: Sequence[Sequence[Sequence[Term]]],
        matrix: NoiseMatrix,
        convention: NoiseConvention,
        seed: int = _DEFAULT_SEED,
        step: float = _DEFAULT_STEP,
    ) -> None:
        self.terms = [[list(component) for component in driver] for driver in terms]
        self.matrix = matrix
        self.convention = convention
        self.seed = seed
        self.step = step

    @property
    def name(self) -> str:
        """The convention's name, as the page states it."""
        return type(self.convention).__name__

    def paths(self) -> PathModel:
        """The part of this the page needs."""
        return PathModel(self.terms, self.name, self.seed, self.step)


class ClosedForms:
    """What is known exactly, to draw beside what was computed.

    ``exponent`` is the top Lyapunov exponent as a function of the parameter
    and ``crest`` the radius of the density's ring, or ``None`` where there is
    none; either may be left out.
    """

    __slots__ = ["crest", "exponent"]

    def __init__(
        self,
        exponent: Callable[[float], float] | None = None,
        crest: Callable[[float], float | None] | None = None,
    ) -> None:
        self.exponent = exponent
        self.crest = crest


class Measurements:
    """What is measured along the parameter.

    ``density_at`` gives the stationary problem at a parameter value on a
    grid, or ``None`` where the system has no density on the plane - below
    the dynamical threshold of an invariant point the whole mass sits on it,
    and asking a grid for that density would return junk. ``exponent_at``
    gives the top Lyapunov exponent. ``exponent_samples`` are the parameter
    values the exponent is drawn at - each is a simulated path and costs
    seconds, so they are chosen rather than taken at every frame.
    """

    __slots__ = ["closed", "density_at", "exponent_at", "exponent_samples"]

    def __init__(
        self,
        density_at: DensityFamily,
        exponent_at: Callable[[float], float],
        exponent_samples: Sequence[float],
        closed: ClosedForms | None = None,
    ) -> None:
        self.density_at = density_at
        self.exponent_at = exponent_at
        self.exponent_samples = list(exponent_samples)
        self.closed = closed or ClosedForms()


class Thresholds:
    """How and where the two thresholds are located.

    ``probe`` is the shape test the phenomenological threshold is found with,
    inside the ``phenomenological`` bracket; the dynamical one is the
    exponent's own zero inside ``dynamical``. ``grid`` is the grid the density
    is solved on while locating, when it should be finer than the one the
    frames are drawn on; see the module docstring for why it usually should.
    """

    __slots__ = ["dynamical", "grid", "phenomenological", "probe", "root_finder"]

    def __init__(
        self,
        probe: DensityProbe,
        phenomenological: tuple[float, float],
        dynamical: tuple[float, float],
        root_finder: ScalarRootFinder,
        grid: DensityGrid | None = None,
    ) -> None:
        self.probe = probe
        self.phenomenological = phenomenological
        self.dynamical = dynamical
        self.root_finder = root_finder
        self.grid = grid


class Diagnostics:
    """Everything the film measures and how, on the grid its frames are drawn on.

    ``centre`` is the reference state: where the probe looks and where the
    crest radius is measured from.
    """

    __slots__ = ["centre", "grid", "measurements", "thresholds"]

    def __init__(
        self,
        measurements: Measurements,
        thresholds: Thresholds,
        grid: DensityGrid,
        centre: Sequence[float] = (0.0, 0.0),
    ) -> None:
        self.measurements = measurements
        self.thresholds = thresholds
        self.grid = grid
        self.centre = [float(value) for value in centre]


class Located:
    """Both thresholds, and every exponent evaluation locating them cost."""

    __slots__ = ["dynamical", "exponents", "phenomenological"]

    def __init__(
        self,
        phenomenological: float,
        dynamical: float,
        exponents: Pairs,
    ) -> None:
        self.phenomenological = phenomenological
        self.dynamical = dynamical
        self.exponents = exponents


class _Memo:
    """A scalar function of the parameter that remembers every answer.

    The exponent is a simulated path and the dearest thing the film computes.
    The curve the page draws and the root finder locating the zero ask about
    overlapping values; remembering them means each is paid for once, and the
    root finder's own evaluations become extra points on the curve.
    """

    __slots__ = ["_function", "_values"]

    def __init__(self, function: Callable[[float], float]) -> None:
        self._function = function
        self._values: dict[float, float] = {}

    def __call__(self, parameter: float) -> float:
        key = float(parameter)
        if key not in self._values:
            self._values[key] = float(self._function(key))
        return self._values[key]

    def pairs(self) -> Pairs:
        """Everything evaluated so far, in parameter order."""
        return sorted(self._values.items())


def locate_thresholds(diagnostics: Diagnostics) -> Located:
    """Locate both bifurcations in the parameter, each inside its own bracket.

    Each is refused rather than guessed when its bracket holds no sign change:
    a film that claims a threshold it did not find would be worse than one
    that fails to build.
    """
    measurements, thresholds = diagnostics.measurements, diagnostics.thresholds
    exponent = _Memo(measurements.exponent_at)
    for value in measurements.exponent_samples:
        exponent(value)
    dynamical = DynamicalThreshold(exponent, thresholds.root_finder).locate(
        *thresholds.dynamical,
    )
    if dynamical is None:
        message = (
            f"the top Lyapunov exponent does not change sign in "
            f"{thresholds.dynamical}; no dynamical bifurcation to mark"
        )
        raise ValueError(message)
    grid = thresholds.grid or diagnostics.grid

    def problem(parameter: float) -> StationaryFokkerPlanck:
        found = measurements.density_at(parameter, grid)
        if found is None:
            message = (
                f"no stationary density at {parameter}; the phenomenological "
                f"bracket must lie where the density exists"
            )
            raise ValueError(message)
        return found

    phenomenological = PhenomenologicalThreshold(
        problem,
        thresholds.probe,
        thresholds.root_finder,
    ).locate(*thresholds.phenomenological)
    if phenomenological is None:
        message = (
            f"the density does not change shape in {thresholds.phenomenological}; "
            f"no phenomenological bifurcation to mark"
        )
        raise ValueError(message)
    return Located(phenomenological, dynamical, exponent.pairs())


def density_window(grid: DensityGrid, window: Box) -> tuple[slice, slice, Box]:
    """The cells of ``grid`` whose centres lie in ``window``, and their extent.

    The density is solved on a box large enough that its reflecting boundary
    is where the density is negligible, and drawn on the stage's own box,
    which is smaller. The extent returned is the cell edges, since the panel
    paints each value over its whole cell.
    """
    slices = []
    extent = []
    for axis, (low, high) in zip(range(grid.dimension), window, strict=True):
        points = grid.axis(axis)
        inside = np.flatnonzero((points >= low) & (points <= high))
        if inside.size == 0:
            message = f"the density window {window} holds no cell of the grid"
            raise ValueError(message)
        first, last = int(inside[0]), int(inside[-1])
        half = 0.5 * float(grid.spacings[axis])
        slices.append(slice(first, last + 1))
        extent.append((float(points[first]) - half, float(points[last]) + half))
    return slices[0], slices[1], (extent[0], extent[1])


def frame_density(
    density: GridDensity,
    window: Box,
    centre: Sequence[float],
) -> Density:
    """A solved density as one frame draws it: windowed, scaled, with its crest.

    The crest is the radius of the ring the interior maxima form once the peak
    has left ``centre``. A ring is reported as a chain of cells around it, so
    the radius is their mean distance from the centre. Maxima within a cell or
    so of the centre are the peak at the centre itself and are not a ring;
    while those are the only maxima there is no crest. The ring counts as
    soon as it is a ring, even while the centre is still higher: the grid
    overstates the centre cells of a singular diffusion, and gating the crest
    on the global maximum would hide the ring for a while after it exists.
    """
    grid = density.grid
    xs, ys, _extent = density_window(grid, window)
    block = density.values[xs, ys]
    peak = float(block.max())
    # Rows of constant y, x running fastest: the layout the field uses.
    values = [float(v) for v in (block / peak).T.ravel()] if peak > 0.0 else []
    centre_point = np.asarray(centre, dtype=float)
    mode = density.dominant_mode()
    cell = float(np.max(grid.spacings))
    maxima = [
        (float(point[0]), float(point[1]))
        for point in density.interior_maxima()
        if window[0][0] <= point[0] <= window[0][1]
        and window[1][0] <= point[1] <= window[1][1]
    ]
    ring = [
        radius
        for radius in (
            float(np.hypot(x - centre_point[0], y - centre_point[1])) for x, y in maxima
        )
        if radius > _AT_CENTRE_CELLS * cell
    ]
    crest = float(np.mean(ring)) if ring else None
    return Density(values, maxima, (float(mode[0]), float(mode[1])), crest, peak)


def check_noise(noise: NoiseModel, view: View, drift: Function, parameter, value):
    """Refuse noise terms that are not the matrix, or a drift that is not Ito.

    The page integrates the terms, not the library's objects, so a mistyped
    coefficient would send the paths through a different equation from the
    one whose density and exponent the panels show. The terms are evaluated at
    random states inside the view's bounds and compared with the matrix; the
    view's drift terms are compared with the Ito drift under the convention,
    which is the drift the Euler-Maruyama step must use. A Stratonovich
    equation whose correction does not vanish therefore needs its corrected
    drift in the view, and this is where that omission is caught.
    """
    generator = np.random.default_rng(0)
    factor = stratonovich_factor(noise.convention)
    saved = parameter.value
    parameter.value = value
    try:
        for _ in range(_CHECKS):
            state = np.array([generator.uniform(lo, hi) for lo, hi in view.bounds])
            columns = noise.matrix.columns_at(state)
            for driver, terms in enumerate(noise.terms):
                claimed = np.array(evaluate_terms(terms, state, value))
                expected = columns[driver]
                scale = 1.0 + float(np.max(np.abs(expected)))
                if float(np.max(np.abs(claimed - expected))) > _AGREEMENT * scale:
                    message = (
                        f"noise driver {driver} disagrees with its matrix at "
                        f"{state.tolist()}: terms give {claimed.tolist()}, the "
                        f"matrix gives {expected.tolist()}"
                    )
                    raise ValueError(message)
            ito = field_at(drift, state) + state_drift_shift(
                noise.matrix.jacobians_at(state),
                columns,
                factor,
            )
            claimed = np.array(evaluate_terms(view.field, state, value))
            scale = 1.0 + float(np.max(np.abs(ito)))
            if float(np.max(np.abs(claimed - ito))) > _AGREEMENT * scale:
                message = (
                    f"the view's drift is not the {noise.name} equation's Ito "
                    f"drift at {state.tolist()}: terms give {claimed.tolist()}, "
                    f"the Ito drift is {ito.tolist()}"
                )
                raise ValueError(message)
    finally:
        parameter.value = saved


def stochastic_stage(
    scene: StageScene,
    continued: Continued,
    noise: NoiseModel,
    diagnostics: Diagnostics,
    describe: Callable[[Frame], str | None] | None = None,
) -> StageScene:
    """Add paths, densities, the exponent and both thresholds to ``scene``.

    ``continued`` is what the scene was staged from; its equation is the drift
    the noise is checked against. The scene must have been staged through a
    :class:`~.stage.View`, since the page integrates the drift from polynomial
    terms; a sampled field cannot be combined with a noise term. ``describe``
    runs after the densities are in place, so a regime can be named by what
    the density does.
    """
    view = scene.lattice.view
    if view is None:
        message = (
            "a stochastic stage integrates its paths from polynomial terms; give "
            "the lattice a View carrying the drift"
        )
        raise ValueError(message)
    drift = continued.equation.derivative
    parameter = drift.parameters[continued.parameter_index]
    check_noise(noise, view, drift, parameter, float(scene.frames[0].parameter))
    measurements = diagnostics.measurements
    xs, ys, extent = density_window(diagnostics.grid, scene.lattice.box)
    shape = (xs.stop - xs.start, ys.stop - ys.start)
    for frame in scene.frames:
        problem = measurements.density_at(frame.parameter, diagnostics.grid)
        if problem is None:
            frame.density = Density([], [], None, None)
        else:
            frame.density = frame_density(
                problem.solve(),
                scene.lattice.box,
                diagnostics.centre,
            )
        if describe is not None:
            frame.label = describe(frame)
    located = locate_thresholds(diagnostics)
    symbol = scene.system.parameter
    centre_x = diagnostics.centre[0]
    scene.timeline.special.extend(
        [
            SpecialPoint(
                located.dynamical,
                centre_x,
                D_BIFURCATION,
                f"D: top exponent crosses zero, {symbol} = {located.dynamical:.3f}",
            ),
            SpecialPoint(
                located.phenomenological,
                centre_x,
                P_BIFURCATION,
                f"P: density craters, {symbol} = {located.phenomenological:.3f}",
            ),
        ],
    )
    parameters = [frame.parameter for frame in scene.frames]
    closed = measurements.closed
    scene.stochastic = StochasticStage(
        noise.paths(),
        DensityWindow(
            extent,
            shape,
            (diagnostics.centre[0], diagnostics.centre[1]),
            _sampled(closed.crest, parameters),
        ),
        ExponentCurve(located.exponents, _sampled(closed.exponent, parameters)),
    )
    return scene


def _sampled(
    closed_form: Callable[[float], float | None] | None,
    parameters: Sequence[float],
) -> Pairs | None:
    """A closed form at every frame, leaving out where it has no value."""
    if closed_form is None:
        return None
    pairs: Pairs = []
    for value in parameters:
        result = closed_form(float(value))
        if result is not None and math.isfinite(result):
            pairs.append((float(value), float(result)))
    return pairs
