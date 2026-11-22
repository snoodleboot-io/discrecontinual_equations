"""The residence-time pages: the three-cell ferroelectric ring read as a sensor.

The ring of :class:`~discrecontinual_equations.systems.coupled_ring.FerroelectricRing`
at ``a = 1`` and ``lambda = -0.5`` has exactly two attractors, the symmetric states
``x_- = -s (1, 1, 1)`` and ``x_+ = +s (1, 1, 1)`` (``s^2 = a - lambda`` at zero
field), and no other stable state - the mixed states of weaker coupling have
gone. With independent noise ``sigma dW_i`` on every cell (``sigma^2 = 1/2``) the
state hops between the two, and a target field ``eps`` lengthens the stay in the
well it favours and shortens the other: the difference of the mean residence times
is the sensor's response. The two wells are separated not by a saddle point but by
the saddle *cycle* of the rotating wave born at ``lambda = -2a``, so no Kramers or
Eyring-Kramers formula holds (the prefactor of an escape through a saddle cycle
oscillates with ``log sigma``); the times here come only from the two routes of
:mod:`~discrecontinual_equations.continuation.first_passage`.

*Grid.* The backward Kolmogorov problem on a ``41^3`` grid on ``[-2.2, 2.2]^3``,
mean time from ``x_-`` to the ball of radius ``0.4`` about ``x_+``; by the
symmetry ``x -> -x``, ``eps -> -eps`` of the ring, the time from ``x_+`` at field
``eps`` is the time from ``x_-`` at ``-eps``, so one solve per field serves both
wells and the response is odd by construction. At zero field the time is 149.14,
145.88, 144.46 and 143.82 on 21, 31, 41 and 51 cells per side - second order,
extrapolating to 142.7 - so the 41 cell value drawn is 1.3% high, which is the
grid's error bar on this page.

*Ensemble.* ``EnsembleResidence`` with 400 paths of length 2000 at ``dt = 0.01``,
half started in each well, the same balls as the readout sets; each estimate is
drawn with its 95% interval. Measured (ensemble +- standard error against the
grid averaged over the entry states, 41^3 cells, itself about 1.3% high):

  - ``eps = 0``: 140.2 +- 2.4 and 140.8 +- 2.5 against 144.2 in both wells;
  - ``eps = 0.05``: 79.4 +- 1.5 against 81.5 (lower), 272.4 +- 5.3 against
    278.8 (upper);
  - ``eps = 0.1``: 49.9 +- 1.2 against 50.2, 571 +- 14 against 589.

The residence step ``dt = 0.01`` is coarse but its bias is below the error bars
here: at zero field 300 paths gave 132.9, 132.6 at ``dt = 0.005, 0.0025`` with
the old, censoring estimator, the same as at 0.01. The sweep takes about 18
minutes on a loaded machine, most of it the nine grid solves.
"""

import functools
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve

from discrecontinual_equations.continuation.first_passage import (
    Ball,
    EnsembleFirstPassage,
    EnsembleResidence,
    GridFirstPassage,
    HalfSpace,
    MeanFirstPassage,
    ResidenceSample,
    TimeSample,
    eyring_kramers_time,
)
from discrecontinual_equations.continuation.fokker_planck import DensityGrid
from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.continuation.quasipotential import (
    EquilibriumAttractor,
    OrderedLineIntegral,
)
from discrecontinual_equations.continuation.stochastic import Ito
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.solver.solver_config import StochasticConfig
from discrecontinual_equations.systems.coupled_ring import FerroelectricRing
from discrecontinual_equations.variable import Variable
from discrecontinual_equations.webplot.scene import Scene
from discrecontinual_equations.webplot.scene_builder import curve_scene

try:  # python -m examples.continuation.pages_residence
    from examples.continuation.systems import First, Second, State, Third
except ImportError:  # run as a script path: only this directory is on sys.path
    from systems import First, Second, State, Third

EPSILON = "ε"
_GAIN = 1.0
_COUPLING = -0.5
_VARIANCE = 0.5
_CELLS = 3
_BOX = 2.2
_GRID_POINTS = 41
_READOUT_RADIUS = 0.4
_GRID_FIELDS = (0.0, 0.025, 0.05, 0.075, 0.1)
_ENSEMBLE_FIELDS = (0.0, 0.05, 0.1)
_DENSITY_FIELDS = (0.0, 0.1)
_PATHS = 400
_HORIZON = 2000.0
_STEP = 0.01
_SEED = 33
_BINS = 30


def _ring_drift(states: np.ndarray, field: float) -> np.ndarray:
    """``a x_i - x_i^3 - lambda x_{i+1} + eps``, row by row."""
    return _GAIN * states - states**3 - _COUPLING * np.roll(states, -1, axis=-1) + field


class StochasticRing(StochasticFunction):
    """The ring with additive noise, evaluated as arrays for the ensemble."""

    field: float = 0.0

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return _ring_drift(np.asarray(point, dtype=float), self.field).tolist()

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return [math.sqrt(_VARIANCE)] * len(point)

    def eval_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> np.ndarray:
        return _ring_drift(points, self.field)

    def diffusion_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> np.ndarray:
        return np.full_like(points, math.sqrt(_VARIANCE))


class _NoiseColumn(DeterministicFunction):
    """Column ``index`` of ``sigma I``."""

    def __init__(self, index: int) -> None:
        states = [State() for _ in range(_CELLS)]
        super().__init__(states, [], [State() for _ in range(_CELLS)], None)
        self._column = [
            math.sqrt(_VARIANCE) if row == index else 0.0 for row in range(_CELLS)
        ]

    def eval(
        self,
        point: list[float],  # noqa: ARG002 (constant field)
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return list(self._column)


def _ring(field: float) -> FerroelectricRing:
    return FerroelectricRing(
        variables=[State() for _ in range(_CELLS)],
        parameters=[
            First(value=_COUPLING),
            Second(value=_GAIN),
            Third(value=field),
        ],
        results=[State() for _ in range(_CELLS)],
        time=None,
    )


def _equation(field: float) -> DifferentialEquation:
    time = Variable(name="t", discretization=[])
    variables = [Variable(name=f"x{i}", discretization=[]) for i in range(_CELLS)]
    derivative = StochasticRing(
        variables,
        [],
        [Variable(name=f"dx{i}/dt", discretization=[]) for i in range(_CELLS)],
        time,
    )
    derivative.field = field
    return DifferentialEquation(
        variables=variables,
        time=time,
        parameters=[],
        derivative=derivative,
    )


def attractors(field: float) -> tuple[np.ndarray, np.ndarray]:
    """The two symmetric states at ``field``, lower then upper."""
    guess = math.sqrt(_GAIN - _COUPLING)
    lower = fsolve(lambda x: _ring_drift(x, field), [-guess] * _CELLS)
    upper = fsolve(lambda x: _ring_drift(x, field), [guess] * _CELLS)
    return np.asarray(lower), np.asarray(upper)


def grid_time_up(field: float, points: int = _GRID_POINTS) -> GridFirstPassage:
    """Mean time from every cell into the ball about ``x_+``, on the grid."""
    grid = DensityGrid.box([-_BOX] * _CELLS, [_BOX] * _CELLS, [points] * _CELLS)
    problem = MeanFirstPassage(
        _ring(field),
        NoiseMatrix([_NoiseColumn(index) for index in range(_CELLS)]),
        Ito(),
        grid,
    )
    _, upper = attractors(field)
    return problem.solve(Ball(upper, _READOUT_RADIUS).on(grid))


def ensemble_residences(field: float, seed: int = _SEED) -> ResidenceSample:
    """Residences in the lower (first) and upper (second) wells, by ensemble."""
    lower, upper = attractors(field)
    return EnsembleResidence(
        StochasticConfig(step_size=_STEP, random_seed=seed),
    ).solve(
        _equation(field),
        (lower, upper),
        _PATHS,
        (Ball(lower, _READOUT_RADIUS), Ball(upper, _READOUT_RADIUS)),
        horizon=_HORIZON,
    )


@dataclass(frozen=True)
class _Reading:
    """Everything the two pages draw at one field."""

    lower: float
    """Grid time from ``x_-`` (the mean residence in the lower well)."""

    upper: float
    """Grid time from ``x_+``."""

    sample: ResidenceSample | None = None
    """The ensemble's residences, at the fields it was run at."""

    lower_entries: float = math.nan
    """Grid time averaged over where the ensemble's lower residences began."""

    upper_entries: float = math.nan
    """The same for the upper well."""


@functools.cache
def _sweep() -> dict[float, _Reading]:
    """Grid times in both wells at every field, and the ensemble where it was run.

    The time from ``x_+`` at ``eps`` is the time from ``x_-`` at ``-eps`` with the
    state reflected, ``tau_+(x; eps) = tau_-(-x; -eps)``, so one grid solve per
    field serves both wells. Each ensemble field has its own seed.
    """
    fields = sorted({*_GRID_FIELDS, *(-value for value in _GRID_FIELDS)})
    solved = {field: grid_time_up(field) for field in fields}
    readings = {}
    for field in fields:
        lower, upper = attractors(field)
        readings[field] = _Reading(
            lower=solved[field].at(lower),
            upper=solved[-field].at(-upper),
        )
    for index, field in enumerate(_ENSEMBLE_FIELDS):
        sample = ensemble_residences(field, seed=_SEED + index)
        reading = readings[field]
        readings[field] = _Reading(
            lower=reading.lower,
            upper=reading.upper,
            sample=sample,
            lower_entries=_mean_over(solved[field], sample.first.starts),
            upper_entries=_mean_over(solved[-field], -sample.second.starts),
        )
    return readings


def _mean_over(times: GridFirstPassage, states: np.ndarray) -> float:
    return float(np.mean([times.at(state) for state in states]))


def _efield_residence_response() -> Scene:
    readings = _sweep()
    grid_line = [
        (field, reading.upper - reading.lower) for field, reading in readings.items()
    ]
    lines = [("grid,_from_the_attractors", "", grid_line, None)]
    ensemble_points, entry_points = [], []
    for field, reading in readings.items():
        if reading.sample is None:
            continue
        low, high = reading.sample.asymmetry_interval()
        # asymmetry is first minus second, lower minus upper: the response is its
        # negative.
        ensemble_points.append((field, -reading.sample.asymmetry))
        entry_points.append((field, reading.upper_entries - reading.lower_entries))
        lines.append(
            ("ensemble_95%_interval", "", [(field, -high), (field, -low)], None)
        )
    markers = [
        ("ensemble", "ensemble residences", ensemble_points, None),
        ("grid", "grid, over the ensemble's entry states", entry_points, None),
    ]
    return curve_scene(
        lines,
        markers,
        (f"target field {EPSILON}", "T+ - T-  (mean residence, upper minus lower)"),
        "Residence-time response of the three-cell ferroelectric ring",
        "The sensor's response curve: the difference of the mean residence times "
        "in the two symmetric states against the target field, a = 1, lambda = "
        "-0.5, sigma^2 = 1/2. Line: backward Kolmogorov first-passage times on a "
        "41^3 grid from the two attractors (1.3% above the grid-converged value "
        "at zero field). Points: 400-path ensemble with 95% intervals, and the "
        "grid's time averaged over where the ensemble's residences began - the "
        "quantity the ensemble estimates, within 0.2% of the attractors' time "
        "here. Zero field: 0.6 +- 6.7; 0.05: 193 +- 11 against the grid's 197; "
        "0.1: 522 +- 28 against 539. The response is odd in the field and zero "
        "at zero field to Monte-Carlo error. The wells are separated by a saddle "
        "cycle, "
        "so no Kramers formula applies.",
    )


def _histogram(times: np.ndarray, top: float) -> list[tuple[float, float]]:
    """A histogram on ``[0, top]`` as a step polyline, normalised by every sample.

    The residences beyond ``top`` are counted in the normalisation, so the steps
    are the density itself on the window rather than the density conditioned on
    falling in it, and the exponential drawn over them needs no correction.
    """
    counts, edges = np.histogram(times, bins=_BINS, range=(0.0, top))
    heights = counts / (times.size * np.diff(edges))
    points = []
    for left, right, height in zip(edges[:-1], edges[1:], heights, strict=True):
        points.extend([(float(left), float(height)), (float(right), float(height))])
    return points


def _efield_residence_densities() -> Scene:
    readings = _sweep()
    strongest = readings[_ENSEMBLE_FIELDS[-1]]
    top = 4.0 * max(strongest.lower, strongest.upper)
    axis = np.linspace(0.0, top, 80)
    lines = []
    for field in _DENSITY_FIELDS:
        reading = readings[field]
        if reading.sample is None:
            continue
        for name, estimate, mean in (
            ("lower", reading.sample.first, reading.lower_entries),
            ("upper", reading.sample.second, reading.upper_entries),
        ):
            kind = f"{name}_well,_{EPSILON}_=_{field:g}"
            lines.append((kind, "", _histogram(estimate.times, top), None))
            lines.append(
                (
                    "exponential_with_the_grid_mean",
                    "",
                    [(float(t), math.exp(-t / mean) / mean) for t in axis],
                    None,
                ),
            )
    return curve_scene(
        lines,
        [],
        ("residence time", "density"),
        "Residence-time densities of the ferroelectric ring",
        "Histograms of the ensemble's residences in each well at zero field and "
        f"at {EPSILON} = {_DENSITY_FIELDS[-1]:g}, against the exponential density "
        "whose mean is the grid's time averaged over the residences' entry "
        "states. At zero field the two wells' densities coincide; the field "
        "stretches one and compresses the other.",
    )


# --- Where the small-noise asymptotics hold ---------------------------------
#
# The double well V = x^4/4 - x^2/2 in one dimension, and the planar gradient
# well V + (1 + x^2) y^2 / 2 (transverse curvature 2 at the minima, 1 at the
# saddle), both with G = sigma I and the time from the left minimum to the
# right one. Every time is divided by the Eyring-Kramers time, so the
# asymptotics hold where a curve reaches 1.

_LINE_WALL = -3.0
_LINE_CELLS = 401
_PLANE_BOX = ([-2.2, -1.8], [2.2, 1.8])
_PLANE_CELLS = 161
_PLANE_RADIUS = 0.3
_BARRIERS = (1.0, 2.0, 4.0, 8.0, 12.0, 16.0)
_ENSEMBLE_BARRIERS = (1.0, 1.5, 2.0)
_ENSEMBLE_PATHS = 2000
_ENSEMBLE_STEP = 0.0025


class _LineDrift(DeterministicFunction):
    """``f = x - x^3``."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x = point[0]
        return [x - x**3]


class _PlaneDrift(DeterministicFunction):
    """``f = -grad V`` for the planar gradient well."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        return [x - x**3 - x * y * y, -(1.0 + x * x) * y]


class _Column(DeterministicFunction):
    """One constant column of ``G``."""

    def __init__(self, column: list[float]) -> None:
        count = len(column)
        super().__init__(
            [State() for _ in range(count)],
            [],
            [State() for _ in range(count)],
            None,
        )
        self._column = column

    def eval(
        self,
        point: list[float],  # noqa: ARG002 (constant field)
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return list(self._column)


class _StochasticPlane(StochasticFunction):
    """The planar gradient well with ``G = sigma I``, batched for the ensemble."""

    sigma: float = 1.0

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        return [x - x**3 - x * y * y, -(1.0 + x * x) * y]

    def diffusion(
        self,
        point: list[float],  # noqa: ARG002 (constant noise)
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        return [self.sigma, self.sigma]

    def eval_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> np.ndarray:
        x, y = points[:, 0], points[:, 1]
        return np.stack([x - x**3 - x * y * y, -(1.0 + x * x) * y], axis=1)

    def diffusion_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> np.ndarray:
        return np.full_like(points, self.sigma)


def _noise(sigma: float, dimension: int) -> NoiseMatrix:
    return NoiseMatrix(
        [
            _Column([sigma if row == column else 0.0 for row in range(dimension)])
            for column in range(dimension)
        ],
    )


def _variance(barrier: float) -> float:
    """``sigma^2`` giving ``Delta U = 2 Delta V / sigma^2``, ``Delta V = 1/4``."""
    return 0.5 / barrier


def _line_time(barrier: float) -> float:
    spacing = (1.0 - _LINE_WALL) / (_LINE_CELLS - 0.5)
    grid = DensityGrid([_LINE_WALL + spacing * (0.5 + np.arange(_LINE_CELLS))])
    problem = MeanFirstPassage(
        _LineDrift([State()], [], [State()], None),
        _noise(math.sqrt(_variance(barrier)), 1),
        Ito(),
        grid,
    )
    return problem.solve(HalfSpace([1.0], 1.0 - 1.0e-9).on(grid)).at([-1.0])


def _plane_time(barrier: float) -> float:
    grid = DensityGrid.box(*_PLANE_BOX, [_PLANE_CELLS, _PLANE_CELLS])
    problem = MeanFirstPassage(
        _PlaneDrift([State(), State()], [], [State(), State()], None),
        _noise(math.sqrt(_variance(barrier)), 2),
        Ito(),
        grid,
    )
    target = Ball((1.0, 0.0), _PLANE_RADIUS).on(grid)
    return problem.solve(target).at([-1.0, 0.0])


def _plane_ensemble(barrier: float, seed: int) -> TimeSample:
    time = Variable(name="t", discretization=[])
    variables = [
        Variable(name="x", discretization=[]),
        Variable(name="y", discretization=[]),
    ]
    derivative = _StochasticPlane(
        variables,
        [],
        [Variable(name=f"d{v.name}/dt", discretization=[]) for v in variables],
        time,
    )
    derivative.sigma = math.sqrt(_variance(barrier))
    equation = DifferentialEquation(
        variables=variables,
        time=time,
        parameters=[],
        derivative=derivative,
    )
    return EnsembleFirstPassage(
        StochasticConfig(step_size=_ENSEMBLE_STEP, random_seed=seed),
    ).solve(
        equation,
        [-1.0, 0.0],
        _ENSEMBLE_PATHS,
        Ball((1.0, 0.0), _PLANE_RADIUS),
        horizon=1.0e5,
    )


def _olim_barrier() -> float:
    """``Delta U`` at ``sigma = 1`` from the ordered-line-integral quasipotential.

    ``U`` scales as ``1 / sigma^2`` for ``G = sigma I``, so one solve at
    ``sigma = 1`` gives the barrier at every noise level; the exact value is
    ``2 Delta V = 1/2``.
    """
    grid = DensityGrid.box([-2.0, -1.5], [1.0, 1.5], [_PLANE_CELLS, _PLANE_CELLS])
    field = OrderedLineIntegral(
        _PlaneDrift([State(), State()], [], [State(), State()], None),
        _noise(1.0, 2),
        grid,
    ).solve(EquilibriumAttractor([-1.0, 0.0]))
    return field.at([0.0, 0.0])


def _line_kramers(barrier: float) -> float:
    return eyring_kramers_time(barrier, 1.0, [[-1.0]], [[2.0]])


def _plane_kramers(barrier: float) -> float:
    return eyring_kramers_time(
        barrier,
        1.0,
        [[-1.0, 0.0], [0.0, 1.0]],
        [[2.0, 0.0], [0.0, 2.0]],
    )


def _kramers_asymptotics() -> Scene:
    line = [
        (barrier, _line_time(barrier) / _line_kramers(barrier)) for barrier in _BARRIERS
    ]
    plane = [
        (barrier, _plane_time(barrier) / _plane_kramers(barrier))
        for barrier in _BARRIERS
    ]
    olim = _olim_barrier() / 0.5
    olim_line = [
        (barrier, _plane_time(barrier) / _plane_kramers(olim * barrier))
        for barrier in _BARRIERS
    ]
    lines = [
        ("one-dimensional_well,_grid", "", line, None),
        ("planar_well,_grid", "", plane, None),
        ("planar_well,_grid_/_EK_with_OLIM_barrier", "", olim_line, None),
        ("asymptote", "", [(_BARRIERS[0], 1.0), (_BARRIERS[-1], 1.0)], None),
    ]
    points = []
    for index, barrier in enumerate(_ENSEMBLE_BARRIERS):
        sample = _plane_ensemble(barrier, seed=_SEED + 100 + index)
        scale = _plane_kramers(barrier)
        low, high = sample.confidence_interval()
        points.append((barrier, sample.mean / scale))
        lines.append(
            (
                "ensemble_95%_interval",
                "",
                [(barrier, low / scale), (barrier, high / scale)],
                None,
            ),
        )
    markers = [("ensemble", "planar well, ensemble", points, None)]
    return curve_scene(
        lines,
        markers,
        (
            "barrier Delta U = 2 Delta V / sigma^2",
            "mean first-passage time / Eyring-Kramers",
        ),
        "Where Kramers' law holds: grid and ensemble against the asymptotics",
        "Mean time from one well to the other, divided by the Eyring-Kramers "
        "time, as the noise falls. The one-dimensional double well (grid, 401 "
        "cells; equal to the exact double integral to 1e-4) and a planar "
        "gradient well whose transverse curvature is 2 at the minima and 1 at "
        "the saddle (grid, 161^2 cells; 2000-path ensemble with 95% intervals at "
        "moderate noise). The excess over 1 halves with the noise variance: the "
        "law is a limit, within 5% only from Delta U of about 10 (line) and 14 "
        "(plane). The third curve takes the barrier from the ordered-line-"
        "integral quasipotential (0.50001 against the exact 1/2) and lies on the "
        "second.",
    )
