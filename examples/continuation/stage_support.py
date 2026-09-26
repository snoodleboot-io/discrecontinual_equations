"""Shared scaffolding for the stage examples.

Every stage does the same four things before it has anything to say about its
own system: fill the component registry, build a differential equation from a
field class, continue a branch in one parameter, and write the page with its
atlas card. Those live here so each example is only the part that differs -
the system, the seeds, and the narration.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.continuation.cycle_continuation import (
    CyclePoint,
    CycleSeed,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable
from discrecontinual_equations.webplot.report import AtlasEntry, PlotReport
from discrecontinual_equations.webplot.stage import StageScene
from discrecontinual_equations.webplot.stage_renderer import StageRenderer

try:  # python -m examples.continuation.<film>
    from examples.continuation.component_registry import ensure_registry
except ImportError:  # run as a script path: only this directory is on sys.path
    from component_registry import ensure_registry

# A traced cycle is kept while it is a cycle: an amplitude that has collapsed
# onto the equilibrium, or a period that has run away, is the continuation
# losing the branch rather than a smaller or slower orbit.
MIN_AMPLITUDE = 1.0e-3
PERIOD_RANGE = (0.5, 400.0)
# How far past the filmed range a kept cycle may sit, as a fraction of it.
_TRIM_MARGIN = 0.02


class Mu(Parameter, name="Mu", abbreviation="mu"):
    """The one continuation parameter of a single-parameter system."""


class State(Variable, name="State", abbreviation="v"):
    """One state coordinate."""


class Time(Variable, name="Time", abbreviation="t"):
    """The time variable."""


def equation(
    field_type: type[DeterministicFunction],
    parameters: list[Parameter],
    count: int = 2,
) -> DifferentialEquation:
    """A differential equation of ``count`` states driven by ``field_type``."""
    return DifferentialEquation(
        variables=[State() for _ in range(count)],
        time=Time(),
        parameters=parameters,
        derivative=field_type(
            variables=[State() for _ in range(count)],
            parameters=parameters,
            results=[State() for _ in range(count)],
            time=None,
        ),
    )


@dataclass(frozen=True, slots=True)
class BranchLimits:
    """How hard an equilibrium continuation is allowed to work.

    These were the reason two films could not use :func:`continue_branch` and
    configured their own continuation instead - which is how one of them came
    to miss a later improvement entirely. A film that needs a finer step or a
    tighter margin says so here rather than building its own config.

    ``margin`` is the room left beyond each end of the span, so a bifurcation
    sitting on the boundary is crossed and detected rather than clipped; a film
    whose branch should stop close to its span asks for a smaller one.
    """

    step: float = 0.02
    margin: float = 0.05
    maximum_points: int = 900


def continue_branch(
    eq: DifferentialEquation,
    seed: Sequence[float],
    span: tuple[float, float],
    detectors: Sequence[str],
    limits: BranchLimits | None = None,
) -> Branch:
    """Continue an equilibrium of ``eq`` across ``span`` from ``seed``.

    ``span`` is travelled in the order it is given: ``(low, high)`` starts at
    the low end and climbs, ``(high, low)`` starts at the high end and
    descends. A branch whose equilibria exist only on one side of a fold has
    to be entered from that side. Either way there is a little room beyond
    each end, so a bifurcation sitting on the boundary is crossed and detected
    rather than clipped.
    """
    limits = limits or BranchLimits()
    start, finish = span
    low, high = min(span), max(span)
    ensure_registry()
    eq.derivative.parameters[0].value = start
    config = ContinuationConfig(
        continuation_parameter_index=0,
        detectors=list(detectors),
        initial_parameter=start,
        direction=1 if finish > start else -1,
        measure="component",
        parameter_lower_bound=low - limits.margin,
        parameter_upper_bound=high + limits.margin,
        maximum_points=limits.maximum_points,
        maximum_step=limits.step,
    )
    return ContinuerBuilder.build(config, eq).solve(eq, list(seed))


def circle_seed(
    radius: float,
    period: float,
    parameter: float,
    intervals: int,
) -> CycleSeed:
    """A circular cycle seed, for a system whose orbit is known to be round."""
    grid = np.linspace(0.0, 1.0, intervals + 1)
    angle = 2.0 * math.pi * grid
    states = np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])
    return CycleSeed(states, period, parameter)


def keep_cycles(
    points: Sequence[CyclePoint],
    minimum_amplitude: float = MIN_AMPLITUDE,
    periods: tuple[float, float] = PERIOD_RANGE,
) -> list[CyclePoint]:
    """The leading run of traced points that are still cycles.

    A continuation that loses the branch reports it by collapsing the orbit
    onto the equilibrium or letting the period run away, so the run stops at
    the first point that is no longer an orbit rather than drawing the wreck.

    The thresholds are arguments because what counts as a lost branch is the
    system's own business: a film whose orbit approaches a homoclinic expects
    long periods and should not accept short ones, which for the shared default
    would look like an ordinary fast cycle.
    """
    kept: list[CyclePoint] = []
    for point in points:
        period = point.solution.period
        if point.amplitude < minimum_amplitude or not (
            periods[0] < period < periods[1]
        ):
            break
        kept.append(point)
    return kept


def within(points: Sequence[CyclePoint], span: tuple[float, float]) -> list[CyclePoint]:
    """Only the traced cycles the film will actually show.

    A trace runs until it loses the branch or exhausts its steps, which can be
    far outside the range being filmed. Those cycles are never drawn on the
    stage but would still stretch the timeline's scale and swell the page.
    """
    low, high = span
    margin = _TRIM_MARGIN * (high - low)
    return [
        point for point in points if low - margin <= point.parameter <= high + margin
    ]


def publish(
    scene: StageScene,
    filename: str,
    entry: AtlasEntry,
    output_dir: str,
) -> Path:
    """Write one stage page, and a one-card atlas only where none exists.

    Running a single film into the directory the full atlas was built in used to
    replace its ``index.html`` with a one-card one, throwing away every other
    card - so a ten-film loop left an index naming only the last of them. The
    page is what the run is for; an index already there belongs to whatever
    built it and is left alone.
    """
    report = PlotReport(StageRenderer(), output_dir)
    path = report.write(scene, filename)
    if not (Path(output_dir) / "index.html").exists():
        report.write_atlas([entry])
    return path
