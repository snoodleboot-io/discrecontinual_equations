"""Shared scaffolding for the stage examples.

Every stage does the same four things before it has anything to say about its
own system: fill the component registry, build a differential equation from a
field class, continue a branch in one parameter, and write the page with its
atlas card. Those live here so each example is only the part that differs -
the system, the seeds, and the narration.
"""

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sweet_tea.registry import Registry

import discrecontinual_equations
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

_registered = False
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


def ensure_registry() -> None:
    """Fill the component registry once; the continuation builders need it."""
    global _registered  # noqa: PLW0603 (module-level guard)
    if _registered:
        return
    Registry.fill_registry(
        path=str(Path(discrecontinual_equations.__file__).parent),
        module="discrecontinual_equations",
        exclude=["*.tests", "*.examples", "*.plot"],
    )
    _registered = True


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


def continue_branch(
    eq: DifferentialEquation,
    seed: Sequence[float],
    span: tuple[float, float],
    detectors: Sequence[str],
    step: float = 0.02,
) -> Branch:
    """Continue an equilibrium of ``eq`` across ``span`` from ``seed``.

    ``span`` is travelled in the order it is given: ``(low, high)`` starts at
    the low end and climbs, ``(high, low)`` starts at the high end and
    descends. A branch whose equilibria exist only on one side of a fold has
    to be entered from that side. Either way there is a little room beyond
    each end, so a bifurcation sitting on the boundary is crossed and detected
    rather than clipped.
    """
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
        parameter_lower_bound=low - 0.05,
        parameter_upper_bound=high + 0.05,
        maximum_points=900,
        maximum_step=step,
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


def keep_cycles(points: Sequence[CyclePoint]) -> list[CyclePoint]:
    """The leading run of traced points that are still cycles.

    A continuation that loses the branch reports it by collapsing the orbit
    onto the equilibrium or letting the period run away, so the run stops at
    the first point that is no longer an orbit rather than drawing the wreck.
    """
    kept: list[CyclePoint] = []
    for point in points:
        period = point.solution.period
        if point.amplitude < MIN_AMPLITUDE or not (
            PERIOD_RANGE[0] < period < PERIOD_RANGE[1]
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
    """Write one stage page and a one-card atlas beside it."""
    report = PlotReport(StageRenderer(), output_dir)
    path = report.write(scene, filename)
    report.write_atlas([entry])
    return path
