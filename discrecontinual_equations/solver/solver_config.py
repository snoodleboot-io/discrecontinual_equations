"""
Base configuration classes for numerical solvers.

This module provides a hierarchy of configuration classes for different types
of numerical solvers. The hierarchy reduces code duplication by providing
common functionality in base classes.

Hierarchy:
- SolverConfig (abstract base)
  ├── FixedStepConfig (fixed step size methods)
  │   ├── DeterministicConfig (base for deterministic solvers)
  │   └── StochasticConfig (base for stochastic solvers with calculus support)
  └── AdaptiveConfig (adaptive step size methods)
"""

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# Refusal tolerance for a span that should be a whole number of steps: the gap
# between ``end - start`` and ``n h``, relative to the larger of |start| and
# |end|. Decimal inputs (up to six decimals, |start| up to 1e4, 1e5 steps) miss
# by at most 5.1e-16 of it in a draw of 200k; a step written as a rounded
# repeating decimal, 1/3 as 0.333333333333, by 1e-12. A span a deliberate
# fraction of a step off misses by far more.
GRID_TOLERANCE = 1e-9
# The same gap may never exceed this fraction of the step. Far from zero,
# ``GRID_TOLERANCE * |t|`` alone grows past the step itself - at ``|t| = 1e6``
# with a step of ``1e-3`` it is a whole step - and a span half a step off would
# be accepted with a last step 1.5 times the others. Rounding at that ``|t|`` is
# a few ulps, ``1e-4`` of a step at ``|t| = 1e9`` and a step of ``1e-3``.
GRID_STEP_FRACTION = 1e-3


def uniform_step_count(start: float, end: float, step: float) -> int:
    """The number of steps of size ``step`` from ``start`` to ``end``.

    A fixed-step solver advances by the same ``step`` every time, so the span
    ``end - start`` must be a whole number ``n`` of steps. ``n`` is the span
    over the step rounded to the nearest integer, and it is accepted when
    ``n * step`` matches the span to rounding - :data:`GRID_TOLERANCE` relative
    to the larger of ``|start|`` and ``|end|``, and never more than
    :data:`GRID_STEP_FRACTION` of the step. Decimal inputs such as
    ``(0, 10, 0.005)`` are multiples only to rounding in binary, which this
    accepts; a span a real fraction of a step off, such as ``(0, 0.8, 0.015)``,
    is refused rather than integrated past ``end`` or with a shorter last step.
    An empty span (``start == end``) has no steps.

    Args:
        start: The first node.
        end: The last node; not before ``start``.
        step: The step, positive.

    Returns:
        The step count ``n``, so the grid has ``n + 1`` nodes.

    Raises:
        ValueError: If the step is not positive, ``end`` is before
            ``start``, or the span is not a whole number of steps.

    References:
        NumPy Developers, "numpy.arange", NumPy Reference (stable). Its length
        is ``ceil((stop - start)/step)``, which "may result in the last element
        of out being greater than stop", and it recommends ``numpy.linspace``
        for a non-integer step: the defect this count replaces (DEQ-129).
        See ``docs/references.md`` (NumPyDevelopers2026, PythonDocs2026).
    """
    if not step > 0:  # a NaN step fails this too
        message = f"step_size must be positive, got {step}"
        raise ValueError(message)
    span = end - start
    if not (np.isfinite(span) and span >= 0):
        message = (
            f"end_time ({end}) must not be before start_time ({start}): "
            "the step is positive"
        )
        raise ValueError(message)
    steps = round(span / step)
    scale = max(abs(start), abs(end))
    tolerance = min(GRID_TOLERANCE * scale, GRID_STEP_FRACTION * step)
    if abs(span - steps * step) > tolerance:
        message = (
            f"the span from {start} to {end} is not a whole number of steps of "
            f"{step} ({span / step:.6g} steps). Every fixed-step solver takes "
            "the same step each time, so it cannot end at end_time; choose "
            "end_time = start_time + n * step_size, or a step_size that divides "
            "the span"
        )
        raise ValueError(message)
    return steps


def uniform_grid(start: float, end: float, step: float) -> np.ndarray:
    """The nodes ``start + k step``, ``k = 0..n``, with the last node ``end``.

    Built from the integer count of :func:`uniform_step_count`, so it has
    exactly ``n + 1`` nodes, never a node past ``end`` - unlike ``np.arange(start,
    end + step, step)``, whose length rounding decides. Node ``k`` is ``start +
    k * step`` in floating point and the last node is set to ``end`` itself,
    which it equals to rounding.

    Args:
        start: The first node.
        end: The last node.
        step: The step; the span must be a whole number of steps.

    Returns:
        The ``n + 1`` nodes, increasing.

    Raises:
        ValueError: As :func:`uniform_step_count`.

    References:
        NumPy Developers, "numpy.arange", NumPy Reference (stable); see
        ``docs/references.md`` (NumPyDevelopers2026).
    """
    steps = uniform_step_count(start, end, step)
    grid = start + step * np.arange(steps + 1, dtype=float)
    grid[-1] = end
    return grid


class SolverConfig(BaseModel):
    """
    Abstract base class for solver configuration objects.

    All solver configurations inherit from this base class and define
    the parameters specific to their numerical method, such as:
    - Time integration range (start_time, end_time)
    - Step size control (fixed or adaptive)
    - Error tolerances (for adaptive methods)
    - Algorithm-specific parameters

    This class provides the common interface that all solver configurations
    must implement, ensuring consistency across different numerical methods.
    """


class FixedStepConfig(SolverConfig):
    """
    Base configuration for solvers with fixed step sizes.

    Provides common time integration parameters and computed properties
    for methods that use constant step sizes.
    """

    start_time: float = Field(
        default=0.0,
        description="Starting time for the integration interval",
    )
    end_time: float = Field(
        default=1.0,
        description="Ending time for the integration interval",
    )
    step_size: float = Field(default=0.01, description="Size of each integration step")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def times(self) -> np.ndarray:
        """The time grid: ``n + 1`` nodes from ``start_time`` to ``end_time``.

        Node ``k`` is ``start_time + k * step_size`` and the last node is
        ``end_time``. The span must be a whole number of steps to rounding; a
        span that is not is refused, not clipped with a shorter last step,
        because every fixed-step solver takes ``dt`` as its step each time - the
        multistep coefficients, the Wiener increments' variance ``dt`` and a
        :class:`BrownianPath` read a whole number of cells at a time all assume
        it (DEQ-129). See :func:`uniform_grid`.

        Raises:
            ValueError: If the span is not a whole number of steps, the step is
                not positive, or ``end_time`` is before ``start_time``.
        """
        return uniform_grid(self.start_time, self.end_time, self.step_size)

    @property
    def n_steps(self) -> int:
        """Number of integration steps; refused as :attr:`times` is."""
        return uniform_step_count(self.start_time, self.end_time, self.step_size)

    @property
    def dt(self) -> float:
        """Alias for step_size for convenience."""
        return self.step_size


class AdaptiveConfig(SolverConfig):
    """
    Base configuration for adaptive step size solvers.

    Provides parameters for error-controlled adaptive integration methods
    like embedded Runge-Kutta pairs.
    """

    start_time: float = Field(
        default=0.0,
        description="Starting time for the integration interval",
    )
    end_time: float = Field(
        default=1.0,
        description="Ending time for the integration interval",
    )
    initial_step_size: float = Field(
        default=0.01,
        description="Initial step size for adaptive integration",
    )
    absolute_tolerance: float = Field(
        default=1e-6,
        description="Absolute error tolerance",
    )
    relative_tolerance: float = Field(
        default=1e-6,
        description="Relative error tolerance",
    )
    min_step_size: float = Field(default=1e-8, description="Minimum allowed step size")
    max_step_size: float = Field(default=1.0, description="Maximum allowed step size")
    safety_factor: float = Field(
        default=0.9,
        description="Safety factor for step size adjustment",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StochasticConfig(FixedStepConfig):
    """
    Base configuration for stochastic differential equation solvers.

    Extends FixedStepConfig with parameters needed for stochastic simulations,
    including random seed for reproducibility and calculus interpretation.
    """

    random_seed: int | None = Field(
        default=None,
        description="Random seed for reproducible stochastic simulations",
    )
    calculus: Literal["ito", "stratonovich"] = Field(
        default="ito",
        description="Calculus interpretation for SDEs",
    )


class DeterministicConfig(FixedStepConfig):
    """
    Base configuration for deterministic differential equation solvers.

    Currently identical to FixedStepConfig but provided for future extension
    and clear semantic distinction between deterministic and stochastic methods.
    """
