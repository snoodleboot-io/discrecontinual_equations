"""Configuration of the Dormand-Prince 5(4) solver: interval, tolerances and step control.

An embedded pair needs a first step, a tolerance and bounds on the step. A step
is accepted when the root-mean-square of the error estimates, each scaled by
``absolute_tolerance + relative_tolerance max(|y|, |y_new|)``, is at most 1;
the next step is ``h S E^(-alpha) E_old^beta`` with ``alpha = 1/q - 0.75 beta``
(a PI controller), its change held to ``[0.2, 10]`` and the step to
``[min_step_size, max_step_size]``. The defaults ``safety_factor = 0.9`` and
``beta = 0.04`` are those of Hairer and Wanner's ``DOPRI5``, whose controller
this is (see ``embedded_runge_kutta``).

References:
    Hairer, E. and Wanner, G. (1996). DOPRI5, Fortran code; ``SAFE`` and
    ``BETA`` defaults. See ``docs/references.md`` (HairerWanner1996, DormandPrince1980).
"""

import numpy as np
from pydantic import ConfigDict, Field

from discrecontinual_equations.solver.solver_config import SolverConfig


class DormandPrinceConfig(SolverConfig):
    """Interval, first step, tolerances and step-size control for ``DormandPrinceSolver``.

    Attributes:
        start_time: Start of the interval.
        end_time: End of the interval; the last step is shortened to land on it.
        initial_step_size: The first step tried.
        absolute_tolerance: Absolute part of the per-component tolerance.
        relative_tolerance: Relative part, times ``|y|``.
        min_step_size: Smallest step proposed. A step rejected at this size
            raises ``StepSizeTooSmallError``.
        max_step_size: Largest step proposed.
        safety_factor: Factor ``S < 1`` on the proposed step.
        beta: The proportional gain of the PI controller: the step scales as
            ``E^(-(1/q - 0.75 beta)) E_old^beta``; 0 gives an integral controller.

    References:
        Hairer and Wanner (1996), ``DOPRI5`` defaults. See
        ``docs/references.md`` (HairerWanner1996).
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
        description="Absolute error tolerance for adaptive stepping",
    )
    relative_tolerance: float = Field(
        default=1e-6,
        description="Relative error tolerance for adaptive stepping",
    )
    min_step_size: float = Field(
        default=1e-8,
        description="Minimum allowed step size to prevent underflow",
    )
    max_step_size: float = Field(
        default=1.0,
        description="Maximum allowed step size to prevent overflow",
    )
    safety_factor: float = Field(
        default=0.9,
        description="Safety factor for step size control (typically 0.8-0.9)",
    )
    beta: float = Field(
        default=0.04,
        description="Proportional gain of the PI step-size controller (DOPRI5: 0.04)",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def times(self) -> np.ndarray:
        """The start time only: an adaptive solver chooses its nodes as it goes."""
        # For adaptive methods, we don't pre-generate times
        # Instead, we'll generate them adaptively during solving
        return np.array([self.start_time])
