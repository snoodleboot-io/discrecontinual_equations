import numpy as np
from pydantic import ConfigDict, Field

from discrecontinual_equations.solver.solver_config import SolverConfig, uniform_grid


class AdamsBashforth3Config(SolverConfig):
    start_time: float = Field(
        default=0.0,
        description="Starting time for the integration interval",
    )
    end_time: float = Field(
        default=1.0,
        description="Ending time for the integration interval",
    )
    number_of_steps: int | None = Field(
        default=None,
        description="Number of integration steps (alternative to step_size)",
    )
    step_size: float | None = Field(
        default=None,
        description="Size of each integration step (alternative to number_of_steps)",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def times(self) -> np.ndarray:
        """The time grid, from ``start_time`` to ``end_time`` inclusive.

        With ``step_size`` it is the shared fixed-step grid of
        :func:`uniform_grid`, which refuses a span that is not a whole number of
        steps; it was ``np.arange(start, end, h)``, which stopped a step short of
        ``end_time`` (DEQ-129).
        """
        if self.number_of_steps is not None:
            return np.linspace(self.start_time, self.end_time, self.number_of_steps)
        return uniform_grid(self.start_time, self.end_time, self.step_size)
