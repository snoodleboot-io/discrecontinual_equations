import numpy as np
from pydantic import ConfigDict, Field

from discrecontinual_equations.solver.solver_config import SolverConfig


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
        if self.number_of_steps is not None:
            return np.linspace(self.start_time, self.end_time, self.number_of_steps)
        return np.arange(self.start_time, self.end_time, self.step_size)
