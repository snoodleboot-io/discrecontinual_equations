import numpy as np
from pydantic import ConfigDict, Field

from discrecontinual_equations.solver.solver_config import SolverConfig


class CashKarpConfig(SolverConfig):
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
        description="PI controller parameter for step size adaptation",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def times(self) -> np.ndarray:
        # For adaptive methods, we don't pre-generate times
        # Instead, we'll generate them adaptively during solving
        return np.array([self.start_time])
