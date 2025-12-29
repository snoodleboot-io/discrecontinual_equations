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
        """Array of time points for integration."""
        return np.arange(
            self.start_time,
            self.end_time + self.step_size,
            self.step_size,
        )

    @property
    def n_steps(self) -> int:
        """Number of integration steps."""
        return len(self.times) - 1

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
