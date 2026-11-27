"""Configuration of the forward Euler solver: a fixed grid and its step.

The grid and its checks come from ``DeterministicConfig``; this class only names
the configuration ``EulerSolver`` takes.

References:
    See ``EulerSolver`` and ``docs/references.md`` (Euler1768).
"""

from discrecontinual_equations.solver.solver_config import DeterministicConfig


class EulerConfig(DeterministicConfig):
    """
    Configuration for the Euler solver.

    Inherits common parameters from DeterministicConfig:
    - start_time, end_time, step_size: Time integration parameters
    - times, n_steps, dt: Computed properties

    The Euler method is the fundamental first-order explicit solver.

    References:
        See ``EulerSolver`` and ``docs/references.md`` (Euler1768).
    """
