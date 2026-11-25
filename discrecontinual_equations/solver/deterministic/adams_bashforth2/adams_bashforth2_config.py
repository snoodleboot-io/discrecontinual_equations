"""Configuration of the two-step Adams-Bashforth solver: a fixed grid and its step.

The grid and its checks come from ``DeterministicConfig``; this class only names
the configuration ``AdamsBashforth2Solver`` takes.

References:
    See ``AdamsBashforth2Solver`` and ``docs/references.md`` (Bashforth1883).
"""

from discrecontinual_equations.solver.solver_config import DeterministicConfig


class AdamsBashforth2Config(DeterministicConfig):
    """
    Configuration for the Adams-Bashforth 2 solver.

    Inherits common parameters from DeterministicConfig:
    - start_time, end_time, step_size: Time integration parameters
    - times, n_steps, dt: Computed properties

    The Adams-Bashforth 2 method uses a 2-step linear multistep approach.

    References:
        See ``AdamsBashforth2Solver`` and ``docs/references.md`` (Bashforth1883).
    """
