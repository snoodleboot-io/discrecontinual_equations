from discrecontinual_equations.solver.solver_config import DeterministicConfig


class AdamsBashforth2Config(DeterministicConfig):
    """
    Configuration for the Adams-Bashforth 2 solver.

    Inherits common parameters from DeterministicConfig:
    - start_time, end_time, step_size: Time integration parameters
    - times, n_steps, dt: Computed properties

    The Adams-Bashforth 2 method uses a 2-step linear multistep approach.
    """
