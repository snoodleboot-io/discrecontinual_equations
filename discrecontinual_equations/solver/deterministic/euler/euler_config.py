from discrecontinual_equations.solver.solver_config import DeterministicConfig


class EulerConfig(DeterministicConfig):
    """
    Configuration for the Euler solver.

    Inherits common parameters from DeterministicConfig:
    - start_time, end_time, step_size: Time integration parameters
    - times, n_steps, dt: Computed properties

    The Euler method is the fundamental first-order explicit solver.
    """
