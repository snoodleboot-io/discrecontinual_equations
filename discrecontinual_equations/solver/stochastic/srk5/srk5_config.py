from discrecontinual_equations.solver.solver_config import StochasticConfig


class SRK5Config(StochasticConfig):
    """
    Configuration for the SRK5 (Stochastic Runge-Kutta order 5) solver.

    Inherits common parameters from StochasticConfig:
    - start_time, end_time, step_size: Time integration parameters
    - random_seed: For reproducible stochastic simulations
    - calculus: Ito or Stratonovich interpretation

    SRK5 implements a 5-stage stochastic Runge-Kutta method for maximum accuracy.
    """
