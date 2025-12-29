from discrecontinual_equations.solver.solver_config import StochasticConfig


class SRK3Config(StochasticConfig):
    """
    Configuration for the SRK3 (Stochastic Runge-Kutta order 3) solver.

    Inherits common parameters from StochasticConfig:
    - start_time, end_time, step_size: Time integration parameters
    - random_seed: For reproducible stochastic simulations
    - calculus: Ito or Stratonovich interpretation

    SRK3 implements a 3-stage stochastic Runge-Kutta method for higher accuracy.
    """
