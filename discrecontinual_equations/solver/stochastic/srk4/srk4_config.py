from discrecontinual_equations.solver.solver_config import StochasticConfig


class SRK4Config(StochasticConfig):
    """
    Configuration for the SRK4 (Stochastic Runge-Kutta order 4) solver.

    Inherits common parameters from StochasticConfig:
    - start_time, end_time, step_size: Time integration parameters
    - random_seed: For reproducible stochastic simulations
    - calculus: Ito or Stratonovich interpretation

    SRK4 implements a 4-stage stochastic Runge-Kutta method for high accuracy.
    """
