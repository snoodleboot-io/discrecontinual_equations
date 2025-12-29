from discrecontinual_equations.solver.solver_config import StochasticConfig


class SRK2Config(StochasticConfig):
    """
    Configuration for the SRK2 (Stochastic Runge-Kutta order 2) solver.

    Inherits common parameters from StochasticConfig:
    - start_time, end_time, step_size: Time integration parameters
    - random_seed: For reproducible stochastic simulations
    - calculus: Ito or Stratonovich interpretation

    SRK2 implements Platen's 2-stage stochastic Runge-Kutta method.
    """
