from discrecontinual_equations.solver.solver_config import StochasticConfig


class MilsteinConfig(StochasticConfig):
    """
    Configuration for the Milstein stochastic differential equation solver.

    Inherits common parameters from StochasticConfig:
    - start_time, end_time, step_size: Time integration parameters
    - random_seed: For reproducible stochastic simulations
    - calculus: Ito or Stratonovich interpretation

    The Milstein method includes diffusion correction terms for higher accuracy.
    """
