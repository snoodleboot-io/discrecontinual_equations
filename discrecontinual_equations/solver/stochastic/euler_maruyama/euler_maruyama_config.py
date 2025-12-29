from discrecontinual_equations.solver.solver_config import StochasticConfig


class EulerMaruyamaConfig(StochasticConfig):
    """
    Configuration for the Euler-Maruyama stochastic differential equation solver.

    Inherits common parameters from StochasticConfig:
    - start_time, end_time, step_size: Time integration parameters
    - random_seed: For reproducible stochastic simulations
    - calculus: Ito or Stratonovich interpretation

    Provides computed properties:
    - times: Array of integration time points
    - n_steps: Number of integration steps
    - dt: Alias for step_size
    """
