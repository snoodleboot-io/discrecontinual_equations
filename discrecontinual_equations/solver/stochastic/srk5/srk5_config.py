"""Configuration of the srk5 solver, which refuses to run."""

from discrecontinual_equations.solver.solver_config import StochasticConfig


class SRK5Config(StochasticConfig):
    """
    Configuration for the SRK5 solver.

    Inherits common parameters from StochasticConfig:
    - start_time, end_time, step_size: Time integration parameters
    - random_seed: For reproducible stochastic simulations
    - calculus: Ito or Stratonovich interpretation

    SRK5 implements nothing: :class:`~.srk5_solver.SRK5Solver` refuses to run,
    because no scheme of the order its name once claimed exists for this
    interface. The configuration is kept so that existing code still imports.

    References:
        Burrage, K. and Burrage, P. M. High strong order explicit Runge-Kutta
        methods for stochastic ordinary differential equations. Applied
        Numerical Mathematics 22(1-3), 81-101, 1996.
    """
