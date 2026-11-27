"""Configuration of the srk4 solver (Platen's weak order 2.0 scheme)."""

from discrecontinual_equations.solver.solver_config import StochasticConfig


class SRK4Config(StochasticConfig):
    """
    Configuration for the SRK4 solver.

    Inherits common parameters from StochasticConfig:
    - start_time, end_time, step_size: Time integration parameters
    - random_seed: For reproducible stochastic simulations
    - calculus: Ito or Stratonovich interpretation

    SRK4 implements Platen's explicit weak order 2.0 scheme for scalar noise,
    Kloeden and Platen (15.1.4); its strong order is 1.0, and the "4" is not
    an order.

    References:
        Kloeden, P. E. and Platen, E. Numerical Solution of Stochastic
        Differential Equations. Springer, 1992, eq. (15.1.4).
        DOI 10.1007/978-3-662-12616-5.
    """
