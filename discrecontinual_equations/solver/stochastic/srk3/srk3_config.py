"""Configuration of the srk3 solver (Platen's strong order 1.5 scheme)."""

from discrecontinual_equations.solver.solver_config import StochasticConfig


class SRK3Config(StochasticConfig):
    """
    Configuration for the SRK3 solver.

    Inherits common parameters from StochasticConfig:
    - start_time, end_time, step_size: Time integration parameters
    - random_seed: For reproducible stochastic simulations
    - calculus: Ito or Stratonovich interpretation

    SRK3 implements Platen's explicit strong order 1.5 scheme, Kloeden and
    Platen (11.2.1), and its several-channel form; the "3" is not an order.

    References:
        Kloeden, P. E. and Platen, E. Numerical Solution of Stochastic
        Differential Equations. Springer, 1992, eq. (11.2.1).
        DOI 10.1007/978-3-662-12616-5.
    """
