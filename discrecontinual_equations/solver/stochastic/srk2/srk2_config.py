"""Configuration of the srk2 solver (Platen's strong order 1.0 scheme)."""

from discrecontinual_equations.solver.solver_config import StochasticConfig


class SRK2Config(StochasticConfig):
    """
    Configuration for the SRK2 solver.

    Inherits common parameters from StochasticConfig:
    - start_time, end_time, step_size: Time integration parameters
    - random_seed: For reproducible stochastic simulations
    - calculus: Ito or Stratonovich interpretation

    SRK2 implements Platen's two-stage explicit scheme, Kloeden and Platen
    (11.1.7). The "2" counts stages: the scheme has strong order 1.0, not 2.

    References:
        Kloeden, P. E. and Platen, E. Numerical Solution of Stochastic
        Differential Equations. Springer, 1992, eq. (11.1.7).
        DOI 10.1007/978-3-662-12616-5.
    """
