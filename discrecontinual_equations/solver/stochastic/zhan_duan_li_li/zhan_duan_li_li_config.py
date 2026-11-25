"""Configuration of the Zhan-Duan-Li-Li theta step: the weight ``theta``."""

from discrecontinual_equations.solver.solver_config import StochasticConfig


class ZhanDuanLiLiConfig(StochasticConfig):
    """
    Configuration for the Zhan-Duan-Li-Li (theta-weighted stochastic Heun) solver.

    Inherits common parameters from StochasticConfig:
    - start_time, end_time, step_size: Time integration parameters
    - random_seed: For reproducible stochastic simulations
    - calculus: "ito" (the default) or "stratonovich", the sense in which the
      equation is written. The solver converges to that calculus's solution for
      every ``method_parameter`` (DEQ-133; before it the calculus was forced to
      "ito" and the step converged to the Stratonovich solution regardless).

    Additional parameters:
    - method_parameter: theta, the weight of the corrector stage. 0 is
      Euler-Maruyama, 0.5 (the default) the stochastic Heun step, strong order 1.0
      for scalar or commuting noise; any other value is strong order 0.5. The
      scheme stays explicit for every theta: theta weights a predicted point, it
      does not make the step implicit.

    References:
        Zhan, Q., Duan, J., Li, X. and Li, Y. Symplectic method for Hamiltonian
        stochastic differential equations with multiplicative Levy noise in the
        sense of Marcus. arXiv:2010.07691 (2020).
    """

    method_parameter: float = 0.5
