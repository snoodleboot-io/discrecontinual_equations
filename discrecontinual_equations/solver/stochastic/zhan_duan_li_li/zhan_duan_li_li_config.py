"""Configuration of the Zhan-Duan-Li-Li theta step: the weight ``theta``."""

from discrecontinual_equations.solver.solver_config import StochasticConfig


class ZhanDuanLiLiConfig(StochasticConfig):
    """
    Configuration for the Zhan-Duan-Li-Li stochastic differential equation solver.

    The solver is a theta-weighted Euler predictor-corrector step for Gaussian
    noise; it is not the symplectic Euler scheme for Marcus SDEs with Levy noise
    of the paper it is named after (arXiv:2010.07691), and for multiplicative
    noise with ``theta > 0`` it does not converge to the Ito solution. See
    :class:`~.zhan_duan_li_li_solver.ZhanDuanLiLiSolver`.

    Inherits common parameters from StochasticConfig:
    - start_time, end_time, step_size: Time integration parameters
    - random_seed: For reproducible stochastic simulations
    - calculus: Set to "ito" (method-specific interpretation)

    Additional parameters:
    - method_parameter: the weight theta of the corrector stage. 0 is
      Euler-Maruyama; 0.5 averages the two stages (the stochastic Heun step);
      1 uses the predicted stage alone. Every value is explicit.

    References:
        Zhan, Q., Duan, J., Li, X. and Li, Y. Symplectic method for Hamiltonian
        stochastic differential equations with multiplicative Levy noise in the
        sense of Marcus. arXiv:2010.07691 (2020).
    """

    method_parameter: float = 0.5  # Default to trapezoidal rule

    def __init__(self, **data):
        super().__init__(**data)
        # Set calculus based on method requirements
        self.calculus = "ito"
