from discrecontinual_equations.solver.solver_config import StochasticConfig


class ZhanDuanLiLiConfig(StochasticConfig):
    """
    Configuration for the Zhan-Duan-Li-Li stochastic differential equation solver.

    This solver implements the method proposed by Zhan, Duan, Li, and Li for solving
    stochastic differential equations. The specific algorithm details should be
    implemented based on the paper: https://arxiv.org/pdf/2010.07691

    Inherits common parameters from StochasticConfig:
    - start_time, end_time, step_size: Time integration parameters
    - random_seed: For reproducible stochastic simulations
    - calculus: Set to "ito" (method-specific interpretation)

    Additional parameters:
    - method_parameter: Theta parameter controlling implicit/explicit balance (0=explicit, 1=implicit, 0.5=trapezoidal)
    """

    method_parameter: float = 0.5  # Default to trapezoidal rule

    def __init__(self, **data):
        super().__init__(**data)
        # Set calculus based on method requirements
        self.calculus = "ito"
