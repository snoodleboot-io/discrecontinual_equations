from pydantic import Field

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

    levy_area_accuracy: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "c1 in the Levy-area truncation rule, L2 error <= c1 dt^(3/2) per step: "
            "smaller draws more Fourier terms (n grows like 1/c1) for a cleaner "
            "order 1 at moderate dt; 0.05 measures slope 0.97 over dt = 2^-3 .. "
            "2^-10 where 1 measures 0.75. Used only for non-commuting noise "
            "without a Wiener source (DEQ-67)."
        ),
    )
