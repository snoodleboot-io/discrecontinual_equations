"""Configuration of the Milstein solver, with the Levy-area accuracy setting."""

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

    ``levy_area_accuracy`` is the constant ``c1`` of the Levy-area truncation:
    the sampler takes the least number ``n`` of Fourier terms whose L2 error per
    step is at most ``c1 dt^(3/2)`` (Mrongowius and Rossler (36); Kastner and
    Rossler (64), (67)). It matters only for non-commuting noise without a
    Wiener source of the caller's.

    References:
        Mrongowius, J. and Rossler, A. On the approximation and simulation of
        iterated stochastic integrals and the corresponding Levy areas in terms
        of a multidimensional Brownian motion. arXiv:2101.09542 (2021), eq. (36).

        Kastner, F. and Rossler, A. An analysis of approximation algorithms for
        iterated stochastic integrals and a Julia and MATLAB simulation toolbox.
        Numerical Algorithms, 2022; arXiv:2201.08424. Eqs. (64)-(67).
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
