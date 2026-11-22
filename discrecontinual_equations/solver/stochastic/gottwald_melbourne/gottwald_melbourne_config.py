from discrecontinual_equations.solver.solver_config import StochasticConfig


def epsilon_exceeds_step(epsilon: float, step_size: float) -> str:
    """The error for an ε longer than the grid step (DEQ-86)."""
    return (
        f"ε must not exceed step_size (epsilon={epsilon}, step_size={step_size}): "
        "each fast step of the map advances slow time by ε, so the path at time t "
        "is z_⌊t/ε⌋ (Gottwald & Melbourne 2021, Theorem 4.1) and cannot resolve a "
        "step shorter than ε. Take ε ≤ step_size; the method converges as ε → 0."
    )


class GottwaldMelbourneConfig(StochasticConfig):
    """
    Configuration for the Gottwald-Melbourne stochastic differential equation solver.

    This solver implements the deterministic homogenisation method for solving
    SDEs driven by α-stable Lévy processes with α ∈ (1,2), based on the Thaler map.

    Inherits common parameters from StochasticConfig:
    - start_time, end_time, step_size: Time integration parameters
    - random_seed: For reproducible stochastic simulations
    - calculus: Set to "ito" (Marcus interpretation is handled internally)

    Additional parameters:
    - alpha: Stability parameter of the α-stable Lévy process (1 < α < 2)
    - eta: Scale parameter of the α-stable Lévy process
    - beta: Skewness parameter of the α-stable Lévy process (-1 ≤ β ≤ 1)
    - epsilon: Homogenisation parameter controlling convergence (small positive
      value, at most step_size: each fast step advances slow time by ε)

    Provides computed properties:
    - times: Array of integration time points
    - n_steps: Number of integration steps
    - dt: Alias for step_size
    - gamma: Computed as 1/α for the Thaler map
    """

    alpha: float = 1.5
    eta: float = 1.0
    beta: float = 0.0
    epsilon: float = 0.01

    def __init__(self, **data):
        super().__init__(**data)
        # Force calculus to "ito" since Thaler method uses Marcus interpretation
        self.calculus = "ito"

        # Validate parameters
        if not (1 < self.alpha < 2):
            raise ValueError("α must be in the range (1, 2)")
        if not (-1 <= self.beta <= 1):
            raise ValueError("β must be in the range [-1, 1]")
        if self.eta <= 0:
            raise ValueError("η must be positive")
        if self.epsilon <= 0:
            raise ValueError("ε must be positive")
        if self.epsilon > self.step_size:
            raise ValueError(epsilon_exceeds_step(self.epsilon, self.step_size))

    @property
    def gamma(self) -> float:
        """γ = 1/α parameter for the Thaler map."""
        return 1.0 / self.alpha
