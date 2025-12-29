from discrecontinual_equations.solver.solver_config import StochasticConfig


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
    - epsilon: Homogenisation parameter controlling convergence (small positive value)

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

    @property
    def gamma(self) -> float:
        """γ = 1/α parameter for the Thaler map."""
        return 1.0 / self.alpha
