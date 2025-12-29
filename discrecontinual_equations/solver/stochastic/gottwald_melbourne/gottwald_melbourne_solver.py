import math
import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_config import (
    GottwaldMelbourneConfig,
)
from discrecontinual_equations.variable import Variable


class GottwaldMelbourneSolver(Solver):
    """
    Gottwald-Melbourne method for solving stochastic differential equations driven by α-stable Lévy processes.

    This solver implements the deterministic homogenisation approach based on the Thaler map,
    which approximates Marcus SDEs with non-Lipschitz coefficients driven by α-stable noise
    for α ∈ (1,2).

    The method uses a fast-slow system where the slow dynamics approximate the SDE solution,
    and the fast dynamics on the Thaler map generate the required α-stable increments.

    References:
    - Gottwald, G. A. and Melbourne, I. "Simulation of non-Lipschitz α-stable stochastic
      differential equations: a method based on deterministic homogenisation"
      arXiv preprint, 2020
    """

    def __init__(self, solver_config: GottwaldMelbourneConfig):
        super().__init__(solver_config=solver_config)

        # Validate parameters
        if not (1 < self.solver_config.alpha < 2):
            raise ValueError("α must be in the range (1, 2)")
        if not (-1 <= self.solver_config.beta <= 1):
            raise ValueError("β must be in the range [-1, 1]")
        if self.solver_config.eta <= 0:
            raise ValueError("η must be positive")
        if self.solver_config.epsilon <= 0:
            raise ValueError("ε must be positive")

        # Set random seed for reproducibility
        if self.solver_config.random_seed is not None:
            np.random.seed(self.solver_config.random_seed)

        # Compute Thaler map parameters
        self._gamma = self.solver_config.gamma
        self._x_star = self._compute_x_star()
        self._d_alpha = self._compute_d_alpha()

    def _compute_x_star(self) -> float:
        """Compute the fixed point x* of the Thaler map."""
        gamma = self._gamma

        # Solve x*^{1-γ} + (1 + x*)^{1-γ} = 2
        # Use numerical solution
        def equation(x):
            return x ** (1 - gamma) + (1 + x) ** (1 - gamma) - 2

        # Binary search for solution
        low, high = 0.0, 1.0
        for _ in range(100):
            mid = (low + high) / 2
            if equation(mid) < 0:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    def _compute_d_alpha(self) -> float:
        """Compute the constant d_α from the paper."""
        alpha = self.solver_config.alpha
        gamma = self._gamma
        g_alpha = math.gamma(1 - alpha) * math.cos(math.pi * alpha / 2)
        return alpha**alpha * (1 - gamma) ** 2 / ((1 - gamma) * g_alpha)

    def _thaler_map(self, x: float) -> float:
        """Compute the Thaler map T(x)."""
        gamma = self._gamma
        if x <= 0:
            return 0.0
        elif x >= 1:
            return 1.0
        else:
            term1 = x ** (1 - gamma)
            term2 = (1 + x) ** (1 - gamma)
            return (term1 + term2 - 1) ** (1 / (1 - gamma)) % 1

    def _compute_observable(self, x: float, chi: int) -> float:
        """Compute the observable v(x) for the fast dynamics."""
        eta = self.solver_config.eta
        gamma = self._gamma
        d_alpha = self._d_alpha

        if x <= self._x_star:
            v_tilde = 1.0
        else:
            v_tilde = (1 - 2 ** (gamma - 1)) ** (-1) * chi

        v = eta * (d_alpha ** (-gamma)) * v_tilde
        return v

    def solve(self, equation: DifferentialEquation, initial_values: list[float]):
        """
        Solve the SDE using the Thaler homogenisation method.

        Args:
            equation: The SDE to solve (must have stochastic derivative function)
            initial_values: Initial values for all variables
        """
        results = [
            Variable(name=f"Integral of {variable.name}")
            for variable in equation.derivative.variables
        ]
        self.solution = Curve(
            time=equation.derivative.time,
            variables=equation.derivative.variables,
            results=results,
        )

        # Initialize
        t = self.solver_config.start_time
        z = np.array(initial_values, dtype=float)

        # Initialize fast dynamics
        x = 0.5  # Start in the middle of [0,1]
        chi = 1 if np.random.random() < (1 + self.solver_config.beta) / 2 else -1

        # Append initial point
        self.solution.append([t, [0] * len(initial_values), z.tolist()])

        # Time stepping loop
        dt = self.solver_config.dt
        epsilon = self.solver_config.epsilon
        n_steps_per_time_step = max(1, int(dt / epsilon))

        for i in range(self.solver_config.n_steps):
            # Current time
            t_current = self.solver_config.times[i]

            # Evolve fast dynamics for n_steps_per_time_step steps
            for _ in range(n_steps_per_time_step):
                # Compute observable
                v = self._compute_observable(x, chi)

                # Evaluate drift and diffusion at current z
                drift = np.array(
                    equation.derivative.eval(point=z.tolist(), time=t_current)
                )
                diffusion = np.array(
                    equation.derivative.diffusion(point=z.tolist(), time=t_current)
                )

                # Update slow variable (z)
                z = z + epsilon * drift + epsilon**self._gamma * diffusion * v

                # Update fast variable (x)
                x = self._thaler_map(x)

                # Update chi based on position
                if x > self._x_star:
                    # In hyperbolic region, sample new chi
                    chi = (
                        1
                        if np.random.random() < (1 + self.solver_config.beta) / 2
                        else -1
                    )

            # Update time
            t_next = self.solver_config.times[i + 1]

            # Append to solution
            self.solution.append([t_next, [0] * len(initial_values), z.tolist()])
