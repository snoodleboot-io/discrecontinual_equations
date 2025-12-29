import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.stochastic.milstein.milstein_config import (
    MilsteinConfig,
)
from discrecontinual_equations.variable import Variable


class MilsteinSolver(Solver):
    """
    Milstein method for solving stochastic differential equations (SDEs).

    The Milstein method is a higher-order method that improves upon Euler-Maruyama
    by including second-order terms from the Ito-Taylor expansion.

    Supports both Ito and Stratonovich interpretations of SDEs.

    For Ito SDEs of the form: dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t
    The discretization is: X_{n+1} = X_n + μ Δt + σ ΔW + (1/2)σ ∂σ/∂x (ΔW² - Δt)

    For Stratonovich SDEs of the form: dX_t = μ(X_t, t) dt + σ(X_t, t) ∘ dW_t
    The Ito equivalent is solved with modified drift: μ_corrected = μ - (1/2)σ ∂σ/∂x

    This method achieves strong order 1.0 and weak order 1.0 convergence.

    References:
    - Milstein, Grigori N. "Numerical Integration of Stochastic Differential Equations"
      Kluwer Academic Publishers, 1995
    """

    def __init__(self, solver_config: MilsteinConfig):
        super().__init__(solver_config=solver_config)

        # Set random seed for reproducibility
        if self.solver_config.random_seed is not None:
            np.random.seed(self.solver_config.random_seed)

    def solve(self, equation: DifferentialEquation, initial_values: list[float]):
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
        y = np.array(initial_values, dtype=float)

        # Append initial point
        self.solution.append([t, [0] * len(initial_values), y.tolist()])

        # Time stepping loop
        for i in range(self.solver_config.n_steps):
            # Current time
            t_current = self.solver_config.times[i]

            # Evaluate drift and diffusion
            drift = np.array(equation.derivative.eval(point=y.tolist(), time=t_current))
            diffusion = np.array(
                equation.derivative.diffusion(point=y.tolist(), time=t_current),
            )

            # Generate Wiener increment: ΔW ~ N(0, dt)
            dt = self.solver_config.dt
            dW = np.random.normal(0, np.sqrt(dt), size=len(y))

            # Milstein correction term
            if len(y) == 1:
                # Numerical differentiation of diffusion w.r.t. x
                eps = 1e-8
                y_plus = y + eps
                y_minus = y - eps
                diffusion_plus = np.array(
                    equation.derivative.diffusion(
                        point=y_plus.tolist(),
                        time=t_current,
                    ),
                )
                diffusion_minus = np.array(
                    equation.derivative.diffusion(
                        point=y_minus.tolist(),
                        time=t_current,
                    ),
                )
                dsigma_dx = (diffusion_plus - diffusion_minus) / (2 * eps)

                if self.solver_config.calculus == "ito":
                    # Ito Milstein: (1/2) σ ∂σ/∂x (ΔW² - Δt)
                    correction = 0.5 * diffusion * dsigma_dx * (dW**2 - dt)
                else:  # stratonovich
                    # Stratonovich Milstein: (1/2) σ ∂σ/∂x ΔW²
                    correction = 0.5 * diffusion * dsigma_dx * dW**2

                milstein_term = correction
            else:
                # For multi-dimensional systems, full Milstein method requires
                # computing the Jacobian of the diffusion term, which is complex.
                # For now, fall back to Euler-Maruyama (without correction)
                milstein_term = np.zeros_like(drift)
                if i == 0:  # Only warn once
                    print(
                        "Warning: Milstein method for multi-dimensional SDEs not fully implemented. Using Euler-Maruyama.",
                    )

            # Milstein step: y_{n+1} = y_n + μ(y_n, t_n) dt + σ(y_n, t_n) dW + correction
            y_new = y + drift * dt + diffusion * dW + milstein_term

            # Update time and state
            t_next = self.solver_config.times[i + 1]
            y = y_new

            # Append to solution
            self.solution.append([t_next, [0] * len(initial_values), y.tolist()])
