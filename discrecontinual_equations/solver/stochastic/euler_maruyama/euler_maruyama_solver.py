import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (
    EulerMaruyamaConfig,
)
from discrecontinual_equations.variable import Variable


class EulerMaruyamaSolver(Solver):
    """
    Euler-Maruyama method for solving stochastic differential equations (SDEs).

    The stochastic analogue of the Euler method for ODEs. Supports both Ito and
    Stratonovich interpretations of SDEs.

    For Ito SDEs of the form: dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t
    The discretization is: X_{n+1} = X_n + μ(X_n, t_n) Δt + σ(X_n, t_n) ΔW_n

    For Stratonovich SDEs of the form: dX_t = μ(X_t, t) dt + σ(X_t, t) ∘ dW_t
    The equivalent Ito form is used with drift correction: μ_corrected = μ - (1/2)σ ∂σ/∂x

    Where ΔW_n ~ N(0, Δt) is a Wiener increment. This method achieves strong order 0.5
    and weak order 1.0 convergence.

    References:
    - Maruyama, Gisiro. "Continuous Markov processes and stochastic equations"
      Rendiconti del Circolo Matematico di Palermo, 1955
    - Kloeden, Peter E. and Platen, Eckhard. "Numerical Solution of Stochastic
      Differential Equations" Springer-Verlag, 1992
    """

    def __init__(self, solver_config: EulerMaruyamaConfig):
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

            # Apply Stratonovich correction if requested
            if self.solver_config.calculus == "stratonovich":
                # For Stratonovich: effective drift = μ - (1/2) σ ∇σ
                # For simplicity, implement numerical differentiation for 1D case
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

                    # Stratonovich correction: subtract (1/2) σ dσ/dx
                    correction = 0.5 * diffusion * dsigma_dx
                    drift = drift - correction
                else:
                    # For multi-dimensional, Stratonovich is more complex
                    # For now, issue a warning and use Ito
                    import warnings

                    warnings.warn(
                        "Stratonovich calculus for multi-dimensional SDEs not implemented. Using Ito interpretation.",
                    )

            # Generate Wiener increment: ΔW ~ N(0, dt)
            dt = self.solver_config.dt
            dW = np.random.normal(0, np.sqrt(dt), size=len(y))

            # Euler-Maruyama step: y_{n+1} = y_n + μ(y_n, t_n) dt + σ(y_n, t_n) dW
            y_new = y + drift * dt + diffusion * dW

            # Update time and state
            t_next = self.solver_config.times[i + 1]
            y = y_new

            # Append to solution
            self.solution.append([t_next, [0] * len(initial_values), y.tolist()])
