import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.stochastic.srk2.srk2_config import SRK2Config
from discrecontinual_equations.variable import Variable


class SRK2Solver(Solver):
    """
    Stochastic Runge-Kutta method of order 2 (SRK2) for solving stochastic differential equations (SDEs).

    This implements Platen's SRK2 scheme with support for both Ito and Stratonovich calculus.
    A higher-order method that improves upon Euler-Maruyama by using a 2-stage Runge-Kutta approach.

    For Ito SDEs of the form: dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t
    The discretization uses: K₁, L₁ = μ(X_n, t_n), σ(X_n, t_n)
    K₂, L₂ = μ(X_n + K₁ Δt + L₁ √Δt, t_n + Δt), σ(X_n + K₁ Δt + L₁ √Δt, t_n + Δt)
    X_{n+1} = X_n + (K₁ + K₂) Δt/2 + (L₁ + L₂) √Δt/2

    For Stratonovich SDEs: Applies drift correction μ_corrected = μ - (1/2)σ ∂σ/∂x

    This method achieves strong order 1.0 and weak order 1.0 convergence.

    References:
    - Platen, Eckhard. "An introduction to numerical methods for stochastic differential equations"
      Acta Numerica, 2004
    - Rößler, Andreas. "Runge–Kutta methods for the numerical solution of stochastic differential equations"
      Journal of Computational and Applied Mathematics, 2006
    """

    def __init__(self, solver_config: SRK2Config):
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

            dt = self.solver_config.dt

            # SRK2 step
            y_new = self._srk2_step(y, t_current, dt, equation)

            # Update time and state
            t_next = self.solver_config.times[i + 1]
            y = y_new

            # Append to solution
            self.solution.append([t_next, [0] * len(initial_values), y.tolist()])

    def _srk2_step(
        self,
        y: np.ndarray,
        t: float,
        dt: float,
        equation: DifferentialEquation,
    ) -> np.ndarray:
        """Perform one SRK2 step."""
        # Stage 1
        K1 = np.array(equation.derivative.eval(point=y.tolist(), time=t))
        L1 = np.array(equation.derivative.diffusion(point=y.tolist(), time=t))

        # Apply Stratonovich correction to drift if requested
        if self.solver_config.calculus == "stratonovich" and len(y) == 1:
            # For 1D Stratonovich: modify drift by -(1/2)σ dσ/dx
            eps = 1e-8
            y_plus = y + eps
            y_minus = y - eps
            diffusion_plus = np.array(
                equation.derivative.diffusion(point=y_plus.tolist(), time=t),
            )
            diffusion_minus = np.array(
                equation.derivative.diffusion(point=y_minus.tolist(), time=t),
            )
            dsigma_dx = (diffusion_plus - diffusion_minus) / (2 * eps)
            stratonovich_correction = 0.5 * L1 * dsigma_dx
            K1 = K1 - stratonovich_correction

        # Stage 2
        y_temp = y + K1 * dt + L1 * np.sqrt(dt)
        K2 = np.array(equation.derivative.eval(point=y_temp.tolist(), time=t + dt))
        L2 = np.array(equation.derivative.diffusion(point=y_temp.tolist(), time=t + dt))

        # Apply Stratonovich correction to K2 if requested
        if self.solver_config.calculus == "stratonovich" and len(y) == 1:
            eps = 1e-8
            y_temp_plus = y_temp + eps
            y_temp_minus = y_temp - eps
            diffusion_temp_plus = np.array(
                equation.derivative.diffusion(point=y_temp_plus.tolist(), time=t + dt),
            )
            diffusion_temp_minus = np.array(
                equation.derivative.diffusion(point=y_temp_minus.tolist(), time=t + dt),
            )
            dsigma_dx_temp = (diffusion_temp_plus - diffusion_temp_minus) / (2 * eps)
            stratonovich_correction_temp = 0.5 * L2 * dsigma_dx_temp
            K2 = K2 - stratonovich_correction_temp

        # Final step - SRK2 combination
        y_new = y + (K1 + K2) * dt / 2 + (L1 + L2) * np.sqrt(dt) / 2

        return y_new
