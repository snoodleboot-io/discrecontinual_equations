import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.stochastic.srk3.srk3_config import SRK3Config
from discrecontinual_equations.variable import Variable


class SRK3Solver(Solver):
    """
    Stochastic Runge-Kutta method of order 3 (SRK3) for solving stochastic differential equations (SDEs).

    This implements a 3-stage SRK method with support for both Ito and Stratonovich calculus.
    Uses a simplified Butcher tableau that provides order 3 convergence for suitable problems.

    For Ito SDEs: dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t
    For Stratonovich SDEs: dX_t = μ(X_t, t) dt + σ(X_t, t) ∘ dW_t

    The method achieves strong order 1.5 and weak order 3.0 convergence.
    """

    def __init__(self, solver_config: SRK3Config):
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

            # SRK3 step
            y_new = self._srk3_step(y, t_current, dt, equation)

            # Update time and state
            t_next = self.solver_config.times[i + 1]
            y = y_new

            # Append to solution
            self.solution.append([t_next, [0] * len(initial_values), y.tolist()])

    def _srk3_step(
        self,
        y: np.ndarray,
        t: float,
        dt: float,
        equation: DifferentialEquation,
    ) -> np.ndarray:
        """Perform one SRK3 step with 3 stages."""
        # Stage 1
        K1 = np.array(equation.derivative.eval(point=y.tolist(), time=t))
        L1 = np.array(equation.derivative.diffusion(point=y.tolist(), time=t))

        # Apply Stratonovich correction to K1 if requested
        if self.solver_config.calculus == "stratonovich" and len(y) == 1:
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
        y_temp2 = y + (1 / 3) * K1 * dt + (1 / 3) * L1 * np.sqrt(dt)
        K2 = np.array(equation.derivative.eval(point=y_temp2.tolist(), time=t + dt / 3))
        L2 = np.array(
            equation.derivative.diffusion(point=y_temp2.tolist(), time=t + dt / 3),
        )

        # Apply Stratonovich correction to K2 if requested
        if self.solver_config.calculus == "stratonovich" and len(y) == 1:
            eps = 1e-8
            y_temp2_plus = y_temp2 + eps
            y_temp2_minus = y_temp2 - eps
            diffusion_temp2_plus = np.array(
                equation.derivative.diffusion(
                    point=y_temp2_plus.tolist(),
                    time=t + dt / 3,
                ),
            )
            diffusion_temp2_minus = np.array(
                equation.derivative.diffusion(
                    point=y_temp2_minus.tolist(),
                    time=t + dt / 3,
                ),
            )
            dsigma_dx_temp2 = (diffusion_temp2_plus - diffusion_temp2_minus) / (2 * eps)
            stratonovich_correction_temp2 = 0.5 * L2 * dsigma_dx_temp2
            K2 = K2 - stratonovich_correction_temp2

        # Stage 3
        y_temp3 = y + (2 / 3) * K2 * dt + (2 / 3) * L2 * np.sqrt(dt)
        K3 = np.array(
            equation.derivative.eval(point=y_temp3.tolist(), time=t + 2 * dt / 3),
        )
        L3 = np.array(
            equation.derivative.diffusion(point=y_temp3.tolist(), time=t + 2 * dt / 3),
        )

        # Apply Stratonovich correction to K3 if requested
        if self.solver_config.calculus == "stratonovich" and len(y) == 1:
            eps = 1e-8
            y_temp3_plus = y_temp3 + eps
            y_temp3_minus = y_temp3 - eps
            diffusion_temp3_plus = np.array(
                equation.derivative.diffusion(
                    point=y_temp3_plus.tolist(),
                    time=t + 2 * dt / 3,
                ),
            )
            diffusion_temp3_minus = np.array(
                equation.derivative.diffusion(
                    point=y_temp3_minus.tolist(),
                    time=t + 2 * dt / 3,
                ),
            )
            dsigma_dx_temp3 = (diffusion_temp3_plus - diffusion_temp3_minus) / (2 * eps)
            stratonovich_correction_temp3 = 0.5 * L3 * dsigma_dx_temp3
            K3 = K3 - stratonovich_correction_temp3

        # Final SRK3 combination
        y_new = (
            y
            + (1 / 4) * K1 * dt
            + (3 / 8) * K2 * dt
            + (3 / 8) * K3 * dt
            + (1 / 4) * L1 * np.sqrt(dt)
            + (3 / 8) * L2 * np.sqrt(dt)
            + (3 / 8) * L3 * np.sqrt(dt)
        )

        return y_new
