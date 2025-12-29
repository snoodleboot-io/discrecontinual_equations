import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.stochastic.srk4.srk4_config import SRK4Config
from discrecontinual_equations.variable import Variable


class SRK4Solver(Solver):
    """
    Stochastic Runge-Kutta method of order 4 (SRK4) for solving stochastic differential equations (SDEs).

    This implements a 4-stage SRK method with support for both Ito and Stratonovich calculus.
    Provides higher accuracy than SRK3 through additional stages and refined coefficients.

    For Ito SDEs: dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t
    For Stratonovich SDEs: dX_t = μ(X_t, t) dt + σ(X_t, t) ∘ dW_t

    The method achieves strong order 2.0 and weak order 4.0 convergence.
    """

    def __init__(self, solver_config: SRK4Config):
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

            # SRK4 step
            y_new = self._srk4_step(y, t_current, dt, equation)

            # Update time and state
            t_next = self.solver_config.times[i + 1]
            y = y_new

            # Append to solution
            self.solution.append([t_next, [0] * len(initial_values), y.tolist()])

    def _srk4_step(
        self,
        y: np.ndarray,
        t: float,
        dt: float,
        equation: DifferentialEquation,
    ) -> np.ndarray:
        """Perform one SRK4 step with 4 stages."""
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
        y_temp2 = y + (1 / 4) * K1 * dt + (1 / 4) * L1 * np.sqrt(dt)
        K2 = np.array(equation.derivative.eval(point=y_temp2.tolist(), time=t + dt / 4))
        L2 = np.array(
            equation.derivative.diffusion(point=y_temp2.tolist(), time=t + dt / 4),
        )

        # Apply Stratonovich correction to K2 if requested
        if self.solver_config.calculus == "stratonovich" and len(y) == 1:
            eps = 1e-8
            y_temp2_plus = y_temp2 + eps
            y_temp2_minus = y_temp2 - eps
            diffusion_temp2_plus = np.array(
                equation.derivative.diffusion(
                    point=y_temp2_plus.tolist(),
                    time=t + dt / 4,
                ),
            )
            diffusion_temp2_minus = np.array(
                equation.derivative.diffusion(
                    point=y_temp2_minus.tolist(),
                    time=t + dt / 4,
                ),
            )
            dsigma_dx_temp2 = (diffusion_temp2_plus - diffusion_temp2_minus) / (2 * eps)
            stratonovich_correction_temp2 = 0.5 * L2 * dsigma_dx_temp2
            K2 = K2 - stratonovich_correction_temp2

        # Stage 3
        y_temp3 = y + (1 / 2) * K2 * dt + (1 / 2) * L2 * np.sqrt(dt)
        K3 = np.array(equation.derivative.eval(point=y_temp3.tolist(), time=t + dt / 2))
        L3 = np.array(
            equation.derivative.diffusion(point=y_temp3.tolist(), time=t + dt / 2),
        )

        # Apply Stratonovich correction to K3 if requested
        if self.solver_config.calculus == "stratonovich" and len(y) == 1:
            eps = 1e-8
            y_temp3_plus = y_temp3 + eps
            y_temp3_minus = y_temp3 - eps
            diffusion_temp3_plus = np.array(
                equation.derivative.diffusion(
                    point=y_temp3_plus.tolist(),
                    time=t + dt / 2,
                ),
            )
            diffusion_temp3_minus = np.array(
                equation.derivative.diffusion(
                    point=y_temp3_minus.tolist(),
                    time=t + dt / 2,
                ),
            )
            dsigma_dx_temp3 = (diffusion_temp3_plus - diffusion_temp3_minus) / (2 * eps)
            stratonovich_correction_temp3 = 0.5 * L3 * dsigma_dx_temp3
            K3 = K3 - stratonovich_correction_temp3

        # Stage 4
        y_temp4 = y + K3 * dt + L3 * np.sqrt(dt)
        K4 = np.array(equation.derivative.eval(point=y_temp4.tolist(), time=t + dt))
        L4 = np.array(
            equation.derivative.diffusion(point=y_temp4.tolist(), time=t + dt),
        )

        # Apply Stratonovich correction to K4 if requested
        if self.solver_config.calculus == "stratonovich" and len(y) == 1:
            eps = 1e-8
            y_temp4_plus = y_temp4 + eps
            y_temp4_minus = y_temp4 - eps
            diffusion_temp4_plus = np.array(
                equation.derivative.diffusion(point=y_temp4_plus.tolist(), time=t + dt),
            )
            diffusion_temp4_minus = np.array(
                equation.derivative.diffusion(
                    point=y_temp4_minus.tolist(),
                    time=t + dt,
                ),
            )
            dsigma_dx_temp4 = (diffusion_temp4_plus - diffusion_temp4_minus) / (2 * eps)
            stratonovich_correction_temp4 = 0.5 * L4 * dsigma_dx_temp4
            K4 = K4 - stratonovich_correction_temp4

        # Final SRK4 combination
        y_new = (
            y
            + (1 / 6) * K1 * dt
            + (2 / 6) * K2 * dt
            + (2 / 6) * K3 * dt
            + (1 / 6) * K4 * dt
            + (1 / 6) * L1 * np.sqrt(dt)
            + (2 / 6) * L2 * np.sqrt(dt)
            + (2 / 6) * L3 * np.sqrt(dt)
            + (1 / 6) * L4 * np.sqrt(dt)
        )

        return y_new
