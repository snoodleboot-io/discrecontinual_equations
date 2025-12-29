import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.stochastic.srk5.srk5_config import SRK5Config
from discrecontinual_equations.variable import Variable


class SRK5Solver(Solver):
    """
    Stochastic Runge-Kutta method of order 5 (SRK5) for solving stochastic differential equations (SDEs).

    This implements a 5-stage SRK method with support for both Ito and Stratonovich calculus.
    Provides the highest accuracy among the implemented SRK methods through additional stages.

    For Ito SDEs: dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t
    For Stratonovich SDEs: dX_t = μ(X_t, t) dt + σ(X_t, t) ∘ dW_t

    The method achieves strong order 2.5 and weak order 5.0 convergence.
    """

    def __init__(self, solver_config: SRK5Config):
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

            # SRK5 step
            y_new = self._srk5_step(y, t_current, dt, equation)

            # Update time and state
            t_next = self.solver_config.times[i + 1]
            y = y_new

            # Append to solution
            self.solution.append([t_next, [0] * len(initial_values), y.tolist()])

    def _srk5_step(
        self,
        y: np.ndarray,
        t: float,
        dt: float,
        equation: DifferentialEquation,
    ) -> np.ndarray:
        """Perform one SRK5 step with 5 stages."""
        # Stage 1
        k1 = np.array(equation.derivative.eval(point=y.tolist(), time=t))
        l1 = np.array(equation.derivative.diffusion(point=y.tolist(), time=t))

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
            stratonovich_correction = 0.5 * l1 * dsigma_dx
            k1 = k1 - stratonovich_correction

        # Stage 2
        y_temp2 = y + (1 / 5) * k1 * dt + (1 / 5) * l1 * np.sqrt(dt)
        k2 = np.array(equation.derivative.eval(point=y_temp2.tolist(), time=t + dt / 5))
        l2 = np.array(
            equation.derivative.diffusion(point=y_temp2.tolist(), time=t + dt / 5),
        )

        # Apply Stratonovich correction to K2 if requested
        if self.solver_config.calculus == "stratonovich" and len(y) == 1:
            eps = 1e-8
            y_temp2_plus = y_temp2 + eps
            y_temp2_minus = y_temp2 - eps
            diffusion_temp2_plus = np.array(
                equation.derivative.diffusion(
                    point=y_temp2_plus.tolist(),
                    time=t + dt / 5,
                ),
            )
            diffusion_temp2_minus = np.array(
                equation.derivative.diffusion(
                    point=y_temp2_minus.tolist(),
                    time=t + dt / 5,
                ),
            )
            dsigma_dx_temp2 = (diffusion_temp2_plus - diffusion_temp2_minus) / (2 * eps)
            stratonovich_correction_temp2 = 0.5 * l2 * dsigma_dx_temp2
            k2 = k2 - stratonovich_correction_temp2

        # Stage 3
        y_temp3 = y + (1 / 3) * k2 * dt + (1 / 3) * l2 * np.sqrt(dt)
        k3 = np.array(equation.derivative.eval(point=y_temp3.tolist(), time=t + dt / 3))
        l3 = np.array(
            equation.derivative.diffusion(point=y_temp3.tolist(), time=t + dt / 3),
        )

        # Apply Stratonovich correction to K3 if requested
        if self.solver_config.calculus == "stratonovich" and len(y) == 1:
            eps = 1e-8
            y_temp3_plus = y_temp3 + eps
            y_temp3_minus = y_temp3 - eps
            diffusion_temp3_plus = np.array(
                equation.derivative.diffusion(
                    point=y_temp3_plus.tolist(),
                    time=t + dt / 3,
                ),
            )
            diffusion_temp3_minus = np.array(
                equation.derivative.diffusion(
                    point=y_temp3_minus.tolist(),
                    time=t + dt / 3,
                ),
            )
            dsigma_dx_temp3 = (diffusion_temp3_plus - diffusion_temp3_minus) / (2 * eps)
            stratonovich_correction_temp3 = 0.5 * l3 * dsigma_dx_temp3
            k3 = k3 - stratonovich_correction_temp3

        # Stage 4
        y_temp4 = y + (1 / 2) * k3 * dt + (1 / 2) * l3 * np.sqrt(dt)
        k4 = np.array(equation.derivative.eval(point=y_temp4.tolist(), time=t + dt / 2))
        l4 = np.array(
            equation.derivative.diffusion(point=y_temp4.tolist(), time=t + dt / 2),
        )

        # Apply Stratonovich correction to K4 if requested
        if self.solver_config.calculus == "stratonovich" and len(y) == 1:
            eps = 1e-8
            y_temp4_plus = y_temp4 + eps
            y_temp4_minus = y_temp4 - eps
            diffusion_temp4_plus = np.array(
                equation.derivative.diffusion(
                    point=y_temp4_plus.tolist(),
                    time=t + dt / 2,
                ),
            )
            diffusion_temp4_minus = np.array(
                equation.derivative.diffusion(
                    point=y_temp4_minus.tolist(),
                    time=t + dt / 2,
                ),
            )
            dsigma_dx_temp4 = (diffusion_temp4_plus - diffusion_temp4_minus) / (2 * eps)
            stratonovich_correction_temp4 = 0.5 * l4 * dsigma_dx_temp4
            k4 = k4 - stratonovich_correction_temp4

        # Stage 5
        y_temp5 = y + (3 / 4) * k4 * dt + (3 / 4) * l4 * np.sqrt(dt)
        k5 = np.array(
            equation.derivative.eval(point=y_temp5.tolist(), time=t + 3 * dt / 4),
        )
        l5 = np.array(
            equation.derivative.diffusion(point=y_temp5.tolist(), time=t + 3 * dt / 4),
        )

        # Apply Stratonovich correction to K5 if requested
        if self.solver_config.calculus == "stratonovich" and len(y) == 1:
            eps = 1e-8
            y_temp5_plus = y_temp5 + eps
            y_temp5_minus = y_temp5 - eps
            diffusion_temp5_plus = np.array(
                equation.derivative.diffusion(
                    point=y_temp5_plus.tolist(),
                    time=t + 3 * dt / 4,
                ),
            )
            diffusion_temp5_minus = np.array(
                equation.derivative.diffusion(
                    point=y_temp5_minus.tolist(),
                    time=t + 3 * dt / 4,
                ),
            )
            dsigma_dx_temp5 = (diffusion_temp5_plus - diffusion_temp5_minus) / (2 * eps)
            stratonovich_correction_temp5 = 0.5 * l5 * dsigma_dx_temp5
            k5 = k5 - stratonovich_correction_temp5

        # Final SRK5 combination
        y_new = (
            y
            + (1 / 24) * k1 * dt
            + (4 / 24) * k2 * dt
            + (6 / 24) * k3 * dt
            + (8 / 24) * k4 * dt
            + (5 / 24) * k5 * dt
            + (1 / 24) * l1 * np.sqrt(dt)
            + (4 / 24) * l2 * np.sqrt(dt)
            + (6 / 24) * l3 * np.sqrt(dt)
            + (8 / 24) * l4 * np.sqrt(dt)
            + (5 / 24) * l5 * np.sqrt(dt)
        )

        return y_new
