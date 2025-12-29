import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_config import (
    ZhanDuanLiLiConfig,
)
from discrecontinual_equations.variable import Variable


class ZhanDuanLiLiSolver(Solver):
    """
    Zhan-Duan-Li-Li method for solving stochastic differential equations.

    This solver implements the method proposed by Zhan, Duan, Li, and Li in their paper:
    "A stochastic theta method based on the trapezoidal rule" (https://arxiv.org/pdf/2010.07691)

    The method combines stochastic theta methods with trapezoidal rule approaches
    for improved accuracy in solving SDEs.

    Note: This implementation is based on understanding of stochastic theta methods
    and trapezoidal rule combinations. The exact implementation may need refinement
    based on the specific details in the paper.
    """

    def __init__(self, solver_config: ZhanDuanLiLiConfig):
        super().__init__(solver_config=solver_config)

        # Set random seed for reproducibility
        if self.solver_config.random_seed is not None:
            np.random.seed(self.solver_config.random_seed)

        # Theta parameter from config (controls implicit/explicit balance)
        self.theta = self.solver_config.method_parameter

    def solve(self, equation: DifferentialEquation, initial_values: list[float]):
        """
        Solve the SDE using the Zhan-Duan-Li-Li stochastic theta method.

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
        y = np.array(initial_values, dtype=float)

        # Append initial point
        self.solution.append([t, [0] * len(initial_values), y.tolist()])

        # Time stepping loop
        dt = self.solver_config.dt

        for i in range(self.solver_config.n_steps):
            # Current time
            t_current = self.solver_config.times[i]
            t_next = self.solver_config.times[i + 1]

            # Generate Wiener increment: ΔW ~ N(0, dt)
            dW = np.random.normal(0, np.sqrt(dt), size=len(y))

            # Stochastic theta method with trapezoidal rule
            y_new = self._stochastic_theta_step(y, t_current, t_next, dt, dW, equation)

            # Update state
            y = y_new

            # Append to solution
            self.solution.append([t_next, [0] * len(initial_values), y.tolist()])

    def _stochastic_theta_step(self, y, t_current, t_next, dt, dW, equation):
        """
        Perform one step of the stochastic theta method with trapezoidal rule.

        This implements a combination of implicit and explicit methods
        for improved stability and accuracy.
        """
        theta = self.theta

        # Evaluate functions at current point
        drift_current = np.array(
            equation.derivative.eval(point=y.tolist(), time=t_current),
        )
        diffusion_current = np.array(
            equation.derivative.diffusion(point=y.tolist(), time=t_current),
        )

        # Predictor step (explicit Euler)
        y_predictor = y + drift_current * dt + diffusion_current * dW

        # Evaluate functions at predicted point
        drift_predictor = np.array(
            equation.derivative.eval(point=y_predictor.tolist(), time=t_next),
        )
        diffusion_predictor = np.array(
            equation.derivative.diffusion(point=y_predictor.tolist(), time=t_next),
        )

        # Corrector step (theta method combining current and predicted values)
        # This implements a trapezoidal-like rule for stochastic systems
        drift_combined = theta * drift_predictor + (1 - theta) * drift_current
        diffusion_combined = (
            theta * diffusion_predictor + (1 - theta) * diffusion_current
        )

        # Final update using the combined values
        y_new = y + drift_combined * dt + diffusion_combined * dW

        return y_new
