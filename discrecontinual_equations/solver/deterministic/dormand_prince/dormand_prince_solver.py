import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.deterministic.dormand_prince.dormand_prince_config import (
    DormandPrinceConfig,
)
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.variable import Variable


class DormandPrinceSolver(Solver):
    """
    Dormand-Prince Runge-Kutta method (5,4) for solving ODEs.

    A 5th-order Runge-Kutta method with embedded 4th-order solution for error estimation.
    Uses 7 stages and provides adaptive step size control through local error estimation.

    The method provides both 5th-order (y5) and 4th-order (y4) solutions, where the
    difference |y5 - y4| serves as an estimate of the local truncation error.

    References:
    - Dormand, John R. and Prince, Peter J. "A family of embedded Runge-Kutta formulae"
      Journal of Computational and Applied Mathematics, 1980
    """

    def __init__(self, solver_config: DormandPrinceConfig):
        super().__init__(solver_config=solver_config)

        # Dormand-Prince coefficients
        self.a = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [1 / 5, 0, 0, 0, 0, 0, 0],
                [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
                [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
                [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
                [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
                [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
            ],
        )

        self.b5 = np.array(
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        )  # 5th order
        self.b4 = np.array(
            [
                5179 / 57600,
                0,
                7571 / 16695,
                393 / 640,
                -92097 / 339200,
                187 / 2100,
                1 / 40,
            ],
        )  # 4th order

        self.c = np.array([0, 1 / 5, 3 / 10, 4 / 5, 7 / 9, 1, 1])

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
        y = np.array(initial_values)
        h = self.solver_config.initial_step_size

        # Append initial point
        self.solution.append([t, [0] * len(initial_values), initial_values])

        # Adaptive integration loop
        while t < self.solver_config.end_time:
            if t + h > self.solver_config.end_time:
                h = self.solver_config.end_time - t

            # Perform one step
            y_new, error, h_new = self._step(equation, t, y, h)

            # Check if step is accepted
            if self._accept_step(error, y, y_new, h):
                # Accept step
                t += h
                y = y_new
                self.solution.append([t, [0] * len(initial_values), y.tolist()])
                h = h_new
            else:
                # Reject step, try smaller step
                h = max(h_new, self.solver_config.min_step_size)
                if h < self.solver_config.min_step_size:
                    raise RuntimeError("Step size became too small")

    def _step(self, equation: DifferentialEquation, t: float, y: np.ndarray, h: float):
        """Perform one Dormand-Prince step."""
        k = np.zeros((7, len(y)))

        # Compute stages
        for i in range(7):
            t_stage = t + self.c[i] * h
            y_stage = y + h * np.dot(self.a[i, : i + 1], k[: i + 1])
            k[i] = np.array(
                equation.derivative.eval(point=y_stage.tolist(), time=t_stage),
            )

        # Compute solutions
        y5 = y + h * np.dot(self.b5, k)  # 5th order solution
        y4 = y + h * np.dot(self.b4, k)  # 4th order solution

        # Error estimate
        error = np.abs(y5 - y4)

        # Step size control
        h_new = self._compute_new_step_size(error, y, h)

        return y5, error, h_new

    def _accept_step(
        self,
        error: np.ndarray,
        y: np.ndarray,
        y_new: np.ndarray,
        h: float,
    ) -> bool:
        """Check if the step should be accepted based on error."""
        # Compute scaled error
        tol = (
            self.solver_config.absolute_tolerance
            + self.solver_config.relative_tolerance
            * np.maximum(np.abs(y), np.abs(y_new))
        )
        scaled_error = np.max(error / tol)

        return scaled_error <= 1.0

    def _compute_new_step_size(
        self,
        error: np.ndarray,
        y: np.ndarray,
        h: float,
    ) -> float:
        """Compute new step size using PI control."""
        tol = (
            self.solver_config.absolute_tolerance
            + self.solver_config.relative_tolerance * np.abs(y)
        )
        scaled_error = np.max(error / tol)

        if scaled_error == 0:
            return self.solver_config.max_step_size

        # PI controller
        h_new = (
            h
            * self.solver_config.safety_factor
            * (1 / scaled_error) ** (1 / 5)
            * (1 / scaled_error) ** (self.solver_config.beta / 5)
        )

        # Limit step size
        h_new = np.clip(
            h_new,
            self.solver_config.min_step_size,
            self.solver_config.max_step_size,
        )

        return h_new
