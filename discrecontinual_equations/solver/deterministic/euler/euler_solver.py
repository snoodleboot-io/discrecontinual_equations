from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.deterministic.euler.euler_config import (
    EulerConfig,
)
from discrecontinual_equations.solver.solver import Solver


class EulerSolver(Solver):
    """
    Euler method for solving ordinary differential equations.

    The Euler method is the simplest numerical method for solving ODEs,
    using the forward difference approximation: y_{n+1} = y_n + h * f(t_n, y_n)

    Reference:
    - Euler, Leonhard. "Institutionum calculi integralis" (1768-1770)
    """

    def __init__(self, solver_config: EulerConfig):
        super().__init__(solver_config=solver_config)

    def solve(
        self,
        equation: DifferentialEquation,
        initial_values: list[float],
    ) -> Curve:
        self.solution = self._new_solution(equation)
        increment = 0
        for time in self.solver_config.times:
            if increment > 0:
                last_value = [
                    result.discretization[-1] for result in self.solution.results
                ]
                evaluated = equation.derivative.eval(point=last_value, time=time)
                next_value = [
                    last + value * self.solver_config.step_size
                    for value, last in zip(evaluated, last_value, strict=False)
                ]
                self.solution.append([time, last_value, next_value])

            else:
                equation.derivative.eval(point=initial_values, time=time)
                self.solution.append([time, [0] * len(initial_values), initial_values])
            increment += 1
        return self.solution
