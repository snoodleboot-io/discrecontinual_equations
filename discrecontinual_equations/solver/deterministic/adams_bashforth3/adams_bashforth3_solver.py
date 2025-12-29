from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.deterministic.adams_bashforth3.adams_bashforth3_config import (
    AdamsBashforth3Config,
)
from discrecontinual_equations.solver.solver import Solver
from discrecontinual_equations.variable import Variable


class AdamsBashforth3Solver(Solver):
    """
    Adams-Bashforth 3rd order method for solving ODEs.

    A 3rd-order linear multistep method using the current and two previous derivative values:
    y_{n+1} = y_n + (h/12) * (23*f_n - 16*f_{n-1} + 5*f_{n-2})

    This method requires two previous steps (startup) and achieves 3rd-order accuracy.

    References:
    - Adams, John Couch. "On the correction of the aberrations of the planets" (1855)
    - Bashforth, Francis. "An attempt to test the theories of capillary action" (1883)
    """

    def __init__(self, solver_config: AdamsBashforth3Config):
        super().__init__(solver_config=solver_config)

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

        # Store derivatives for multi-step method
        derivatives = []

        increment = 0
        for time in self.solver_config.times:
            if increment == 0:
                # Initial step
                equation.derivative.eval(point=initial_values, time=time)
                self.solution.append([time, [0] * len(initial_values), initial_values])
                derivatives.append(
                    equation.derivative.eval(point=initial_values, time=time),
                )
            elif increment == 1:
                # Second step using Euler
                last_value = [
                    result.discretization[-1] for result in self.solution.results
                ]
                evaluated = equation.derivative.eval(point=last_value, time=time)
                next_value = [
                    last + value * self.solver_config.step_size
                    for value, last in zip(evaluated, last_value, strict=False)
                ]
                self.solution.append([time, last_value, next_value])
                derivatives.append(evaluated)
            elif increment == 2:
                # Third step using Adams-Bashforth 2
                last_value = [
                    result.discretization[-1] for result in self.solution.results
                ]

                # AB2 for bootstrap: y_{n+1} = y_n + (h/2) * (3*f(t_n, y_n) - f(t_{n-1}, y_{n-1}))
                # f(t_n, y_n) is derivatives[-1], f(t_{n-1}, y_{n-1}) is derivatives[-2]
                next_value = [
                    last
                    + (self.solver_config.step_size / 2)
                    * (3 * derivatives[-1][i] - derivatives[-2][i])
                    for i, last in enumerate(last_value)
                ]
                self.solution.append([time, last_value, next_value])

                # Evaluate derivative at new point for next step
                current_derivative = equation.derivative.eval(
                    point=next_value,
                    time=time,
                )
                derivatives.append(current_derivative)
            else:
                # Adams-Bashforth 3rd order
                last_value = [
                    result.discretization[-1] for result in self.solution.results
                ]

                # AB3: y_{n+1} = y_n + (h/12) * (23*f(t_n, y_n) - 16*f(t_{n-1}, y_{n-1}) + 5*f(t_{n-2}, y_{n-2}))
                # f(t_n, y_n) is derivatives[-1], f(t_{n-1}, y_{n-1}) is derivatives[-2], f(t_{n-2}, y_{n-2}) is derivatives[-3]
                next_value = [
                    last
                    + (self.solver_config.step_size / 12)
                    * (
                        23 * derivatives[-1][i]
                        - 16 * derivatives[-2][i]
                        + 5 * derivatives[-3][i]
                    )
                    for i, last in enumerate(last_value)
                ]
                self.solution.append([time, last_value, next_value])

                # Evaluate derivative at new point for next step
                current_derivative = equation.derivative.eval(
                    point=next_value,
                    time=time,
                )
                derivatives.append(current_derivative)

            increment += 1
