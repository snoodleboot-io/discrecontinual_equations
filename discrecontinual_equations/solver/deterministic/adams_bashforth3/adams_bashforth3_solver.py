"""The three-step Adams-Bashforth method on a fixed grid.

Adams's idea (Bashforth 1883, p. 18) is to integrate, over the next step, the
polynomial that interpolates the derivative values already computed. With the
backward differences of ``f_n = f(t_n, y_n)`` his step reads
``y_{n+1} - y_n = h (f_n + 1/2 Df_n + 5/12 D^2 f_n + 3/8 D^3 f_n + ...)``.
A multistep method reuses one new field evaluation per step, so it is cheap
per step, but it needs a fixed step and past values that the initial
condition does not supply.

Stopping after the second difference gives
``y_{n+1} = y_n + h/12 (23 f_n - 16 f_{n-1} + 5 f_{n-2})``, third order when
its two starting values are accurate to ``O(h^3)``.

Limit, found while documenting and not changed here: they are not. The first
step is forward Euler and the second Adams-Bashforth 2, and the first carries a
local error ``O(h^2)``; as in the two-step solver the stored derivative for
node 1 is ``f(t_1, y_0)``, so ``f_1`` is never evaluated. Each defect is
``O(h^2)`` once, and the result is second order, not third: on ``y' = -y^2``
and ``y' = y cos t`` the error falls as ``h^2``, as for Adams-Bashforth 2.
Starting values of order 3 (a Runge-Kutta start) would restore the order.

References:
    Bashforth, F. (1883). An Attempt to Test the Theories of Capillary Action
    ..., with an Explanation of the Method of Integration ..., by J. C. Adams.
    Cambridge University Press. P. 18: Adams's step formula, here truncated
    after the second difference. (The code's earlier citation of Adams (1855)
    was not checked and is dropped.)

    See ``docs/references.md`` (Bashforth1883).
"""

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.deterministic.adams_bashforth3.adams_bashforth3_config import (
    AdamsBashforth3Config,
)
from discrecontinual_equations.solver.solver import Solver


class AdamsBashforth3Solver(Solver):
    """Three-step Adams-Bashforth, ``y_{n+1} = y_n + h/12 (23 f_n - 16 f_{n-1} + 5 f_{n-2})``.

    On the config's fixed grid, started with one Euler and one
    Adams-Bashforth 2 step; because of that start the result is second order
    (module docstring).

    References:
        Bashforth (1883), p. 18. See ``docs/references.md`` (Bashforth1883).
    """

    def __init__(self, solver_config: AdamsBashforth3Config):
        """Store the configuration.

        Args:
            solver_config: The interval and ``step_size`` (or
                ``number_of_steps``).
        """
        super().__init__(solver_config=solver_config)

    def solve(
        self,
        equation: DifferentialEquation,
        initial_values: list[float],
    ) -> Curve:
        """Integrate over the config's fixed grid ``times``.

        Args:
            equation: The equation; its ``derivative.eval(point, time)`` is the
                field ``f(t, y)``.
            initial_values: The state at ``times[0]``.

        Returns:
            The curve on ``times``: ``results`` holds the state ``y_n``;
            ``variables`` holds the state one node earlier, ``y_{n-1}`` (zeros
            at the first node). It is also kept as ``solution``.
        """
        self.solution = self._new_solution(equation)

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
        return self.solution
