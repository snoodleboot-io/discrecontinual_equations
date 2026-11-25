"""The two-step Adams-Bashforth method on a fixed grid.

Adams's idea (Bashforth 1883, p. 18) is to integrate, over the next step, the
polynomial that interpolates the derivative values already computed. With the
backward differences of ``f_n = f(t_n, y_n)`` his step reads
``y_{n+1} - y_n = h (f_n + 1/2 Df_n + 5/12 D^2 f_n + 3/8 D^3 f_n + ...)``.
A multistep method reuses one new field evaluation per step, so it is cheap
per step, but it needs a fixed step and past values that the initial
condition does not supply.

Stopping after the first difference gives
``y_{n+1} = y_n + h/2 (3 f_n - f_{n-1})``, second order. The first step is
taken with forward Euler (local error ``O(h^2)``, enough for second order).

Departure, found while documenting and not changed here: the history is
misaligned at the start. The derivative stored for node 1 is the one the Euler
step used, ``f(t_1, y_0)``, not ``f(t_1, y_1)``, so steps 2 and 3 use it in
place of ``f_1``. It costs a fixed number of ``O(h^2)`` local errors and the
global order stays 2 (on ``y' = -y^2`` and ``y' = y cos t`` the error falls
as ``h^2``).

References:
    Bashforth, F. (1883). An Attempt to Test the Theories of Capillary Action
    ..., with an Explanation of the Method of Integration ..., by J. C. Adams.
    Cambridge University Press. P. 18: Adams's step formula, here truncated
    after the first difference. (The code's earlier citation of Adams (1855),
    "On the correction of the aberrations of the planets", was not checked
    and is dropped.)

    See ``docs/references.md`` (Bashforth1883).
"""

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.deterministic.adams_bashforth2.adams_bashforth2_config import (
    AdamsBashforth2Config,
)
from discrecontinual_equations.solver.solver import Solver


class AdamsBashforth2Solver(Solver):
    """Two-step Adams-Bashforth, ``y_{n+1} = y_n + h/2 (3 f_n - f_{n-1})``.

    Second order, on the config's fixed grid, started with one Euler step (see
    the module docstring for how the start is stored).

    References:
        Bashforth (1883), p. 18. See ``docs/references.md`` (Bashforth1883).
    """

    def __init__(self, solver_config: AdamsBashforth2Config):
        """Store the configuration.

        Args:
            solver_config: The interval and ``step_size``.
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
            else:
                # Adams-Bashforth 2nd order
                last_value = [
                    result.discretization[-1] for result in self.solution.results
                ]

                # AB2: y_{n+1} = y_n + (h/2) * (3*f(t_n, y_n) - f(t_{n-1}, y_{n-1}))
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

            increment += 1
        return self.solution
