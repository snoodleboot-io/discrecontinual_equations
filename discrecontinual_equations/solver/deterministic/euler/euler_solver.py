"""The explicit (forward) Euler method on a fixed grid.

The simplest one-step method: over one short step, treat the slope as constant
and follow it. Euler (1768, Problem 85, section 650) puts ``V = A`` at the
point ``(a, b)`` and steps to ``b' = b + A (a' - a)``, then repeats from
``(a', b')``. In the usual notation, ``y_{n+1} = y_n + h f(t_n, y_n)``. The
local error is ``O(h^2)`` and the global error ``O(h)``: halving the step halves
the error. Euler's own corollaries (sections 652-653) name both sides: smaller
steps are more accurate, but the per-step errors add up, and they are largest
where ``V`` changes fastest. On ``y' = lambda y`` the method is stable only for
``|1 + h lambda| <= 1``, so stiff or oscillatory problems need small steps.

Departure from Euler's formula: the field is evaluated at the *new* node,
``f(t_{n+1}, y_n)``, not ``f(t_n, y_n)``. For an autonomous field this changes
nothing. For one that depends on ``t`` it is a different method, still first
order: on ``y' = y cos t`` the error still halves with ``h``.

References:
    Euler, L. (1768). Institutionum calculi integralis, vol. 1, part I,
    section II, chapter VII, Problem 85, sections 650-653. Read in I. Bruce's
    English translation (17centurymaths.com, ``intcalvol1/part2ch7.pdf``).

    See ``docs/references.md`` (Euler1768).
"""

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.deterministic.euler.euler_config import (
    EulerConfig,
)
from discrecontinual_equations.solver.solver import Solver


class EulerSolver(Solver):
    """Forward Euler, ``y_{n+1} = y_n + h f(t_{n+1}, y_n)``, on a fixed grid.

    First order. The grid is the config's ``times`` and the step its
    ``step_size``; see the module docstring for the evaluation time.

    References:
        Euler (1768), Problem 85, section 650. See ``docs/references.md``
        (Euler1768).
    """

    def __init__(self, solver_config: EulerConfig):
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
