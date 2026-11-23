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

The field is evaluated at the old node, ``f(t_n, y_n)``, as Euler's formula
has it. Until DEQ-141 it was evaluated at ``f(t_{n+1}, y_n)``: the same method
for an autonomous field, a different (still first-order) one for a field that
depends on ``t`` (``y' = y cos t`` to ``t = 2`` with ``h = 0.1``: error 2.5e-1,
now 7.5e-2).

References:
    Euler, L. (1768). Institutionum calculi integralis, vol. 1, part I,
    section II, chapter VII, Problem 85, sections 650-653. Read in I. Bruce's
    English translation (17centurymaths.com, ``intcalvol1/part2ch7.pdf``).

    See ``docs/references.md`` (Euler1768).
"""

from itertools import pairwise

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.deterministic.euler.euler_config import (
    EulerConfig,
)
from discrecontinual_equations.solver.solver import Solver


class EulerSolver(Solver):
    """Forward Euler, ``y_{n+1} = y_n + h f(t_n, y_n)``, on a fixed grid.

    First order. The grid is the config's ``times`` and the step its
    ``step_size``.

    References:
        Euler (1768), Problem 85, section 650. See ``docs/references.md``
        (Euler1768).

    Complexity:
        ``O(K (n + C_f))`` time and ``O(K n)`` memory for ``K`` steps, all of it
        Python lists. Each ``solve`` builds a curve of its own, so solving the
        same equation again costs the same and leaves the first curve as it was.
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
        times = self.solver_config.times
        step = self.solver_config.step_size
        value = list(initial_values)
        self.solution.append([times[0], [0] * len(value), value])
        for start, end in pairwise(times):
            slope = equation.derivative.eval(point=value, time=start)
            following = [
                last + rate * step for rate, last in zip(slope, value, strict=True)
            ]
            self.solution.append([end, value, following])
            value = following
        return self.solution
