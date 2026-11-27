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

The start and the history are those of
:mod:`~discrecontinual_equations.solver.deterministic.adams_bashforth`: classical
Runge-Kutta starting steps and ``f`` at each computed state (DEQ-141).

References:
    Bashforth, F. (1883). An Attempt to Test the Theories of Capillary Action
    ..., with an Explanation of the Method of Integration ..., by J. C. Adams.
    Cambridge University Press. P. 18: Adams's step formula, here truncated
    after the second difference. (The code's earlier citation of Adams (1855)
    was not checked and is dropped.)

    See ``docs/references.md`` (Bashforth1883).
"""

from discrecontinual_equations.solver.deterministic.adams_bashforth import (
    AdamsBashforthSolver,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth3.adams_bashforth3_config import (
    AdamsBashforth3Config,
)


class AdamsBashforth3Solver(AdamsBashforthSolver):
    """Three-step Adams-Bashforth, ``y_{n+1} = y_n + h/12 (23 f_n - 16 f_{n-1} + 5 f_{n-2})``.

    Third order, on the config's fixed grid, started with two classical
    Runge-Kutta steps (see :mod:`~discrecontinual_equations.solver.deterministic.adams_bashforth`).

    References:
        Bashforth (1883), p. 18. See ``docs/references.md`` (Bashforth1883).
    """

    weights = (23 / 12, -16 / 12, 5 / 12)

    def __init__(self, solver_config: AdamsBashforth3Config) -> None:
        """Store the configuration.

        Args:
            solver_config: The interval and its fixed step.
        """
        super().__init__(solver_config=solver_config)
