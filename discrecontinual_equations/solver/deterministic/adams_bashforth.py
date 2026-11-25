"""The explicit Adams-Bashforth methods on a fixed grid, with a Runge-Kutta start.

Adams's idea (Bashforth 1883, p. 18) is to integrate, over the next step, the
polynomial that interpolates the derivatives already computed. A ``k``-step
method advances with ``y_{n+1} = y_n + h sum_j beta_j f_{n-j}``, ``j = 0..k-1``,
where ``f_m = f(t_m, y_m)`` is the field at a *computed* state: one new field
evaluation per step, which is what makes the method cheap. Its local error is
``O(h^(k+1))`` and its global error ``O(h^k)``.

The ``k - 1`` values ``y_1 .. y_{k-1}`` that the initial condition does not
supply are computed here with the classical fourth-order Runge-Kutta method,
whose local error ``O(h^5)`` on a fixed number of steps is below the method's
own error for ``k <= 4``, so the start does not lower the order. Each
derivative in the history is evaluated at its node's own state.

Until DEQ-141 the start was a forward Euler step (local error ``O(h^2)``),
followed for three steps by Adams-Bashforth 2, and the derivative stored for
node 1 was ``f(t_1, y_0)`` - the one the Euler step had used - not
``f(t_1, y_1)``. Adams-Bashforth 2 kept its order 2, but Adams-Bashforth 3
was second order (``y' = y cos t`` to ``t = 2``: orders 1.97, 1.99, 2.00 as
``h`` halves from 0.1; now 3).

References:
    Bashforth, F. (1883). An Attempt to Test the Theories of Capillary Action
    ..., with an Explanation of the Method of Integration ..., by J. C. Adams.
    Cambridge University Press. P. 18: Adams's step formula
    ``y_1 - y_0 = w {q_0 + 1/2 Dq_0 + 5/12 D^2 q_0 + ...}``; truncated after
    ``Dq_0`` it gives the weights ``(3/2, -1/2)``, after ``D^2 q_0``
    ``(23/12, -16/12, 5/12)``.

    Kutta, W. (1901). Beitrag zur naeherungsweisen Integration totaler
    Differentialgleichungen. Zeitschrift fuer Mathematik und Physik
    46:435-453. The classical fourth-order method used for the start (not
    read: its tableau is checked to be exactly fourth order by the rooted-tree
    conditions of Butcher (1963) in the package's tests).

    See ``docs/references.md`` (Bashforth1883, Kutta1901, Butcher1963).
"""

from typing import ClassVar, Protocol, cast

import numpy as np

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver import Solver


class _FixedGrid(Protocol):
    """The one configuration field a fixed-grid solver reads."""

    @property
    def times(self) -> np.ndarray: ...


def _field(
    equation: DifferentialEquation,
    t: float,
    y: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        equation.derivative.eval(point=y.tolist(), time=float(t)),
        dtype=float,
    )


def classical_runge_kutta_step(
    equation: DifferentialEquation,
    t: float,
    y: np.ndarray,
    h: float,
) -> np.ndarray:
    """One step of the classical fourth-order Runge-Kutta method from ``(t, y)``.

    ``k1 = f(t, y)``, ``k2 = f(t + h/2, y + h k1/2)``, ``k3 = f(t + h/2,
    y + h k2/2)``, ``k4 = f(t + h, y + h k3)`` and ``y + h (k1 + 2 k2 + 2 k3 +
    k4)/6``: local error ``O(h^5)``.

    References:
        Kutta (1901). See ``docs/references.md`` (Kutta1901).
    """
    k1 = _field(equation, t, y)
    k2 = _field(equation, t + h / 2, y + h / 2 * k1)
    k3 = _field(equation, t + h / 2, y + h / 2 * k2)
    k4 = _field(equation, t + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class AdamsBashforthSolver(Solver):
    """A ``k``-step Adams-Bashforth method; a subclass names its ``weights``.

    ``weights`` are ``beta_0 .. beta_{k-1}``, newest derivative first, so
    ``k = len(weights)``. The grid is the config's ``times``, which must be
    evenly spaced; the step is read off it. The first ``k - 1`` steps are
    classical Runge-Kutta steps (module docstring).

    References:
        Bashforth (1883), p. 18; Kutta (1901). See ``docs/references.md``
        (Bashforth1883, Kutta1901).
    """

    weights: ClassVar[tuple[float, ...]]

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
        times = np.asarray(cast("_FixedGrid", self.solver_config).times, dtype=float)
        steps = len(self.weights)
        y = np.array(initial_values, dtype=float)
        self.solution.append([times[0], [0.0] * len(y), y.tolist()])

        history = [_field(equation, times[0], y)]
        for n in range(len(times) - 1):
            h = times[n + 1] - times[n]
            if n + 1 < steps:
                y_next = classical_runge_kutta_step(equation, times[n], y, h)
            else:
                y_next = y + h * sum(
                    beta * history[-1 - j] for j, beta in enumerate(self.weights)
                )
            self.solution.append([times[n + 1], y.tolist(), y_next.tolist()])
            y = y_next
            if n + 2 < len(times):
                history.append(_field(equation, times[n + 1], y))
                del history[:-steps]
        return self.solution
