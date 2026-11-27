"""Observed convergence orders of the deterministic solvers on known solutions.

The global error at ``t = 2`` is measured at halving step sizes, and the order
is ``log2(e(h) / e(h/2))``. The test fields have closed-form solutions and
depend on ``t``, so a wrong node ``c_i`` or a field evaluated at the wrong time
shows up as a lost order (DEQ-137, DEQ-141):

* ``y' = y cos t``, ``y(0) = 1``: ``y = exp(sin t)``;
* ``y' = -2 t y^2``, ``y(0) = 1``: ``y = 1 / (1 + t^2)``, where ``t`` and ``y``
  enter together;
* ``y' = -y^2``, ``y(0) = 1``: ``y = 1 / (1 + t)``, autonomous, for contrast.

Adaptive solvers are held to a fixed step by setting the first, smallest and
largest step equal and the tolerance out of reach. The embedded row of a pair
is measured with :func:`rk_fixed`, an explicit Runge-Kutta loop written here
that reads only the solver's ``a`` and ``c`` and the row it is given.
"""

import math
from collections.abc import Callable

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.variable import Variable

END = 2.0


class Oscillating(DeterministicFunction):
    """``y' = y cos t``."""

    exact = staticmethod(lambda t: math.exp(math.sin(t)))

    def eval(self, point, time=None):
        return [point[0] * math.cos(time)]


class Mixed(DeterministicFunction):
    """``y' = -2 t y^2``."""

    exact = staticmethod(lambda t: 1.0 / (1.0 + t * t))

    def eval(self, point, time=None):
        return [-2.0 * time * point[0] ** 2]


class Autonomous(DeterministicFunction):
    """``y' = -y^2``."""

    exact = staticmethod(lambda t: 1.0 / (1.0 + t))

    def eval(self, point, time=None):  # noqa: ARG002
        return [-point[0] ** 2]


def equation(field: type[DeterministicFunction]) -> DifferentialEquation:
    t, y, r = Variable(name="t"), Variable(name="y"), Variable(name="r")
    return DifferentialEquation(
        variables=[y],
        time=t,
        parameters=[],
        derivative=field([y], [], [r], t),
    )


def forced(solver: type, config: type) -> Callable[[float], object]:
    """An adaptive solver held to the fixed step ``h`` on ``[0, END]``."""
    return lambda h: solver(
        config(
            start_time=0.0,
            end_time=END,
            initial_step_size=h,
            min_step_size=h,
            max_step_size=h,
            absolute_tolerance=1e30,
            relative_tolerance=0.0,
        ),
    )


def orders(
    make: Callable[[float], object],
    field: type[DeterministicFunction],
    steps: list[float],
) -> list[float]:
    """Observed orders of ``make(h).solve`` between consecutive ``steps``."""
    errors = []
    for h in steps:
        curve = make(h).solve(equation(field), [1.0])
        assert abs(curve.time.discretization[-1] - END) < 1e-12
        errors.append(abs(curve.results[0].discretization[-1] - field.exact(END)))
    return _slopes(errors, steps)


def rk_fixed(
    a: np.ndarray,
    c: np.ndarray,
    weights: np.ndarray,
    field: type[DeterministicFunction],
    steps: list[float],
) -> list[float]:
    """Observed orders of the explicit Runge-Kutta method ``(a, c, weights)``."""
    f = equation(field).derivative
    errors = []
    for h in steps:
        y = np.array([1.0])
        n = round(END / h)
        for step in range(n):
            t = step * h
            k = np.zeros((len(weights), 1))
            for i in range(len(weights)):
                k[i] = f.eval(
                    point=(y + h * (a[i, :i] @ k[:i])).tolist(), time=t + c[i] * h
                )
            y = y + h * (weights @ k)
        errors.append(abs(y[0] - field.exact(END)))
    return _slopes(errors, steps)


def _slopes(errors: list[float], steps: list[float]) -> list[float]:
    return [
        math.log(errors[i] / errors[i + 1]) / math.log(steps[i] / steps[i + 1])
        for i in range(len(errors) - 1)
    ]
