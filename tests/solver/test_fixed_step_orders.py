"""Forward Euler and Adams-Bashforth 2 and 3: their orders and where f is evaluated.

DEQ-141: Euler evaluated ``f(t_{n+1}, y_n)``, and the Adams-Bashforth solvers
started with an Euler step and stored ``f(t_1, y_0)`` for node 1, which left
Adams-Bashforth 3 at second order. The orders are measured on ``y' = y cos t``,
whose field depends on ``t``, over four halvings of the step.
"""

import math
from typing import ClassVar

import numpy as np
import pytest

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.solver.deterministic.adams_bashforth import (
    classical_runge_kutta_step,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth2.adams_bashforth2_config import (  # noqa: E501
    AdamsBashforth2Config,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth2.adams_bashforth2_solver import (  # noqa: E501
    AdamsBashforth2Solver,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth3.adams_bashforth3_config import (  # noqa: E501
    AdamsBashforth3Config,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth3.adams_bashforth3_solver import (  # noqa: E501
    AdamsBashforth3Solver,
)
from discrecontinual_equations.solver.deterministic.euler.euler_config import (
    EulerConfig,
)
from discrecontinual_equations.solver.deterministic.euler.euler_solver import (
    EulerSolver,
)
from discrecontinual_equations.variable import Variable
from tests.solver.deterministic_orders import (
    END,
    Autonomous,
    Oscillating,
    equation,
    orders,
)

STEPS = [0.1, 0.05, 0.025, 0.0125]


def euler(h: float) -> EulerSolver:
    return EulerSolver(EulerConfig(start_time=0.0, end_time=END, step_size=h))


def adams2(h: float) -> AdamsBashforth2Solver:
    return AdamsBashforth2Solver(
        AdamsBashforth2Config(start_time=0.0, end_time=END, step_size=h),
    )


def adams3(h: float) -> AdamsBashforth3Solver:
    # number_of_steps is the node count of the grid (np.linspace).
    return AdamsBashforth3Solver(
        AdamsBashforth3Config(
            start_time=0.0,
            end_time=END,
            number_of_steps=round(END / h) + 1,
        ),
    )


@pytest.mark.parametrize(
    ("make", "order", "low", "high"),
    [
        pytest.param(euler, 1, 0.95, 1.05, id="euler"),
        pytest.param(adams2, 2, 1.9, 2.1, id="adams_bashforth2"),
        pytest.param(adams3, 3, 2.9, 3.5, id="adams_bashforth3"),
    ],
)
@pytest.mark.parametrize("field", [Oscillating, Autonomous])
def test_measured_order(make, order, low, high, field):  # noqa: ARG001
    observed = orders(make, field, STEPS)
    assert all(low < p < high for p in observed), observed


class Ramp(DeterministicFunction):
    """``y' = t``: the step's slope is the time it is evaluated at."""

    def eval(self, point, time=None):  # noqa: ARG002
        return [time]


def ramp() -> DifferentialEquation:
    t, y, r = Variable(name="t"), Variable(name="y"), Variable(name="r")
    return DifferentialEquation(
        variables=[y],
        time=t,
        parameters=[],
        derivative=Ramp([y], [], [r], t),
    )


def test_euler_evaluates_the_field_at_the_old_node():
    # y_{n+1} = y_n + 0.5 t_n: 0, 0.25, 0.75, 1.5 at t = 0.5 .. 2. Evaluated at
    # the new node it gave 0.25, 0.75, 1.5, 2.5.
    curve = euler(0.5).solve(ramp(), [0.0])
    assert curve.results[0].discretization[1:] == [0.0, 0.25, 0.75, 1.5]


class Counted(DeterministicFunction):
    """``y' = -y``, recording every ``(t, y)`` it is evaluated at."""

    seen: ClassVar[list[tuple[float, float]]] = []

    def eval(self, point, time=None):
        Counted.seen.append((time, point[0]))
        return [-point[0]]


@pytest.mark.parametrize(
    ("make", "start_steps"),
    [
        pytest.param(adams2, 1, id="adams_bashforth2"),
        pytest.param(adams3, 2, id="adams_bashforth3"),
    ],
)
def test_history_is_the_field_at_the_computed_states(make, start_steps):
    """After the Runge-Kutta start, each derivative is ``f(t_n, y_n)`` at a node.

    Every evaluation past the start is at a grid node and at the state the
    solver returned for it, so the history holds ``f_n`` and not, as before,
    ``f(t_1, y_0)``.
    """
    t, y, r = Variable(name="t"), Variable(name="y"), Variable(name="r")
    equation = DifferentialEquation(
        variables=[y],
        time=t,
        parameters=[],
        derivative=Counted([y], [], [r], t),
    )
    Counted.seen = []
    curve = make(0.25).solve(equation, [1.0])
    times = curve.time.discretization
    states = curve.results[0].discretization
    for node in range(1, len(times) - 1):
        assert (times[node], states[node]) in Counted.seen
    assert (times[1], states[0]) not in Counted.seen
    # One evaluation at t_0, four per Runge-Kutta step, then one per inner node.
    assert len(Counted.seen) == 1 + 4 * start_steps + (len(times) - 2)


def test_starting_step_is_fourth_order_runge_kutta():
    """The start the Adams-Bashforth solvers use is the classical method itself.

    Its order is measured on ``y' = y cos t`` by stepping
    :func:`classical_runge_kutta_step` alone: 4. Adams-Bashforth 3 keeps its
    order with any start of local error ``O(h^3)``, so its measured order does
    not tell a second-order start from the classical one; this does.
    """
    field = equation(Oscillating)
    errors = []
    for h in STEPS:
        y = np.array([1.0])
        for step in range(round(END / h)):
            y = classical_runge_kutta_step(field, step * h, y, h)
        errors.append(abs(y[0] - Oscillating.exact(END)))
    observed = [math.log2(errors[i] / errors[i + 1]) for i in range(len(STEPS) - 1)]
    assert all(3.9 < p < 4.2 for p in observed), observed
