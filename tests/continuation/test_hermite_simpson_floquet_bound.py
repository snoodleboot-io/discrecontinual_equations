"""Hermite-Simpson's Floquet error bounds the multipliers' error (DEQ-105).

The reference is built outside the code under test: the Rossler cycle that
winds ``turns`` times is a fixed point of the ``turns``-th return map to the
section ``y = 0``, found by Newton on that map with each return integrated by
DOP853 at ``rtol = 1e-12`` together with its variational equation. Its
nontrivial multipliers are the eigenvalues of the return map's Jacobian, and
the trivial one is exactly one.
"""

import itertools
from unittest import TestCase, mock

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CycleSeed,
    _multiplier_disagreement,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import Alpha, State, Time

_A = _B = 0.2


class RosslerField(DeterministicFunction):
    """Rossler's system with ``a = b = 0.2``; ``c`` is the parameter."""

    def eval(self, point, time=None):  # noqa: ARG002
        c = self.parameters[0].value
        return [
            -point[1] - point[2],
            point[0] + _A * point[1],
            _B + point[2] * (point[0] - c),
        ]


def _equation(c: float) -> DifferentialEquation:
    parameters = [Alpha(value=c)]
    return DifferentialEquation(
        variables=[State(), State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=RosslerField(
            variables=[State(), State(), State()],
            parameters=parameters,
            results=[State(), State(), State()],
            time=None,
        ),
    )


def _variational(_t, u, c):
    x, y, z = u[:3]
    jacobian = np.array([[0.0, -1.0, -1.0], [1.0, _A, 0.0], [z, 0.0, x - c]])
    flow = [-y - z, x + _A * y, _B + z * (x - c)]
    return np.concatenate([flow, (jacobian @ u[3:].reshape(3, 3)).ravel()])


def _upward(_t, u, _c):
    return u[1]


_upward.direction = 1.0


def _returns(point: np.ndarray, c: float, turns: int):
    """``turns`` returns to ``y = 0`` from ``(x, 0, z)``: image, Jacobian, time."""
    state, transition, elapsed = np.array([point[0], 0.0, point[1]]), np.eye(3), 0.0
    normal = np.array([0.0, 1.0, 0.0])
    for _ in range(turns):
        run = solve_ivp(
            _variational,
            (0.0, 60.0),
            np.concatenate([state, transition.ravel()]),
            args=(c,),
            method="DOP853",
            rtol=1e-12,
            atol=1e-13,
            events=_upward,
        )
        index = next(i for i, t in enumerate(run.t_events[0]) if t > 1e-6)
        elapsed += run.t_events[0][index]
        end = run.y_events[0][index]
        flow = _variational(0.0, end, c)[:3]
        projection = np.eye(3) - np.outer(flow, normal) / (normal @ flow)
        transition = projection @ end[3:].reshape(3, 3)
        state = np.array([end[0], 0.0, end[2]])
    image = np.array([state[0], state[2]])
    return image, transition[np.ix_([0, 2], [0, 2])], elapsed


def _reference(c: float, turns: int, intervals: int):
    """Nodes, period and the full set of multipliers of the attracting cycle."""
    settled = solve_ivp(
        _variational,
        (0.0, 2000.0),
        np.concatenate([[1.0, 1.0, 0.0], np.eye(3).ravel()]),
        args=(c,),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        events=_upward,
    )
    point = settled.y_events[0][-1][[0, 2]]
    for _ in range(30):
        image, jacobian, period = _returns(point, c, turns)
        if np.linalg.norm(image - point) < 1e-12:
            break
        point = point - np.linalg.solve(jacobian - np.eye(2), image - point)
    start = np.array([point[0], 0.0, point[1]])
    orbit = solve_ivp(
        lambda t, u: _variational(t, np.concatenate([u, np.eye(3).ravel()]), c)[:3],
        (0.0, period),
        start,
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
        dense_output=True,
    )
    nodes = orbit.sol(np.linspace(0.0, period, intervals + 1)).T
    return nodes, period, np.concatenate([[1.0], np.linalg.eigvals(jacobian)])


def _true_error(computed: np.ndarray, exact: np.ndarray) -> float:
    """Worst relative gap under the pairing that makes it smallest."""
    return min(
        max(
            abs(computed[i] - exact[j]) / max(1.0, abs(exact[j]))
            for i, j in enumerate(order)
        )
        for order in itertools.permutations(range(len(exact)))
    )


def _point(c: float, turns: int, per_turn: int = 80, doubled=None):
    intervals = turns * per_turn
    nodes, period, exact = _reference(c, turns, intervals)
    continuation = CycleContinuation(_equation(c), 0, intervals, 1, 0.0)
    seed = CycleSeed(nodes, period, c)
    if doubled is None:
        trace, _ = continuation.trace(seed, 0.1, 0)
    else:
        with mock.patch.object(CycleContinuation, "_doubled_multipliers", doubled):
            trace, _ = continuation.trace(seed, 0.1, 0)
    return trace[0], exact


class TestBoundOnMultiTurnCycles(TestCase):
    """The Rossler cascade at 80 nodes a turn, where the drift read 1.2-43x low.

    The trivial drift read 8.4e-6 against a true 1.6e-5 on the two-turn cycle,
    and 2.3e-5 against 1.7e-4 on the four-turn one.
    """

    def _over_read(self, c: float, turns: int) -> float:
        point, exact = _point(c, turns)
        true = _true_error(point.multipliers, exact)
        assert point.floquet_error >= true, (c, turns, point.floquet_error, true)
        return point.floquet_error / true

    def test_one_turn(self):
        assert self._over_read(2.8, 1) < 100.0

    def test_two_turns(self):
        assert self._over_read(3.8, 2) < 100.0

    def test_four_turns(self):
        assert self._over_read(4.1, 4) < 100.0


class TestEstimateReadings(TestCase):
    """Each reading counts on its own, and a failed doubled solve refuses."""

    def test_the_drift_alone_when_the_doubled_mesh_agrees(self):
        def agreeing(continuation, states, period, parameter):
            return continuation._multipliers(states, period, parameter)  # noqa: SLF001

        point, _ = _point(2.8, 1, 40, agreeing)
        drift = float(np.min(np.abs(point.multipliers - 1.0)))
        assert drift > 0.0
        assert point.floquet_error == drift

    def test_twice_the_gap_when_it_is_the_larger(self):
        def shifted(continuation, states, period, parameter):
            reading = continuation._multipliers  # noqa: SLF001
            multipliers = reading(states, period, parameter)
            return multipliers + 0.01 * (np.abs(multipliers) > 0.5)

        point, _ = _point(2.8, 1, 40, shifted)
        assert abs(point.floquet_error - 0.02) < 1e-9

    def test_a_failed_doubled_solve_reads_infinite_and_is_refused(self):
        def failing(continuation, states, period, parameter):  # noqa: ARG001
            return np.full(3, np.nan)

        point, _ = _point(2.8, 1, 40, failing)
        assert np.all(np.isfinite(point.multipliers))
        assert point.floquet_error == float("inf")
        assert not point.resolved()


class TestDisagreementScale(TestCase):
    def test_a_reading_that_lost_the_trivial_multiplier_is_not_capped_at_one(self):
        """Van der Pol at ``mu = 5`` on 12 nodes read 1540 for multipliers 1 and 0."""
        gap = _multiplier_disagreement(np.array([1540.0, 0.0]), np.array([1.0, 1e-30]))
        assert abs(gap - 1539.0) < 1e-9

    def test_either_order_gives_the_same_gap(self):
        first, second = np.array([140.0, 1.0, 0.2]), np.array([1.0, 0.25, 7.0])
        assert _multiplier_disagreement(first, second) == _multiplier_disagreement(
            second,
            first,
        )


class TestUnreadableMultipliers(TestCase):
    def test_a_monodromy_with_no_finite_multipliers_reads_infinite(self):
        def unreadable(*_reading):
            return np.full(3, np.nan)

        intervals = 40
        nodes, period, _ = _reference(2.8, 1, intervals)
        continuation = CycleContinuation(_equation(2.8), 0, intervals, 1, 0.0)
        with mock.patch.object(CycleContinuation, "_multipliers", unreadable):
            trace, _ = continuation.trace(CycleSeed(nodes, period, 2.8), 0.1, 0)
        assert trace[0].floquet_error == float("inf")
        assert not trace[0].resolved()


class TestTrapezoidalKeepsTheDrift(TestCase):
    def test_a_trapezoidal_point_reads_the_drift_without_a_doubled_solve(self):
        """Only Hermite-Simpson and Gauss-Legendre pay for the doubled mesh."""

        def forbidden(*_reading):
            message = "trapezoidal must not re-solve on the doubled mesh"
            raise AssertionError(message)

        intervals = 40
        nodes, period, _ = _reference(2.8, 1, intervals)
        continuation = CycleContinuation(
            _equation(2.8),
            0,
            intervals,
            1,
            0.0,
            collocation="trapezoidal",
        )
        with mock.patch.object(CycleContinuation, "_doubled_multipliers", forbidden):
            trace, _ = continuation.trace(CycleSeed(nodes, period, 2.8), 0.1, 0)
        drift = float(np.min(np.abs(trace[0].multipliers - 1.0)))
        assert trace[0].floquet_error == drift
