"""The monodromy's ``df/dx`` is exact, or a relative central difference (DEQ-128).

The reference is closed-form, not computed by the code under test. On

    x' = a x (1 - x^2 - y^2) - w y,
    y' = a y (1 - x^2 - y^2) + w x,
    z' = (k x - c) z + s z^2

the unit circle in ``z = 0`` is a cycle of period ``T = 2 pi / w``. Its
multipliers are ``1``, ``exp(-2 a T)`` (the radial direction: ``d/dr`` of
``a r (1 - r^2)`` is ``-2a`` at ``r = 1``) and ``exp(-c T)`` (the ``z``
direction: ``z' = (k cos(w t) - c) z`` on the cycle, and ``cos`` integrates to
zero over a period). ``s z^2`` vanishes on the cycle and so moves neither, but
its curvature ``2 s`` is what a forward difference at step ``h`` turns into a
bias of ``h s`` in ``df_z/dz``: about ``5e-6`` at ``h = 1e-7`` and ``s = 50``,
the same on every mesh, so the doubled-mesh gap could not see it and the
estimate read eleven times low at 40 intervals.
"""

import itertools
import math
from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CycleSeed,
)
from discrecontinual_equations.continuation.periodic_orbit import (
    HermiteSimpsonOrbit,
    PeriodicOrbit,
    PeriodicOrbitSolution,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import Alpha, State, Time

_RATE = 0.3
_SPEED = 1.0
_COUPLING = 0.8
_DECAY = 0.1
_CURVATURE = 50.0
_PERIOD = 2.0 * math.pi / _SPEED
_EXACT = np.array(
    [1.0, math.exp(-2.0 * _RATE * _PERIOD), math.exp(-_DECAY * _PERIOD)],
)


class KnownCycle(DeterministicFunction):
    """The closed-form cycle above, in ``+ - *`` only: duals and jets both pass."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x, y, z = point[0], point[1], point[2]
        rate = self.parameters[0].value
        radial = 1.0 - x * x - y * y
        return [
            rate * x * radial - _SPEED * y,
            rate * y * radial + _SPEED * x,
            (_COUPLING * x - _DECAY) * z + _CURVATURE * z * z,
        ]


class OpaqueKnownCycle(KnownCycle):
    """The same field through ``math.fsum``, which neither duals nor jets pass.

    Its Jacobian can only be differenced, so it exercises the fallback.
    """

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x, y, z = (float(value) for value in point[:3])
        rate = self.parameters[0].value
        radial = math.fsum([1.0, -x * x, -y * y])
        return [
            rate * x * radial - _SPEED * y,
            rate * y * radial + _SPEED * x,
            (_COUPLING * x - _DECAY) * z + _CURVATURE * z * z,
        ]


def _field(kind: type = KnownCycle) -> DeterministicFunction:
    return kind(
        variables=[State(), State(), State()],
        parameters=[Alpha(value=_RATE)],
        results=[State(), State(), State()],
        time=None,
    )


def _equation(kind: type = KnownCycle) -> DifferentialEquation:
    field = _field(kind)
    return DifferentialEquation(
        variables=[State(), State(), State()],
        time=Time(),
        parameters=field.parameters,
        derivative=field,
    )


def _nodes(intervals: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, intervals + 1)
    return np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])


def _true_error(computed: np.ndarray) -> float:
    """Worst gap to the exact multipliers, relative to ``max(1, |mu|)``."""
    return min(
        max(
            abs(computed[i] - _EXACT[j]) / max(1.0, abs(_EXACT[j]))
            for i, j in enumerate(order)
        )
        for order in itertools.permutations(range(_EXACT.size))
    )


def _traced(intervals: int, kind: type = KnownCycle):
    continuation = CycleContinuation(_equation(kind), 0, intervals, 1, 0.0)
    trace, _ = continuation.trace(CycleSeed(_nodes(intervals), _PERIOD, _RATE), 0.1, 0)
    return trace[0]


class TestMonodromyBelowTheOldFloor(TestCase):
    """On the exact nodes the monodromy's error keeps falling at fourth order.

    With the forward difference it sat at 1.68e-5 from 40 intervals on; with
    the exact Jacobian it reads 3.18e-6 at 40 and 1.99e-7 at 80, a factor of
    16.0 per doubling, and the central difference reads the same to 1e-10.
    """

    def _error(self, intervals: int, kind: type = KnownCycle) -> float:
        orbit = PeriodicOrbit(_field(kind), intervals, 1, 0.0)
        solution = PeriodicOrbitSolution(_nodes(intervals), _PERIOD)
        return _true_error(orbit.floquet_multipliers(solution))

    def test_the_exact_jacobian_has_no_floor(self):
        coarse, fine = self._error(40), self._error(80)
        assert fine < 2.2e-7, fine
        assert 15.0 < coarse / fine < 17.0, coarse / fine

    def test_the_central_difference_fallback_has_no_floor_at_this_level(self):
        coarse, fine = self._error(40, OpaqueKnownCycle), self._error(
            80,
            OpaqueKnownCycle,
        )
        assert fine < 2.2e-7, fine
        assert 15.0 < coarse / fine < 17.0, coarse / fine

    def test_hermite_simpson_orbit_reads_the_same_monodromy(self):
        orbit = HermiteSimpsonOrbit(_field(), 80, 1, 0.0)
        multipliers = orbit.floquet_multipliers(
            PeriodicOrbitSolution(_nodes(80), _PERIOD),
        )
        assert _true_error(multipliers) < 2.2e-7


class TestFloquetErrorBoundsBelowTheOldFloor(TestCase):
    """``floquet_error`` is at least the true error at every mesh (DEQ-128).

    With the forward-difference Jacobian it read 1.49e-6 against a true
    1.65e-5 at 40 intervals and 9.4e-8 against 1.67e-5 at 80. Now it reads
    1.87-1.88 times the true error at 20 to 320, the ``2 (1 - 2^-4)`` that
    order four predicts.
    """

    def _check(self, intervals: int, kind: type = KnownCycle) -> float:
        point = _traced(intervals, kind)
        true = _true_error(point.multipliers)
        assert point.floquet_error >= true, (intervals, point.floquet_error, true)
        return point.floquet_error / true

    def test_the_over_read_is_what_order_four_predicts(self):
        assert 1.8 < self._check(80) < 1.95

    def test_above_the_old_floor(self):
        assert self._check(20) < 4.0

    def test_below_the_old_floor(self):
        for intervals in (40, 80):
            assert self._check(intervals) < 4.0

    def test_below_the_old_floor_by_central_differences(self):
        assert self._check(80, OpaqueKnownCycle) < 4.0


class _Cubic(DeterministicFunction):
    """``(x^3, x y)`` through ``math.fsum``: differenced, never differentiated."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x, y = float(point[0]), float(point[1])
        return [math.fsum([x * x * x]), math.fsum([x * y])]


class TestFallbackStep(TestCase):
    """The fallback steps relative to the state, centrally (finite_difference.py)."""

    def _orbit(self) -> PeriodicOrbit:
        field = _Cubic(
            variables=[State(), State()],
            parameters=[],
            results=[State(), State()],
            time=None,
        )
        return PeriodicOrbit(field, 4)

    def test_a_large_state_keeps_its_digits(self):
        # An absolute step of 1e-7 at x = 1e6 is under one ulp of x: the
        # difference reads zero. The relative central step keeps 1e-9.
        state = np.array([1.0e6, 3.0])
        exact = np.array([[3.0e12, 0.0], [3.0, 1.0e6]])
        jacobian = self._orbit()._jacobian(state)  # noqa: SLF001
        assert np.max(np.abs(jacobian - exact) / np.abs(exact).max(axis=1)[:, None]) < (
            1e-9
        )

    def test_the_central_difference_is_second_order(self):
        # A forward difference errs by h f''/2 = 3 h x: 3e-7 at x = 1. The
        # central one by h^2 f'''/6 = h^2, about 4e-11 at h = eps^(1/3).
        state = np.array([1.0, 2.0])
        jacobian = self._orbit()._jacobian(state)  # noqa: SLF001
        assert abs(jacobian[0, 0] - 3.0) < 1e-9
        assert abs(jacobian[1, 1] - 1.0) < 1e-9

    def test_every_row_of_a_batch_is_the_fallback(self):
        states = np.array([[1.0, 2.0], [-2.0, 0.5], [1.0e6, 3.0]])
        orbit = self._orbit()
        batch = orbit._jacobians(states)  # noqa: SLF001
        for row, state in zip(batch, states, strict=True):
            single = orbit._jacobian(state)  # noqa: SLF001
            np.testing.assert_allclose(row, single, rtol=1e-12)


class TestExactJacobianPaths(TestCase):
    """Where the field allows it the Jacobian is the analytic one, to rounding."""

    def _analytic(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        radial = 1.0 - x * x - y * y
        return np.array(
            [
                [_RATE * (radial - 2.0 * x * x), -2.0 * _RATE * x * y - _SPEED, 0.0],
                [-2.0 * _RATE * x * y + _SPEED, _RATE * (radial - 2.0 * y * y), 0.0],
                [_COUPLING * z, 0.0, _COUPLING * x - _DECAY + 2.0 * _CURVATURE * z],
            ],
        )

    def test_one_state(self):
        state = np.array([0.3, -0.7, 0.2])
        jacobian = PeriodicOrbit(_field(), 4)._jacobian(state)  # noqa: SLF001
        np.testing.assert_allclose(jacobian, self._analytic(state), atol=1e-13)

    def test_a_batch(self):
        states = np.random.default_rng(3).standard_normal((7, 3))
        batch = PeriodicOrbit(_field(), 4)._jacobians(states)  # noqa: SLF001
        for row, state in zip(batch, states, strict=True):
            np.testing.assert_allclose(row, self._analytic(state), atol=1e-12)
