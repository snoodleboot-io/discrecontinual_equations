"""Stiff cycles start from the documented seed path (DEQ-54).

The documented path is :meth:`CycleContinuation.trace` from a :class:`CycleSeed`
whose states are an integrated orbit sampled on the mesh. On van der Pol at
``mu = 5`` and ``10`` under Gauss-Legendre collocation it used to return
nothing: the trapezoidal stage of the seed solve landed on another cycle, or on
none, and the scheme's own Newton could not recover. The reference is DOP853
at ``rtol = 1e-13``, independent of the code under test.
"""

import math
from unittest import TestCase

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.cycle_continuation import (
    _NOISE_FLOOR,
    CycleContinuation,
    CycleSeed,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from tests.continuation.fields import Alpha, State, Time, VanDerPolField


def _equation(mu: float) -> DifferentialEquation:
    parameters = [Alpha(value=mu)]
    return DifferentialEquation(
        variables=[State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=VanDerPolField(
            variables=[State(), State()],
            parameters=parameters,
            results=[State(), State()],
            time=None,
        ),
    )


def _reference(mu: float):
    """Van der Pol's period and orbit by DOP853 at ``rtol = 1e-13``."""

    def field(_t, u):
        return [u[1], mu * (1.0 - u[0] ** 2) * u[1] - u[0]]

    settled = solve_ivp(
        field,
        [0.0, 400.0],
        [2.0, 0.0],
        method="DOP853",
        rtol=1e-12,
        atol=1e-13,
    )

    def section(_t, u):
        return u[1]

    section.direction = -1.0
    run = solve_ivp(
        field,
        [0.0, 200.0],
        settled.y[:, -1],
        method="DOP853",
        rtol=1e-13,
        atol=1e-14,
        events=section,
        dense_output=True,
    )
    start, end = run.t_events[0][0], run.t_events[0][1]
    return end - start, lambda s: run.sol(start + s * (end - start)).T


def _start(mu: float, intervals: int, points: int, period=None):
    reference, orbit = _reference(mu)
    continuation = CycleContinuation(
        _equation(mu),
        0,
        intervals,
        phase_index=1,
        collocation="gauss_legendre",
        collocation_points=points,
    )
    seed = CycleSeed(
        orbit(np.linspace(0.0, 1.0, intervals + 1)),
        reference if period is None else period,
        mu,
    )
    points_, _ = continuation.trace(seed, 0.05, 0)
    return points_, reference


class TestStiffStart(TestCase):
    """Each case started on a collapsed or a wrong cycle, or none, before DEQ-54.

    Before: ``mu = 5`` on 32 intervals reached period 2.2e-5, ``mu = 6`` on 40
    reached -8.8e-4, and ``mu = 10`` on 80 returned no point. The bound is the
    scheme's own error on that mesh - Gauss-Legendre Newton started from the
    DOP853 orbit sampled at every collocation point lands on periods 2.4e-4,
    3.1e-4 and 1.1e-3 from the reference - rounded up, so the check says the
    start is that discrete cycle and not merely some cycle.
    """

    def _check(self, mu: float, intervals: int, bound: float):
        points, reference = _start(mu, intervals, 4)
        assert len(points) == 1, (mu, intervals)
        error = abs(points[0].solution.period - reference) / reference
        assert error < bound, (mu, intervals, error)

    def test_van_der_pol_at_mu_5(self):
        self._check(5.0, 32, 3.0e-4)

    def test_van_der_pol_at_mu_6(self):
        self._check(6.0, 40, 4.0e-4)

    def test_van_der_pol_at_mu_10(self):
        self._check(10.0, 80, 1.5e-3)


class TestRefusal(TestCase):
    def test_a_negative_seed_period_is_refused(self):
        points, _ = _start(1.0, 20, 4, period=-6.66)
        assert points == []

    def test_a_collapsed_seed_is_refused_rather_than_continued(self):
        """Van der Pol at ``mu = 2`` on 8 intervals collapsed onto a point."""
        points, _ = _start(2.0, 8, 4)
        for point in points:
            states = point.solution.states
            assert point.solution.period > 0.0
            assert np.ptp(states) > _NOISE_FLOOR


class TestNonStiffUnchanged(TestCase):
    def test_the_staged_solve_is_still_what_starts_a_mild_cycle(self):
        """Bit for bit: the staged path answers first wherever it already did."""
        reference, orbit = _reference(1.0)
        for scheme in ("trapezoidal", "hermite_simpson", "gauss_legendre"):
            continuation = CycleContinuation(
                _equation(1.0),
                0,
                40,
                phase_index=1,
                collocation=scheme,
            )
            continuation._dimension = 2  # noqa: SLF001 (trace sets it)
            seed = CycleSeed(orbit(np.linspace(0.0, 1.0, 41)), reference, 1.0)
            nodes = 40 * continuation.collocation_points + 1
            staged = continuation._staged_seed(seed, nodes)  # noqa: SLF001
            solved = continuation._solve_seed(seed, nodes)  # noqa: SLF001
            np.testing.assert_array_equal(solved, staged)
            assert math.isclose(solved[2 * nodes], reference, rel_tol=1e-2)
