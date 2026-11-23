"""The periodic-orbit solvers and the cycle seed cost O(N) per Newton step (DEQ-49).

Each Newton step used to finite-difference every unknown against the whole
residual - ``O(N^2)`` field calls - and solve the dense result. The Jacobian is
now differenced a column group at a time (Curtis, Powell and Reid) and solved
sparse. The checks here are against a dense difference and a dense Newton
written out below, not against the code under test.
"""

import math
from unittest import TestCase

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation import periodic_orbit
from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CycleSeed,
)
from discrecontinual_equations.continuation.periodic_orbit import (
    HermiteSimpsonOrbit,
    PeriodicOrbit,
    RobustPeriodicOrbit,
    _column_groups,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from tests.continuation.fields import Alpha, State, Time, VanDerPolField

_SOLVERS = (PeriodicOrbit, HermiteSimpsonOrbit, RobustPeriodicOrbit)


class _CountedVanDerPol(VanDerPolField):
    """Van der Pol, counting its own evaluations."""

    calls = 0

    def eval(
        self,
        point: list[float],
        time: float | None = None,
    ) -> list[float]:
        type(self).calls += 1
        return super().eval(point, time)


def _equation(mu: float) -> DifferentialEquation:
    parameters = [Alpha(value=mu)]
    return DifferentialEquation(
        variables=[State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=_CountedVanDerPol(
            variables=[State(), State()],
            parameters=parameters,
            results=[State(), State()],
            time=None,
        ),
    )


def _reference(mu: float):
    """Van der Pol's cycle by DOP853: its period, and the orbit on ``[0, 1]``."""

    def field(_t, u):
        return [u[1], mu * (1.0 - u[0] ** 2) * u[1] - u[0]]

    settled = solve_ivp(field, [0.0, 100.0], [2.0, 0.0], rtol=1e-12, atol=1e-12)

    def section(_t, u):
        return u[1]

    section.direction = -1.0
    run = solve_ivp(
        field,
        [0.0, 60.0],
        settled.y[:, -1],
        method="DOP853",
        rtol=1e-12,
        atol=1e-12,
        events=section,
        dense_output=True,
    )
    start, end = run.t_events[0][0], run.t_events[0][1]
    return end - start, lambda s: run.sol(start + s * (end - start)).T


def _dense_difference(orbit: PeriodicOrbit, unknowns, nodes, dimension):
    """Every unknown perturbed alone: the Jacobian the solvers used to build."""
    residual = orbit._residual(unknowns, nodes, dimension)  # noqa: SLF001
    jacobian = np.empty((residual.size, unknowns.size))
    for j in range(unknowns.size):
        shifted = unknowns.copy()
        shifted[j] += periodic_orbit._STEP  # noqa: SLF001
        jacobian[:, j] = (
            orbit._residual(shifted, nodes, dimension) - residual  # noqa: SLF001
        ) / periodic_orbit._STEP  # noqa: SLF001
    return residual, jacobian


class TestGroupedDifference(TestCase):
    def setUp(self):
        self.period, self.orbit = _reference(1.0)

    def test_matches_the_dense_difference_bit_for_bit(self):
        """Grouping changes how many residuals are taken, not one Jacobian entry."""
        rng = np.random.default_rng(3)
        intervals = 12
        for kind in _SOLVERS:
            orbit = kind(_equation(1.0).derivative, intervals, phase_index=1)
            states = self.orbit(np.linspace(0.0, 1.0, intervals + 1))
            states = states + 1e-2 * rng.standard_normal(states.shape)
            unknowns = np.concatenate([states.ravel(), [self.period * 1.01]])
            residual, dense = _dense_difference(orbit, unknowns, intervals + 1, 2)
            nodes = intervals + 1
            grouped = orbit._difference_jacobian(  # noqa: SLF001
                unknowns,
                nodes,
                2,
                residual,
            )
            np.testing.assert_array_equal(grouped.toarray(), dense, kind.__name__)

    def test_groups_never_share_a_row(self):
        """The property the exactness rests on, checked on an irregular pattern."""
        rng = np.random.default_rng(5)
        dense = rng.random((30, 25)) < 0.15
        rows, columns = np.nonzero(dense)
        groups = _column_groups(rows, columns, 25)
        for group in np.unique(groups):
            assert dense[:, groups == group].sum(axis=1).max() <= 1

    def test_a_handful_of_groups_whatever_the_mesh(self):
        """About three per state component and one for the period, at any N."""
        for intervals in (10, 40, 160):
            pattern = periodic_orbit._difference_pattern  # noqa: SLF001
            _, _, groups = pattern(intervals + 1, 2, 1)
            assert groups.max() + 1 <= 8


class TestSingularJacobian(TestCase):
    def test_falls_back_to_least_squares_on_the_same_matrix(self):
        """At an equilibrium the period column is zero and the LU has no answer.

        Every node on van der Pol's origin makes the field, and so the period's
        column, vanish. The sparse LU returns ``nan`` there; the update must
        still be the dense least-squares one, finite, as before DEQ-49.
        """
        intervals = 10
        orbit = PeriodicOrbit(
            _equation(1.0).derivative,
            intervals,
            phase_index=1,
            phase_value=0.5,
        )
        unknowns = np.concatenate([np.zeros(2 * (intervals + 1)), [6.0]])
        residual, dense = _dense_difference(orbit, unknowns, intervals + 1, 2)
        update = orbit._newton_update(  # noqa: SLF001
            unknowns,
            intervals + 1,
            2,
            residual,
        )
        expected, *_ = np.linalg.lstsq(dense, -residual, rcond=None)
        assert np.all(np.isfinite(update))
        assert np.linalg.norm(update) > 0.1
        np.testing.assert_allclose(update, expected, atol=1e-12)


class TestNewtonCost(TestCase):
    """Field calls per Newton iteration grow as N, not N^2, from 40 to 320."""

    def setUp(self):
        self.period, self.orbit = _reference(1.0)
        self.saved = periodic_orbit._ITERATIONS  # noqa: SLF001

    def tearDown(self):
        periodic_orbit._ITERATIONS = self.saved  # noqa: SLF001

    def _calls(self, kind, intervals: int) -> int:
        periodic_orbit._ITERATIONS = 1  # noqa: SLF001
        orbit = kind(_equation(1.0).derivative, intervals, phase_index=1)
        seed = 1.01 * self.orbit(np.linspace(0.0, 1.0, intervals + 1))
        _CountedVanDerPol.calls = 0
        orbit.solve(seed, self.period)
        return _CountedVanDerPol.calls

    def test_field_calls_grow_linearly(self):
        for kind in _SOLVERS:
            ratio = self._calls(kind, 320) / self._calls(kind, 40)
            # Linear is 8; the dense difference was 61 to 64.
            assert ratio < 9.0, (kind.__name__, ratio)


class TestSolutionsUnchanged(TestCase):
    """The sparse solve converges to the dense Newton's cycle, to its tolerance."""

    def test_matches_a_dense_newton(self):
        period, orbit_at = _reference(1.0)
        intervals = 40
        seed = 1.02 * orbit_at(np.linspace(0.0, 1.0, intervals + 1))
        for kind in (PeriodicOrbit, HermiteSimpsonOrbit):
            orbit = kind(_equation(1.0).derivative, intervals, phase_index=1)
            unknowns = np.concatenate([seed.ravel(), [period]])
            for _ in range(periodic_orbit._ITERATIONS):  # noqa: SLF001
                residual, jacobian = _dense_difference(
                    orbit,
                    unknowns,
                    intervals + 1,
                    2,
                )
                if np.linalg.norm(residual) < periodic_orbit._TOLERANCE:  # noqa: SLF001
                    break
                update, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
                unknowns = unknowns + update
            solution = orbit.solve(seed, period)
            assert solution is not None
            states = unknowns[:-1].reshape(-1, 2)
            assert np.max(np.abs(solution.states - states)) < 1e-9, kind.__name__
            assert abs(solution.period - unknowns[-1]) < 1e-9, kind.__name__

    def test_robust_solver_matches_the_plain_one(self):
        period, orbit_at = _reference(1.0)
        seed = 1.02 * orbit_at(np.linspace(0.0, 1.0, 41))
        plain = PeriodicOrbit(_equation(1.0).derivative, 40, phase_index=1)
        robust = RobustPeriodicOrbit(_equation(1.0).derivative, 40, phase_index=1)
        first = plain.solve(seed, period)
        second = robust.solve(seed, period)
        assert first is not None
        assert second is not None
        assert abs(first.period - second.period) < 1e-9


class TestSeedCost(TestCase):
    """The cycle seed's field calls grow as N, and it costs no more than a step."""

    def test_seed_field_calls_grow_linearly(self):
        period, orbit_at = _reference(1.0)
        counts = []
        for intervals in (40, 320):
            continuation = CycleContinuation(
                _equation(1.0),
                0,
                intervals,
                phase_index=1,
            )
            continuation._dimension = 2  # noqa: SLF001
            seed = CycleSeed(
                orbit_at(np.linspace(0.0, 1.0, intervals + 1)),
                period,
                1.0,
            )
            _CountedVanDerPol.calls = 0
            solved = continuation._solve_seed(seed, intervals + 1)  # noqa: SLF001
            assert solved is not None
            counts.append(_CountedVanDerPol.calls)
        assert counts[1] / counts[0] < 9.0, counts

    def test_seed_period_is_right(self):
        period, orbit_at = _reference(1.0)
        continuation = CycleContinuation(_equation(1.0), 0, 160, phase_index=1)
        seed = CycleSeed(orbit_at(np.linspace(0.0, 1.0, 161)), period, 1.0)
        points, _ = continuation.trace(seed, 0.05, 0)
        assert math.isclose(points[0].solution.period, period, rel_tol=1e-6)
