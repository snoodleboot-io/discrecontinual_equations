"""A trace restarts from a point it traced, and a seed that fails says so (DEQ-106).

Rossler's period-2 cycle crosses the section ``y = 0`` upward twice a period,
so the point phase condition has two solutions on it. Seeded at one crossing,
the trapezoidal stage of the seed solve could carry the cycle to the other:
the same cycle, period and multipliers, but every node half a turn away. On
Gauss-Legendre with 40 intervals at ``c = 2.85`` the restart came back 1.44
off. The target scheme's own Newton from the seed stays at the seed's crossing.
"""

import time
import warnings
from typing import ClassVar
from unittest import TestCase

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CycleSeed,
    CycleSeedWarning,
)
from discrecontinual_equations.continuation.period_doubling import (
    CascadeLevel,
    PeriodDoublingCascade,
)
from tests.continuation.test_period_doubling import _attractor_seed, _rossler

_ROSSLER = 0.2
_GAUSS_POINTS = 4
# A traced point is converged to a residual of 1e-10; a restart that reproduces
# it lands within a few times that of its states.
_REPRODUCED = 1.0e-8


def _rhs(c: float):
    def rhs(_time: float, u: np.ndarray) -> list[float]:
        return [-u[1] - u[2], u[0] + _ROSSLER * u[1], _ROSSLER + u[2] * (u[0] - c)]

    return rhs


def _section(_time: float, u: np.ndarray) -> float:
    return float(u[1])


_section.direction = 1.0  # type: ignore[attr-defined]


def _seed_at_crossing(c: float, intervals: int, turns: int, offset: int) -> CycleSeed:
    """``turns`` turns of the attractor from its ``offset``-th last crossing."""
    settled = solve_ivp(
        _rhs(c),
        (0.0, 400.0),
        [1.0, 1.0, 0.0],
        rtol=1e-10,
        atol=1e-12,
        events=_section,
    )
    first = -1 - turns - offset
    start, end = settled.t_events[0][first], settled.t_events[0][-1 - offset]
    run = solve_ivp(
        _rhs(c),
        (0.0, end - start),
        settled.y_events[0][first],
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
    )
    times = np.linspace(0.0, end - start, intervals + 1)
    return CycleSeed(run.sol(times).T, float(end - start), c)


def _continuation(c: float, intervals: int, scheme: str) -> CycleContinuation:
    return CycleContinuation(_rossler(c), 0, intervals, 1, 0.0, collocation=scheme)


def _gap(first, second) -> float:
    states = np.max(np.abs(first.solution.states - second.solution.states))
    return float(max(states, abs(first.solution.period - second.solution.period)))


class TestRestartFromAPoint(TestCase):
    """``CycleSeed.from_point`` carries every state, and the trace starts there."""

    def _restart(self, c: float, intervals: int, scheme: str, offset: int) -> float:
        seed = _seed_at_crossing(c, intervals, 2, offset)
        (point,), _ = _continuation(c, intervals, scheme).trace(seed, 0.01, 0)
        (again,), _ = _continuation(c, intervals, scheme).trace(
            CycleSeed.from_point(point),
            0.01,
            0,
        )
        return _gap(again, point)

    def test_gauss_legendre_keeps_the_interior_points(self):
        # The full states have 4 * 40 + 1 rows: they used to reach the
        # trapezoidal stage as if they were 41 and raise an IndexError.
        assert self._restart(2.85, 40, "gauss_legendre", 1) < _REPRODUCED

    def test_hermite_simpson(self):
        assert self._restart(2.85, 40, "hermite_simpson", 1) < _REPRODUCED

    def test_a_converged_point_is_used_as_it_is(self):
        seed = _seed_at_crossing(2.85, 40, 2, 1)
        (point,), _ = _continuation(2.85, 40, "gauss_legendre").trace(seed, 0.01, 0)
        (again,), _ = _continuation(2.85, 40, "gauss_legendre").trace(
            CycleSeed.from_point(point),
            0.01,
            0,
        )
        np.testing.assert_array_equal(again.solution.states, point.solution.states)

    def test_the_seed_carries_the_point(self):
        seed = _seed_at_crossing(2.85, 20, 2, 1)
        (point,), _ = _continuation(2.85, 20, "gauss_legendre").trace(seed, 0.01, 0)
        restart = CycleSeed.from_point(point)
        assert restart.states.shape == (_GAUSS_POINTS * 20 + 1, 3)
        assert restart.period == point.solution.period
        assert restart.parameter == point.parameter


class TestRestartFromMeshNodes(TestCase):
    """Only the mesh states of a Gauss-Legendre point: the phase is kept too.

    These are the cases where the trapezoidal stage moved the cycle to its
    other crossing of the section: 0.46, 1.44 and 1.03 off.
    """

    def _restart(self, c: float, intervals: int) -> float:
        seed = _seed_at_crossing(c, intervals, 2, 1)
        (point,), _ = _continuation(c, intervals, "gauss_legendre").trace(seed, 0.01, 0)
        mesh = CycleSeed(
            point.solution.states[::_GAUSS_POINTS],
            point.solution.period,
            c,
        )
        (again,), _ = _continuation(c, intervals, "gauss_legendre").trace(mesh, 0.01, 0)
        return _gap(again, point)

    def test_close_to_the_doubling(self):
        assert self._restart(2.84, 40) < _REPRODUCED

    def test_further_past_it(self):
        assert self._restart(2.85, 40) < _REPRODUCED

    def test_on_a_coarse_mesh(self):
        assert self._restart(2.873, 20) < _REPRODUCED


class TestFailingSeed(TestCase):
    """A seed nothing converges from is refused quickly, with a warning."""

    def _hopeless(self, intervals: int) -> CycleSeed:
        good = _attractor_seed(2.7, intervals)
        centre = good.states.mean(axis=0)
        return CycleSeed(centre + 0.1 * (good.states - centre), 2.0 * good.period, 2.7)

    def test_refused_in_well_under_a_second_with_a_warning(self):
        continuation = _continuation(2.7, 80, "gauss_legendre")
        start = time.perf_counter()
        with pytest.warns(CycleSeedWarning, match="no cycle"):
            points, bifurcations = continuation.trace(self._hopeless(80), 0.01, 0)
        assert time.perf_counter() - start < 2.0
        assert points == []
        assert bifurcations == []

    def test_a_seed_that_converges_warns_nothing(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", CycleSeedWarning)
            points, _ = _continuation(2.7, 20, "hermite_simpson").trace(
                _attractor_seed(2.7, 20),
                0.01,
                0,
            )
        assert len(points) == 1

    def test_a_seed_of_the_wrong_length_is_refused_by_name(self):
        seed = _attractor_seed(2.7, 30)
        with pytest.raises(ValueError, match="31 states"):
            _continuation(2.7, 20, "gauss_legendre").trace(seed, 0.01, 0)


@pytest.mark.slow  # the whole four-level cascade: about 7 s, then 45 restarts
class TestRestartAnywhereOnTheCascade(TestCase):
    """Every point of Rossler's four-branch cascade restarts to itself (DEQ-106).

    The cascade page's settings: 20 Gauss-Legendre intervals a turn, doubling
    with the period to 160. Before DEQ-106 the full states raised, and from the
    mesh states one point of 45 - the first past the first doubling - came back
    on the other crossing, 2.18 off.
    """

    levels: ClassVar[list[CascadeLevel]]
    cascade: ClassVar[PeriodDoublingCascade]

    @classmethod
    def setUpClass(cls) -> None:
        cls.cascade = PeriodDoublingCascade(_rossler(2.7), 0, 20, 1, 0.0)
        cls.levels = cls.cascade.follow(
            _attractor_seed(2.7, 20),
            1.0,
            4,
            points_per_gap=8,
        )

    def _worst(self, mesh_only: bool) -> tuple[float, float]:
        worst, slowest = 0.0, 0.0
        for index, level in enumerate(self.levels):
            for point in level.points:
                states = point.solution.states
                seed = (
                    CycleSeed(
                        states[::_GAUSS_POINTS],
                        point.solution.period,
                        point.parameter,
                    )
                    if mesh_only
                    else CycleSeed.from_point(point)
                )
                start = time.perf_counter()
                (again,), _ = self.cascade.continuation(index).trace(seed, 0.01, 0)
                slowest = max(slowest, time.perf_counter() - start)
                worst = max(worst, _gap(again, point))
        return worst, slowest

    def test_from_every_point(self):
        worst, slowest = self._worst(mesh_only=False)
        assert worst < _REPRODUCED
        assert slowest < 5.0

    def test_from_every_point_s_mesh_states(self):
        worst, slowest = self._worst(mesh_only=True)
        assert worst < _REPRODUCED
        assert slowest < 5.0
