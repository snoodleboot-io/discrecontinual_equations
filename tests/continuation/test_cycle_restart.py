"""A trace restarts from a point it traced, and a seed that fails says so (DEQ-106).

Rossler's period-2 cycle crosses the section ``y = 0`` upward twice a period,
so the point phase condition has two solutions on it. Seeded at one crossing,
the trapezoidal stage of the seed solve could carry the cycle to the other:
the same cycle, period and multipliers, but every node half a turn away. On
Gauss-Legendre with 40 intervals at ``c = 2.85`` the restart came back 1.44
off. The target scheme's own Newton from the seed stays at the seed's crossing.
"""

import math
import time
import warnings
from typing import ClassVar
from unittest import TestCase, mock

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation import cycle_continuation
from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CycleSeed,
    CycleSeedWarning,
)
from discrecontinual_equations.continuation.period_doubling import (
    CascadeLevel,
    PeriodDoublingCascade,
)
from tests.continuation.cycle_fields import circle, fold_equation
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

    def test_refused_with_a_warning(self):
        continuation = _continuation(2.7, 80, "gauss_legendre")
        with pytest.warns(CycleSeedWarning, match="no cycle"):
            points, bifurcations = continuation.trace(self._hopeless(80), 0.01, 0)
        assert points == []
        assert bifurcations == []

    @pytest.mark.timing
    def test_refused_in_well_under_a_second(self):
        # 0.15-0.5 s measured from 20 to 320 intervals; the 2 s is headroom
        # for a loaded machine, not a claim.
        continuation = _continuation(2.7, 320, "gauss_legendre")
        seed = self._hopeless(320)
        start = time.perf_counter()
        with pytest.warns(CycleSeedWarning):
            continuation.trace(seed, 0.01, 0)
        assert time.perf_counter() - start < 2.0

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


class TestRestartGuards(TestCase):
    """The restart takes only a seed that is nearly a solution.

    The fold of cycles at ``mu = -0.24`` has cycles of radius 0.632 and 0.775.
    From circles of radius 0.64 and 0.78 the restart lands on the nearer in
    three or four solves; from 0.68 and 0.70, between them, the first step does
    not halve the residual and it stops; from 0.9 Newton reaches the outer cycle,
    but 14% of the seed's extent away, and the staged solve is left to decide.
    Radii are pinned to 2e-6: the 40-interval Hermite-Simpson cycles sit
    6.7e-7 and 8.2e-7 off the exact ones.
    """

    def _restart(
        self,
        radius: float,
        period: float = 2.0 * math.pi,
    ) -> tuple[np.ndarray | None, int]:
        continuation = CycleContinuation(fold_equation(-0.24), 0, 40, 1, 0.0)
        continuation._dimension = 2  # noqa: SLF001 (trace sets it)
        continuation._mesh = np.linspace(0.0, 1.0, 41)  # noqa: SLF001
        seed = CycleSeed(circle(radius, 40), period, -0.24)
        solve = CycleContinuation._solve_bordered  # noqa: SLF001
        with mock.patch.object(
            CycleContinuation,
            "_solve_bordered",
            autospec=True,
            side_effect=solve,
        ) as spy:
            unknowns = continuation._restarted(seed, seed.states, 41)  # noqa: SLF001
        return unknowns, spy.call_count

    def _radius(self, unknowns: np.ndarray) -> float:
        states = unknowns[:82].reshape(41, 2)
        return float(np.max(np.hypot(states[:, 0], states[:, 1])))

    def test_a_near_seed_lands_on_the_nearer_cycle(self):
        inner, _ = self._restart(0.64)
        outer, _ = self._restart(0.78)
        assert inner is not None
        assert outer is not None
        assert abs(self._radius(inner) - math.sqrt(0.4)) < 2e-6
        assert abs(self._radius(outer) - math.sqrt(0.6)) < 2e-6

    def test_a_step_that_does_not_halve_the_residual_stops_it(self):
        # From 0.68 the first step lowers the residual to 0.81 of itself:
        # less than the half asked for, so the second solve is never made.
        # From 0.70 the first step raises it 4.7 times.
        for radius in (0.68, 0.70):
            unknowns, solves = self._restart(radius)
            assert unknowns is None
            assert solves == 1, radius

    def test_a_cycle_too_far_from_the_seed_is_left_to_the_staged_solve(self):
        unknowns, _ = self._restart(0.9)
        assert unknowns is None
        with mock.patch.object(
            cycle_continuation,
            "_RESTART_MOVE",
            1.0e9,
        ):
            reached, _ = self._restart(0.9)
        assert reached is not None
        assert abs(self._radius(reached) - math.sqrt(0.6)) < 2e-6

    def test_a_period_more_than_twice_off_is_refused(self):
        # From the outer cycle's own states with its period scaled, Newton
        # reaches the cycle at 2 pi every time; the seed's period window
        # (a factor of two either way) decides whether that is the cycle meant.
        outer = math.sqrt(0.6)
        kept, _ = self._restart(outer, 1.9 * 2.0 * math.pi)
        assert kept is not None
        for factor in (0.45, 2.5):
            refused, _ = self._restart(outer, factor * 2.0 * math.pi)
            assert refused is None, factor

    def test_a_singular_jacobian_ends_it_without_a_dense_solve(self):
        # A restart never pays for the dense least-squares fallback: a seed
        # whose sparse LU fails is handed to the staged solve instead.
        with (
            mock.patch.object(
                cycle_continuation,
                "splu",
                side_effect=RuntimeError("singular"),
            ),
            mock.patch.object(
                cycle_continuation.np.linalg,
                "lstsq",
                wraps=np.linalg.lstsq,
            ) as least_squares,
        ):
            unknowns, solves = self._restart(0.78)
        assert unknowns is None
        assert solves == 1
        least_squares.assert_not_called()


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

    def _worst(self, *, mesh_only: bool) -> tuple[float, float]:
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
