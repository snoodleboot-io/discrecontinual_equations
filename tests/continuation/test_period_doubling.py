"""Following a period-doubling cascade: locating each doubling, switching, and delta.

The acceptance oracles of DEQ-42. The Rossler system at ``a = b = 0.2`` has its
first period-doubling at ``c = 2.832445`` - Wilczak and Zgliczynski prove it lies
in ``[2.83244, 2.832446]`` (arXiv:0712.1123, section 6.3, their ``a`` being this
``c``) - and the second in ``[3.83735812, 3.837358168411]`` (section 6.4). The
deeper doublings are checked against an independent reference: Poincare-map
shooting on the section ``y = 0`` with scipy's DOP853 at ``rtol = 1e-12``, the
period-``2^k`` point a fixed point of the ``2^k``-th return map and each doubling
the root, by brentq, of its flip eigenvalue plus one. Nothing in it uses the
collocation under test.
"""

import math
from typing import ClassVar
from unittest import TestCase

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.cycle_continuation import (
    CyclePoint,
    CycleSeed,
)
from discrecontinual_equations.continuation.period_doubling import (
    CascadeLevel,
    PeriodDoublingCascade,
    _first_doubling,
    _halves_differ,
    feigenbaum_ratios,
    flip_multiplier,
    locate_doubling,
)
from discrecontinual_equations.continuation.periodic_orbit import (
    PeriodicOrbitSolution,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import Alpha, State, Time

FEIGENBAUM_DELTA = 4.669201609102990671853
# Wilczak and Zgliczynski's enclosures of the first two doublings.
FIRST_DOUBLING = (2.83244, 2.832446)
SECOND_DOUBLING = (3.83735812, 3.837358168411)
# The independent reference's doublings (Poincare shooting, see the module).
REFERENCE_DOUBLINGS = (
    2.8324450278621,
    3.8373581684065,
    4.1242147320743,
    4.1869969993951,
    4.2005372929837,
    4.2034406286571,
)
REFERENCE_RATIOS = (3.5031903, 4.5690698, 4.6366991, 4.6637024)
_FLOQUET_TOLERANCE = 2.0e-2
_ROSSLER = 0.2


class Rossler(DeterministicFunction):
    """x' = -y - z, y' = x + a y, z' = b + z (x - c), a = b = 0.2, c the parameter."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        c = self.parameters[0].value
        x, y, z = point[0], point[1], point[2]
        return [-y - z, x + _ROSSLER * y, _ROSSLER + z * (x - c)]


def _rossler(value: float) -> DifferentialEquation:
    parameters = [Alpha(value=value)]
    return DifferentialEquation(
        variables=[State() for _ in range(3)],
        time=Time(),
        parameters=parameters,
        derivative=Rossler(
            variables=[State() for _ in range(3)],
            parameters=parameters,
            results=[State() for _ in range(3)],
            time=None,
        ),
    )


def _attractor_seed(c: float, intervals: int, turns: int = 1) -> CycleSeed:
    """``turns`` turns of the attracting cycle, from y = 0 upward, by DOP853."""

    def rhs(_time: float, u: np.ndarray) -> list[float]:
        return [-u[1] - u[2], u[0] + _ROSSLER * u[1], _ROSSLER + u[2] * (u[0] - c)]

    def section(_time: float, u: np.ndarray) -> float:
        return float(u[1])

    section.direction = 1.0  # type: ignore[attr-defined]
    settled = solve_ivp(
        rhs, (0.0, 400.0), [1.0, 1.0, 0.0], rtol=1e-10, atol=1e-12, events=section
    )
    start, end = settled.t_events[0][-1 - turns], settled.t_events[0][-1]
    turn = solve_ivp(
        rhs,
        (0.0, end - start),
        settled.y_events[0][-1 - turns],
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
    )
    times = np.linspace(0.0, end - start, intervals + 1)
    return CycleSeed(turn.sol(times).T, float(end - start), c)


def _point(parameter: float, multipliers: list[complex]) -> CyclePoint:
    angle = np.linspace(0.0, 2.0 * math.pi, 9)
    states = np.column_stack([np.cos(angle), np.sin(angle)])
    return CyclePoint(
        parameter,
        PeriodicOrbitSolution(states, 2.0 * math.pi),
        np.array(multipliers, dtype=complex),
    )


class TestFeigenbaumRatios(TestCase):
    def test_a_geometric_cascade_gives_its_ratio(self):
        doublings = [3.0 - 2.0 * FEIGENBAUM_DELTA**-k for k in range(6)]
        ratios = feigenbaum_ratios(doublings)
        assert len(ratios) == 4
        np.testing.assert_allclose(ratios, FEIGENBAUM_DELTA, rtol=1e-12)

    def test_the_ratio_is_earlier_gap_over_later(self):
        assert feigenbaum_ratios([0.0, 4.0, 5.0]) == [4.0]


class TestFlipMultiplier(TestCase):
    def test_picks_the_real_negative_multiplier_nearest_minus_one(self):
        point = _point(0.0, [1.0, -0.3, -0.95, 1e-6])
        assert flip_multiplier(point) == -0.95

    def test_of_two_on_the_negative_axis_the_one_nearer_minus_one(self):
        point = _point(0.0, [1.0, -7.0, -1.05])
        assert flip_multiplier(point) == -1.05

    def test_a_complex_pair_is_not_a_flip(self):
        point = _point(0.0, [1.0, -0.9 + 0.2j, -0.9 - 0.2j])
        assert math.isnan(flip_multiplier(point))


class TestLocateDoubling(TestCase):
    """Inverse interpolation against a flip multiplier with a known root."""

    @staticmethod
    def _flip(c: float) -> float:
        offset = c - 1.3
        return -1.0 - offset - 0.8 * offset**2 + 0.5 * offset**3

    def _points(self, spacing: float) -> list[CyclePoint]:
        values = 1.3 - 3.37 * spacing + spacing * np.arange(8)
        return [_point(c, [1.0, self._flip(c), 1e-5]) for c in values]

    def test_fourth_order_in_the_spacing(self):
        errors = [abs(locate_doubling(self._points(h)) - 1.3) for h in (0.04, 0.02)]
        assert errors[0] < 1e-5
        # A cubic through four points: 2^4 per halving, less a little.
        assert errors[0] / errors[1] > 12.0

    def test_beats_the_linear_estimate(self):
        points = self._points(0.04)
        crossing = next(
            i
            for i in range(len(points) - 1)
            if (self._flip(points[i].parameter) + 1.0)
            * (self._flip(points[i + 1].parameter) + 1.0)
            <= 0.0
        )
        lower, upper = points[crossing], points[crossing + 1]
        before = -self._flip(lower.parameter) - 1.0
        after = -self._flip(upper.parameter) - 1.0
        linear = lower.parameter + before / (before - after) * (
            upper.parameter - lower.parameter
        )
        assert abs(locate_doubling(points) - 1.3) < 0.05 * abs(linear - 1.3)

    def test_none_without_a_crossing(self):
        points = [_point(c, [1.0, -0.5 - 0.1 * c]) for c in range(4)]
        assert locate_doubling(points) is None


class TestHalvesDiffer(TestCase):
    def test_an_old_cycle_twice_is_not_new(self):
        angle = np.linspace(0.0, 4.0 * math.pi, 17)
        states = np.column_stack([np.cos(angle), np.sin(angle)])
        point = CyclePoint(0.0, PeriodicOrbitSolution(states, 4.0), np.ones(2))
        assert not _halves_differ(point, 1.0)

    def test_a_doubled_cycle_is_new(self):
        angle = np.linspace(0.0, 4.0 * math.pi, 17)
        states = np.column_stack(
            [np.cos(angle) + 0.1 * np.cos(0.5 * angle), np.sin(angle)],
        )
        point = CyclePoint(0.0, PeriodicOrbitSolution(states, 4.0), np.ones(2))
        assert _halves_differ(point, 1.0)


class TestRosslerCascade(TestCase):
    """Rossler's first four doublings: located, switched between, and delta.

    Twenty intervals per turn under Gauss-Legendre collocation, the mesh
    doubling with the period to 160 on the period-8 branch, and eight points
    per predicted gap - the cascade page's settings. Measured against the
    reference this puts the four doublings 6.3e-8, 2.2e-7, 4.5e-8 and 1.5e-7
    off and the two ratios within 5e-6 of the reference's.
    """

    levels: ClassVar[list[CascadeLevel]]

    @classmethod
    def setUpClass(cls) -> None:
        cascade = PeriodDoublingCascade(_rossler(2.7), 0, 20, 1, 0.0)
        seed = _attractor_seed(2.7, 20)
        cls.levels = cascade.follow(seed, 1.0, 4, points_per_gap=8)

    def _doublings(self) -> list[float]:
        return [float(level.doubling) for level in self.levels]  # type: ignore[arg-type]

    def test_four_branches_each_doubled(self):
        assert [level.multiple for level in self.levels] == [1, 2, 4, 8]
        assert all(level.doubling is not None for level in self.levels)

    def test_the_first_doubling_lies_in_the_proven_enclosure(self):
        first = self.levels[0].doubling
        assert FIRST_DOUBLING[0] <= first <= FIRST_DOUBLING[1]
        assert abs(first - REFERENCE_DOUBLINGS[0]) < 5.0e-7

    def test_every_doubling_agrees_with_the_reference(self):
        np.testing.assert_allclose(
            self._doublings(),
            REFERENCE_DOUBLINGS[:4],
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_the_ratios_agree_with_the_reference_and_approach_delta(self):
        ratios = feigenbaum_ratios(self._doublings())
        np.testing.assert_allclose(ratios, REFERENCE_RATIOS[:2], rtol=0.0, atol=1e-4)
        errors = [abs(ratio - FEIGENBAUM_DELTA) for ratio in ratios]
        assert errors[1] < errors[0]

    def test_the_mesh_doubles_with_the_period(self):
        for coarse, fine in zip(self.levels, self.levels[1:], strict=False):
            assert fine.intervals == 2 * coarse.intervals
            nodes = fine.points[0].solution.states.shape[0] - 1
            assert nodes == fine.intervals * 4
            period = coarse.points[-1].solution.period
            assert abs(fine.points[0].solution.period / period - 2.0) < 0.05

    def test_each_switch_lands_on_a_new_cycle_past_the_doubling(self):
        for coarse, fine in zip(self.levels, self.levels[1:], strict=False):
            start = fine.points[0]
            assert _halves_differ(start, start.amplitude)
            assert start.parameter > coarse.doubling

    def test_every_point_is_resolved(self):
        for level in self.levels:
            resolved = level.resolved()
            assert resolved.refused == 0
            assert len(resolved.points) == len(level.points)

    def test_the_flip_crosses_minus_one_with_the_trivial_at_one(self):
        for level in self.levels:
            crossed = _first_doubling(level.points)
            assert crossed is not None
            pair = level.points[crossed : crossed + 2]
            for point in pair:
                trivial = point.multipliers[np.argmin(np.abs(point.multipliers - 1.0))]
                assert abs(trivial - 1.0) < _FLOQUET_TOLERANCE
                assert point.resolved()
            before, after = (flip_multiplier(point) for point in pair)
            assert before > -1.0 > after


class TestSecondDoubling(TestCase):
    """The second doubling to the collocation's accuracy, from a period-2 seed.

    Wilczak and Zgliczynski's enclosure is ``[3.83735812, 3.837358168411]`` and
    the reference sits 4.5e-12 inside its upper end; on eighty intervals for the
    two turns this lands 6.8e-10 above the reference, so just outside it. On
    forty it was 2.5e-7 above: the collocation, not the interpolation, is the
    limit there, and with ``m = 4`` it falls by about 2^8 per mesh doubling.
    """

    def test_agrees_with_the_reference_to_a_few_parts_in_ten_billion(self):
        cascade = PeriodDoublingCascade(_rossler(3.8), 0, 80, 1, 0.0)
        seed = _attractor_seed(3.8, 80, turns=2)
        (level,) = cascade.follow(seed, 0.25, 1)
        assert abs(level.doubling - REFERENCE_DOUBLINGS[1]) < 5.0e-9
