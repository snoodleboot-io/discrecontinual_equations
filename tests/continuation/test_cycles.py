"""Cycle bifurcations, the SNIC, and the Floquet error a mesh carries."""

import math
from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CyclePoint,
    CycleSeed,
)
from discrecontinual_equations.continuation.periodic_orbit import (
    PeriodicOrbit,
    PeriodicOrbitSolution,
)
from discrecontinual_equations.continuation.region_analysis import (
    _jacobian,
    classify_equilibrium,
    find_equilibria,
)
from discrecontinual_equations.continuation.snic import characterize_snic
from discrecontinual_equations.differential_equation import DifferentialEquation
from tests.continuation.fields import (
    Alpha,
    FoldOfCyclesField,
    NeimarkSackerField,
    PeriodDoublingField,
    SnicField,
    State,
    Time,
)

_SADDLE_JACOBIAN = np.array([[0.0, 1.0], [1.0, 0.0]])


class TestCycleBifurcations(TestCase):
    """Fold of cycles, Neimark-Sacker, and period-doubling against oracles."""

    def _nodes(self, intervals: int) -> np.ndarray:
        return np.linspace(0.0, 1.0, intervals + 1)

    def _circle(self, radius: float, intervals: int, extra: int) -> np.ndarray:
        nodes = self._nodes(intervals)
        columns = [
            radius * np.cos(2.0 * math.pi * nodes),
            radius * np.sin(2.0 * math.pi * nodes),
        ]
        columns += [np.zeros(intervals + 1) for _ in range(extra)]
        return np.column_stack(columns)

    def _equation(self, field_type: type, dimension: int, value: float):
        parameters = [Alpha(value=value)]
        return DifferentialEquation(
            variables=[State() for _ in range(dimension)],
            time=Time(),
            parameters=parameters,
            derivative=field_type(
                variables=[State() for _ in range(dimension)],
                parameters=parameters,
                results=[State() for _ in range(dimension)],
                time=None,
            ),
        )

    def test_fold_of_cycles(self):
        intervals = 60
        mu0 = -0.1
        radius = math.sqrt((1.0 + math.sqrt(1.0 + 4.0 * mu0)) / 2.0)
        seed = CycleSeed(self._circle(radius, intervals, 0), 2.0 * math.pi, mu0)
        continuation = CycleContinuation(
            self._equation(FoldOfCyclesField, 2, mu0),
            0,
            intervals,
            phase_index=1,
            phase_value=0.0,
        )
        _points, bifurcations = continuation.trace(seed, 0.03, 120, direction=-1.0)
        folds = [b for b in bifurcations if b.kind == "fold_of_cycles"]
        assert folds
        assert any(abs(b.parameter + 0.25) < 5.0e-3 for b in folds)

    def test_neimark_sacker(self):
        intervals = 60
        seed = CycleSeed(
            self._circle(math.sqrt(0.5), intervals, 2),
            2.0 * math.pi,
            -0.1,
        )
        continuation = CycleContinuation(
            self._equation(NeimarkSackerField, 4, -0.1),
            0,
            intervals,
            phase_index=1,
            phase_value=0.0,
        )
        _points, bifurcations = continuation.trace(seed, 0.02, 12)
        torus = [b for b in bifurcations if b.kind == "neimark_sacker"]
        assert torus
        assert any(abs(b.parameter) < 5.0e-3 for b in torus)

    def test_period_doubling(self):
        intervals = 60
        seed = CycleSeed(
            self._circle(math.sqrt(0.5), intervals, 2),
            2.0 * math.pi,
            -0.1,
        )
        continuation = CycleContinuation(
            self._equation(PeriodDoublingField, 4, -0.1),
            0,
            intervals,
            phase_index=1,
            phase_value=0.0,
        )
        _points, bifurcations = continuation.trace(seed, 0.02, 12)
        flips = [b for b in bifurcations if b.kind == "period_doubling"]
        assert flips
        assert any(abs(b.parameter) < 5.0e-3 for b in flips)


class TestSnic(TestCase):
    """Saddle-node on an invariant circle: period law, detector, and structure."""

    def _snic(self, mu: float) -> DifferentialEquation:
        parameters = [Alpha(value=mu)]
        return DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=SnicField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )

    def test_period_matches_oracle(self):
        mu, intervals = 1.3, 200
        nodes = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [np.cos(2.0 * math.pi * nodes), np.sin(2.0 * math.pi * nodes)],
        )
        oracle = 2.0 * math.pi / math.sqrt(mu * mu - 1.0)
        orbit = PeriodicOrbit(
            self._snic(mu).derivative,
            intervals,
            phase_index=1,
            phase_value=0.0,
        )
        solution = orbit.solve(seed, oracle * 1.02)
        assert solution is not None
        assert abs(solution.period - oracle) / oracle < 1.0e-3

    def test_detector_identifies_inverse_square_root(self):
        mus = [1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.15, 1.20]
        periods = [2.0 * math.pi / math.sqrt(m * m - 1.0) for m in mus]
        result = characterize_snic(mus, periods)
        assert result.is_inverse_square_root
        assert abs(result.critical_parameter - 1.0) < 1.0e-2
        assert abs(result.exponent + 0.5) < 0.08

    def test_detector_rejects_logarithmic_divergence(self):
        mus = [1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.15, 1.20]
        periods = [-3.0 * math.log(m - 1.0) for m in mus]
        assert not characterize_snic(mus, periods).is_inverse_square_root

    def test_saddle_node_lies_on_the_circle(self):
        seeds = [
            np.array([0.6, 0.8]),
            np.array([-0.6, 0.8]),
            np.array([0.6, -0.8]),
        ]
        equilibria = find_equilibria(self._snic(0.8).derivative, seeds)
        assert len(equilibria) == 2
        labels = sorted(
            classify_equilibrium(_jacobian(self._snic(0.8).derivative, point))
            for point in equilibria
        )
        assert labels == ["saddle", "stable node"]
        for point in equilibria:
            assert abs(float(np.hypot(point[0], point[1])) - 1.0) < 1.0e-6


class TestFloquetError(TestCase):
    """The trivial multiplier's drift from one is the accuracy of the whole set."""

    def test_reports_the_multiplier_nearest_one(self):
        point = CyclePoint(
            0.0,
            PeriodicOrbitSolution(np.zeros((3, 2)), 6.0),
            np.array([0.911, 6.596]),
        )
        assert abs(point.floquet_error - 0.089) < 1.0e-12

    def test_is_zero_for_an_exact_trivial_multiplier(self):
        point = CyclePoint(
            0.0,
            PeriodicOrbitSolution(np.zeros((3, 2)), 6.0),
            np.array([1.0, 0.5 + 0.5j]),
        )
        assert point.floquet_error == 0.0
