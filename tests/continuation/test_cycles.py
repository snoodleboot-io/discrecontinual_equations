"""Cycle bifurcations, the SNIC, and the Floquet error a mesh carries."""

import math
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CyclePoint,
    CycleSeed,
    _Arc,
    resolved_branch,
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
    HopfField,
    NeimarkSackerField,
    PeriodDoublingField,
    SnicField,
    State,
    Time,
    VanDerPolField,
)

_SADDLE_JACOBIAN = np.array([[0.0, 1.0], [1.0, 0.0]])
# A forward difference with step 1e-7 is itself only good to about 1e-7 times the
# second derivative, so this is the finite-difference oracle's own floor, not the
# analytic Jacobian's. DEQ-15 measured the trapezoidal blocks at this level.
_JACOBIAN_AGREEMENT = 1.0e-6


def _equation(field_type: type, dimension: int, value: float) -> DifferentialEquation:
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


class TestCollocationJacobian(TestCase):
    """The analytic sparse Jacobian of each scheme against finite differences.

    Checked off the cycle, on a perturbed seed, so that every term is exercised:
    on the exact cycle several of the Hermite-Simpson chain-rule terms are small
    enough that a wrong sign in one would pass unnoticed.
    """

    def _system(self, scheme: str) -> tuple[CycleContinuation, np.ndarray, int]:
        intervals, dimension = 12, 4
        nodes = intervals + 1
        grid = np.linspace(0.0, 1.0, nodes)
        states = np.column_stack(
            [
                0.7 * np.cos(2.0 * math.pi * grid),
                0.7 * np.sin(2.0 * math.pi * grid),
                0.1 * np.cos(4.0 * math.pi * grid),
                0.05 * np.sin(2.0 * math.pi * grid),
            ],
        )
        states += 0.05 * np.random.default_rng(7).standard_normal(states.shape)
        unknowns = np.concatenate([states.ravel(), [5.9], [-0.07]])
        continuation = CycleContinuation(
            _equation(NeimarkSackerField, dimension, -0.07),
            0,
            intervals,
            phase_index=1,
            collocation=scheme,
        )
        continuation._dimension = dimension  # noqa: SLF001 (trace sets this)
        return continuation, unknowns, nodes

    def _disagreement(self, scheme: str) -> float:
        continuation, unknowns, nodes = self._system(scheme)
        tangent = np.random.default_rng(11).standard_normal(unknowns.size)
        tangent /= np.linalg.norm(tangent)
        arc = _Arc(unknowns - 0.01 * tangent, tangent, 0.01)
        residual = continuation._augmented(unknowns, nodes, arc)  # noqa: SLF001
        analytic = continuation._bordered(unknowns, nodes, tangent)  # noqa: SLF001
        numerical = continuation._numerical_jacobian(  # noqa: SLF001
            unknowns,
            nodes,
            arc,
            residual,
        )
        return float(np.max(np.abs(analytic.toarray() - numerical)))

    def test_trapezoidal_matches_finite_differences(self):
        assert self._disagreement("trapezoidal") < _JACOBIAN_AGREEMENT

    def test_hermite_simpson_matches_finite_differences(self):
        assert self._disagreement("hermite_simpson") < _JACOBIAN_AGREEMENT

    def test_hermite_simpson_has_the_same_sparsity_pattern(self):
        # Each interval still touches only its own two nodes, so the shared
        # assembly is right for both: same nonzeros, same places.
        patterns = []
        for scheme in ("trapezoidal", "hermite_simpson"):
            continuation, unknowns, nodes = self._system(scheme)
            border = np.ones(unknowns.size)
            matrix = continuation._bordered(unknowns, nodes, border)  # noqa: SLF001
            patterns.append(matrix.toarray() != 0.0)
        assert np.array_equal(patterns[0], patterns[1])

    def test_rejects_an_unknown_scheme(self):
        with pytest.raises(ValueError, match="collocation must be one of"):
            CycleContinuation(
                _equation(HopfField, 2, 0.5),
                0,
                10,
                collocation="gauss",
            )

    def test_defaults_to_hermite_simpson(self):
        continuation = CycleContinuation(_equation(HopfField, 2, 0.5), 0, 10)
        assert continuation.collocation == "hermite_simpson"


class TestCollocationOrder(TestCase):
    """Doubling the mesh divides the error by 16 under Hermite-Simpson, 4 under
    trapezoidal. Measured on the continued branch, not a single solve: the Hopf
    normal form has period exactly 2 pi at every mu, so the period of the last
    traced point is the continuation's own discretisation error outright."""

    def _period_error(self, scheme: str, intervals: int) -> float:
        mu, steps = 0.5, 4
        grid = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [
                math.sqrt(mu) * np.cos(2.0 * math.pi * grid),
                math.sqrt(mu) * np.sin(2.0 * math.pi * grid),
            ],
        )
        continuation = CycleContinuation(
            _equation(HopfField, 2, mu),
            0,
            intervals,
            phase_index=1,
            collocation=scheme,
        )
        points, _ = continuation.trace(CycleSeed(seed, 6.0, mu), 0.05, steps)
        assert len(points) == steps + 1
        return abs(points[-1].solution.period - 2.0 * math.pi)

    def _orders(self, scheme: str) -> list[float]:
        errors = [self._period_error(scheme, n) for n in (10, 20, 40)]
        return [math.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)]

    def test_hermite_simpson_is_fourth_order(self):
        for order in self._orders("hermite_simpson"):
            assert 3.7 < order < 4.3

    def test_trapezoidal_is_second_order(self):
        for order in self._orders("trapezoidal"):
            assert 1.8 < order < 2.3

    def test_hermite_simpson_beats_trapezoidal_on_the_same_mesh(self):
        hermite = self._period_error("hermite_simpson", 20)
        trapezoidal = self._period_error("trapezoidal", 20)
        assert hermite < 0.01 * trapezoidal


class TestSeedSolve(TestCase):
    """The first point is reached in two stages: second order, then the scheme.

    Van der Pol at ``mu = 0.08`` from the normal form's circle of radius two is
    the case that found this. The fourth-order Newton started from that circle
    takes the period to -2592 on its first step and never returns, where the
    trapezoidal one, from the same seed, recovers; the fourth-order cycle is then
    a few quadratic steps from the trapezoidal one. The period it lands on is
    ``2 pi (1 + mu^2 / 16)`` to the order shown, which trapezoidal at 80 nodes
    misses by 3e-3 - so the check also says which scheme the first point is on.
    """

    def test_fourth_order_reaches_a_seed_its_own_newton_cannot(self):
        mu, intervals = 0.08, 80
        grid = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [2.0 * np.cos(2.0 * math.pi * grid), 2.0 * np.sin(2.0 * math.pi * grid)],
        )
        continuation = CycleContinuation(
            _equation(VanDerPolField, 2, mu),
            0,
            intervals,
            phase_index=1,
        )
        points, _ = continuation.trace(CycleSeed(seed, 2.0 * math.pi, mu), 0.04, 2)
        assert len(points) == 3
        expected = 2.0 * math.pi * (1.0 + mu * mu / 16.0)
        assert abs(points[0].solution.period - expected) < 1.0e-4


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


class TestResolvedBranch(TestCase):
    """The gate that refuses cycle points whose multipliers cannot be believed."""

    def _point(
        self,
        parameter: float,
        trivial: float,
        nontrivial: complex,
    ) -> CyclePoint:
        return CyclePoint(
            parameter,
            PeriodicOrbitSolution(np.zeros((3, 2)), 6.0),
            np.array([trivial, nontrivial]),
        )

    def test_keeps_a_branch_that_is_resolved_throughout(self):
        points = [self._point(0.1 * i, 1.0 - 1.0e-4, 0.5) for i in range(5)]
        branch = resolved_branch(points, tolerance=1.0e-2)
        assert len(branch.points) == 5
        assert branch.refused == 0
        assert branch.worst_error < 1.0e-2

    def test_cuts_the_tail_a_trace_could_not_resolve(self):
        good = [self._point(0.1 * i, 1.0 - 1.0e-4, 0.5) for i in range(3)]
        bad = [self._point(0.3 + 0.1 * i, 0.4, 0.5) for i in range(4)]
        branch = resolved_branch(good + bad, tolerance=1.0e-2)
        assert len(branch.points) == 3
        assert branch.refused == 4
        assert branch.tolerance == 1.0e-2

    def test_refuses_every_point_when_none_is_resolved(self):
        branch = resolved_branch(
            [self._point(0.1 * i, 0.4, 0.5) for i in range(4)],
            tolerance=1.0e-2,
        )
        assert branch.points == []
        assert branch.refused == 4
        assert branch.worst_error == 0.0

    def test_keeps_the_resolved_middle_of_a_parameter_sorted_branch(self):
        # Two traces from one seed, sorted by parameter, put the worst-resolved
        # point first: a leading run would refuse the whole branch.
        points = (
            [self._point(-0.46, 0.4, 0.5)]
            + [self._point(-0.40 + 0.01 * i, 1.0 - 1.0e-4, 0.5) for i in range(4)]
            + [self._point(-0.25, 0.3, 0.5)]
        )
        branch = resolved_branch(points, tolerance=1.0e-2)
        assert len(branch.points) == 4
        assert branch.refused == 2
        assert [round(p.parameter, 2) for p in branch.points] == [
            -0.40,
            -0.39,
            -0.38,
            -0.37,
        ]

    def test_takes_the_longest_run_when_there_is_more_than_one(self):
        resolved = 1.0 - 1.0e-4
        trivia = [resolved, resolved, 0.4, resolved, resolved, resolved]
        points = [self._point(0.1 * i, t, 0.5) for i, t in enumerate(trivia)]
        branch = resolved_branch(points, tolerance=1.0e-2)
        assert len(branch.points) == 3
        assert branch.refused == 3
        assert abs(branch.points[0].parameter - 0.3) < 1.0e-12

    def test_does_not_interpolate_a_crossing_into_the_refused_tail(self):
        # A fold of cycles across the cut: the kept run stops before it, so the
        # crossing must not be reported from a pair whose far end was refused.
        resolved = self._point(0.0, 1.0 - 1.0e-6, 0.5)
        unresolved = self._point(0.1, 0.4, 1.5)
        branch = resolved_branch([resolved, unresolved], tolerance=1.0e-2)
        assert len(branch.points) == 1
        assert branch.bifurcations == []

    def test_still_classifies_a_crossing_inside_the_kept_run(self):
        branch = resolved_branch(
            [
                self._point(0.0, 1.0 - 1.0e-6, 0.5),
                self._point(0.1, 1.0 - 1.0e-6, 1.5),
            ],
            tolerance=1.0e-2,
        )
        assert len(branch.points) == 2
        assert [b.kind for b in branch.bifurcations] == ["fold_of_cycles"]

    def test_point_level_gate_reads_the_trivial_multiplier(self):
        assert self._point(0.0, 0.999, 6.0).resolved(tolerance=1.0e-2)
        assert not self._point(0.0, 0.911, 6.0).resolved(tolerance=1.0e-2)

    def test_empty_trace_is_an_empty_resolved_branch(self):
        branch = resolved_branch([])
        assert branch.points == []
        assert branch.refused == 0
        assert branch.bifurcations == []
