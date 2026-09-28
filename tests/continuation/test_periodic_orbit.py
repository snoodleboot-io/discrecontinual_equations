"""The periodic orbit solvers, and what it takes to resolve a stiff cycle.

Also expensive: mesh adaptation re-solves as it equidistributes, and the
resolution search solves again at each doubling of the mesh, so one test
here is many whole continuations.
"""

import math
from unittest import TestCase

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.connecting_orbit import (
    HomoclinicOrbit,
    MeshSpec,
    OrbitResolution,
)
from discrecontinual_equations.continuation.periodic_orbit import (
    AdaptivePeriodicOrbit,
    AnalyticPeriodicOrbit,
    HermiteSimpsonOrbit,
    PeriodicOrbit,
    ResolutionLevel,
    ResolutionSettings,
    RobustPeriodicOrbit,
    _certified,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from tests.continuation.fields import (
    Alpha,
    HopfField,
    SnicField,
    State,
    Time,
    VanDerPolField,
    _saddle,
)

_SADDLE_JACOBIAN = np.array([[0.0, 1.0], [1.0, 0.0]])


_SADDLE_JACOBIAN = np.array([[0.0, 1.0], [1.0, 0.0]])


class TestPeriodicOrbit(TestCase):
    """Limit cycle and Floquet multipliers against the Hopf normal form.

    The supercritical Hopf has an exact cycle of radius sqrt(mu), period 2 pi, a
    trivial Floquet multiplier +1, and a nontrivial multiplier exp(-4 pi mu).
    """

    def _hopf(self, mu: float) -> DifferentialEquation:
        parameters = [Alpha(value=mu)]
        return DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=HopfField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )

    def test_cycle_radius_period_and_floquet(self):
        mu, intervals = 0.25, 80
        nodes = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [
                math.sqrt(mu) * np.cos(2.0 * math.pi * nodes),
                math.sqrt(mu) * np.sin(2.0 * math.pi * nodes),
            ],
        )
        orbit = PeriodicOrbit(
            self._hopf(mu).derivative,
            intervals,
            phase_index=1,
            phase_value=0.0,
        )
        solution = orbit.solve(seed, 6.0)
        assert solution is not None
        radius = float(np.mean(np.linalg.norm(solution.states, axis=1)))
        assert abs(radius - math.sqrt(mu)) < 1.0e-3
        assert abs(solution.period - 2.0 * math.pi) < 1.0e-2
        multipliers = sorted(orbit.floquet_multipliers(solution).real)
        assert abs(multipliers[1] - 1.0) < 1.0e-2
        assert abs(multipliers[0] - math.exp(-4.0 * math.pi * mu)) < 5.0e-3


class TestOrbitResolution(TestCase):
    """``estimate_orbit_error`` against the exact homoclinic ``1.5 sech^2(t/2)``.

    The orbit has two independent resolutions - mesh spacing and half-length - and
    the point of the estimate is that either can be the limiting one. Each table
    below records the true error beside both components, so the reason the estimate
    must come from the larger of them is visible rather than asserted.
    """

    @staticmethod
    def _solve(half: float, intervals: int):
        times = np.linspace(-half, half, intervals + 1)
        seed = np.zeros((intervals + 1, 2))
        seed[:, 0] = 1.2 / np.cosh(0.45 * times) ** 2
        seed[:, 1] = -0.9 * np.tanh(0.45 * times) / np.cosh(0.45 * times) ** 2
        orbit = HomoclinicOrbit(
            _saddle(),
            np.zeros(2),
            _SADDLE_JACOBIAN,
            MeshSpec(half, intervals, phase_index=1, phase_value=0.0),
        )
        solution = orbit.solve(seed)
        sech = 1.0 / np.cosh(solution.times / 2.0) ** 2
        exact = np.column_stack(
            [1.5 * sech, -1.5 * sech * np.tanh(solution.times / 2.0)],
        )
        true_error = float(np.max(np.abs(solution.states - exact)))
        return orbit, solution, true_error

    def test_estimate_tracks_true_error_where_truncation_is_ample(self):
        """At T = 15 the estimate runs 0.75x the true error at every resolution.

        That factor is not a coincidence to be tuned away: trapezoidal collocation
        is second order, so doubling the mesh moves the answer by E - E/4 = 0.75 E.
        The estimate therefore bounds the *refined* orbit's error by 3x while
        running slightly under the returned orbit's own error.

            N     true       discr      est/true
            40    6.06e-2    4.50e-2    0.74
            80    1.72e-2    1.28e-2    0.75
            160   4.37e-3    3.27e-3    0.75
        """
        for intervals in (40, 80, 160):
            orbit, solution, true_error = self._solve(15.0, intervals)
            resolution = orbit.estimate_orbit_error(solution)
            ratio = resolution.estimate / true_error
            assert 0.6 < ratio < 0.9
            assert resolution.limited_by == "spacing"

    def test_truncation_component_catches_a_boundary_limited_orbit(self):
        """At T = 5 refining the mesh stops helping, and only truncation sees it.

        Past N = 160 the error is pinned at ~2.7e-4 by the truncated interval, so
        the discretisation component collapses while the orbit does not improve.
        Reporting it alone would be 12x optimistic at N = 640.

            N     true       discr      trunc      limited_by
            320   2.73e-4    9.18e-5    2.72e-4    length
            640   2.78e-4    2.29e-5    2.72e-4    length
        """
        for intervals in (320, 640):
            orbit, solution, true_error = self._solve(5.0, intervals)
            resolution = orbit.estimate_orbit_error(solution)
            assert resolution.discretisation < 0.5 * true_error
            assert resolution.limited_by == "length"
            assert 0.8 < resolution.estimate / true_error < 1.3

    def test_discretisation_component_catches_a_coarse_mesh(self):
        """At h = 0.25 truncation is finished and says so; the mesh is the problem.

        Extending T from 15 to 20 shifts the orbit by ~5e-11 while its true error is
        7.7e-3. The truncation component is right about its own term and useless as
        an error estimate, which is why ``estimate`` takes the larger.
        """
        orbit, solution, true_error = self._solve(15.0, 120)
        resolution = orbit.estimate_orbit_error(solution)
        assert resolution.truncation < 1.0e-8
        assert resolution.truncation < 1.0e-5 * true_error
        assert resolution.limited_by == "spacing"
        assert 0.6 < resolution.estimate / true_error < 0.9

    def test_reports_large_where_either_resolution_is_badly_wrong(self):
        """A grossly under-resolved orbit is reported as such, in either parameter."""
        orbit, solution, _ = self._solve(20.0, 40)
        coarse = orbit.estimate_orbit_error(solution)
        assert coarse.estimate > 1.0e-2
        assert coarse.limited_by == "spacing"
        orbit, solution, _ = self._solve(2.0, 200)
        short = orbit.estimate_orbit_error(solution)
        assert short.estimate > 1.0e-2
        assert short.limited_by == "length"

    def test_estimate_leaves_the_solver_mesh_unchanged(self):
        """The estimate re-solves on other meshes without disturbing this solver.

        Asserted through behaviour rather than by reading the mesh: the second call
        passes the same solution back, and the node-count guard would reject it if
        the first call had left the solver on one of its refined meshes.
        """
        orbit, solution, _ = self._solve(10.0, 80)
        first = orbit.estimate_orbit_error(solution)
        second = orbit.estimate_orbit_error(solution)
        assert first.discretisation > 0.0
        assert second.discretisation == pytest.approx(first.discretisation)
        assert second.truncation == pytest.approx(first.truncation)

    def test_rejects_an_odd_mesh(self):
        """Odd intervals have no matching centre node when doubled.

        The phase condition pins the centre node, so on an odd mesh the doubled
        mesh's centre sits half a step away and the two orbits come out translated
        relative to each other. The shift would then measure that translation - an
        O(h) quantity - rather than the O(h^2) error.
        """
        orbit, solution, _ = self._solve(10.0, 81)
        with pytest.raises(ValueError, match="even"):
            orbit.estimate_orbit_error(solution)

    def test_rejects_a_solution_from_another_mesh(self):
        """A solution with the wrong node count is a caller error, not a resolution."""
        orbit, _, _ = self._solve(10.0, 80)
        _, other, _ = self._solve(10.0, 40)
        with pytest.raises(ValueError, match="nodes"):
            orbit.estimate_orbit_error(other)

    def test_resolution_prefers_the_larger_component(self):
        """``estimate`` is the max and ``limited_by`` names it, with ties to spacing."""
        assert OrbitResolution(3.0, 1.0).estimate == 3.0
        assert OrbitResolution(3.0, 1.0).limited_by == "spacing"
        assert OrbitResolution(1.0, 3.0).estimate == 3.0
        assert OrbitResolution(1.0, 3.0).limited_by == "length"
        assert OrbitResolution(2.0, 2.0).limited_by == "spacing"


class TestAnalyticPeriodicOrbit(TestCase):
    """Exact, analytically-assembled BVP Jacobian via automatic differentiation."""

    def _field(self, field_type, mu: float) -> object:
        return field_type(
            variables=[State(), State()],
            parameters=[Alpha(value=mu)],
            results=[State(), State()],
            time=None,
        )

    def test_analytic_jacobian_matches_finite_difference(self):
        mu, intervals, dimension = 0.5, 12, 2
        solver = AnalyticPeriodicOrbit(
            self._field(HopfField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        )
        nodes = intervals + 1
        grid = np.linspace(0.0, 1.0, nodes)
        states = np.column_stack(
            [
                math.sqrt(mu) * np.cos(2.0 * math.pi * grid),
                math.sqrt(mu) * np.sin(2.0 * math.pi * grid),
            ],
        )
        unknowns = np.concatenate([states.flatten(), [6.3]])
        analytic = solver._analytic_jacobian(unknowns, nodes, dimension)  # noqa: SLF001
        base = solver._residual(unknowns, nodes, dimension)  # noqa: SLF001
        finite = np.zeros((base.size, unknowns.size))
        step = 1.0e-7
        for j in range(unknowns.size):
            shifted = unknowns.copy()
            shifted[j] += step
            finite[:, j] = (
                solver._residual(shifted, nodes, dimension) - base  # noqa: SLF001
            ) / step
        assert np.max(np.abs(analytic - finite)) < 1.0e-5

    def test_matches_base_solver_on_hopf(self):
        mu, intervals = 0.5, 60
        grid = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [
                math.sqrt(mu) * np.cos(2.0 * math.pi * grid),
                math.sqrt(mu) * np.sin(2.0 * math.pi * grid),
            ],
        )
        analytic = AnalyticPeriodicOrbit(
            self._field(HopfField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed.copy(), 6.3)
        base = PeriodicOrbit(
            self._field(HopfField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed.copy(), 6.3)
        assert analytic is not None
        assert abs(analytic.period - base.period) < 1.0e-7

    def test_solves_polynomial_van_der_pol(self):
        mu, intervals = 1.0, 200
        grid = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [2.0 * np.cos(2.0 * math.pi * grid), -2.0 * np.sin(2.0 * math.pi * grid)],
        )
        analytic = AnalyticPeriodicOrbit(
            self._field(VanDerPolField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed.copy(), 6.6)
        base = PeriodicOrbit(
            self._field(VanDerPolField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed.copy(), 6.6)
        assert analytic is not None
        assert abs(analytic.period - base.period) < 1.0e-6


class TestAdaptivePeriodicOrbit(TestCase):
    """Mesh adaptation for stiff relaxation oscillations, vs an integrated oracle."""

    def _field(self, field_type, mu: float) -> object:
        return field_type(
            variables=[State(), State()],
            parameters=[Alpha(value=mu)],
            results=[State(), State()],
            time=None,
        )

    def _van_der_pol_oracle(self, mu: float, intervals: int):
        def flow(_t, point):
            x, y = point
            return [y, mu * (1.0 - x * x) * y - x]

        settled = solve_ivp(
            flow,
            [0.0, 100.0 * mu],
            [2.0, 0.0],
            method="Radau",
            rtol=1.0e-9,
            atol=1.0e-11,
        ).y[:, -1]

        def section(_t, point):
            return point[1]

        section.direction = 1.0
        cycle = solve_ivp(
            flow,
            [0.0, 20.0 * mu],
            settled,
            method="Radau",
            rtol=1.0e-10,
            atol=1.0e-12,
            events=section,
            dense_output=True,
        )
        crossings = cycle.t_events[0]
        period = crossings[1] - crossings[0]
        times = crossings[0] + np.linspace(0.0, period, intervals + 1)
        return period, cycle.sol(times).T

    def test_beats_uniform_on_stiff_cycle(self):
        mu, intervals = 10.0, 200
        period, seed = self._van_der_pol_oracle(mu, intervals)
        uniform = AnalyticPeriodicOrbit(
            self._field(VanDerPolField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed.copy(), period)
        adaptive = AdaptivePeriodicOrbit(
            self._field(VanDerPolField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed.copy(), period)
        assert uniform is not None
        assert adaptive is not None
        uniform_error = abs(uniform.period - period) / period
        adaptive_error = abs(adaptive.period - period) / period
        assert uniform_error > 0.03
        assert adaptive_error < 5.0e-3
        assert adaptive_error < 0.1 * uniform_error

    def test_no_regression_on_smooth_cycle(self):
        mu, intervals = 0.5, 60
        grid = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [
                math.sqrt(mu) * np.cos(2.0 * math.pi * grid),
                math.sqrt(mu) * np.sin(2.0 * math.pi * grid),
            ],
        )
        adaptive = AdaptivePeriodicOrbit(
            self._field(HopfField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed, 6.3)
        assert adaptive is not None
        assert abs(adaptive.period - 2.0 * math.pi) / (2.0 * math.pi) < 1.0e-2

    def test_error_estimate_flags_an_under_resolved_cycle(self):
        """The estimate separates a resolved cycle from an under-resolved one.

        A converged solve only proves the discrete system was satisfied; nothing in
        the residual gate knows whether the mesh resolves the jump layers, so too
        coarse a mesh converges confidently to the wrong cycle. Refining is what
        catches it: an under-resolved solution either fails to seed the finer mesh
        at all, or shifts a long way when it does.
        """
        mu, coarse, fine = 8.0, 50, 200

        def estimate(intervals: int) -> tuple[float | None, float]:
            period, seed = self._van_der_pol_oracle(mu, intervals)
            orbit = AdaptivePeriodicOrbit(
                self._field(VanDerPolField, mu),
                intervals,
                phase_index=1,
                phase_value=0.0,
            )
            solution = orbit.solve(seed.copy(), period)
            if solution is None:
                return None, 1.0
            true_error = abs(solution.period - period) / period
            return orbit.estimate_period_error(
                solution.states,
                solution.period,
            ), true_error

        coarse_estimate, coarse_error = estimate(coarse)
        # Either it cannot be refined at all, or refining moves it a long way.
        # Both are the detector doing its job; which one occurs is not the contract.
        assert coarse_estimate is None or coarse_estimate > 1.0e-2

        fine_estimate, fine_error = estimate(fine)
        assert fine_estimate is not None
        assert fine_estimate < 1.0e-2
        # It must not flatter the true error - conservative is the useful direction.
        assert fine_estimate >= fine_error / 10.0
        # And it must actually discriminate.
        assert fine_error < coarse_error

    def test_error_estimate_is_small_on_a_well_resolved_cycle(self):
        """A control: the check must not cry wolf on a cycle that is fine.

        A detector that flags everything is worthless, so the smooth, amply-resolved
        case has to come back with a small estimate - not merely a non-None one.
        """
        mu, intervals = 2.0, 200
        period, seed = self._van_der_pol_oracle(mu, intervals)
        orbit = AdaptivePeriodicOrbit(
            self._field(VanDerPolField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        )
        solution = orbit.solve(seed.copy(), period)
        assert solution is not None
        estimate = orbit.estimate_period_error(solution.states, solution.period)
        assert estimate is not None
        assert estimate < 1.0e-3

    def test_error_estimate_is_available_without_mesh_adaptation(self):
        """The check belongs to every cycle solver, not only the adaptive one.

        AnalyticPeriodicOrbit has the same blind spot - it reports a converged
        discrete solve and cannot tell whether the mesh resolves the orbit.
        """
        mu, intervals = 2.0, 200
        period, seed = self._van_der_pol_oracle(mu, intervals)
        orbit = AnalyticPeriodicOrbit(
            self._field(VanDerPolField, mu),
            intervals,
            phase_index=1,
            phase_value=0.0,
        )
        solution = orbit.solve(seed.copy(), period)
        assert solution is not None
        estimate = orbit.estimate_period_error(solution.states, solution.period)
        assert estimate is not None
        assert estimate < 1.0e-2

    def test_error_estimate_rejects_a_mismatched_solution(self):
        """Passing a solution from a different solver is a caller error, not a nan."""
        mu, intervals = 2.0, 100
        period, seed = self._van_der_pol_oracle(mu, intervals)
        orbit = AdaptivePeriodicOrbit(
            self._field(VanDerPolField, mu),
            intervals * 2,
            phase_index=1,
            phase_value=0.0,
        )
        with pytest.raises(ValueError, match="nodes"):
            orbit.estimate_period_error(seed, period)
        with pytest.raises(ValueError, match="positive"):
            orbit.estimate_period_error(
                np.zeros((intervals * 2 + 1, 2)),
                -1.0,
            )

    def test_resolution_search_reports_the_mesh_that_worked(self):
        """The search returns every level it measured, and certifies from them.

        One continuation call, for the wiring: the table comes back whether or not
        anything is certified, and a certified cycle meets its own estimate. The
        certification rule itself is covered against measured tables in
        TestResolutionCertification, without paying for continuation.
        """
        mu, start = 12.0, 200
        period, seed = self._van_der_pol_oracle(6.0, start)
        orbit = AdaptivePeriodicOrbit(
            self._field(VanDerPolField, 6.0),
            start,
            phase_index=1,
            phase_value=0.0,
        )
        study = orbit.continue_to_resolved(
            0,
            mu,
            seed.copy(),
            period,
            ResolutionSettings(tolerance=1.0e-2, doublings=2),
        )
        assert [level.intervals for level in study.levels] == [200, 400, 800]
        assert all(level.period is not None for level in study.levels)
        assert study.levels[0].agreement is None
        assert all(level.agreement is not None for level in study.levels[1:])
        assert study.best is study.levels[-1]

        resolved = study.resolved
        assert resolved is not None
        assert resolved.intervals == study.levels[-1].intervals
        oracle_period, _ = self._van_der_pol_oracle(mu, resolved.intervals)
        true_error = abs(resolved.period - oracle_period) / oracle_period
        assert true_error <= resolved.estimate

    def test_continuation_reaches_extreme_stiffness(self):
        """Continuation reaches mu = 16, a stiffness a cold solve cannot.

        At 400 intervals the period error is about 5e-4. Under the earlier mesh
        monitor 200 intervals stalled at mu ~ 11.7; that was the monitor failing to
        adapt, not a resolution limit, and 200 now arrives at about 2.5e-3. See
        FUTURE_WORK.md section 1b.
        """
        intervals, target = 400, 16.0
        period, seed = self._van_der_pol_oracle(target, intervals)
        cold = AdaptivePeriodicOrbit(
            self._field(VanDerPolField, target),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed.copy(), period)
        cold_failed = cold is None or abs(cold.period - period) / period > 0.1
        assert cold_failed
        start_period, start_seed = self._van_der_pol_oracle(6.0, intervals)
        continued = AdaptivePeriodicOrbit(
            self._field(VanDerPolField, 6.0),
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).continue_to(0, target, start_seed.copy(), start_period)
        assert continued is not None
        assert abs(continued.period - period) / period < 1.0e-2


class TestResolutionCertification(TestCase):
    """The certification rule, against resolution tables measured on van der Pol.

    Each table below was produced by continuing to mu = 12 and comparing against the
    integrated oracle; the true errors are recorded beside the agreements so the
    reason each case must or must not certify is visible. They were measured under
    the earlier mesh monitor, whose erratic convergence is exactly the behaviour the
    rule has to withstand, so they stay as the rule's regression data.
    """

    @staticmethod
    def _levels(rows):
        return [ResolutionLevel(n, period, agreement) for n, period, agreement in rows]

    def test_a_single_agreement_does_not_certify(self):
        # 400 nodes agree with 200 to 1.128e-3, but the true 400-node error is
        # 1.209e-3. Certifying on that one agreement would report an estimate the
        # answer does not meet.
        levels = self._levels(
            [
                (100, 21.409699, None),  # true error 3.378e-2
                (200, 22.210033, 3.603e-2),  # true error 2.338e-3
                (400, 22.185012, 1.128e-3),  # true error 1.209e-3
            ],
        )
        assert _certified(levels, 2.0e-3) is None

    def test_two_agreements_certify_with_the_larger(self):
        levels = self._levels(
            [
                (100, 21.409699, None),
                (200, 22.210033, 3.603e-2),
                (400, 22.185012, 1.128e-3),
                (800, 22.160267, 1.117e-3),  # true error 9.231e-5
            ],
        )
        estimate = _certified(levels, 2.0e-3)
        assert estimate == 1.128e-3
        assert estimate >= 9.231e-5

    def test_one_agreement_outside_tolerance_breaks_the_chain(self):
        # Measured from a 200-node seed: the 200/400 shift is large, so 800 has
        # only one agreement behind it even though it is accurate to 1.08e-4.
        levels = self._levels(
            [
                (200, 21.955683, None),  # true error 9.141e-3
                (400, 22.145518, 8.572e-3),  # true error 5.733e-4
                (800, 22.155835, 4.656e-4),  # true error 1.077e-4
            ],
        )
        assert _certified(levels, 2.0e-3) is None
        assert _certified(levels, 1.0e-2) == 8.572e-3

    def test_a_stall_breaks_the_chain(self):
        levels = self._levels(
            [
                (200, 22.210033, None),
                (400, None, None),
                (800, 22.160267, None),
                (1600, 22.158500, 8.0e-5),
            ],
        )
        assert _certified(levels, 2.0e-3) is None

    def test_too_few_levels_cannot_certify(self):
        assert _certified(self._levels([(200, 22.2, None)]), 1.0) is None
        assert _certified([], 1.0) is None


class TestHermiteSimpsonOrbit(TestCase):
    """Fourth-order collocation for limit cycles, checked by convergence order."""

    def _hopf(self, mu: float) -> DifferentialEquation:
        parameters = [Alpha(value=mu)]
        return DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=HopfField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )

    def _period_error(self, solver_type, intervals: int, mu: float) -> float:
        nodes = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [
                math.sqrt(mu) * np.cos(2.0 * math.pi * nodes),
                math.sqrt(mu) * np.sin(2.0 * math.pi * nodes),
            ],
        )
        solution = solver_type(
            self._hopf(mu).derivative,
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed, 6.0)
        return abs(solution.period - 2.0 * math.pi)

    def test_fourth_order_convergence(self):
        mu = 0.5
        errors = [self._period_error(HermiteSimpsonOrbit, n, mu) for n in (20, 40, 80)]
        orders = [
            math.log(errors[k] / errors[k + 1]) / math.log(2.0)
            for k in range(len(errors) - 1)
        ]
        for order in orders:
            assert order > 3.5

    def test_more_accurate_than_trapezoidal(self):
        mu, intervals = 0.5, 40
        hermite = self._period_error(HermiteSimpsonOrbit, intervals, mu)
        trapezoidal = self._period_error(PeriodicOrbit, intervals, mu)
        assert hermite < 1.0e-4
        assert hermite < 0.01 * trapezoidal


class TestRobustPeriodicOrbit(TestCase):
    """Newton robustness against collapse to the trivial (zero-period) solution."""

    def _hopf(self, mu: float) -> DifferentialEquation:
        parameters = [Alpha(value=mu)]
        return DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=HopfField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )

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

    def test_matches_plain_solver_on_regular_cycle(self):
        mu, intervals = 0.25, 80
        nodes = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [
                np.sqrt(mu) * np.cos(2.0 * math.pi * nodes),
                np.sqrt(mu) * np.sin(2.0 * math.pi * nodes),
            ],
        )
        solution = RobustPeriodicOrbit(
            self._hopf(mu).derivative,
            intervals,
            phase_index=1,
            phase_value=0.0,
        ).solve(seed, 6.0)
        assert solution is not None
        assert abs(solution.period - 2.0 * math.pi) < 1.0e-2
        radius = float(np.mean(np.linalg.norm(solution.states, axis=1)))
        assert abs(radius - math.sqrt(mu)) < 1.0e-3

    def test_rescues_collapse_near_snic(self):
        intervals = 120
        nodes = np.linspace(0.0, 1.0, intervals + 1)
        seed = np.column_stack(
            [np.cos(2.0 * math.pi * nodes), np.sin(2.0 * math.pi * nodes)],
        )
        plain_failures = 0
        for mu in (1.06, 1.08, 1.10, 1.12):
            oracle = 2.0 * math.pi / math.sqrt(mu * mu - 1.0)
            plain = PeriodicOrbit(
                self._snic(mu).derivative,
                intervals,
                phase_index=1,
                phase_value=0.0,
            ).solve(seed.copy(), oracle * 1.05)
            robust = RobustPeriodicOrbit(
                self._snic(mu).derivative,
                intervals,
                phase_index=1,
                phase_value=0.0,
            ).solve(seed.copy(), oracle * 1.05)
            assert robust is not None
            assert abs(robust.period - oracle) / oracle < 1.0e-3
            if plain is None or abs(plain.period - oracle) / oracle > 0.5:
                plain_failures += 1
        assert plain_failures >= 1
