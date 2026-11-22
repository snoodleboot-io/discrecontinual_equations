"""Slow manifolds and the van der Pol canard, by boundary-value problem.

Every oracle here is computed independently of the code under test:

* the linear system ``x' = -x + eps y``, ``y' = eps`` (``x' = -x/eps + y`` on the
  slow time) has the exact slow manifold ``x = eps y - eps^2``;
* van der Pol's slow manifold as a graph over the *fast* variable,
  ``y = Y_0(x) + eps Y_1(x) + ...``, was expanded symbolically (SymPy, from
  ``eps (a - x) = Y' (Y - x^3/3 + x)``) and is the coefficient tables below - a
  different graph, solved a different way, from the series under test;
* the canard parameter ``a_c = 1 - eps/8 - 3 eps^2/32 - 173 eps^3/1024 -
  7593 eps^4/16384 - ...`` comes from the same symbolic expansion with ``a`` solved
  order by order for regularity at the fold; through ``eps^3`` it is the
  series Brons (arXiv 1209.1109, eq. 9) quotes from Zvonkin and Shubin (1984)
  for this system, and its first term is Krupa and
  Szmolyan's (3.8) with ``A = -1`` for this system, and its value at
  ``eps = 0.05`` is the ``0.993491`` of Desroches et al. (2012), Fig. 3;
* trajectories are checked against SciPy's DOP853 at tolerance ``1e-13``.
"""

import math
from unittest import TestCase, expectedFailure

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.normal_form import first_lyapunov
from discrecontinual_equations.continuation.slow_manifold import (
    _SEED_NODES,
    BoundarySettings,
    CanardSearch,
    FastSlowSystem,
    FenichelExpansion,
    MaximalCanard,
    OrbitSegment,
    SlowManifold,
    SlowManifoldError,
    _coarsened,
)
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import Alpha, BetaOne, State, Time

_EPSILON = 0.05
# Desroches et al. (2012), Fig. 3: every orbit of the explosion at eps = 0.05.
_REVIEW_CANARD = 0.993491
# a_c = sum_k CANARD[k] eps^k, from the symbolic expansion.
_CANARD = (1.0, -1.0 / 8.0, -3.0 / 32.0, -173.0 / 1024.0, -7593.0 / 16384.0)
_ATTRACTING_X, _REPELLING_X = 2.0, 0.0
_FOLD = 1.0


class VanDerPol(FastSlowSystem):
    """``x' = y - x^3/3 + x``, ``y' = eps (a - x)``: Desroches et al. (2.7)."""

    def fast(self, x, y, epsilon, parameter):  # noqa: ARG002 (protocol)
        return y - x**3 * (1.0 / 3.0) + x

    def slow(self, x, y, epsilon, parameter):  # noqa: ARG002 (protocol)
        return parameter - x


class Linear(FastSlowSystem):
    """``x' = -x + eps y``, ``y' = eps``: the ticket's explicit oracle."""

    def fast(self, x, y, epsilon, parameter):  # noqa: ARG002 (protocol)
        return -1.0 * x + epsilon * y

    def slow(self, x, y, epsilon, parameter):  # noqa: ARG002 (protocol)
        return 1.0 + 0.0 * x


def _graph_over_x(x: float, a: float, epsilon: float) -> float:
    """Van der Pol's slow manifold as ``y(x)`` to ``eps^3``, from SymPy."""
    y0 = x**3 / 3.0 - x
    y1 = -(x - a) / ((x - 1.0) * (x + 1.0))
    y2 = (x - a) * (x * x - 2.0 * a * x + 1.0) / ((x - 1.0) ** 4 * (x + 1.0) ** 4)
    y3 = (
        -2.0
        * (x - a)
        * (
            9.0 * a * a * x * x
            + a * a
            - 11.0 * a * x**3
            - 9.0 * a * x
            + 3.0 * x**4
            + 6.0 * x * x
            + 1.0
        )
        / ((x - 1.0) ** 7 * (x + 1.0) ** 7)
    )
    return y0 + epsilon * y1 + epsilon**2 * y2 + epsilon**3 * y3


def _series(epsilon: float, order: int) -> float:
    return sum(c * epsilon**k for k, c in enumerate(_CANARD[: order + 1]))


def _integrate(system, start, time, times=None):
    def rhs(_t, state):
        return system.field(state, _PARAMETER[0])

    return solve_ivp(
        rhs,
        (0.0, time),
        start,
        method="DOP853",
        rtol=1.0e-13,
        atol=1.0e-14,
        t_eval=times,
        dense_output=True,
    )


# The parameter the integration oracle runs at; set per test.
_PARAMETER = [0.0]


def _manifolds(system, parameter, order=3):
    expansion = FenichelExpansion(system, order)
    attracting = expansion.at(
        _ATTRACTING_X**3 / 3.0 - _ATTRACTING_X,
        _ATTRACTING_X,
        parameter,
    )
    repelling = expansion.at(0.0, _REPELLING_X, parameter)
    return expansion, SlowManifold(system, attracting), SlowManifold(system, repelling)


def _slope(epsilons, errors) -> float:
    return float(np.polyfit(np.log(epsilons), np.log(errors), 1)[0])


class TestFenichelExpansion(TestCase):
    def test_linear_system_is_exact(self):
        system = Linear(0.1)
        point = FenichelExpansion(system, 4).at(0.7, 0.0, 0.0)
        assert np.allclose(point.coefficients, [0.0, 0.7, -1.0, 0.0, 0.0], atol=1e-15)
        assert abs(point.fast - (0.1 * 0.7 - 0.01)) < 1.0e-15
        assert point.attracting

    def test_invariance_residual_vanishes_through_the_order(self):
        system = VanDerPol(_EPSILON)
        expansion = FenichelExpansion(system, 4)
        point = expansion.at(0.9, 2.1, 0.99)
        residual = expansion.residual(point, 0.99)
        terms = residual.coefficients.values()
        assert max(abs(value) for value in terms) < 1.0e-13

    def test_matches_the_graph_over_the_fast_variable(self):
        # Both graphs are truncated at eps^3, so they disagree at eps^4.
        epsilons = [0.04, 0.02, 0.01, 0.005]
        a, slow = 0.99, 1.2
        misses = []
        for epsilon in epsilons:
            system = VanDerPol(epsilon)
            point = FenichelExpansion(system, 3).at(slow, 2.0, a)
            misses.append(abs(_graph_over_x(point.fast, a, epsilon) - slow))
        assert 3.7 < _slope(epsilons, misses) < 4.3

    def test_each_order_is_needed(self):
        # Truncating at eps^k leaves an eps^(k+1) miss: the order is real.
        epsilons = [0.04, 0.02, 0.01]
        for order in (0, 1, 2):
            misses = []
            for epsilon in epsilons:
                point = FenichelExpansion(VanDerPol(epsilon), order).at(1.2, 2.0, 0.99)
                exact = FenichelExpansion(VanDerPol(epsilon), 5).at(1.2, 2.0, 0.99)
                misses.append(abs(point.fast - exact.fast))
            assert abs(_slope(epsilons, misses) - (order + 1)) < 0.2

    def test_repelling_branch_is_recognised(self):
        point = FenichelExpansion(VanDerPol(_EPSILON), 2).at(0.0, 0.1, 0.99)
        assert not point.attracting
        assert point.rate > 0.0

    def test_refuses_a_fold_and_a_negative_order(self):
        with pytest.raises(SlowManifoldError):
            FenichelExpansion(VanDerPol(_EPSILON), 2).at(-2.0 / 3.0, 1.0, 0.99)
        with pytest.raises(ValueError, match="non-negative"):
            FenichelExpansion(VanDerPol(_EPSILON), -1)


class TestSlowManifoldBoundaryValue(TestCase):
    def test_linear_manifold_follows_the_exact_one(self):
        system = Linear(0.1)
        start = FenichelExpansion(system, 2).at(0.0, 0.0, 0.0)
        segment = SlowManifold(system, start, index=1).sweep(0.0, 2.0)[-1]
        x, y = segment.states[:, 0], segment.states[:, 1]
        assert np.max(np.abs(x - (0.1 * y - 0.01))) < 1.0e-8
        assert abs(segment.end[1] - 2.0) < 1.0e-12

    def test_attracting_manifold_agrees_with_integration(self):
        system = VanDerPol(_EPSILON)
        _PARAMETER[0] = 0.9935
        _, attracting, _ = _manifolds(system, 0.9935)
        segment = attracting.sweep(0.9935, _FOLD)[-1]
        assert segment.time > 0.0
        oracle = _integrate(system, segment.start, segment.time, segment.times)
        assert np.max(np.abs(oracle.y.T - segment.states)) < 1.0e-7

    def test_points_on_the_manifolds_flow_along_them(self):
        # Start the integrator from points *on* each computed manifold - in the
        # direction the manifold attracts - and it must reproduce the rest of it.
        system = VanDerPol(_EPSILON)
        _PARAMETER[0] = 0.9935
        _, attracting, repelling = _manifolds(system, 0.9935)
        for manifold in (attracting, repelling):
            segment = manifold.sweep(0.9935, _FOLD)[-1]
            middle = len(segment.mesh) // 2
            elapsed = segment.time - segment.times[middle]
            oracle = _integrate(system, segment.states[middle], elapsed)
            assert np.max(np.abs(oracle.y[:, -1] - segment.end)) < 1.0e-7

    def test_repelling_manifold_runs_backward(self):
        system = VanDerPol(_EPSILON)
        _, _, repelling = _manifolds(system, 0.9935)
        segment = repelling.sweep(0.9935, _FOLD)[-1]
        assert repelling.sign == -1.0
        assert segment.time < 0.0
        assert abs(segment.end[0] - _FOLD) < 1.0e-10

    def test_family_members_end_on_the_manifold(self):
        system = VanDerPol(_EPSILON)
        _, attracting, _ = _manifolds(system, 0.9935)
        family = attracting.sweep(0.9935, _FOLD)
        assert len(family) > 3
        final = family[-1]
        for member in family[:-1]:
            # Each member ends where the final segment crosses the same x.
            order = np.argsort(final.states[:, 0])
            curve = CubicSpline(final.states[order, 0], final.states[order, 1])
            assert abs(member.end[1] - curve(member.end[0])) < 1.0e-8

    def test_start_order_matters_only_exponentially(self):
        # Fenichel: segments from two different start points, an O(eps^2) apart,
        # meet at the fold to far better than eps^2.
        system = VanDerPol(_EPSILON)
        ends = []
        for order in (1, 4):
            _, attracting, _ = _manifolds(system, 0.9935, order)
            ends.append(attracting.sweep(0.9935, _FOLD)[-1].end)
        assert abs(ends[0][1] - ends[1][1]) < 1.0e-9

    def test_following_the_parameter_matches_a_fresh_sweep(self):
        system = VanDerPol(_EPSILON)
        _, attracting, _ = _manifolds(system, 0.99)
        segment = attracting.sweep(0.99, _FOLD)[-1]
        followed = attracting.follow(segment, [0.992, 0.994], _FOLD)[-1]
        fresh = attracting.sweep(0.994, _FOLD)[-1]
        assert followed.parameter == 0.994
        assert abs(followed.end[1] - fresh.end[1]) < 1.0e-9

    def test_a_section_against_the_flow_is_refused(self):
        system = VanDerPol(_EPSILON)
        _, attracting, _ = _manifolds(system, 0.9935)
        with pytest.raises(SlowManifoldError):
            attracting.sweep(0.9935, 3.0)

    def test_an_unreachable_section_stalls_rather_than_lies(self):
        system = VanDerPol(_EPSILON)
        _, _, repelling = _manifolds(system, 0.9)
        # Backward along S_r at a = 0.9 the flow winds onto the equilibrium at
        # x = 0.9, so a section at x = 0.95 is never reached.
        with pytest.raises(SlowManifoldError):
            repelling.sweep(0.9, 0.95)


class TestCanard(TestCase):
    def _search(self, epsilon, parameter):
        system = VanDerPol(epsilon)
        expansion, attracting, repelling = _manifolds(system, parameter)
        return system, expansion, CanardSearch(attracting, repelling, _FOLD)

    def test_canard_at_the_review_value(self):
        _, _, search = self._search(_EPSILON, 0.9935)
        canard = search.locate(0.993, 0.994)
        assert abs(canard.parameter - _REVIEW_CANARD) < 5.0e-7
        assert abs(canard.gap) < 1.0e-12
        assert abs(canard.attracting.end[1] - canard.repelling.end[1]) < 1.0e-12

    def test_the_gap_changes_sign_through_the_canard(self):
        _, _, search = self._search(_EPSILON, 0.9935)
        below = search.gap(*search.segments(0.993))
        above = search.gap(*search.segments(0.994))
        assert below < 0.0 < above

    def test_concatenated_problem_agrees_with_the_search(self):
        system, expansion, search = self._search(_EPSILON, 0.9935)
        canard = search.locate(0.993, 0.994)
        ends = (
            tuple(canard.attracting.start),
            tuple(canard.repelling.start),
        )
        orbit = MaximalCanard(system, expansion, ends).solve(
            MaximalCanard.join(canard),
        )
        assert abs(orbit.parameter - canard.parameter) < 1.0e-11
        assert orbit.segment.time > 0.0
        assert np.allclose(orbit.segment.start, canard.attracting.start)
        assert np.allclose(orbit.segment.end, canard.repelling.start)

    def test_joined_halves_are_one_forward_orbit(self):
        _, _, search = self._search(_EPSILON, 0.9935)
        canard = search.locate(0.993, 0.994)
        joined = MaximalCanard.join(canard)
        assert joined.mesh[0] == 0.0
        assert abs(joined.mesh[-1] - 1.0) < 1.0e-15
        assert np.all(np.diff(joined.mesh) > 0.0)
        assert (
            abs(joined.time - (canard.attracting.time - canard.repelling.time)) < 1e-12
        )

    def test_canard_parameter_matches_the_series_to_its_order(self):
        # Continue the maximal canard in eps and measure the error of each
        # truncation of the asymptotic series: truncated after eps^k it must
        # fall as eps^(k+1), and what is left after eps^4 must be the next
        # coefficient, -436777/262144 (from the same symbolic expansion).
        epsilons = [0.04, 0.03, 0.025, 0.02, 0.015, 0.0125, 0.01]
        system, expansion, search = self._search(epsilons[0], 0.995)
        canard = search.locate(0.994, 0.996)
        ends = (tuple(canard.attracting.start), tuple(canard.repelling.start))
        settings = BoundarySettings(tolerance=1.0e-10)
        orbits = MaximalCanard(system, expansion, ends, settings).follow(
            MaximalCanard.join(canard),
            epsilons,
        )
        located = np.array([orbit.parameter for orbit in orbits])
        assert [orbit.epsilon for orbit in orbits] == epsilons
        for order in (1, 2, 3, 4):
            errors = [
                abs(a - _series(e, order))
                for a, e in zip(located, epsilons, strict=True)
            ]
            assert abs(_slope(epsilons, errors) - (order + 1)) < 0.25, (order, errors)
        # Beyond the coefficients this file states: the symbolic expansion goes
        # on -436777/262144 eps^5 - 15340159/2097152 eps^6. Where eps^7 is
        # negligible what is left after eps^4 is those two terms, and the
        # located parameter carries it to 20% at a few times 1e-12.
        fifth, sixth = -436777.0 / 262144.0, -15340159.0 / 2097152.0
        for a, e in zip(located, epsilons, strict=True):
            if e <= 0.02:
                remainder = a - _series(e, 4) - fifth * e**5
                assert abs(remainder / (sixth * e**6) - 1.0) < 0.2, (e, remainder)

    def test_a_search_needs_one_manifold_of_each_kind(self):
        system = VanDerPol(_EPSILON)
        _, attracting, _ = _manifolds(system, 0.9935)
        with pytest.raises(ValueError, match="attract"):
            CanardSearch(attracting, attracting, _FOLD)


class TestOrbitSegment(TestCase):
    def test_times_scale_the_mesh(self):
        segment = OrbitSegment(
            np.array([0.0, 0.5, 1.0]),
            np.zeros((3, 2)),
            -4.0,
            0.0,
            0.0,
        )
        assert np.allclose(segment.times, [0.0, -2.0, -4.0])


class ForcedVanDerPol(DeterministicFunction):
    """``x' = y - x^3/3 + x``, ``y' = eps (a - x)`` as a library field."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        a, epsilon = self.parameters[0].value, self.parameters[1].value
        x, y = point[0], point[1]
        return [y - x * x * x / 3.0 + x, epsilon * (a - x)]


def _kuznetsov_l1(epsilon: float) -> float:
    """Kuznetsov's ``l_1`` for this system at ``a = 1``, written out by hand.

    In ``X = x - 1``, ``Y = y + 2/3`` the field is ``A = [[0, 1], [-eps, 0]]``
    plus ``B(u, v) = (-2 u_1 v_1, 0)`` and ``C(u, v, w) = (-2 u_1 v_1 w_1, 0)``;
    ``q`` and ``p`` are normalised by ``<q, q> = <p, q> = 1`` (``<p, q> =
    conj(p) . q``).
    """
    omega = math.sqrt(epsilon)
    a = np.array([[0.0, 1.0], [-epsilon, 0.0]])

    def b(u, v):
        return np.array([-2.0 * u[0] * v[0], 0.0])

    def c(u, v, w):
        return np.array([-2.0 * u[0] * v[0] * w[0], 0.0])

    q = np.array([1.0, 1j * omega]) / math.sqrt(1.0 + epsilon)
    p = np.array([-1j * omega, 1.0])
    p = p / np.conj(np.conj(p) @ q)
    h11 = np.linalg.solve(a, b(q, q.conj()))
    h20 = np.linalg.solve(2j * omega * np.eye(2) - a, b(q, q))
    g21 = np.conj(p) @ (c(q, q, q.conj()) - 2.0 * b(q, h11) + b(q.conj(), h20))
    return float(g21.real / (2.0 * omega))


def _forced_problem(epsilon: float) -> TwoParameterProblem:
    parameters = [Alpha(value=1.0), BetaOne(value=epsilon)]
    eq = DifferentialEquation(
        variables=[State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=ForcedVanDerPol(
            variables=[State(), State()],
            parameters=parameters,
            results=[State(), State()],
            time=None,
        ),
    )
    return TwoParameterProblem(eq, 0, 1, 0.0, AutomaticDifferentiation())


class TestSingularHopf(TestCase):
    """The ticket's third oracle: the Hopf at ``a = 1`` and its ``l_1``.

    At ``a = 1`` the shift ``X = x - 1``, ``Y = y + 2/3`` and ``u = X``,
    ``v = -Y / omega`` with ``omega = sqrt(eps)`` give ``u' = -omega v - u^2 -
    u^3/3``, ``v' = omega u``, and Guckenheimer and Holmes' (3.4.11) reads
    ``16 a = f_uuu = -2``: ``a = -1/8``, supercritical, as Desroches et al.
    state. Kuznetsov's normalisation scales that by ``|q|^2`` and gives
    ``l_1 = -1 / (2 omega (1 + eps))``.
    """

    def test_hopf_at_a_equals_one(self):
        for epsilon in (0.05, 0.2):
            problem = _forced_problem(epsilon)
            state = np.array([1.0, -2.0 / 3.0])
            eigenvalues = np.linalg.eigvals(problem.state_jacobian(state, 1.0, epsilon))
            omega = math.sqrt(epsilon)
            assert np.max(np.abs(eigenvalues.real)) < 1.0e-12
            assert np.allclose(sorted(eigenvalues.imag), [-omega, omega])

    def test_first_lyapunov_coefficient_by_hand(self):
        for epsilon in (0.05, 0.2):
            expected = -1.0 / (2.0 * math.sqrt(epsilon) * (1.0 + epsilon))
            assert abs(_kuznetsov_l1(epsilon) - expected) < 1.0e-12

    @expectedFailure
    def test_library_first_lyapunov_has_the_supercritical_sign(self):
        # normal_form.first_lyapunov normalises p by <p, q> instead of its
        # conjugate, so <p, q> = c / conj(c); here c is imaginary and the
        # normalisation is -1, flipping the sign (+2.13 against -2.13 at
        # eps = 0.05). A normal Jacobian has a real c, which is why the
        # library's own Hopf tests do not see it. Reported, not fixed here.
        epsilon = 0.05
        state = np.array([1.0, -2.0 / 3.0])
        l1 = first_lyapunov(
            _forced_problem(epsilon), state, 1.0, epsilon, math.sqrt(epsilon)
        )
        assert abs(l1 - _kuznetsov_l1(epsilon)) < 1.0e-10


class TestSeedResampling(TestCase):
    def test_a_long_seed_is_resampled_by_arclength(self):
        # solve_bvp only adds nodes, so a seed is cut down before every solve -
        # onto a mesh that crowds where the orbit moves: here a jump at s = 0.5.
        mesh = np.linspace(0.0, 1.0, 4001)
        x = np.tanh((mesh - 0.5) / 0.01)
        states = np.column_stack([x, mesh])
        seed = OrbitSegment(mesh, states, 3.0, 0.7, 0.0)
        coarse = _coarsened(seed)
        assert len(coarse.mesh) == _SEED_NODES
        assert coarse.mesh[0] == 0.0
        assert coarse.mesh[-1] == 1.0
        assert (coarse.time, coarse.parameter) == (3.0, 0.7)
        near_jump = np.sum(np.abs(coarse.mesh - 0.5) < 0.05)
        assert near_jump > 2.5 * 0.1 * _SEED_NODES
        assert np.allclose(coarse.states[:, 0], np.interp(coarse.mesh, mesh, x))
        short = OrbitSegment(mesh[:50], states[:50], 1.0, 0.0, 0.0)
        assert _coarsened(short) is short
