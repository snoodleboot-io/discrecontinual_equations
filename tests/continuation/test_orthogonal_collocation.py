"""Orthogonal collocation at Gauss-Legendre points, and its Floquet multipliers.

The acceptance oracles of DEQ-34: the scheme is selectable beside the other two,
its period converges at ``2m`` per mesh doubling, the multipliers read off its own
Jacobian agree with the Hermite monodromy and inherit the ``2m``, and on the
Bogdanov-Takens orbit nearest the homoclinic it gets the Floquet error the
fourth-order schemes could not.
"""

import math
from typing import ClassVar
from unittest import TestCase

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import root

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CyclePoint,
    CycleSeed,
    _Arc,
    _jacobian_monodromy,
    _multiplier_disagreement,
)
from discrecontinual_equations.continuation.periodic_orbit import (
    PeriodicOrbitSolution,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from tests.continuation.fields import (
    Alpha,
    BetaOne,
    BetaTwo,
    BogdanovTakensField,
    FoldOfCyclesField,
    HopfField,
    NeimarkSackerField,
    PeriodDoublingField,
    State,
    Time,
    VanDerPolField,
)

# The finite-difference oracle's own floor, as for the other two schemes.
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


def _circle(radius: float, intervals: int) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, intervals + 1)
    return np.column_stack(
        [
            radius * np.cos(2.0 * math.pi * grid),
            radius * np.sin(2.0 * math.pi * grid),
        ],
    )


def _orders(errors: list[float]) -> list[float]:
    return [math.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)]


class TestSelection(TestCase):
    """The scheme is chosen by name beside the other two, with its own point count."""

    def test_is_selectable_with_four_points_by_default(self):
        continuation = CycleContinuation(
            _equation(HopfField, 2, 0.5),
            0,
            10,
            collocation="gauss_legendre",
        )
        assert continuation.collocation == "gauss_legendre"
        assert continuation.collocation_points == 4

    def test_takes_its_point_count(self):
        for points in (2, 3, 4):
            continuation = CycleContinuation(
                _equation(HopfField, 2, 0.5),
                0,
                10,
                collocation="gauss_legendre",
                collocation_points=points,
            )
            assert continuation.collocation_points == points

    def test_the_other_schemes_have_one_point_per_interval(self):
        for scheme in ("trapezoidal", "hermite_simpson"):
            continuation = CycleContinuation(
                _equation(HopfField, 2, 0.5),
                0,
                10,
                collocation=scheme,
            )
            assert continuation.collocation_points == 1

    def test_refuses_a_point_count_for_a_scheme_without_one(self):
        with pytest.raises(ValueError, match="collocation_points"):
            CycleContinuation(
                _equation(HopfField, 2, 0.5),
                0,
                10,
                collocation="hermite_simpson",
                collocation_points=3,
            )

    def test_refuses_a_point_count_that_is_not_a_positive_integer(self):
        for points in (0, -1, 2.0, True, "4"):
            with pytest.raises(ValueError, match="positive integer"):
                CycleContinuation(
                    _equation(HopfField, 2, 0.5),
                    0,
                    10,
                    collocation="gauss_legendre",
                    collocation_points=points,  # type: ignore[arg-type]
                )

    def test_the_default_is_still_hermite_simpson(self):
        continuation = CycleContinuation(_equation(HopfField, 2, 0.5), 0, 10)
        assert continuation.collocation == "hermite_simpson"


class TestJacobian(TestCase):
    """The wider interval block against finite differences, on and off uniform."""

    def _disagreement(self, points: int, mesh: np.ndarray | None) -> float:
        intervals, dimension = 6, 4
        nodes = intervals * points + 1
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
            collocation="gauss_legendre",
            collocation_points=points,
        )
        continuation._dimension = dimension  # noqa: SLF001 (trace sets this)
        if mesh is not None:
            continuation._mesh = mesh  # noqa: SLF001 (no public way to grade it)
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

    def test_matches_finite_differences_on_a_uniform_mesh(self):
        for points in (2, 3, 4):
            assert self._disagreement(points, None) < _JACOBIAN_AGREEMENT

    def test_matches_finite_differences_on_a_graded_mesh(self):
        # The continuation runs it uniform, but nothing in the blocks assumes so.
        mesh = np.array([0.0, 0.05, 0.2, 0.3, 0.6, 0.85, 1.0])
        for points in (2, 4):
            assert self._disagreement(points, mesh) < _JACOBIAN_AGREEMENT


class TestPeriodOrder(TestCase):
    """Oracle 1: the exact Hopf cycle's period error falls ``2^(2m)`` per doubling.

    The Hopf normal form has period exactly 2 pi at every mu, so the last traced
    point's period is the continuation's discretisation error outright, as in the
    fourth-order test beside the other schemes. The meshes are coarser for larger
    ``m`` so that the error stays above the corrector's 1e-10 tolerance.
    """

    _MESHES: ClassVar[dict[int, tuple[int, ...]]] = {
        2: (8, 16, 32),
        3: (4, 8, 16),
        4: (3, 6, 12),
    }

    def _period_error(self, points: int, intervals: int) -> float:
        mu, steps = 0.5, 4
        continuation = CycleContinuation(
            _equation(HopfField, 2, mu),
            0,
            intervals,
            phase_index=1,
            collocation="gauss_legendre",
            collocation_points=points,
        )
        seed = CycleSeed(_circle(math.sqrt(mu), intervals), 6.0, mu)
        trace, _ = continuation.trace(seed, 0.05, steps)
        assert len(trace) == steps + 1
        return abs(trace[-1].solution.period - 2.0 * math.pi)

    def test_order_is_twice_the_point_count(self):
        for points, meshes in self._MESHES.items():
            errors = [self._period_error(points, n) for n in meshes]
            for order in _orders(errors):
                assert abs(order - 2 * points) < 0.5, (points, errors)


class TestJacobianMultipliers(TestCase):
    """Oracle 2: multipliers off the collocation Jacobian, against the oracles.

    On the Hopf cycle the nontrivial multiplier is ``exp(-4 pi mu)`` exactly, so
    its error is measured outright; the Hermite monodromy of DEQ-29, walking the
    cubic interpolant between the same mesh points, is the cross-check. The
    trivial multiplier cannot measure the order on that cycle - the normal form
    is rotation-equivariant, so the discrete map carries the rotation's generator
    onto itself and the trivial multiplier is one to rounding - so its drift is
    measured on van der Pol, whose orbit has no such symmetry.
    """

    def _hopf_point(self, points: int, intervals: int, mu: float = 0.25):
        continuation = CycleContinuation(
            _equation(HopfField, 2, mu),
            0,
            intervals,
            phase_index=1,
            collocation="gauss_legendre",
            collocation_points=points,
        )
        seed = CycleSeed(_circle(math.sqrt(mu), intervals), 2.0 * math.pi, mu)
        trace, _ = continuation.trace(seed, 0.05, 0)
        return continuation, trace[0]

    def _nontrivial_error(self, points: int, intervals: int) -> float:
        _, point = self._hopf_point(points, intervals)
        nontrivial = np.sort(point.multipliers.real)[0]
        return abs(nontrivial - math.exp(-4.0 * math.pi * 0.25))

    def test_nontrivial_multiplier_converges_at_twice_the_point_count(self):
        for points, meshes in {2: (8, 16, 32), 3: (8, 16, 32), 4: (3, 6, 12)}.items():
            errors = [self._nontrivial_error(points, n) for n in meshes]
            for order in _orders(errors):
                assert abs(order - 2 * points) < 0.5, (points, errors)

    def test_agrees_with_the_hermite_monodromy_to_its_order(self):
        # The cross-check is fourth order whatever m is, so agreement is to the
        # lower of the two orders: the Hermite monodromy's own error.
        gaps = []
        for intervals in (10, 20, 40):
            continuation, point = self._hopf_point(4, intervals)
            hermite = np.sort(continuation.hermite_multipliers(point).real)
            jacobian = np.sort(point.multipliers.real)
            exact = np.array([math.exp(-math.pi), 1.0])
            assert np.max(np.abs(jacobian - exact)) < np.max(np.abs(hermite - exact))
            gaps.append(float(np.max(np.abs(jacobian - hermite))))
        for order in _orders(gaps):
            assert order > 3.5, gaps

    def test_trivial_multiplier_is_exact_on_the_equivariant_cycle(self):
        # Exact for the exact discrete cycle; the solve stops at a residual
        # of 1e-10, and the drift follows where it stopped: 1.7e-15 at m = 3
        # on 8 intervals but 1.5e-11 on 16 before DEQ-106, 1.2e-12 and 1.2e-15
        # after it, when the seed is corrected on Gauss-Legendre directly.
        _, point = self._hopf_point(3, 8)
        assert np.min(np.abs(point.multipliers - 1.0)) < 1.0e-10

    def _van_der_pol(self, points: int, intervals: int):
        mu = 1.0
        continuation = CycleContinuation(
            _equation(VanDerPolField, 2, mu),
            0,
            intervals,
            phase_index=1,
            collocation="gauss_legendre",
            collocation_points=points,
        )
        # Clockwise, as van der Pol turns, so the period comes out positive.
        seed = CycleSeed(_circle(2.0, intervals) * np.array([1.0, -1.0]), 6.66, mu)
        trace, _ = continuation.trace(seed, 0.05, 0)
        return trace[0]

    def test_trivial_multiplier_drift_falls_at_least_at_twice_the_point_count(self):
        # Faster, in fact: 1.0e-1, 3.4e-4, 1.4e-9 for m = 2 at 10, 20, 40 -
        # orders 8.2 and 17.9 - because the discrete map is the flow of a
        # modified field to all orders, so its trivial multiplier stays at one.
        for points in (2, 3):
            drifts = [
                float(np.min(np.abs(self._van_der_pol(points, n).multipliers - 1.0)))
                for n in (10, 20, 40)
            ]
            for order in _orders(drifts):
                assert order > 2 * points - 0.5, (points, drifts)

    def test_floquet_error_is_not_the_trivial_drift_and_covers_the_true_error(self):
        # The reason the point carries its error: at m = 2 on 40 intervals the
        # trivial drift is 1.4e-9 and the nontrivial multiplier is off by 8e-8,
        # so the drift would call the set fifty times better than it is.
        reference = np.min(np.abs(self._van_der_pol(4, 160).multipliers))
        point = self._van_der_pol(2, 40)
        truth = abs(float(np.min(np.abs(point.multipliers))) - reference)
        drift = float(np.min(np.abs(point.multipliers - 1.0)))
        assert drift < truth < point.floquet_error


def _bogdanov_takens(b1: float) -> DifferentialEquation:
    parameters = [BetaOne(value=b1), BetaTwo(value=0.5)]
    return DifferentialEquation(
        variables=[State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=BogdanovTakensField(
            variables=[State(), State()],
            parameters=parameters,
            results=[State(), State()],
            time=None,
        ),
    )


def _unstable_cycle(
    equation: DifferentialEquation,
    b1: float,
    intervals: int,
) -> tuple[np.ndarray, float, float]:
    """The film's seed: the unstable cycle, as an attractor of reversed time.

    Also the exact nontrivial multiplier, the oracle: a planar cycle's is
    ``exp`` of the divergence ``b2 + x`` integrated over one period, carried
    along the same run as a third component.
    """

    def reversed_flow(_t: float, state: np.ndarray) -> list[float]:
        return [-v for v in equation.derivative.eval(point=list(state), time=None)]

    settled = solve_ivp(
        reversed_flow,
        [0.0, 400.0],
        [-math.sqrt(-b1) + 0.05, 0.0],
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.05,
    ).y[:, -1]

    def section(_t: float, state: np.ndarray) -> float:
        return float(state[1])

    section.direction = -1.0  # type: ignore[attr-defined] (solve_ivp reads it)

    def with_integral(t: float, state: np.ndarray) -> list[float]:
        return [*reversed_flow(t, state[:2]), float(state[0])]

    run = solve_ivp(
        with_integral,
        [0.0, 200.0],
        [*settled, 0.0],
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-13,
        events=section,
        dense_output=True,
        max_step=0.05,
    )
    crossings = run.t_events[0]
    period = float(crossings[1] - crossings[0])
    times = crossings[0] + np.linspace(0.0, period, intervals + 1)
    integral = float(run.sol(crossings[1])[2] - run.sol(crossings[0])[2])
    exact = math.exp(0.5 * period + integral)
    return run.sol(times)[:2].T[::-1], period, exact


class TestBogdanovTakensOrbit(TestCase):
    """Oracle 3 on the film's hardest orbit, b1 = -0.4554 near the homoclinic.

    Hermite-Simpson on 80 uniform nodes leaves 0.089% there, the film's worst;
    the bar is 0.01% on 80 mesh intervals. Measured against the exact
    multiplier, not against the scheme's own estimate of its error: the larger
    of the trivial multiplier's drift and the nontrivial one's relative error.
    """

    _B1 = -0.4554

    def _error(self, collocation: str, intervals: int = 80) -> float:
        equation = _bogdanov_takens(self._B1)
        states, period, exact = _unstable_cycle(equation, self._B1, intervals)
        continuation = CycleContinuation(
            equation,
            0,
            intervals,
            phase_index=1,
            collocation=collocation,
        )
        trace, _ = continuation.trace(CycleSeed(states, period, self._B1), 0.02, 0)
        multipliers = trace[0].multipliers
        trivial = int(np.argmin(np.abs(multipliers - 1.0)))
        nontrivial = multipliers[1 - trivial]
        return max(
            float(abs(multipliers[trivial] - 1.0)),
            float(abs(nontrivial - exact)) / exact,
        )

    def test_hermite_simpson_is_the_baseline(self):
        assert self._error("hermite_simpson") > 1.0e-4

    def test_gauss_legendre_on_80_intervals_meets_the_bar(self):
        # 1.2e-10 measured (the point's own bound reads 1.9e-7), against
        # 8.9e-4 under Hermite-Simpson.
        assert self._error("gauss_legendre") < 1.0e-4

    def test_gauss_legendre_meets_it_on_hermite_simpsons_unknowns(self):
        # 20 intervals of four points are the 81 states of 80 nodes:
        # 1.3e-5 measured (the bound reads 8.1e-5).
        assert self._error("gauss_legendre", intervals=20) < 1.0e-4


class TestSingularBlock(TestCase):
    """An interval block too singular to solve refuses the point, not the trace."""

    def test_reads_no_multipliers_and_reports_an_infinite_error(self):
        intervals, points, mu = 6, 2, 0.5
        continuation = CycleContinuation(
            _equation(HopfField, 2, mu),
            0,
            intervals,
            phase_index=1,
            collocation="gauss_legendre",
            collocation_points=points,
        )
        continuation._dimension = 2  # noqa: SLF001 (trace sets this)
        nodes = intervals * points + 1
        grid = np.linspace(0.0, 1.0, nodes)
        states = math.sqrt(mu) * np.column_stack(
            [np.cos(2.0 * math.pi * grid), np.sin(2.0 * math.pi * grid)],
        )
        unknowns = np.concatenate([states.ravel(), [2.0 * math.pi], [mu]])
        block = np.zeros((intervals, points * 2, (points + 1) * 2))
        continuation._assembled = (unknowns, block)  # noqa: SLF001 (the cache)
        point = continuation._point(unknowns, nodes)  # noqa: SLF001
        assert np.all(np.isnan(point.multipliers))
        assert point.floquet_error == math.inf
        assert not point.resolved()


def _gauss_nodes(points: int) -> np.ndarray:
    """Gauss-Legendre nodes on [0, 1] by Golub-Welsch, not by ``leggauss``."""
    k = np.arange(1, points)
    off = k / np.sqrt(4.0 * k * k - 1.0)
    return 0.5 * (np.linalg.eigvalsh(np.diag(off, 1) + np.diag(off, -1)) + 1.0)


def _lagrange(grid: np.ndarray, at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lagrange basis on ``grid`` and its derivative at ``at``, by products."""
    values = np.ones((at.size, grid.size))
    slopes = np.zeros((at.size, grid.size))
    for j, node in enumerate(grid):
        others = np.delete(grid, j)
        scale = np.prod(node - others)
        for i, z in enumerate(at):
            values[i, j] = np.prod(z - others) / scale
            slopes[i, j] = (
                sum(np.prod(z - np.delete(others, k)) for k in range(others.size))
                / scale
            )
    return values, slopes


def _van_der_pol_field(mu: float):
    def flow(_t: float, state: np.ndarray) -> list[float]:
        return [state[1], mu * (1.0 - state[0] ** 2) * state[1] - state[0]]

    return flow


def _van_der_pol_reference(mu: float):
    """Period and dense orbit, starting where ``y`` falls through zero, by DOP853."""
    flow = _van_der_pol_field(mu)
    settled = solve_ivp(
        flow,
        [0.0, 100.0],
        [2.0, 0.0],
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-13,
    ).y[:, -1]

    def section(_t: float, state: np.ndarray) -> float:
        return float(state[1])

    section.direction = -1.0  # type: ignore[attr-defined] (solve_ivp reads it)
    run = solve_ivp(
        flow,
        [0.0, 30.0],
        settled,
        method="DOP853",
        rtol=1.0e-13,
        atol=1.0e-14,
        events=section,
        dense_output=True,
    )
    start, end = run.t_events[0][0], run.t_events[0][1]
    period = float(end - start)
    return period, lambda fraction: run.sol(start + fraction * period).T


def _van_der_pol_point(points: int, intervals: int, mu: float = 1.0):
    period, orbit = _van_der_pol_reference(mu)
    continuation = CycleContinuation(
        _equation(VanDerPolField, 2, mu),
        0,
        intervals,
        phase_index=1,
        collocation="gauss_legendre",
        collocation_points=points,
    )
    seed = CycleSeed(orbit(np.linspace(0.0, 1.0, intervals + 1)), period, mu)
    trace, _ = continuation.trace(seed, 0.05, 0)
    return continuation, trace[0], period, orbit


class TestIndependentDiscretisation(TestCase):
    """The scheme against one built here from scratch, on van der Pol at mu = 1.

    Nodes by Golub-Welsch rather than ``leggauss``, the basis by products rather
    than a Vandermonde inverse: if the two agree, the nodes, the basis, the
    local time scaling ``h T`` and the interval windows are all what AUTO's
    collocation says they are.
    """

    # An odd count, so that a sign flipped in every transfer matrix shows.
    _POINTS, _INTERVALS, _MU = 3, 7, 1.0

    def _scheme(self):
        nodes = _gauss_nodes(self._POINTS)
        values, slopes = _lagrange(np.linspace(0.0, 1.0, self._POINTS + 1), nodes)
        return values, slopes, 1.0 / self._INTERVALS, _van_der_pol_field(self._MU)

    def test_its_solution_satisfies_the_independent_collocation_equations(self):
        _, point, _, _ = _van_der_pol_point(self._POINTS, self._INTERVALS)
        values, slopes, step, flow = self._scheme()
        states, period = point.solution.states, point.solution.period
        defects = []
        for i in range(self._INTERVALS):
            window = states[i * self._POINTS : (i + 1) * self._POINTS + 1]
            for value, slope in zip(values @ window, slopes @ window, strict=True):
                defects.append(slope - step * period * np.array(flow(0.0, value)))
        assert np.max(np.abs(defects)) < 1.0e-9

    def test_condensed_monodromy_is_the_derivative_of_the_discrete_period_map(self):
        # The map is stepped here interval by interval, each step a nonlinear
        # solve for the interval's other points, and differentiated by central
        # differences - nothing of the branch's Jacobian is used.
        continuation, point, _, _ = _van_der_pol_point(self._POINTS, self._INTERVALS)
        values, slopes, step, flow = self._scheme()
        states, period = point.solution.states, point.solution.period
        width = self._POINTS

        def advance(start: np.ndarray) -> np.ndarray:
            current = start
            for i in range(self._INTERVALS):

                def defect(rest: np.ndarray, left=current) -> np.ndarray:
                    window = np.vstack([left, rest.reshape(width, 2)])
                    return np.concatenate(
                        [
                            slope - step * period * np.array(flow(0.0, value))
                            for value, slope in zip(
                                values @ window,
                                slopes @ window,
                                strict=True,
                            )
                        ],
                    )

                guess = states[i * width + 1 : (i + 1) * width + 1] + (
                    current - states[i * width]
                )
                current = root(defect, guess.ravel(), tol=1.0e-14).x[-2:]
            return current

        shift = 1.0e-6
        differenced = np.column_stack(
            [
                (advance(states[0] + shift * e) - advance(states[0] - shift * e))
                / (2.0 * shift)
                for e in np.eye(2)
            ],
        )
        unknowns = np.concatenate([states.ravel(), [period], [self._MU]])
        continuation._blocks(unknowns, states.shape[0])  # noqa: SLF001
        assembled = continuation._assembled  # noqa: SLF001 (the cached blocks)
        assert assembled is not None
        condensed = _jacobian_monodromy(assembled[1], 2)
        assert np.max(np.abs(condensed - differenced)) < 1.0e-7


class TestSuperconvergence(TestCase):
    """De Boor and Swartz: order ``2m`` at the mesh points, ``m + 1`` between.

    Against van der Pol at mu = 1 by DOP853 at rtol 1e-13, with ``m = 3``. The
    gap between the two is the whole point of collocating at Gauss points: the
    polynomial in between is only as good as its degree, but its values at the
    mesh are those of the ``m``-stage Gauss Runge-Kutta method. Measured 3.5e-9
    and 1.35e-5 on 80 intervals, orders 5.94 and 3.79 from 40.
    """

    def _errors(self, intervals: int) -> tuple[float, float]:
        points = 3
        _, point, period, orbit = _van_der_pol_point(points, intervals)
        states = point.solution.states
        mesh_error = float(
            np.max(np.abs(states[::points] - orbit(point.mesh[::points]))),
        )
        fine = np.linspace(0.0, 1.0, 21)
        values, _ = _lagrange(np.linspace(0.0, 1.0, points + 1), fine)
        uniform = 0.0
        for i in range(intervals):
            polynomial = values @ states[i * points : (i + 1) * points + 1]
            exact = orbit((i + fine) / intervals)
            uniform = max(uniform, float(np.max(np.abs(polynomial - exact))))
        assert abs(point.solution.period - period) < 1.0e-6
        return mesh_error, uniform

    def test_mesh_points_converge_at_2m_and_the_polynomial_at_m_plus_one(self):
        coarse, fine = self._errors(40), self._errors(80)
        mesh_order = math.log2(coarse[0] / fine[0])
        uniform_order = math.log2(coarse[1] / fine[1])
        assert mesh_order > 5.5, (coarse, fine)
        assert 3.4 < uniform_order < 4.6, (coarse, fine)
        assert fine[0] < 1.0e-3 * fine[1]


class TestExactBranches(TestCase):
    """Whole continued branches whose multipliers are known in closed form.

    The fold-of-cycles normal form has ``r^2 = s`` with ``mu = s^2 - s``, period
    2 pi, and nontrivial multiplier ``exp(4 pi s (1 - 2 s))``: the branch must
    round the fold at ``mu = -1/4`` by pseudo-arclength, every multiplier must be
    right along the way, and the point's error must cover it. The four-
    dimensional fields add a complex pair ``exp(2 pi (nu +- i omega))``.
    """

    def _circle_states(self, radius: float, intervals: int, extra: int):
        return np.column_stack(
            [_circle(radius, intervals), np.zeros((intervals + 1, extra))],
        )

    def test_rounds_the_fold_with_exact_multipliers_and_a_covering_error(self):
        intervals, mu = 8, -0.1
        radius = math.sqrt((1.0 + math.sqrt(1.0 + 4.0 * mu)) / 2.0)
        continuation = CycleContinuation(
            _equation(FoldOfCyclesField, 2, mu),
            0,
            intervals,
            phase_index=1,
            collocation="gauss_legendre",
            collocation_points=4,
        )
        seed = CycleSeed(_circle(radius, intervals), 2.0 * math.pi, mu)
        points, bifurcations = continuation.trace(seed, 0.06, 60, direction=-1.0)
        lower = 0
        for point in points:
            squared = float(np.max(np.sum(point.solution.states**2, axis=1)))
            root_ = math.sqrt(max(1.0 + 4.0 * point.parameter, 0.0))
            s = (1.0 + root_) / 2.0 if squared > 0.5 else (1.0 - root_) / 2.0
            lower += squared < 0.5
            assert abs(squared - s) < 1.0e-6
            exact = np.array([1.0, math.exp(4.0 * math.pi * s * (1.0 - 2.0 * s))])
            error = _multiplier_disagreement(point.multipliers, exact)
            assert error < 1.0e-5
            assert error < point.floquet_error
        assert lower > 10  # it went round the fold onto the unstable cycles
        folds = [b.parameter for b in bifurcations if b.kind == "fold_of_cycles"]
        assert len(folds) == 1
        assert abs(folds[0] + 0.25) < 2.0e-4

    def test_complex_multipliers_of_the_torus_and_flip_fields(self):
        for field_type, rotation, kind in (
            (NeimarkSackerField, 0.3, "neimark_sacker"),
            (PeriodDoublingField, 0.5, "period_doubling"),
        ):
            intervals = 8
            continuation = CycleContinuation(
                _equation(field_type, 4, -0.1),
                0,
                intervals,
                phase_index=1,
                collocation="gauss_legendre",
                collocation_points=4,
            )
            states = self._circle_states(math.sqrt(0.5), intervals, 2)
            points, bifurcations = continuation.trace(
                CycleSeed(states, 2.0 * math.pi, -0.1),
                0.04,
                12,
            )
            for point in points:
                nu = point.parameter
                exact = np.array(
                    [
                        1.0,
                        math.exp(-2.0 * math.pi),
                        np.exp(2.0 * math.pi * complex(nu, rotation)),
                        np.exp(2.0 * math.pi * complex(nu, -rotation)),
                    ],
                )
                assert _multiplier_disagreement(point.multipliers, exact) < 1.0e-7
            assert [b.kind for b in bifurcations] == [kind]
            assert abs(bifurcations[0].parameter) < 2.0e-3


class TestMultiplierDisagreement(TestCase):
    """Relative to the larger of one and the multiplier, absolute below one."""

    def test_is_relative_for_a_large_multiplier(self):
        gap = _multiplier_disagreement(np.array([140.0, 1.0]), np.array([1.0, 140.14]))
        assert abs(gap - 1.0e-3) < 1.0e-12

    def test_is_absolute_for_a_small_one(self):
        gap = _multiplier_disagreement(np.array([1.0e-3, 1.0]), np.array([1.0, 2.0e-3]))
        assert abs(gap - 1.0e-3) < 1.0e-15


class TestCollapsedCycle(TestCase):
    """A cycle collapsed onto one state is not resolved, whatever the scheme.

    An under-resolved seed can converge to the degenerate solutions every
    collocation admits: all states equal at period zero, or an equilibrium at
    any period. The monodromy is then the identity or within ``|T|`` of it, so
    the trivial drift and the Hermite disagreement both read about zero and
    the point used to pass :meth:`CyclePoint.resolved` - on van der Pol at
    ``mu = 2`` seeded from the exact orbit on 8 intervals, under all three
    schemes.
    """

    def test_a_point_with_no_extent_reports_an_infinite_error(self):
        states = np.tile([1.3, 0.0], (9, 1))
        solution = PeriodicOrbitSolution(states, -4.0e-20)
        for error in (None, 0.0):
            point = CyclePoint(0.5, solution, np.ones(2, dtype=complex), None, error)
            assert point.floquet_error == math.inf
            assert not point.resolved()

    def test_the_collapse_an_under_resolved_seed_reaches_is_refused(self):
        """Since DEQ-54 the seed path refuses the collapse before it is a point.

        Whatever comes back is a cycle with extent and a positive period, and
        if the collapse came back anyway the gate would still refuse it.
        """
        period, orbit = _van_der_pol_reference(2.0)
        continuation = CycleContinuation(
            _equation(VanDerPolField, 2, 2.0),
            0,
            8,
            phase_index=1,
            collocation="gauss_legendre",
            collocation_points=4,
        )
        seed = CycleSeed(orbit(np.linspace(0.0, 1.0, 9)), period, 2.0)
        points, _ = continuation.trace(seed, 0.05, 0)
        for point in points:
            assert point.solution.period > 0.0
            assert np.ptp(point.solution.states) > 1.0e-6


class TestGateIsSound(TestCase):
    """``resolved()`` must not pass a cycle whose multipliers are worse than it says.

    Along the fold-of-cycles branch on coarse meshes, where the Hermite
    cross-check alone read up to 2.1 times low at ``m = 2`` on 4 intervals and
    6.1 times low at ``m = 3`` on 3, and let cycles with true errors of 3.1e-2
    and 3.4e-2 through the default 2e-2. The true error is taken against the
    nearer of the two exact cycles at the point's parameter - the more
    charitable reading close to the fold, where the discrete fold sits a little
    away from the exact one - and points past the exact fold, where there is no
    exact cycle, are skipped.
    """

    def _worst_ratio(self, points_per_interval: int, intervals: int) -> float:
        mu = -0.1
        radius = math.sqrt((1.0 + math.sqrt(1.0 + 4.0 * mu)) / 2.0)
        continuation = CycleContinuation(
            _equation(FoldOfCyclesField, 2, mu),
            0,
            intervals,
            phase_index=1,
            collocation="gauss_legendre",
            collocation_points=points_per_interval,
        )
        seed = CycleSeed(_circle(radius, intervals), 2.0 * math.pi, mu)
        points, _ = continuation.trace(seed, 0.06, 60, direction=-1.0)
        worst, checked = 0.0, 0
        for point in points:
            discriminant = 1.0 + 4.0 * point.parameter
            if discriminant < 0.0:
                continue
            truth = min(
                _multiplier_disagreement(
                    point.multipliers,
                    np.array([1.0, math.exp(4.0 * math.pi * s * (1.0 - 2.0 * s))]),
                )
                for s in (
                    (1.0 + math.sqrt(discriminant)) / 2.0,
                    (1.0 - math.sqrt(discriminant)) / 2.0,
                )
            )
            worst = max(worst, truth / point.floquet_error)
            checked += 1
        assert checked > 50
        return worst

    def test_covers_the_true_error_at_two_points_on_four_intervals(self):
        assert self._worst_ratio(2, 4) <= 1.0

    def test_covers_the_true_error_at_three_points_on_three_intervals(self):
        assert self._worst_ratio(3, 3) <= 1.0
