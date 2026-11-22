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

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CycleSeed,
    _Arc,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from tests.continuation.fields import (
    Alpha,
    BetaOne,
    BetaTwo,
    BogdanovTakensField,
    HopfField,
    NeimarkSackerField,
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
        _, point = self._hopf_point(3, 8)
        assert np.min(np.abs(point.multipliers - 1.0)) < 1.0e-12

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
