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
    _equidistributed,
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

    def test_refuses_mesh_adaptation_on_a_scheme_that_cannot_use_it(self):
        with pytest.raises(ValueError, match="adapt"):
            CycleContinuation(
                _equation(HopfField, 2, 0.5),
                0,
                10,
                collocation="hermite_simpson",
                adapt=3,
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
            continuation._mesh = mesh  # noqa: SLF001 (adaptation sets this)
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
        for points, meshes in {2: (8, 16, 32), 3: (8, 16, 32)}.items():
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
) -> tuple[np.ndarray, float]:
    """The film's seed: the unstable cycle, as an attractor of reversed time."""

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
    run = solve_ivp(
        reversed_flow,
        [0.0, 200.0],
        settled,
        rtol=1.0e-10,
        atol=1.0e-12,
        events=section,
        dense_output=True,
        max_step=0.05,
    )
    crossings = run.t_events[0]
    period = float(crossings[1] - crossings[0])
    times = crossings[0] + np.linspace(0.0, period, intervals + 1)
    return run.sol(times).T[::-1], period


class TestBogdanovTakensOrbit(TestCase):
    """Oracle 3 on the film's hardest orbit, b1 = -0.4554 near the homoclinic.

    Hermite-Simpson on 80 uniform nodes leaves 0.089% there, the film's worst;
    the bar is 0.01% on 80 mesh intervals.
    """

    _B1 = -0.4554

    def _error(self, collocation: str, adapt: int, intervals: int = 80) -> float:
        equation = _bogdanov_takens(self._B1)
        states, period = _unstable_cycle(equation, self._B1, intervals)
        extra = {"adapt": adapt} if adapt else {}
        continuation = CycleContinuation(
            equation,
            0,
            intervals,
            phase_index=1,
            collocation=collocation,
            **extra,
        )
        trace, _ = continuation.trace(CycleSeed(states, period, self._B1), 0.02, 0)
        return trace[0].floquet_error

    def test_hermite_simpson_is_the_baseline(self):
        assert self._error("hermite_simpson", 0) > 1.0e-4

    def test_gauss_legendre_on_an_equidistributed_mesh_meets_the_bar(self):
        assert self._error("gauss_legendre", 1) < 1.0e-4

    def test_gauss_legendre_meets_it_on_hermite_simpsons_unknowns(self):
        # 20 intervals of four points are the 81 mesh states of 80 nodes.
        assert self._error("gauss_legendre", 0, intervals=20) < 1.0e-4


class TestEquidistribution(TestCase):
    """The Russell-Christiansen mesh, and a trace that moves its mesh as it goes."""

    def test_puts_the_monitor_equally_into_every_interval(self):
        mesh = np.linspace(0.0, 1.0, 9)
        # A fourth derivative that jumps hard between the third and fourth
        # intervals and is flat elsewhere: the monitor is large only there.
        highest = np.zeros((8, 2))
        highest[3:] = 1.0
        moved = _equidistributed(mesh, highest, 4)
        assert moved[0] == 0.0
        assert moved[-1] == 1.0
        assert np.all(np.diff(moved) > 0.0)
        steps = np.diff(moved)
        assert steps.min() < 0.5 * steps.max()

    def test_a_round_cycle_keeps_a_nearly_uniform_mesh(self):
        mu, intervals = 0.5, 12
        continuation = CycleContinuation(
            _equation(HopfField, 2, mu),
            0,
            intervals,
            phase_index=1,
            collocation="gauss_legendre",
            collocation_points=3,
            adapt=1,
        )
        seed = CycleSeed(_circle(math.sqrt(mu), intervals), 6.0, mu)
        trace, _ = continuation.trace(seed, 0.05, 3)
        assert len(trace) == 4
        steps = np.diff(continuation.mesh)
        assert steps.max() / steps.min() < 1.5
        assert abs(trace[-1].solution.period - 2.0 * math.pi) < 1.0e-5

    def test_grades_the_mesh_toward_the_saddle_on_bogdanov_takens(self):
        b1 = -0.4554
        equation = _bogdanov_takens(b1)
        states, period = _unstable_cycle(equation, b1, 40)
        continuation = CycleContinuation(
            equation,
            0,
            40,
            phase_index=1,
            collocation="gauss_legendre",
            adapt=1,
        )
        trace, _ = continuation.trace(CycleSeed(states, period, b1), 0.02, 0)
        steps = np.diff(trace[0].mesh)
        assert steps.max() / steps.min() > 2.0
        assert trace[0].solution.states.shape == (40 * 4 + 1, 2)
