"""Manifolds of cycles by Fourier-Taylor parameterization (DEQ-38).

The oracles are in ``cycle_fields``: cycles whose stable and unstable manifolds
are known in closed form, pulled back through a polynomial diffeomorphism so
every Fourier and Taylor coefficient is exercised. Independently of them, the
charts are checked against the flow itself (DOP853 at tight tolerance), and the
Floquet-normal-form solve against a dense Fourier-Galerkin solve of the
homological equations that never forms the Floquet basis.
"""

import math
from functools import cache
from itertools import pairwise
from unittest import TestCase

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CycleSeed,
)
from discrecontinual_equations.continuation.cycle_manifold import (
    CycleManifold,
    CycleManifoldChart,
    FloquetBundles,
    FourierCycle,
)
from discrecontinual_equations.continuation.jet import Jet
from discrecontinual_equations.continuation.manifold import (
    ManifoldSelection,
    StableManifold,
    UnstableManifold,
)
from discrecontinual_equations.continuation.periodic_orbit import (
    PeriodicOrbitSolution,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.cycle_fields import (
    MU,
    OMEGA,
    TWIST_A,
    TWIST_B,
    Conjugated,
    SaddleCycle,
    TwistedCycle,
    circle,
    conjugate,
    fold_equation,
    saddle_stable,
    saddle_unstable,
    twisted_stable,
)

PERIOD = 2.0 * math.pi / OMEGA


def _plain(field_type: type[DeterministicFunction]) -> DeterministicFunction:
    return field_type(variables=[], parameters=[], results=[])


def _ring_seed(shape, nodes: int = 41, period: float = PERIOD) -> PeriodicOrbitSolution:
    """A rough collocation-style seed: a perturbed circle, mapped by ``shape``."""
    states = np.array(
        [
            shape(
                np.array(
                    [
                        1.02 * math.cos(2.0 * math.pi * s),
                        math.sin(2.0 * math.pi * s),
                        0.01,
                    ],
                ),
            )
            for s in np.linspace(0.0, 1.0, nodes)
        ],
    )
    return PeriodicOrbitSolution(states, 1.01 * period)


@cache
def _saddle_cycle() -> tuple[DeterministicFunction, FourierCycle]:
    field = Conjugated(_plain(SaddleCycle))
    return field, FourierCycle.refine(field, _ring_seed(conjugate), 24)


@cache
def _saddle_chart(stable: bool, order: int = 14, harmonics: int = 48):  # noqa: FBT001
    field, cycle = _saddle_cycle()
    selection = StableManifold() if stable else UnstableManifold()
    return CycleManifold(order=order, harmonics=harmonics).compute_on(
        field,
        cycle,
        selection,
    )


def _align(chart: CycleManifoldChart, exact, mapped=conjugate) -> tuple[float, float]:
    """The phase offset and ``sigma`` scale taking the chart onto ``exact``.

    The chart's phase origin is wherever the refined cycle put it, and its
    ``sigma`` is the exact one times the bundle's scaling (Castelli et al. I,
    Remark 2.8); both are read off the cycle and the first-order term only.
    """
    start = chart.point(0.0, 0.0)
    grid = np.linspace(0.0, PERIOD, 4001)
    rough = grid[
        np.argmin([np.linalg.norm(mapped(exact(t, 0.0)) - start) for t in grid])
    ]
    offset = minimize_scalar(
        lambda t: np.linalg.norm(mapped(exact(t, 0.0)) - start),
        bracket=(rough - 1.0e-3, rough + 1.0e-3),
        tol=1.0e-12,
    ).x
    step = 1.0e-6
    slope = (mapped(exact(offset, step)) - mapped(exact(offset, -step))) / (2 * step)
    first = chart.grid[1][0]
    return float(offset), float(first @ slope / (slope @ slope))


def _flow(field, start: np.ndarray, time: float) -> np.ndarray:
    solution = solve_ivp(
        lambda _t, x: field.eval(point=list(x), time=None),
        (0.0, time),
        start,
        method="DOP853",
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    return solution.y[:, -1]


class TestSaddleCycleOracle(TestCase):
    """The ticket's linear-cycle oracle, made nonlinear and conjugated."""

    def test_cycle_refines_to_the_exact_period(self):
        field, cycle = _saddle_cycle()
        assert abs(cycle.period - PERIOD) < 1.0e-12
        assert cycle.residual(field) < 1.0e-11

    def test_floquet_exponents_are_exact(self):
        field, cycle = _saddle_cycle()
        bundles = FloquetBundles.compute(field, cycle, 48)
        np.testing.assert_allclose(bundles.exponents, [-2.0, 0.0, MU], atol=1.0e-10)
        assert bundles.orientable == [True, True, True]

    def test_stable_chart_matches_the_closed_form(self):
        chart = _saddle_chart(stable=True)
        offset, scale = _align(chart, saddle_stable)
        assert abs(chart.exponent + 2.0) < 1.0e-10
        for theta in np.linspace(0.0, 2.0 * PERIOD, 9):
            for sigma in (-0.1, 0.05, 0.1):
                exact = conjugate(saddle_stable(theta + offset, scale * sigma))
                assert np.linalg.norm(chart.point(theta, sigma) - exact) < 1.0e-9

    def test_unstable_chart_matches_the_closed_form(self):
        chart = _saddle_chart(stable=False)
        offset, scale = _align(chart, saddle_unstable)
        assert abs(chart.exponent - MU) < 1.0e-10
        for theta in np.linspace(0.0, 2.0 * PERIOD, 9):
            for sigma in (-0.1, 0.05, 0.1):
                exact = conjugate(saddle_unstable(theta + offset, scale * sigma))
                assert np.linalg.norm(chart.point(theta, sigma) - exact) < 1.0e-9

    def test_coefficients_are_the_fourier_taylor_expansion(self):
        chart = _saddle_chart(stable=True)
        coefficients = chart.coefficients
        assert coefficients.shape == (chart.order + 1, 97, 3)
        theta, sigma = 0.7, 0.08
        wavenumbers = np.arange(-48, 49)
        modes = np.exp(1j * math.pi * wavenumbers * theta / chart.period)
        series = np.einsum("nkd,k,n->d", coefficients, modes, sigma ** np.arange(15))
        assert np.max(np.abs(series.imag)) < 1.0e-12
        assert np.linalg.norm(series.real - chart.point(theta, sigma)) < 1.0e-12

    def test_orientable_chart_has_period_t(self):
        chart = _saddle_chart(stable=True)
        difference = chart.point(0.4 + PERIOD, 0.1) - chart.point(0.4, 0.1)
        assert np.linalg.norm(difference) < 1.0e-12


class TestMobiusBand(TestCase):
    """A twisted bundle: multipliers negative, chart antiperiodic in sigma."""

    @classmethod
    def setUpClass(cls):
        cls.field = Conjugated(_plain(TwistedCycle))
        cls.cycle = FourierCycle.refine(cls.field, _ring_seed(conjugate), 24)
        cls.chart = CycleManifold(order=8, harmonics=48).compute_on(
            cls.field,
            cls.cycle,
            StableManifold(),
        )

    def test_multipliers_are_negative(self):
        bundles = FloquetBundles.compute(self.field, self.cycle, 48)
        np.testing.assert_allclose(
            bundles.exponents, [TWIST_A, 0.0, TWIST_B], atol=1e-10
        )
        assert bundles.orientable == [False, True, False]
        np.testing.assert_allclose(
            bundles.multipliers,
            [-math.exp(TWIST_A * PERIOD), 1.0, -math.exp(TWIST_B * PERIOD)],
            rtol=1.0e-10,
        )

    def test_chart_is_a_mobius_band(self):
        assert not self.chart.orientable
        for theta in (0.1, 1.7, 3.3):
            flipped = self.chart.point(theta + PERIOD, -0.1)
            assert np.linalg.norm(flipped - self.chart.point(theta, 0.1)) < 1.0e-12

    def test_chart_matches_the_closed_form(self):
        offset, scale = _align(self.chart, twisted_stable)
        for theta in np.linspace(0.0, 2.0 * PERIOD, 11):
            for sigma in (-0.2, 0.1, 0.2):
                exact = conjugate(twisted_stable(theta + offset, scale * sigma))
                assert np.linalg.norm(self.chart.point(theta, sigma) - exact) < 1.0e-10


class TestConvergence(TestCase):
    """The a-posteriori invariance error falls geometrically in order and modes."""

    def test_invariance_error_decays_geometrically_with_order(self):
        # The stable chart is (1 - 2 c sigma)^(-1/2) under h: radius of
        # convergence 1 / (2 c), so at radius nu the error falls like
        # (2 c nu)^N per order.
        _, scale = _align(_saddle_chart(stable=True), saddle_stable)
        radius = 0.25
        orders = [4, 6, 8, 10, 12, 14]
        errors = [
            _saddle_chart(stable=True, order=order).invariance_error(radius)
            for order in orders
        ]
        assert all(later < earlier for earlier, later in pairwise(errors))
        slope = np.polyfit(orders, np.log(errors), 1)[0]
        expected = math.log(2.0 * abs(scale) * radius)
        assert abs(slope - expected) < 0.25 * abs(expected)

    def test_invariance_error_decays_geometrically_with_modes(self):
        # The rational conjugation gives the cycle and its bundles infinitely
        # many Fourier modes, decaying geometrically; at a radius small enough
        # that the order-8 truncation is far below, the error tracks them.
        field, _ = _saddle_cycle()
        harmonics = [12, 16, 20, 24]
        errors = []
        for count in harmonics:
            cycle = FourierCycle.refine(field, _ring_seed(conjugate), count // 2)
            chart = CycleManifold(order=8, harmonics=count).compute_on(
                field,
                cycle,
                StableManifold(),
            )
            errors.append(chart.invariance_error(0.02))
        ratios = [later / earlier for earlier, later in pairwise(errors)]
        assert all(ratio < 0.2 for ratio in ratios)

    def test_pointwise_defect_is_bounded_by_the_invariance_error(self):
        chart = _saddle_chart(stable=True, order=8)
        bound = chart.invariance_error(0.2)
        for theta in np.linspace(0.0, 2.0 * PERIOD, 7):
            assert np.max(np.abs(chart.defect(theta, 0.2))) <= bound


class TestAgainstTheFlow(TestCase):
    """The conjugacy, checked against DOP853 rather than any closed form."""

    def test_stable_chart_conjugates_the_flow(self):
        field, _ = _saddle_cycle()
        chart = _saddle_chart(stable=True)
        for theta, sigma in ((0.3, 0.1), (2.9, -0.08), (6.1, 0.05)):
            flowed = _flow(field, chart.point(theta, sigma), 1.5)
            assert np.linalg.norm(flowed - chart.flow_image(theta, sigma, 1.5)) < 1.0e-9

    def test_unstable_chart_conjugates_the_backward_flow(self):
        field, _ = _saddle_cycle()
        chart = _saddle_chart(stable=False)
        for theta, sigma in ((0.3, 0.1), (4.0, -0.1)):
            flowed = _flow(field, chart.point(theta, sigma), -2.0)
            assert (
                np.linalg.norm(flowed - chart.flow_image(theta, sigma, -2.0)) < 1.0e-9
            )

    def test_repelling_cycle_of_the_fold_film(self):
        """A cycle from the library's own continuation, consumed through its API."""
        mu = -0.15
        inner = math.sqrt((1.0 - math.sqrt(1.0 + 4.0 * mu)) / 2.0)
        eq = fold_equation(mu)
        continuation = CycleContinuation(eq, 0, 60, phase_index=1, phase_value=0.0)
        points, _ = continuation.trace(
            CycleSeed(circle(inner, 60), 2.0 * math.pi, mu),
            0.01,
            1,
        )
        point = points[0]
        field = eq.derivative
        field.parameters[0].value = point.parameter
        chart = CycleManifold(order=10, harmonics=32).compute(
            field,
            point.solution,
            UnstableManifold(),
            mesh=point.mesh,
        )
        # r' = g(r) r with g = mu + r^2 - r^4 vanishing on the cycle, so the
        # radial exponent is g'(r) r = 2 r^2 - 4 r^4 at the cycle's radius.
        radius = math.sqrt((1.0 - math.sqrt(1.0 + 4.0 * point.parameter)) / 2.0)
        assert abs(chart.exponent - (2.0 * radius**2 - 4.0 * radius**4)) < 1.0e-9
        sigma = 0.5 * chart.radius(1.0e-12)
        for theta in (0.0, 2.0, 5.0):
            flowed = _flow(field, chart.point(theta, sigma), -3.0)
            assert (
                np.linalg.norm(flowed - chart.flow_image(theta, sigma, -3.0)) < 1.0e-9
            )


def _jet_remainder(field, coefficients: np.ndarray, order: int) -> np.ndarray:
    """``R_n`` at each node with the library's scalar ``Jet``, not ``GridSeries``."""
    points, dimension = coefficients.shape[1], coefficients.shape[2]
    remainder = np.zeros((points, dimension))
    for node in range(points):
        jets = [
            Jet([float(coefficients[n, node, i]) for n in range(order)] + [0.0])
            for i in range(dimension)
        ]
        values = field.eval(point=jets, time=None)
        remainder[node] = [float(np.real(v.coefficients[order])) for v in values]
    return remainder


class TestAgainstDirectGalerkin(TestCase):
    """Equation (21) against a dense solve of the homological equation (18).

    ``(d/dtheta + n lambda - Df(gamma)) a_n = R_n`` is solved as one dense
    Fourier-collocation system, with no Floquet basis, and ``R_n`` from the
    scalar Jet. Agreement checks the reducibility, not just the oracle.
    """

    def test_coefficients_agree(self):
        field, cycle = _saddle_cycle()
        chart = CycleManifold(order=6, harmonics=48).compute_on(
            field,
            cycle,
            StableManifold(),
        )
        grid = chart.grid
        points, dimension = grid.shape[1], grid.shape[2]
        length = 2.0 * chart.period
        wavenumbers = np.fft.fftfreq(points, 1.0 / points)
        spectrum = np.fft.fft(np.eye(points), axis=0)
        derivative = np.real(
            np.fft.ifft(
                (2j * math.pi * wavenumbers / length)[:, None] * spectrum, axis=0
            ),
        )
        jacobians = []
        for node in range(points):
            columns = []
            for column in range(dimension):
                jets = [
                    Jet([grid[0, node, i], 1.0 if i == column else 0.0])
                    for i in range(dimension)
                ]
                columns.append(
                    [float(np.real(v.coefficients[1])) for v in field.eval(jets)]
                )
            jacobians.append(np.array(columns).T)
        reference = grid[:2].copy()
        for order in range(2, chart.order + 1):
            matrix = np.kron(derivative, np.eye(dimension))
            matrix += order * chart.exponent * np.eye(points * dimension)
            for node in range(points):
                block = slice(node * dimension, (node + 1) * dimension)
                matrix[block, block] -= jacobians[node]
            padded = np.concatenate([reference, np.zeros((1, points, dimension))])
            remainder = _jet_remainder(field, padded, order)
            solved = np.linalg.solve(matrix, remainder.ravel()).reshape(
                points, dimension
            )
            reference = np.concatenate([reference, solved[None]])
            scale = np.max(np.abs(solved))
            assert np.max(np.abs(grid[order] - solved)) < 1.0e-9 * max(1.0, scale)


class _Nothing(ManifoldSelection):
    def indices(self, values):  # noqa: ARG002 (base signature)
        return []


class _Both(ManifoldSelection):
    def indices(self, values):
        return [i for i in range(len(values)) if values[i] != 0.0]


class RotatingCycle(DeterministicFunction):
    """A cycle whose normal plane turns a third of a revolution: complex multipliers."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x, y, z = point[0], point[1], point[2]
        radius = (x * x + y * y) ** 0.5
        offset = radius - 1.0
        turn = OMEGA / 3.0
        rdot = -0.5 * offset - turn * z
        zdot = -0.5 * z + turn * offset
        return [rdot * x / radius - OMEGA * y, rdot * y / radius + OMEGA * x, zdot]


class TestRefusals(TestCase):
    def test_selection_must_pick_exactly_one_exponent(self):
        field, cycle = _saddle_cycle()
        for selection in (_Nothing(), _Both()):
            with pytest.raises(ValueError, match="selection picks"):
                CycleManifold(order=3, harmonics=16).compute_on(field, cycle, selection)

    def test_complex_multipliers_are_refused(self):
        field = _plain(RotatingCycle)
        cycle = FourierCycle.refine(field, _ring_seed(lambda v: v), 12)
        with pytest.raises(ValueError, match="real Floquet exponents"):
            CycleManifold(order=3, harmonics=24).compute_on(
                field, cycle, StableManifold()
            )


class LorenzField(DeterministicFunction):
    """Lorenz at sigma = 10, beta = 8/3, rho = 22."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x, y, z = point[0], point[1], point[2]
        return [10.0 * (y - x), 22.0 * x - y - x * z, x * y - 8.0 / 3.0 * z]


class TestPublishedLorenzOrbit(TestCase):
    """Castelli, Lessard and Mireles James II, Section 4.3: rho = 22.

    They report the saddle cycle round ``C+`` with period ``T = 0.764386427``
    and stable exponent ``-13.861695717713566``, both rigorously enclosed.
    """

    def test_period_and_stable_exponent_match(self):
        field = _plain(LorenzField)
        centre = np.array([math.sqrt(8.0 / 3.0 * 21.0)] * 2 + [21.0])
        jacobian = np.array(
            [
                [-10.0, 10.0, 0.0],
                [1.0, -1.0, -centre[0]],
                [centre[1], centre[0], -8.0 / 3.0],
            ],
        )
        values, vectors = np.linalg.eig(jacobian)
        pair = int(np.argmax(values.imag))
        turn = np.exp(2j * math.pi * np.linspace(0.0, 1.0, 81))
        seed = centre + 10.0 * np.real(np.outer(turn, vectors[:, pair]))
        cycle = FourierCycle.refine(
            field,
            PeriodicOrbitSolution(seed, 2.0 * math.pi / values[pair].imag),
            40,
        )
        assert abs(cycle.period - 0.764386427) < 1.0e-9
        chart = CycleManifold(order=4, harmonics=80).compute_on(
            field,
            cycle,
            StableManifold(),
        )
        assert abs(chart.exponent + 13.861695717713566) < 1.0e-9


class TestScaling(TestCase):
    """The chart is unique only up to the bundle's scale (paper I, Remark 2.8)."""

    def test_rescaling_stretches_sigma(self):
        chart = _saddle_chart(stable=True, order=8)
        stretched = chart.rescaled(0.4)
        for theta, sigma in ((0.2, 0.1), (3.0, -0.2)):
            expected = chart.point(theta, 0.4 * sigma)
            assert np.linalg.norm(stretched.point(theta, sigma) - expected) < 1e-14

    def test_radius_is_where_the_last_term_reaches_the_tolerance(self):
        chart = _saddle_chart(stable=True, order=8)
        reach = chart.radius(1.0e-10)
        last = np.max(np.abs(chart.grid[-1])) * reach**chart.order
        assert abs(last - 1.0e-10) < 1.0e-14
