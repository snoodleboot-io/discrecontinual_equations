"""Covariant Lyapunov vectors and finite-time exponents, against independent oracles.

Every expected value here comes from outside the code under test: closed-form
eigenvectors (Kuptsov and Parlitz 2012, Sec. V A, and ``numpy.linalg.eig``),
the tangent flow re-integrated by SciPy's DOP853 with the analytic Lorenz
Jacobian, the trace of that Jacobian, the published Lorenz spectrum, the
published angle statistics of the Henon and Lozi maps (Ginelli et al. 2007,
Fig. 1a) and of the Lorenz splitting (Kuptsov 2012, Fig. 1), Wolfe and
Samelson's observation that the second Lorenz vector is the flow direction,
and for the noisy pitchfork the exact law ``log|v(T)| / T = a - s^2/2 +
s W(T) / T`` of its linearisation.
"""

import math
from unittest import TestCase

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.covariant_lyapunov import (
    AngleDistribution,
    CovariantLyapunovSettings,
    FiniteTimeExponents,
    FlowTangent,
    MapTangent,
    covariant_lyapunov_vectors,
    stochastic_finite_time_exponents,
)
from discrecontinual_equations.continuation.lyapunov import (
    LyapunovSettings,
    MatrixNoiseSystem,
    StochasticLyapunovSettings,
    lyapunov_spectrum,
)
from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.continuation.root_finder import (
    RootFinderSettings,
    Secant,
)
from discrecontinual_equations.continuation.stochastic import Ito
from discrecontinual_equations.continuation.stochastic_threshold import (
    DynamicalThreshold,
    MeanTopExponent,
)
from tests.continuation.clv_fields import (
    HenonMap,
    KuptsovParlitzField,
    LoziMap,
    MatrixField,
)
from tests.continuation.fields import Diffusion, Drift, LorenzField, State, _scalar

# Lorenz at sigma = 10, rho = 28, beta = 8/3: Sprott's spectrum, which Wolfe and
# Samelson (0.91, 0, -14.58) and Kuptsov (0.90, 0, -14.57) reproduce, and the
# trace of the Jacobian, which the exponents must sum to exactly.
_LORENZ = (0.9056, 0.0, -14.5723)
_LORENZ_TRACE = -(10.0 + 1.0 + 8.0 / 3.0)


def _function(function_type, dimension):
    return function_type(
        variables=[State() for _ in range(dimension)],
        parameters=[],
        results=[State() for _ in range(dimension)],
        time=None,
    )


def _matrix_field(matrix):
    dimension = len(matrix)
    return MatrixField(
        matrix,
        [State() for _ in range(dimension)],
        [],
        [State() for _ in range(dimension)],
    )


def _aligned(vector, reference):
    """``|cos|`` of the angle between two directions, signs ignored."""
    return abs(vector @ reference) / (
        np.linalg.norm(vector) * np.linalg.norm(reference)
    )


def _lorenz_jacobian(state):
    x, y, z = state
    return np.array([[-10.0, 10.0, 0.0], [28.0 - z, -1.0, -x], [y, x, -8.0 / 3.0]])


def _lorenz_field(_time, state):
    x, y, z = state
    return [10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z]


def _carried(start, vector, span, backward):
    """Carry ``vector`` along the Lorenz orbit from ``start`` with SciPy alone.

    The orbit is always integrated forward (backward it is unstable at the
    rate 14.6); the tangent is then integrated along that orbit, forward from
    0 to ``span`` or backward from ``span`` to 0. Returns the orbit's end and
    the carried vector.
    """
    orbit = solve_ivp(
        _lorenz_field,
        (0.0, span),
        start,
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-12,
        dense_output=True,
    )
    interval = (span, 0.0) if backward else (0.0, span)
    tangent = solve_ivp(
        lambda time, v: _lorenz_jacobian(orbit.sol(time)) @ v,
        interval,
        vector,
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-14,
    )
    return orbit.y[:, -1], tangent.y[:, -1]


class TestLinearFlows(TestCase):
    """For ``x' = A x`` the covariant vectors are the eigenvectors, at every time."""

    def test_the_kuptsov_parlitz_example(self):
        result = covariant_lyapunov_vectors(
            FlowTangent(_function(KuptsovParlitzField, 3), dt=0.05, steps=2),
            np.zeros(3),
            settings=CovariantLyapunovSettings(150, 100, 150),
        )
        # Eq. 70 of the paper: the columns of Gamma.
        third = 1.0 / math.sqrt(3.0)
        expected = np.array([[1.0, third, 0.0], [0.0, third, 0.0], [0.0, third, 1.0]])
        for basis in result.vectors:
            for j in range(3):
                assert _aligned(basis[:, j], expected[:, j]) > 1.0 - 1.0e-10
        assert np.allclose(result.exponents, [1.0, -1.0, -3.0], atol=1.0e-4)
        assert np.allclose(result.covariant_exponents, [1.0, -1.0, -3.0], atol=1.0e-4)

    def test_the_finite_time_exponents_do_not_depend_on_the_window(self):
        result = covariant_lyapunov_vectors(
            FlowTangent(_function(KuptsovParlitzField, 3), dt=0.05, steps=2),
            np.zeros(3),
            settings=CovariantLyapunovSettings(50, 100, 50),
        )
        for window in (1, 7, 50):
            finite = result.finite_time_exponents(window)
            assert finite.window == pytest.approx(window * 0.1)
            assert np.allclose(finite.values, [1.0, -1.0, -3.0], atol=1.0e-4)

    def test_a_non_normal_matrix_against_numpy_eigenvectors(self):
        # Eigenvectors 20 degrees apart: a Gram-Schmidt basis is nowhere near them.
        directions = np.array([[1.0, math.cos(0.35)], [0.0, math.sin(0.35)]])
        matrix = directions @ np.diag([0.4, -0.7]) @ np.linalg.inv(directions)
        values, vectors = np.linalg.eig(matrix)
        order = np.argsort(values.real)[::-1]
        result = covariant_lyapunov_vectors(
            FlowTangent(_matrix_field(matrix), dt=0.05, steps=2),
            np.zeros(2),
            settings=CovariantLyapunovSettings(100, 50, 300),
        )
        for basis in result.vectors:
            for j, column in enumerate(order):
                assert _aligned(basis[:, j], vectors[:, column].real) > 1.0 - 1.0e-9
        assert np.allclose(result.exponents, values.real[order], atol=1.0e-4)
        angle = result.angles(0, 1)
        assert np.allclose(angle.values, 0.35, atol=1.0e-6)

    def test_fewer_vectors_are_the_leading_ones(self):
        result = covariant_lyapunov_vectors(
            FlowTangent(_function(KuptsovParlitzField, 3), dt=0.05, steps=2),
            np.zeros(3),
            count=2,
            settings=CovariantLyapunovSettings(50, 20, 50),
        )
        assert result.vectors.shape == (20, 3, 2)
        third = 1.0 / math.sqrt(3.0)
        assert _aligned(result.vectors[5][:, 1], np.array([third] * 3)) > 1 - 1.0e-10
        assert np.allclose(result.exponents, [1.0, -1.0], atol=1.0e-4)


class TestLorenz(TestCase):
    """One Lorenz run of 300 time units shared by every check on it."""

    @classmethod
    def setUpClass(cls):
        cls.result = covariant_lyapunov_vectors(
            FlowTangent(_function(LorenzField, 3), dt=0.01),
            np.array([1.0, 1.0, 1.0]),
            settings=CovariantLyapunovSettings(200, 3000, 200),
        )

    def test_the_spectrum_matches_the_published_one(self):
        exponents = self.result.exponents
        assert abs(exponents[0] - _LORENZ[0]) < 0.05
        assert abs(exponents[1] - _LORENZ[1]) < 0.02
        assert abs(exponents[2] - _LORENZ[2]) < 0.1

    def test_the_exponents_sum_to_the_trace(self):
        assert abs(self.result.exponents.sum() - _LORENZ_TRACE) < 1.0e-3
        assert abs(self.result.covariant_exponents.sum() - _LORENZ_TRACE) < 1.0e-2

    def test_covariant_growth_rates_are_benettins_up_to_the_boundary_term(self):
        # sum log gamma_j - sum log|R_jj| = log|c_jj(0)| - log|c_jj(T)|, and
        # c_jj is the cosine of an angle the record never closes to zero, so
        # the gap is bounded by a few units over T.
        gap = np.abs(self.result.covariant_exponents - self.result.exponents)
        assert np.all(gap < 0.05)

    def test_the_spectrum_agrees_with_the_library_estimator(self):
        benettin = lyapunov_spectrum(
            _function(LorenzField, 3),
            np.array([1.0, 1.0, 1.0]),
            settings=LyapunovSettings(dt=0.01, horizon=20000, transient=1000),
        ).exponents
        assert np.all(
            np.abs(self.result.exponents - benettin) < np.array([0.1, 0.05, 0.2])
        )

    def test_vectors_are_covariant_under_an_independent_integration(self):
        # The expanding and neutral vectors are carried forward from sample n
        # and must land on those at n + lag; the contracting one is carried
        # backward from n + lag and must land on the one at n. Carrying the
        # stable vector forward would amplify any error along v1 by exp(15.5 t),
        # and what would then be measured is the integrator, not covariance.
        result = self.result
        for n in (0, 377, 1500, 2990):
            for lag in (1, 7):
                span = lag * result.interval
                for j in range(3):
                    backward = j == 2
                    source, target = (n + lag, n) if backward else (n, n + lag)
                    end, carried = _carried(
                        result.states[n],
                        result.vectors[source][:, j],
                        span,
                        backward,
                    )
                    assert np.allclose(end, result.states[n + lag], atol=1.0e-3)
                    assert (
                        _aligned(carried, result.vectors[target][:, j]) > 1.0 - 1.0e-8
                    )

    def test_the_neutral_vector_is_the_flow_direction(self):
        # Wolfe and Samelson (2007, Sec. 4.1) found 0.02 degrees.
        lorenz = _function(LorenzField, 3)
        field = np.array([lorenz.eval(list(x)) for x in self.result.states])
        cosines = [
            _aligned(field[n], self.result.vectors[n][:, 1]) for n in range(len(field))
        ]
        angles = np.degrees(np.arccos(np.clip(cosines, 0.0, 1.0)))
        assert np.median(angles) < 0.02
        assert np.max(angles) < 0.5

    def test_the_stable_direction_never_meets_the_centre_unstable_plane(self):
        # Kuptsov (2012), Fig. 1: p(d2) stays off the origin (the Lorenz
        # attractor is singular hyperbolic) while p(d1) touches it.
        stable = self.result.splitting_angles(2)
        unstable = self.result.splitting_angles(1)
        assert stable.minimum > 0.1
        assert unstable.minimum < 0.1 * stable.minimum
        pair = self.result.angles(0, 2)
        assert pair.minimum >= stable.minimum - 1.0e-12

    def test_finite_time_exponents_tile_back_to_the_covariant_ones(self):
        for window in (10, 100, 1000):
            finite = self.result.finite_time_exponents(window)
            assert finite.values.shape == (3000 // window, 3)
            assert np.allclose(finite.mean, self.result.covariant_exponents)
        short = self.result.finite_time_exponents(10)
        # Over one time unit the leading exponent is often negative; over ten
        # it is not, and the spread shrinks with the window.
        assert 1.0 - short.fraction_positive()[0] > 0.05
        longer = self.result.finite_time_exponents(100)
        assert longer.spread[0] < 0.5 * short.spread[0]


class TestPlanarMaps(TestCase):
    """Ginelli et al. (2007), Fig. 1a: Henon reaches tangency, Lozi does not."""

    def _angles(self, map_type):
        result = covariant_lyapunov_vectors(
            MapTangent(_function(map_type, 2)),
            np.array([0.1, 0.1]),
            settings=CovariantLyapunovSettings(200, 8000, 100),
        )
        return result, result.angles(0, 1)

    def test_henon_has_near_tangencies(self):
        result, angles = self._angles(HenonMap)
        assert abs(result.exponents[0] - 0.4192) < 0.02
        assert abs(result.exponents.sum() - math.log(0.3)) < 1.0e-6
        assert angles.minimum < 0.02

    def test_lozi_is_bounded_away_from_tangency(self):
        result, angles = self._angles(LoziMap)
        assert abs(result.exponents.sum() - math.log(0.3)) < 1.0e-6
        assert angles.minimum > 0.3


class TestNoisyPitchfork(TestCase):
    """``dx = (a x - x^3) dt + s x dW`` (Ito) at its invariant origin.

    The linearisation's finite-time exponent is ``a - s^2/2 + s W(T)/T``:
    normal, with spread ``s / sqrt(T)``, and with a median that crosses zero at
    ``a = s^2/2`` - the dynamical threshold :class:`DynamicalThreshold` locates
    from the exponent itself.
    """

    _SIGMA = 0.6

    def _system(self, a):
        return MatrixNoiseSystem(
            _scalar(Drift, a, self._SIGMA),
            NoiseMatrix([_scalar(Diffusion, a, self._SIGMA)]),
            Ito(),
        )

    def _finite(self, a, windows=24):
        return stochastic_finite_time_exponents(
            self._system(a),
            np.array([0.0]),
            500,
            windows,
            StochasticLyapunovSettings(dt=0.01, transient=0, seed=40),
        )

    def _secant(self):
        return Secant(
            RootFinderSettings(
                value_tolerance=1.0e-6,
                fraction_tolerance=1.0e-3,
                max_iterations=4,
            ),
        )

    def test_the_distribution_has_the_exact_spread(self):
        finite = self._finite(0.1, windows=60)
        assert finite.window == pytest.approx(5.0)
        expected = self._SIGMA / math.sqrt(finite.window)
        # 60 draws: the sample spread is within 25 percent of the truth at
        # better than three standard errors.
        assert abs(finite.spread[0] - expected) < 0.25 * expected
        assert abs(
            finite.mean[0] - (0.1 - 0.5 * self._SIGMA**2)
        ) < 3.0 * expected / math.sqrt(60)

    def test_the_median_crosses_zero_at_the_dynamical_threshold(self):
        exact = 0.5 * self._SIGMA**2
        median = DynamicalThreshold(
            lambda a: float(self._finite(a).median[0]),
            self._secant(),
        ).locate(0.0, 0.4)
        assert median is not None
        exponent = DynamicalThreshold(
            MeanTopExponent(
                self._system,
                np.array([0.0]),
                StochasticLyapunovSettings(dt=0.01, horizon=12000, transient=0),
                seeds=(0,),
            ).at,
            self._secant(),
        ).locate(0.0, 0.4)
        assert exponent is not None
        # The median of 24 windows of 5 has a Monte-Carlo error of
        # 1.25 s / sqrt(120) = 0.07, the exponent over 120 of s / sqrt(120)
        # = 0.055; both are centred on s^2 / 2.
        assert abs(median - exact) < 0.2
        assert abs(exponent - exact) < 0.17
        assert abs(median - exponent) < 0.25


class TestRefusals(TestCase):
    def test_a_window_longer_than_the_record_is_refused(self):
        result = covariant_lyapunov_vectors(
            FlowTangent(_function(KuptsovParlitzField, 3), dt=0.05, steps=2),
            np.zeros(3),
            settings=CovariantLyapunovSettings(5, 10, 5),
        )
        with pytest.raises(ValueError, match="does not fit"):
            result.finite_time_exponents(11)
        with pytest.raises(ValueError, match="does not fit"):
            result.finite_time_exponents(0)
        with pytest.raises(ValueError, match="both sides"):
            result.splitting_angles(3)

    def test_bad_counts_and_settings_are_refused(self):
        dynamics = FlowTangent(_function(KuptsovParlitzField, 3))
        with pytest.raises(ValueError, match="count"):
            covariant_lyapunov_vectors(dynamics, np.zeros(3), count=4)
        with pytest.raises(ValueError, match="sample"):
            CovariantLyapunovSettings(samples=0)
        with pytest.raises(ValueError, match="negative"):
            CovariantLyapunovSettings(transient=-1)
        with pytest.raises(ValueError, match="step"):
            FlowTangent(_function(KuptsovParlitzField, 3), steps=0)

    def test_a_window_off_the_renormalisation_grid_is_refused(self):
        system = MatrixNoiseSystem(
            _scalar(Drift, 0.1, 0.5),
            NoiseMatrix([_scalar(Diffusion, 0.1, 0.5)]),
            Ito(),
        )
        with pytest.raises(ValueError, match="multiple"):
            stochastic_finite_time_exponents(system, np.zeros(1), 15, 3)
        with pytest.raises(ValueError, match="window"):
            stochastic_finite_time_exponents(system, np.zeros(1), 10, 0)


class TestSummaries(TestCase):
    def test_angle_and_finite_time_summaries(self):
        angles = AngleDistribution(np.array([0.1, 0.2, 0.3, 0.4]))
        assert angles.minimum == pytest.approx(0.1)
        assert angles.median == pytest.approx(0.25)
        assert angles.fraction_below(0.25) == pytest.approx(0.5)
        assert angles.quantile(1.0) == pytest.approx(0.4)
        finite = FiniteTimeExponents(
            np.array([[1.0, -2.0], [-1.0, -4.0], [3.0, 0.0]]), 2.0
        )
        assert np.allclose(finite.median, [1.0, -2.0])
        assert np.allclose(finite.mean, [1.0, -2.0])
        assert np.allclose(finite.fraction_positive(), [2.0 / 3.0, 0.0])
        assert np.allclose(finite.quantile(0.0), [-1.0, -4.0])
        assert finite.window == 2.0
