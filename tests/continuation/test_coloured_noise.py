"""Coloured noise by augmentation, against closed forms and the white-noise limit.

Oracle 2 (Gaussian): the OU-driven linear system's augmented pair has a stationary
covariance from its Lyapunov equation, ``var x = c^2 / (2 k (1 + k tau))`` in closed
form (Hänggi and Jung, Adv. Chem. Phys. 89, eqs. 3.28-3.33); the grid Fokker-Planck
solver and an ensemble of simulated paths are both held to it. Oracle 1 (white
limit): the coloured pitchfork's thresholds tend to the white Stratonovich ones the
library already locates - the dynamical one at ``a = 0``, the phenomenological one at
``a = s^2 / 2`` - and two exact statements hold at every ``tau`` on the way.
"""

from unittest import TestCase

import numpy as np
import pytest
from scipy import stats
from scipy.linalg import solve_continuous_lyapunov

from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.lyapunov import (
    MatrixNoiseSystem,
    StochasticLyapunovSettings,
)
from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.continuation.root_finder import (
    Bisection,
    RootFinderSettings,
    Secant,
)
from discrecontinual_equations.continuation.stochastic import (
    Ito,
    LyapunovExponent,
    Stratonovich,
)
from discrecontinual_equations.continuation.stochastic_threshold import (
    DynamicalThreshold,
    MeanTopExponent,
)
from discrecontinual_equations.continuation.topological_modes import (
    KernelDensity,
    ModeCountThreshold,
    TopologicalModes,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.coloured_noise import ColouredNoise
from discrecontinual_equations.solver.solver_config import StochasticConfig
from discrecontinual_equations.solver.stochastic.ensemble import StochasticEnsemble
from discrecontinual_equations.variable import Variable
from tests.continuation.coloured_fields import (
    LinearRelaxation,
    Pitchfork,
    PitchforkDrift,
    PitchforkNoise,
    scalar,
    state_noise,
    white_pitchfork_samples,
)

SIGMA = 0.6
P_THRESHOLD = SIGMA * SIGMA / 2.0
RATE, AMPLITUDE = 1.0, 1.0


def _closed_form_variance(tau: float) -> float:
    return AMPLITUDE**2 / (2.0 * RATE * (1.0 + RATE * tau))


def _augmented_lyapunov(tau: float, regulariser: float) -> np.ndarray:
    drift = np.array([[-RATE, AMPLITUDE], [0.0, -1.0 / tau]])
    noise = np.array([[0.0, regulariser], [1.0 / tau, 0.0]])
    return solve_continuous_lyapunov(drift, -noise @ noise.T)


def _equation(function) -> DifferentialEquation:
    return DifferentialEquation(
        variables=function.variables,
        time=Variable(name="t", discretization=[]),
        parameters=function.parameters,
        derivative=function,
    )


def _ensemble_final(coloured, initial, horizon, step, seed):
    config = StochasticConfig(
        start_time=0.0,
        end_time=horizon,
        step_size=step,
        random_seed=seed,
    )
    return StochasticEnsemble(config).solve(_equation(coloured), initial).final


class TestAugmentation(TestCase):
    def test_batch_and_pointwise_forms_agree(self):
        coloured = ColouredNoise(scalar(Pitchfork, 0.2, SIGMA), 0.3)
        points = np.random.default_rng(0).normal(size=(7, 2))
        rows = np.array([coloured.eval(list(point)) for point in points])
        assert np.allclose(coloured.eval_batch(points), rows, rtol=0.0, atol=1e-15)
        x, eta = points[:, 0], points[:, 1]
        assert np.allclose(rows[:, 0], 0.2 * x - x**3 + SIGMA * x * eta)
        assert np.allclose(rows[:, 1], -eta / 0.3)
        matrix = np.array(coloured.diffusion(list(points[0])))
        assert np.array_equal(matrix, [[0.0], [1.0 / 0.3]])
        assert np.array_equal(coloured.diffusion_batch(points)[3], matrix)

    def test_driver_columns_are_the_noise_matrix(self):
        coloured = ColouredNoise(scalar(Pitchfork, 0.2, SIGMA), 0.25)
        columns = NoiseMatrix(coloured.driver_columns()).columns_at(np.zeros(2))
        assert np.array_equal(columns, [[0.0, 4.0]])

    def test_stationary_drivers_have_the_stationary_variance(self):
        coloured = ColouredNoise(scalar(Pitchfork, 0.2, SIGMA), 0.4)
        states = coloured.stationary_drivers(
            np.zeros((200000, 1)),
            np.random.default_rng(1),
        )
        assert abs(np.var(states[:, 1]) / 1.25 - 1.0) < 0.01

    def test_refusals(self):
        with pytest.raises(ValueError, match="positive"):
            ColouredNoise(scalar(Pitchfork, 0.2, SIGMA), 0.0)
        coloured = ColouredNoise(scalar(Pitchfork, 0.2, SIGMA), 0.4, drivers=2)
        with pytest.raises(ValueError, match="vector form"):
            coloured.eval([0.1, 0.0, 0.0])


class TestLinearGaussianOracle(TestCase):
    """Oracle 2: the augmented pair is Gaussian, and its covariance is known."""

    tau = 0.5

    def _grid_covariance(self, regulariser: float) -> np.ndarray:
        coloured = ColouredNoise(scalar(LinearRelaxation, RATE, AMPLITUDE), self.tau)
        noise = NoiseMatrix(
            [*coloured.driver_columns(), state_noise(coloured.variables, regulariser)],
        )
        grid = DensityGrid.box([-3.5, -5.0], [3.5, 5.0], [91, 91])
        return StationaryFokkerPlanck(coloured, noise, Ito(), grid).solve().covariance()

    def test_lyapunov_solution_is_the_closed_form(self):
        exact = _augmented_lyapunov(self.tau, 0.0)
        assert abs(exact[0, 0] - _closed_form_variance(self.tau)) < 1e-12
        assert abs(exact[1, 1] - 0.5 / self.tau) < 1e-12

    def test_grid_density_has_the_lyapunov_covariance(self):
        # The augmented diffusion is singular (noise on eta only), which the grid
        # scheme refuses; a white column of size e on x makes it elliptic and
        # adds e^2 / (2k) to var x, exactly, through the same Lyapunov equation.
        for regulariser in (0.3, 0.5):
            grid = self._grid_covariance(regulariser)
            exact = _augmented_lyapunov(self.tau, regulariser)
            assert np.max(np.abs(grid - exact)) < 1.5e-2 * np.max(np.abs(exact))

    def test_removing_the_regulariser_recovers_the_closed_form(self):
        # var x is linear in e^2, so two solves extrapolate to e = 0.
        low, high = 0.3, 0.5
        at_low = self._grid_covariance(low)[0, 0]
        at_high = self._grid_covariance(high)[0, 0]
        slope = (at_high - at_low) / (high**2 - low**2)
        extrapolated = at_low - slope * low**2
        assert abs(extrapolated / _closed_form_variance(self.tau) - 1.0) < 2.0e-2

    def test_the_unregularised_pair_is_refused_by_the_grid(self):
        coloured = ColouredNoise(scalar(LinearRelaxation, RATE, AMPLITUDE), self.tau)
        grid = DensityGrid.box([-3.0, -3.0], [3.0, 3.0], [11, 11])
        solver = StationaryFokkerPlanck(
            coloured,
            NoiseMatrix(coloured.driver_columns()),
            Ito(),
            grid,
        )
        with pytest.raises(ValueError, match="singular"):
            solver.solve()

    def test_simulated_paths_have_the_closed_form_variance(self):
        coloured = ColouredNoise(scalar(LinearRelaxation, RATE, AMPLITUDE), self.tau)
        paths = 6000
        initial = coloured.stationary_drivers(
            np.zeros((paths, 1)),
            np.random.default_rng(5),
        )
        final = _ensemble_final(coloured, initial, 8.0, 0.01, 6)
        exact = _augmented_lyapunov(self.tau, 0.0)
        sample = np.cov(final.T)
        # Sampling error of a variance: sqrt(2 / paths) ~ 1.8%; Euler adds O(dt).
        assert abs(sample[0, 0] / exact[0, 0] - 1.0) < 0.06
        assert abs(sample[0, 1] / exact[0, 1] - 1.0) < 0.08


class TestWhiteNoiseLimit(TestCase):
    """Oracle 1: the coloured pitchfork's thresholds against the white ones."""

    def _exponent(self, tau: float) -> MeanTopExponent:
        def family(rate: float) -> MatrixNoiseSystem:
            coloured = ColouredNoise(scalar(Pitchfork, rate, SIGMA), tau)
            return MatrixNoiseSystem(
                coloured,
                NoiseMatrix(coloured.driver_columns()),
                Ito(),
            )

        step = min(0.01, tau / 10.0)
        settings = StochasticLyapunovSettings(
            dt=step,
            horizon=int(60.0 / step),
            transient=0,
        )
        return MeanTopExponent(family, np.array([0.0, 0.0]), settings, seeds=(3,))

    def test_dynamical_threshold_is_the_white_one_at_every_tau(self):
        # The trivial solution linearises to (a + s eta) dx: its exponent is a plus
        # s times the time-average of eta, the same offset at every a. White
        # Stratonovich: exactly a (LyapunovExponent below), threshold 0.
        white = LyapunovExponent(
            scalar(PitchforkDrift, 0.0, SIGMA),
            scalar(PitchforkNoise, 0.0, SIGMA),
            Stratonovich(),
        )
        assert abs(white.at(0.0)) < 1e-6
        for tau in (0.05, 1.0):
            exponent = self._exponent(tau)
            # Linear in a up to the Euler step: the estimate is the mean of
            # log|1 + (a + s eta) dt| / dt, whose curvature in a is O(dt).
            assert abs(exponent.at(0.3) - exponent.at(0.1) - 0.2) < 1e-3
            located = DynamicalThreshold(
                exponent.at,
                Secant(RootFinderSettings(1e-10, 1e-6, 20)),
            ).locate(-0.3, 0.3)
            assert located is not None
            # Offset: s times the mean of eta over T = 60, whose sd is 1/sqrt(T).
            assert abs(located) < 3.0 * SIGMA / np.sqrt(60.0)

    def test_second_moment_is_the_white_one_at_every_tau(self):
        # log x^2 is stationary and drifts at 2 (a - x^2) + 2 s eta: <x^2> = a.
        rate = 0.25
        for tau in (0.05, 1.0):
            coloured = ColouredNoise(scalar(Pitchfork, rate, SIGMA), tau)
            paths = 4000
            start = np.full((paths, 1), 0.5)
            initial = coloured.stationary_drivers(start, np.random.default_rng(7))
            final = _ensemble_final(coloured, initial, 30.0, min(0.01, tau / 10), 8)
            spread = np.std(final[:, 0] ** 2) / np.sqrt(paths)
            assert abs(np.mean(final[:, 0] ** 2) - rate) < 4.0 * spread + 2e-3

    def test_density_tends_to_the_white_one(self):
        # Kolmogorov-Smirnov distance of |x| from the white law x^2/s^2 ~ Gamma(a/s^2)
        # shrinks as tau does.
        rate = 0.25
        distances = []
        for tau in (0.5, 0.05):
            coloured = ColouredNoise(scalar(Pitchfork, rate, SIGMA), tau)
            paths = 4000
            initial = coloured.stationary_drivers(
                np.full((paths, 1), 0.5),
                np.random.default_rng(9),
            )
            final = _ensemble_final(coloured, initial, 30.0, min(0.01, tau / 10), 10)
            scaled = final[:, 0] ** 2 / SIGMA**2
            distances.append(
                stats.kstest(scaled, stats.gamma(rate / SIGMA**2).cdf).statistic
            )
        assert distances[1] < 0.5 * distances[0]
        assert distances[1] < 2.0 / np.sqrt(4000)

    def test_phenomenological_threshold_tends_to_the_white_one(self):
        # The same detector, the same bandwidth and the same number of samples on
        # both sides: exact white samples, and coloured paths at tau = 0.05 with
        # common random numbers across the parameter.
        paths, width, tau = 6000, 0.1, 0.05
        grid = DensityGrid.box([-2.5], [2.5], [251])
        generator = np.random.default_rng(13)
        signs = np.where(np.arange(paths) % 2 == 0, 1.0, -1.0)

        def coloured_modes(rate: float) -> TopologicalModes:
            coloured = ColouredNoise(scalar(Pitchfork, rate, SIGMA), tau)
            initial = coloured.stationary_drivers(
                (0.5 * signs)[:, None],
                np.random.default_rng(14),
            )
            final = _ensemble_final(coloured, initial, 25.0, 0.005, 15)
            return KernelDensity(final[:, 0], grid, bandwidth=width).modes()

        uniforms = generator.random(paths)

        def white_modes(rate: float) -> TopologicalModes:
            samples = white_pitchfork_samples(rate, SIGMA, uniforms, signs)
            return KernelDensity(samples, grid, bandwidth=width).modes()

        bisection = Bisection(RootFinderSettings(1e-14, 4e-3, 30))
        white = ModeCountThreshold(white_modes, bisection).locate(0.1, 0.4)
        coloured = ModeCountThreshold(coloured_modes, bisection).locate(0.1, 0.4)
        assert white is not None
        assert coloured is not None
        assert abs(white - (P_THRESHOLD + width**2)) < 0.04
        assert abs(coloured - white) < 0.03
