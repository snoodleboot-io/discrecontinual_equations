"""Stochastic bifurcation, and the Lyapunov exponents both notions rest on.

The pitchfork with multiplicative noise has a phenomenological
bifurcation at a = s^2 / 2 and a dynamical one at a = 0; the stationary
density and the exponent are checked against both thresholds.
"""

from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.lyapunov import (
    LyapunovSettings,
    StochasticLyapunovSettings,
    StochasticSystem,
    lyapunov_spectrum,
    stochastic_lyapunov,
)
from discrecontinual_equations.continuation.stochastic import (
    DensityModes,
    Ito,
    LyapunovExponent,
    StationaryDensity,
    Stratonovich,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from tests.continuation.fields import (
    Alpha,
    ConstantDiffusion,
    DiagonalField,
    Diffusion,
    Drift,
    LorenzField,
    SpiralField,
    State,
    Time,
    _scalar,
)

_SADDLE_JACOBIAN = np.array([[0.0, 1.0], [1.0, 0.0]])


class TestLyapunovSpectrum(TestCase):
    """Numerical Lyapunov exponents and the dynamical (D) bifurcation."""

    def _spiral(self, mu: float) -> DifferentialEquation:
        parameters = [Alpha(value=mu)]
        return DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=SpiralField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )

    def test_top_exponent_tracks_parameter(self):
        settings = LyapunovSettings(dt=0.01, horizon=6000, transient=100)
        for mu in (-0.2, 0.0, 0.1):
            spectrum = lyapunov_spectrum(
                self._spiral(mu).derivative,
                np.array([1.0, 0.0]),
                settings=settings,
            )
            assert abs(spectrum.top - mu) < 5.0e-3

    def test_dynamical_bifurcation_sign_change(self):
        settings = LyapunovSettings(dt=0.01, horizon=6000, transient=100)
        below = lyapunov_spectrum(
            self._spiral(-0.15).derivative,
            np.array([1.0, 0.0]),
            settings=settings,
        )
        above = lyapunov_spectrum(
            self._spiral(0.1).derivative,
            np.array([1.0, 0.0]),
            settings=settings,
        )
        assert below.top < 0.0 < above.top

    def test_diagonal_spectrum_is_exact(self):
        equation = DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=[],
            derivative=DiagonalField(
                variables=[State(), State()],
                parameters=[],
                results=[State(), State()],
                time=None,
            ),
        )
        spectrum = lyapunov_spectrum(
            equation.derivative,
            np.array([1.0, 1.0]),
            settings=LyapunovSettings(dt=0.01, horizon=20000, transient=50),
        )
        assert abs(spectrum.exponents[0] + 0.5) < 1.0e-2
        assert abs(spectrum.exponents[1] + 1.0) < 1.0e-2

    def test_lorenz_top_exponent(self):
        equation = DifferentialEquation(
            variables=[State(), State(), State()],
            time=Time(),
            parameters=[],
            derivative=LorenzField(
                variables=[State(), State(), State()],
                parameters=[],
                results=[State(), State(), State()],
                time=None,
            ),
        )
        spectrum = lyapunov_spectrum(
            equation.derivative,
            np.array([1.0, 1.0, 1.0]),
            count=1,
            settings=LyapunovSettings(dt=0.005, horizon=30000, transient=3000),
        )
        assert abs(spectrum.top - 0.906) < 5.0e-2


class TestStochasticLyapunov(TestCase):
    """Simulated Lyapunov exponent of a noisy flow and the stochastic D-bifurcation."""

    def _system(self, convention, alpha: float, sigma: float) -> StochasticSystem:
        return StochasticSystem(
            _scalar(Drift, alpha, sigma),
            _scalar(Diffusion, alpha, sigma),
            convention,
        )

    def _top(self, convention, alpha: float, sigma: float) -> float:
        settings = StochasticLyapunovSettings(
            dt=0.005,
            horizon=80000,
            transient=0,
            seed=7,
        )
        return stochastic_lyapunov(
            self._system(convention, alpha, sigma),
            np.array([0.0]),
            settings=settings,
        ).top

    def test_convention_shift_is_half_diffusion_squared(self):
        # Stratonovich minus Ito equals s^2 / 2 exactly (shared Brownian path cancels).
        alpha, sigma = 0.3, 0.6
        difference = self._top(Stratonovich(), alpha, sigma) - self._top(
            Ito(),
            alpha,
            sigma,
        )
        assert abs(difference - 0.5 * sigma * sigma) < 2.0e-2

    def test_ito_matches_closed_form(self):
        alpha, sigma = 0.3, 0.6
        oracle = LyapunovExponent(
            _scalar(Drift, alpha, sigma),
            _scalar(Diffusion, alpha, sigma),
            Ito(),
        ).at(0.0)
        assert abs(self._top(Ito(), alpha, sigma) - oracle) < 5.0e-2

    def test_noise_shifts_dynamical_bifurcation(self):
        # Deterministically neutral (alpha = 0) but stabilised by noise; unstable above.
        sigma = 0.6
        assert self._top(Ito(), 0.0, sigma) < 0.0
        assert self._top(Ito(), 2.0 * 0.5 * sigma * sigma + 0.18, sigma) > 0.0

    def test_zero_noise_recovers_drift(self):
        assert abs(self._top(Ito(), 0.2, 0.0) - 0.2) < 1.0e-2

    def test_spectrum_on_noisy_lorenz_attractor(self):
        # Full spectrum on a bounded chaotic attractor under additive noise. The sum
        # equals the exact phase-space divergence -(10 + 1 + 8/3); the top stays
        # positive (chaos persists) and the middle exponent is the neutral flow.
        divergence = -(10.0 + 1.0 + 8.0 / 3.0)
        settings = StochasticLyapunovSettings(
            dt=0.005,
            horizon=25000,
            transient=5000,
            seed=7,
        )
        for sigma in (0.0, 0.3):
            system = StochasticSystem(
                LorenzField(
                    variables=[State(), State(), State()],
                    parameters=[Alpha(value=sigma)],
                    results=[State(), State(), State()],
                    time=None,
                ),
                ConstantDiffusion(
                    variables=[State(), State(), State()],
                    parameters=[Alpha(value=sigma)],
                    results=[State(), State(), State()],
                    time=None,
                ),
                Ito(),
            )
            spectrum = stochastic_lyapunov(
                system,
                np.array([1.0, 1.0, 1.0]),
                count=3,
                settings=settings,
            ).exponents
            assert spectrum[0] > 0.6
            assert abs(spectrum[1]) < 0.1
            assert abs(float(spectrum.sum()) - divergence) < 0.5


class TestStochasticBifurcation(TestCase):
    def test_phenomenological_threshold(self):
        sigma = 0.6
        grid = np.linspace(1.0e-3, 3.0, 3000)
        drift = _scalar(Drift, 0.0, sigma)
        diffusion = _scalar(Diffusion, 0.0, sigma)
        below = self._dominant_mode(drift, diffusion, grid, 0.10)
        above = self._dominant_mode(drift, diffusion, grid, 0.30)
        assert below < 0.05
        assert above > 0.05
        threshold = self._mode_threshold(drift, diffusion, grid)
        assert abs(threshold - sigma * sigma / 2.0) < 1.0e-2

    def test_dynamical_threshold(self):
        sigma = 0.6
        drift = _scalar(Drift, 0.0, sigma)
        diffusion = _scalar(Diffusion, 0.0, sigma)
        exponent = LyapunovExponent(drift, diffusion, Stratonovich())
        for alpha in (-0.2, 0.0, 0.25):
            drift.parameters[0].value = alpha
            assert abs(exponent.at(0.0) - alpha) < 1.0e-4

    def _dominant_mode(self, drift, diffusion, grid, alpha):
        drift.parameters[0].value = alpha
        xs, density = StationaryDensity(
            drift,
            diffusion,
            Stratonovich(),
            grid,
        ).evaluate()
        return DensityModes(xs, density).dominant_mode()

    def _mode_threshold(self, drift, diffusion, grid):
        low, high = 0.0, 0.5
        for _ in range(30):
            mid = 0.5 * (low + high)
            if self._dominant_mode(drift, diffusion, grid, mid) > 0.02:
                high = mid
            else:
                low = mid
        return 0.5 * (low + high)
