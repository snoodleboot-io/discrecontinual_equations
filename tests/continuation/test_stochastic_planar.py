"""Stochastic bifurcation above one dimension, against closed-form planar densities.

Every density here is checked against an exact stationary solution, not against
recorded output. Three families supply them.

*Linear drift with additive noise.* ``dx = A x dt + S dW`` is Gaussian with covariance
from ``A Sigma + Sigma A^T + D = 0``. For a rotating ``A`` with isotropic noise
symmetry forces ``Sigma = s^2 / (2 |mu|) I``; for a diagonal ``A`` and correlated noise
the solution is ``Sigma_ij = D_ij / (a_i + a_j)``, which is the only oracle that
exercises the off-diagonal part of the discretisation.

*Radially symmetric drift with isotropic noise.* If the drift is a radial field plus a
rotation and ``D = b(r) I``, the rotation is divergence-free and orthogonal to every
radial gradient, so it drops out of the stationary equation entirely and the radial
balance ``F p = (1/2) (b p)'`` integrates to ``p = (C / b) exp(integral 2 F / b dr)``.
That is the one-dimensional closed form of :mod:`~...continuation.stochastic`
reappearing in the radius, and it gives exact planar densities for the noisy Hopf
system:

* ``b = s^2`` (additive): ``p propto exp((mu r^2 - r^4 / 2) / s^2)``. The joint density
  peaks at ``r = sqrt(mu)`` for ``mu > 0`` and at the origin otherwise, so the
  phenomenological threshold is ``mu = 0`` - additive noise does *not* move it.
* ``b = s^2 (1 + k r^2)``: ``p propto (1 + k r^2)^(A - 1) exp(-r^2 / (k s^2))`` with
  ``A = (mu + 1/k) / (k s^2)``, whose peak leaves the origin at ``mu = k s^2`` (Ito) or
  ``mu = k s^2 / 2`` (Stratonovich). Multiplicative noise does move the threshold, by a
  known amount, in a case that stays elliptic everywhere.

For the dynamical side, ``dx = A x dt + s x dW`` factorises as
``x = exp(s W - s^2 t / 2) y`` with ``y' = A y``, so its top exponent is
``max Re eig(A) - s^2 / 2`` (Ito) or ``max Re eig(A)`` (Stratonovich) exactly, and with
``G = diag(s_i x_i)`` the components are independent geometric Brownian motions with
spectrum ``{a_i - s_i^2 / 2}``.
"""

import math
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.lyapunov import (
    MatrixNoiseSystem,
    StochasticLyapunovSettings,
    StochasticSystem,
    matrix_noise_lyapunov,
    stochastic_lyapunov,
)
from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.continuation.root_finder import (
    RootFinderSettings,
    Secant,
)
from discrecontinual_equations.continuation.stochastic import (
    Ito,
    StationaryDensity,
    Stratonovich,
)
from discrecontinual_equations.continuation.stochastic_threshold import (
    DynamicalThreshold,
    MeanTopExponent,
    PhenomenologicalThreshold,
    RadialCrater,
    StochasticScan,
    ThresholdCurve,
)
from tests.continuation.fields import (
    Alpha,
    ConstantNoiseColumn,
    DiagonalField,
    DiagonalGrowthField,
    DiagonalMultiplicativeColumn,
    Diffusion,
    Drift,
    HopfField,
    IsotropicNoiseColumn,
    Kappa,
    ScaledStateColumn,
    Sigma,
    SpiralField,
    State,
    _scalar,
)

_PLANE = 2


def _planar(field_type, parameters):
    return field_type(
        variables=[State(), State()],
        parameters=parameters,
        results=[State(), State()],
        time=None,
    )


def _indexed(column_type, index, parameters):
    return column_type(
        index,
        [State(), State()],
        parameters,
        [State(), State()],
        None,
    )


def _isotropic(sigma, kappa):
    """``s sqrt(1 + k r^2) I`` as two independent columns."""
    return NoiseMatrix(
        [
            _indexed(
                IsotropicNoiseColumn,
                index,
                [Alpha(value=0.0), Sigma(value=sigma), Kappa(value=kappa)],
            )
            for index in range(_PLANE)
        ],
    )


def _hopf_density(mu, sigma, kappa, convention, grid):
    return StationaryFokkerPlanck(
        _planar(HopfField, [Alpha(value=mu)]),
        _isotropic(sigma, kappa),
        convention,
        grid,
    )


def _radial_oracle(grid, profile):
    """Normalise a radial profile of ``r^2`` onto the grid for comparison."""
    first, second = np.meshgrid(grid.axis(0), grid.axis(1), indexing="ij")
    raw = profile(first * first + second * second)
    return raw / (raw.sum() * grid.cell_volume)


def _secant():
    return Secant(
        RootFinderSettings(
            value_tolerance=1.0e-7,
            fraction_tolerance=1.0e-5,
            max_iterations=6,
        ),
    )


class TestPlanarStationaryDensity(TestCase):
    """The grid solve of the stationary Fokker-Planck equation, against exact ones."""

    def test_rotating_ornstein_uhlenbeck_is_gaussian(self):
        mu, sigma = -1.0, 1.0
        grid = DensityGrid.box([-4.0, -4.0], [4.0, 4.0], [81, 81])
        density = StationaryFokkerPlanck(
            _planar(SpiralField, [Alpha(value=mu)]),
            _isotropic(sigma, 0.0),
            Ito(),
            grid,
        ).solve()
        variance = sigma * sigma / (2.0 * abs(mu))
        assert abs(density.mass() - 1.0) < 1.0e-9
        assert density.minimum() > 0.0
        covariance = density.covariance()
        assert abs(covariance[0][0] - variance) < 5.0e-3
        assert abs(covariance[1][1] - variance) < 5.0e-3
        assert abs(covariance[0][1]) < 1.0e-6
        exact = _radial_oracle(
            grid,
            lambda squared: np.exp(-squared / (2.0 * variance)),
        )
        error = np.max(np.abs(density.values - exact)) / exact.max()
        assert error < 5.0e-3
        assert np.allclose(density.dominant_mode(), [0.0, 0.0])
        assert len(density.interior_maxima()) == 1

    def test_refinement_is_second_order(self):
        # The error is discretisation error, so halving the spacing must quarter it.
        # That is the whole reason this route was taken over an ensemble: a sampling
        # error would not shrink on a known power of anything.
        mu, sigma = -1.0, 1.0
        variance = sigma * sigma / (2.0 * abs(mu))
        errors = []
        for count in (41, 81):
            grid = DensityGrid.box([-4.0, -4.0], [4.0, 4.0], [count, count])
            density = StationaryFokkerPlanck(
                _planar(SpiralField, [Alpha(value=mu)]),
                _isotropic(sigma, 0.0),
                Ito(),
                grid,
            ).solve()
            errors.append(abs(density.covariance()[0][0] - variance))
        assert errors[1] < errors[0]
        assert errors[0] / errors[1] > 3.5

    def test_correlated_noise_fills_the_off_diagonal(self):
        # Sigma_ij = D_ij / (a_i + a_j) for a diagonal drift; the only check on the
        # cross-diffusion stencil, which the radially symmetric oracles never touch.
        grid = DensityGrid.box([-5.0, -3.5], [5.0, 3.5], [101, 101])
        noise = NoiseMatrix(
            [
                _planar(ConstantNoiseColumn, [Sigma(value=1.0), Sigma(value=0.5)]),
                _planar(
                    ConstantNoiseColumn,
                    [Sigma(value=0.0), Sigma(value=math.sqrt(0.75))],
                ),
            ],
        )
        density = StationaryFokkerPlanck(
            _planar(DiagonalField, []),
            noise,
            Ito(),
            grid,
        ).solve()
        covariance = density.covariance()
        assert abs(covariance[0][0] - 1.0 / 1.0) < 5.0e-3
        assert abs(covariance[1][1] - 1.0 / 2.0) < 5.0e-3
        assert abs(covariance[0][1] - 0.5 / 1.5) < 5.0e-3

    def test_noisy_hopf_additive_density_matches_closed_form(self):
        mu, sigma = 0.5, 0.5
        grid = DensityGrid.box([-2.5, -2.5], [2.5, 2.5], [101, 101])
        density = _hopf_density(mu, sigma, 0.0, Ito(), grid).solve()
        exact = _radial_oracle(
            grid,
            lambda squared: np.exp(
                (mu * squared - 0.5 * squared * squared) / (sigma * sigma),
            ),
        )
        assert np.max(np.abs(density.values - exact)) / exact.max() < 5.0e-3
        radius = float(np.linalg.norm(density.dominant_mode()))
        assert abs(radius - math.sqrt(mu)) < 2.0 * grid.spacings[0]

    def test_a_ring_of_maxima_is_reported_as_a_chain(self):
        # The crater of a rotationally symmetric density is a circle of maxima, not a
        # point, and interior_maxima says so rather than picking one arbitrarily. Every
        # cell it returns sits on the analytic ring radius sqrt(mu).
        grid = DensityGrid.box([-2.5, -2.5], [2.5, 2.5], [81, 81])
        density = _hopf_density(0.5, 0.5, 0.0, Ito(), grid).solve()
        maxima = density.interior_maxima()
        assert len(maxima) > 8
        radii = [float(np.linalg.norm(position)) for position in maxima]
        assert max(radii) - min(radii) < 2.0 * grid.spacings[0]
        assert all(
            abs(radius - math.sqrt(0.5)) < 2.0 * grid.spacings[0] for radius in radii
        )

    def test_noisy_hopf_below_the_threshold_peaks_at_the_origin(self):
        grid = DensityGrid.box([-2.5, -2.5], [2.5, 2.5], [81, 81])
        density = _hopf_density(-0.3, 0.5, 0.0, Ito(), grid).solve()
        assert np.allclose(density.dominant_mode(), [0.0, 0.0])

    def test_state_dependent_noise_density_matches_closed_form(self):
        # Both conventions, since Stratonovich adds the radial drift k s^2 r / 2 and
        # the same closed form must then be reached from a shifted mu.
        kappa, sigma = 1.0, 0.6
        grid = DensityGrid.box([-2.5, -2.5], [2.5, 2.5], [101, 101])
        for convention, shift in (
            (Ito(), 0.0),
            (Stratonovich(), 0.5 * kappa * sigma**2),
        ):
            mu = 0.6
            density = _hopf_density(mu, sigma, kappa, convention, grid).solve()
            power = (mu + shift + 1.0 / kappa) / (kappa * sigma * sigma) - 1.0
            exact = _radial_oracle(
                grid,
                lambda squared, power=power: (
                    (1.0 + kappa * squared) ** power
                    * np.exp(-squared / (kappa * sigma * sigma))
                ),
            )
            assert np.max(np.abs(density.values - exact)) / exact.max() < 5.0e-3

    def test_one_dimensional_grid_agrees_with_the_closed_form(self):
        # The grid solve has to reproduce the formula it replaces where that formula
        # still applies, or the two notions of P-bifurcation would not be the same one.
        alpha, sigma = 0.3, 0.6
        axis = np.linspace(0.05, 3.0, 601)
        drift = _scalar(Drift, alpha, sigma)
        diffusion = _scalar(Diffusion, alpha, sigma)
        gridded = StationaryFokkerPlanck(
            drift,
            NoiseMatrix([diffusion]),
            Stratonovich(),
            DensityGrid([axis]),
        ).solve()
        _xs, closed = StationaryDensity(
            drift,
            diffusion,
            Stratonovich(),
            axis,
        ).evaluate()
        assert np.max(np.abs(gridded.values - closed)) / closed.max() < 5.0e-3

    def test_degenerate_diffusion_is_refused(self):
        # Scalar noise on a plane has D of rank one, so there is no density to find;
        # returning the junk a singular operator produces would be far worse.
        grid = DensityGrid.box([-2.0, -2.0], [2.0, 2.0], [21, 21])
        shared = NoiseMatrix(
            [_planar(ConstantNoiseColumn, [Sigma(value=0.4), Sigma(value=0.4)])],
        )
        problem = StationaryFokkerPlanck(
            _planar(SpiralField, [Alpha(value=-1.0)]),
            shared,
            Ito(),
            grid,
        )
        with pytest.raises(ValueError, match="singular"):
            problem.solve()

    def test_graded_axis_is_refused(self):
        with pytest.raises(ValueError, match="uniformly spaced"):
            DensityGrid([np.array([0.0, 1.0, 3.0])])


class TestPhenomenologicalThreshold(TestCase):
    """Locating the planar P-bifurcation, against thresholds known in closed form."""

    def _locate(self, sigma, kappa, convention, bracket):
        grid = DensityGrid.box([-2.5, -2.5], [2.5, 2.5], [61, 61])
        threshold = PhenomenologicalThreshold(
            lambda mu: _hopf_density(mu, sigma, kappa, convention, grid),
            RadialCrater([0.0, 0.0]),
            _secant(),
        )
        return threshold.locate(*bracket)

    def test_additive_noise_does_not_move_the_threshold(self):
        located = self._locate(0.5, 0.0, Ito(), (-0.4, 0.4))
        assert located is not None
        assert abs(located) < 1.0e-2

    def test_multiplicative_noise_moves_it_by_a_known_amount(self):
        kappa, sigma = 1.0, 0.6
        located = self._locate(sigma, kappa, Ito(), (0.0, 0.8))
        assert located is not None
        assert abs(located - kappa * sigma * sigma) < 1.0e-2

    def test_the_convention_halves_the_shift(self):
        kappa, sigma = 1.0, 0.6
        located = self._locate(sigma, kappa, Stratonovich(), (-0.2, 0.6))
        assert located is not None
        assert abs(located - 0.5 * kappa * sigma * sigma) < 1.0e-2

    def test_an_unbracketed_threshold_is_reported_as_missing(self):
        assert self._locate(0.5, 0.0, Ito(), (0.3, 0.9)) is None


def _multiplicative(sigma, convention):
    def family(mu):
        return MatrixNoiseSystem(
            _planar(SpiralField, [Alpha(value=mu)]),
            NoiseMatrix(
                [
                    _planar(
                        ScaledStateColumn,
                        [Alpha(value=mu), Sigma(value=sigma)],
                    ),
                ],
            ),
            convention,
        )

    return family


def _dynamical(sigma, convention, horizon, seeds):
    estimator = MeanTopExponent(
        _multiplicative(sigma, convention),
        np.array([1.0, 0.0]),
        StochasticLyapunovSettings(dt=0.005, horizon=horizon, transient=0),
        seeds,
    )
    return DynamicalThreshold(estimator.at, _secant())


class TestDynamicalThreshold(TestCase):
    """Locating the planar D-bifurcation, with common random numbers."""

    def test_the_convention_gap_is_exactly_half_the_variance(self):
        # The two conventions see the same Brownian path, so the Monte-Carlo offset
        # that displaces each located threshold is identical and cancels in the
        # difference. That difference is s^2 / 2 exactly, and is by far the most
        # accurate statement available about an absolute threshold located this way.
        sigma, horizon = 1.0, 15000
        ito = _dynamical(sigma, Ito(), horizon, (7,)).locate(-1.0, 1.5)
        stratonovich = _dynamical(sigma, Stratonovich(), horizon, (7,)).locate(
            -1.0,
            1.5,
        )
        assert ito is not None
        assert stratonovich is not None
        assert abs((ito - stratonovich) - 0.5 * sigma * sigma) < 1.0e-2

    def test_the_absolute_threshold_sits_at_half_the_variance(self):
        # Averaging four independent paths over a horizon of 75 time units leaves a
        # standard error of s / sqrt(4 T) = 0.041 on the exponent and so on the
        # located threshold; the tolerance is three of those plus the Euler-Maruyama
        # bias of order s^4 dt. It still separates s^2 / 2 from zero, which is the
        # claim that noise shifts the planar D-bifurcation at all.
        sigma = 1.0
        located = _dynamical(sigma, Ito(), 15000, (2, 3, 5, 7)).locate(-1.0, 1.5)
        assert located is not None
        assert abs(located - 0.5 * sigma * sigma) < 0.18
        assert located > 0.1

    def test_the_threshold_curve_is_the_expected_parabola(self):
        # Followed in the noise amplitude, the threshold is mu = s^2 / 2 displaced by
        # one common offset that is linear in s, so the located curve is
        # s^2 / 2 - c s for an unknown c. Its second difference over equally spaced s
        # kills both the unknown and the linear term and leaves d^2 exactly, which
        # pins the quadratic coefficient at 1/2 without ever estimating c.
        step = 0.3
        controls = (0.6, 0.9, 1.2)
        curve = ThresholdCurve(
            lambda sigma: _dynamical(sigma, Ito(), 15000, (7,)),
            (-1.0, 1.5),
            0.5,
        )
        points = curve.trace(controls)
        assert [point.control for point in points] == list(controls)
        located = [point.parameter for point in points]
        assert located[0] < located[1] < located[2]
        second = located[0] - 2.0 * located[1] + located[2]
        assert abs(second - step * step) < 2.0e-2


class TestMatrixNoiseLyapunov(TestCase):
    """The spectrum of a flow driven by several independent Brownian motions."""

    def _settings(self, horizon=20000):
        return StochasticLyapunovSettings(
            dt=0.005,
            horizon=horizon,
            transient=0,
            seed=7,
        )

    def test_diagonal_multiplicative_spectrum(self):
        first, second = 0.3, -0.5
        sigmas = (0.4, 0.8)
        system = MatrixNoiseSystem(
            _planar(
                DiagonalGrowthField,
                [Alpha(value=first), Alpha(value=second)],
            ),
            NoiseMatrix(
                [
                    _indexed(
                        DiagonalMultiplicativeColumn,
                        index,
                        [Sigma(value=sigmas[0]), Sigma(value=sigmas[1])],
                    )
                    for index in range(_PLANE)
                ],
            ),
            Ito(),
        )
        spectrum = matrix_noise_lyapunov(
            system,
            np.array([1.0, 1.0]),
            settings=self._settings(),
        ).exponents
        assert abs(spectrum[0] - (first - 0.5 * sigmas[0] ** 2)) < 0.15
        assert abs(spectrum[1] - (second - 0.5 * sigmas[1] ** 2)) < 0.3

    def test_additive_matrix_noise_leaves_the_spectrum_exact(self):
        # With additive noise the amplitude has no Jacobian, so the tangent equation
        # carries no noise at all: the spectrum of a linear drift is its eigenvalue
        # real parts with no Monte-Carlo error whatsoever, only the Euler bias.
        mu = -0.3
        system = MatrixNoiseSystem(
            _planar(SpiralField, [Alpha(value=mu)]),
            NoiseMatrix(
                [
                    _planar(ConstantNoiseColumn, [Sigma(value=0.4), Sigma(value=0.0)]),
                    _planar(ConstantNoiseColumn, [Sigma(value=0.0), Sigma(value=0.4)]),
                ],
            ),
            Ito(),
        )
        spectrum = matrix_noise_lyapunov(
            system,
            np.array([1.0, 0.0]),
            settings=self._settings(),
        ).exponents
        assert abs(spectrum[0] - mu) < 1.0e-2
        assert abs(spectrum[1] - mu) < 1.0e-2

    def test_one_column_reproduces_the_scalar_estimator(self):
        # Scalar noise is the one-driver case, not a separate numerical method; if the
        # two ever disagreed, the density and the exponent would be describing
        # different systems.
        alpha, sigma = 0.3, 0.6
        drift = _scalar(Drift, alpha, sigma)
        diffusion = _scalar(Diffusion, alpha, sigma)
        scalar_top = stochastic_lyapunov(
            StochasticSystem(drift, diffusion, Stratonovich()),
            np.array([0.5]),
            settings=self._settings(horizon=4000),
        ).top
        matrix_top = matrix_noise_lyapunov(
            MatrixNoiseSystem(drift, NoiseMatrix([diffusion]), Stratonovich()),
            np.array([0.5]),
            settings=self._settings(horizon=4000),
        ).top
        assert abs(scalar_top - matrix_top) < 1.0e-8


class TestStochasticScan(TestCase):
    """Both diagnostics along one parameter range, which is what decides the order."""

    def test_scan_reports_density_and_exponent_together(self):
        sigma = 0.5
        grid = DensityGrid.box([-2.5, -2.5], [2.5, 2.5], [61, 61])
        estimator = MeanTopExponent(
            lambda mu: MatrixNoiseSystem(
                _planar(HopfField, [Alpha(value=mu)]),
                _isotropic(sigma, 0.0),
                Ito(),
            ),
            np.array([0.4, 0.0]),
            StochasticLyapunovSettings(dt=0.005, horizon=6000, transient=1000),
            (7,),
        )
        samples = StochasticScan(
            lambda mu: _hopf_density(mu, sigma, 0.0, Ito(), grid),
            estimator.at,
        ).run([-0.3, 0.1, 0.5])
        assert [sample.parameter for sample in samples] == [-0.3, 0.1, 0.5]
        radii = [float(np.linalg.norm(s.density.dominant_mode())) for s in samples]
        assert radii[0] == 0.0
        assert radii[0] <= radii[1] < radii[2]
        assert abs(radii[2] - math.sqrt(0.5)) < 2.0 * grid.spacings[0]
        assert samples[0].exponent < 0.0
        assert samples[0].exponent < samples[-1].exponent
