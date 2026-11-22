"""Mode counting by superlevel persistence, on grids and from samples.

The references are independent of the code under test: persistence is checked
against connected components counted level by level with ``scipy.ndimage.label``;
the grid count against :class:`DensityModes`; the kernel estimate against a direct
sum over samples; its standard error against the scatter of repeated estimates; and
the sample-based threshold of the multiplicative pitchfork against the closed form
``a = s^2 / 2``, shifted by exactly the amount the kernel smoothing predicts.
"""

from unittest import TestCase

import numpy as np
import pytest
from scipy import ndimage, stats
from scipy.optimize import brentq

from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.continuation.root_finder import (
    Bisection,
    RootFinderSettings,
)
from discrecontinual_equations.continuation.stochastic import (
    DensityModes,
    Ito,
    StationaryDensity,
    Stratonovich,
)
from discrecontinual_equations.continuation.stochastic_threshold import (
    PhenomenologicalThreshold,
)
from discrecontinual_equations.continuation.topological_modes import (
    KernelDensity,
    ModeCountThreshold,
    PersistenceProbe,
    TopologicalModes,
    superlevel_persistence,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.coloured_fields import (
    PitchforkDrift,
    PitchforkNoise,
    scalar,
    white_pitchfork_samples,
)
from tests.continuation.fields import Alpha, Sigma, State

SIGMA = 0.6
P_THRESHOLD = SIGMA * SIGMA / 2.0


def _bisection(tolerance: float = 1.0e-4) -> Bisection:
    return Bisection(RootFinderSettings(1.0e-14, tolerance, 60))


def _components(values: np.ndarray, level: float) -> int:
    structure = np.ones((3,) * values.ndim)
    return int(ndimage.label(values >= level, structure=structure)[1])


class TestSuperlevelPersistence(TestCase):
    def test_two_peaks_pair_the_lower_with_the_saddle(self):
        values = np.array([0.0, 2.0, 1.0, 3.0, 0.5])
        pairs = superlevel_persistence(values)
        finite = [pair for pair in pairs if not pair.essential]
        assert len(finite) == 1
        assert (finite[0].birth, finite[0].death) == (2.0, 1.0)
        assert finite[0].birth_cell == (1,)
        assert finite[0].death_cell == (2,)
        essential = [pair for pair in pairs if pair.essential]
        assert (essential[0].birth, essential[0].death) == (3.0, 0.0)

    def test_cells_meeting_at_a_corner_are_one_component(self):
        values = np.array([[1.0, 0.0], [0.0, 1.0]])
        pairs = superlevel_persistence(values)
        assert sum(pair.persistence > 0.0 for pair in pairs) == 1

    def test_betti_numbers_match_components_at_every_level(self):
        generator = np.random.default_rng(4)
        for shape in ((40,), (9, 11), (5, 6, 4)):
            values = generator.random(shape)
            pairs = superlevel_persistence(values)
            for level in np.unique(values):
                alive = sum(
                    pair.birth >= level and (pair.essential or pair.death < level)
                    for pair in pairs
                )
                assert alive == _components(values, level), (shape, level)

    def test_a_plateau_is_one_mode(self):
        modes = TopologicalModes(np.array([0.0, 1.0, 1.0, 1.0, 0.0]))
        assert modes.count() == 1

    def test_a_component_counts_only_above_its_significance_bar(self):
        values = np.array([0.0, 1.0, 0.8, 0.9, 0.0])
        uncertainty = np.full(5, 0.05)
        # Persistence 0.1 against z * hypot(0.05, 0.05) = 0.0707 z.
        assert TopologicalModes(values, uncertainty, significance=1.0).count() == 2
        assert TopologicalModes(values, uncertainty, significance=3.0).count() == 1
        assert TopologicalModes(values, uncertainty).count() == 1
        margin = TopologicalModes(values, uncertainty, significance=1.0).margin()
        assert abs(margin - (0.1 - np.hypot(0.05, 0.05))) < 1e-12

    def test_sampling_wiggles_in_a_unimodal_estimate_are_not_modes(self):
        samples = np.random.default_rng(8).normal(0.0, 1.0, 3000)
        grid = DensityGrid.box([-5.0], [5.0], [1001])
        kde = KernelDensity(samples, grid, bandwidth=0.04)
        assert len(superlevel_persistence(kde.density().values)) > 5
        assert kde.modes().count() == 1

    def test_refuses_an_empty_array(self):
        with pytest.raises(ValueError, match="at least one value"):
            superlevel_persistence(np.array([]))


def _white_density(rate: float, grid: np.ndarray) -> np.ndarray:
    drift = scalar(PitchforkDrift, rate, SIGMA)
    noise = scalar(PitchforkNoise, rate, SIGMA)
    return StationaryDensity(drift, noise, Stratonovich(), grid).evaluate()[1]


def _mirrored_geometric_grid() -> np.ndarray:
    """``±x`` with ``x`` geometric from ``1e-6`` to ``2.5``.

    The density's log has a ``log |x|`` term, and a uniform grid through the origin
    integrates ``1 / x`` across its first cells badly enough to put a spurious dip
    there. A geometric grid integrates it to ``(log r)^3 / 12`` per cell, and the
    two cells either side of the origin cancel by symmetry. Both counts are
    index-based, so the grid need not be uniform.
    """
    half = np.geomspace(1.0e-6, 2.5, 1500)
    return np.concatenate([-half[::-1], half])


class TestGridAgreement(TestCase):
    """On an exact grid density the persistent count is the local-maximum count."""

    grid = _mirrored_geometric_grid()

    def test_counts_agree_across_the_threshold(self):
        for rate in np.linspace(0.05, 0.45, 21):
            density = _white_density(float(rate), self.grid)
            maxima = len(DensityModes(self.grid, density).interior_maxima())
            assert TopologicalModes(density).count() == maxima, rate

    def test_located_thresholds_agree_and_match_the_closed_form(self):
        def maxima(rate: float) -> float:
            density = _white_density(rate, self.grid)
            return len(DensityModes(self.grid, density).interior_maxima()) - 1.5

        by_maxima = _bisection().locate(maxima, 0.1, 0.3)
        by_persistence = ModeCountThreshold(
            lambda rate: TopologicalModes(_white_density(rate, self.grid)),
            _bisection(),
        ).locate(0.1, 0.3)
        assert by_persistence is not None
        assert by_maxima is not None
        assert abs(by_persistence - by_maxima) < 1.0e-3
        assert abs(by_persistence - P_THRESHOLD) < 1.0e-3


class _PlanarPitchfork(DeterministicFunction):
    """``(a x - x^3, -y)``: bimodal in ``x`` exactly when ``a > 0`` (additive noise)."""

    def eval(self, point, time=None):  # noqa: ARG002
        a = self.parameters[0].value
        return [a * point[0] - point[0] ** 3, -point[1]]


class _Column(DeterministicFunction):
    def __init__(self, column):
        super().__init__([State(), State()], [], [State(), State()], None)
        self._column = column

    def eval(self, point, time=None):  # noqa: ARG002
        return list(self._column)


class TestPersistenceProbe(TestCase):
    def test_planar_density_splits_at_zero_with_additive_noise(self):
        grid = DensityGrid.box([-2.0, -2.0], [2.0, 2.0], [61, 41])

        def density_at(rate: float) -> StationaryFokkerPlanck:
            drift = _PlanarPitchfork(
                [State(), State()],
                [Alpha(value=rate), Sigma(value=0.5)],
                [State(), State()],
                None,
            )
            noise = NoiseMatrix([_Column([0.5, 0.0]), _Column([0.0, 0.5])])
            return StationaryFokkerPlanck(drift, noise, Ito(), grid)

        located = PhenomenologicalThreshold(
            density_at,
            PersistenceProbe(),
            _bisection(1.0e-3),
        ).locate(-0.3, 0.4)
        assert located is not None
        # Exact threshold 0; the grid (spacing 1/15) delays it by O(h^2).
        assert 0.0 <= located < 2.0e-2


class TestKernelDensity(TestCase):
    def test_binned_estimate_matches_the_direct_sum(self):
        generator = np.random.default_rng(1)
        samples = np.concatenate(
            [generator.normal(-1.0, 0.3, 3000), generator.normal(1.0, 0.5, 2000)],
        )
        grid = DensityGrid.box([-3.0], [3.5], [651])
        estimate = KernelDensity(samples, grid, bandwidth=0.15).density().values
        direct = stats.gaussian_kde(samples, bw_method=0.15 / samples.std(ddof=1))(
            grid.axis(0),
        )
        assert np.max(np.abs(estimate - direct)) < 2.0e-3 * np.max(direct)

    def test_linear_binning_preserves_the_sample_mean(self):
        # Splitting each sample between its two nodes in proportion to proximity
        # keeps its first moment, and a symmetric kernel keeps the bins'.
        samples = np.random.default_rng(6).normal(0.3, 0.7, 3000)
        grid = DensityGrid.box([-4.0], [4.0], [81])
        estimate = KernelDensity(samples, grid, bandwidth=0.3).density()
        # Exact up to kernel mass leaving the grid edge and FFT round-off (~1e-7);
        # the swapped split misplaces it by ~spacing / sqrt(n) ~ 1e-3.
        assert abs(float(estimate.mean()[0]) - float(np.mean(samples))) < 1e-6

    def test_scott_bandwidth_is_the_default(self):
        generator = np.random.default_rng(2)
        samples = generator.normal(0.0, 2.0, (500, 2))
        grid = DensityGrid.box([-8.0, -8.0], [8.0, 8.0], [41, 41])
        widths = KernelDensity(samples, grid).bandwidths
        expected = samples.std(axis=0, ddof=1) * 500 ** (-1.0 / 6.0)
        assert np.allclose(widths, expected)

    def test_standard_error_matches_the_scatter_of_repeats(self):
        grid = DensityGrid.box([-4.0], [4.0], [401])
        count, width = 4000, 0.2
        estimates, errors = [], []
        for seed in range(200):
            samples = np.random.default_rng(100 + seed).normal(0.0, 1.0, count)
            kde = KernelDensity(samples, grid, bandwidth=width)
            estimates.append(kde.density().values)
            errors.append(kde.standard_error())
        centre = 200
        scatter = float(np.std([row[centre] for row in estimates], ddof=1))
        predicted = float(np.mean([row[centre] for row in errors]))
        # 200 repeats measure a standard deviation to about 5%; the asymptotic
        # p R(K) / (n h) alone overstates this one by 17%.
        assert abs(scatter / predicted - 1.0) < 0.1

    def test_two_dimensional_bimodal_samples_have_two_modes(self):
        generator = np.random.default_rng(3)
        samples = np.concatenate(
            [
                generator.normal([-1.5, 0.0], 0.4, (3000, 2)),
                generator.normal([1.5, 0.5], 0.4, (3000, 2)),
            ],
        )
        grid = DensityGrid.box([-4.0, -3.0], [4.0, 3.5], [81, 66])
        modes = KernelDensity(samples, grid, bandwidth=0.25).modes()
        assert modes.count() == 2
        peaks = sorted(location[0] for location in modes.locations(grid))
        assert abs(peaks[0] + 1.5) < 0.2
        assert abs(peaks[1] - 1.5) < 0.2

    def test_refuses_bad_input(self):
        grid = DensityGrid.box([-1.0], [1.0], [11])
        with pytest.raises(ValueError, match="positive"):
            KernelDensity(np.zeros(10), grid, bandwidth=0.0)
        with pytest.raises(ValueError, match="at least two"):
            KernelDensity(np.zeros(1), grid, bandwidth=0.1)
        with pytest.raises(ValueError, match="dimension"):
            KernelDensity(np.zeros((10, 2)), grid, bandwidth=0.1)


def _smoothed_white_density(
    rate: float, width: float, points: np.ndarray
) -> np.ndarray:
    """``p * K_h`` of the white pitchfork by quadrature: the estimate's expectation."""
    fine = np.linspace(-4.0, 4.0, 16000)
    exact = np.abs(fine) ** (2.0 * rate / SIGMA**2 - 1.0) * np.exp(
        -(fine**2) / SIGMA**2
    )
    exact /= np.trapezoid(exact, fine)
    kernel = stats.norm.pdf((points[:, None] - fine[None, :]) / width) / width
    return kernel @ exact * (fine[1] - fine[0])


class TestSampledThreshold(TestCase):
    """Oracle 3: the pitchfork's P-threshold from samples alone."""

    count = 40000
    width = 0.08

    def _modes_at(self, uniforms, signs, grid):
        def modes_at(rate: float) -> TopologicalModes:
            samples = white_pitchfork_samples(rate, SIGMA, uniforms, signs)
            return KernelDensity(samples, grid, bandwidth=self.width).modes()

        return modes_at

    def test_smoothing_shifts_the_threshold_by_exactly_h_squared(self):
        # Independent of the detector: where p * K_h stops peaking at the origin.
        def curvature(rate: float) -> float:
            values = _smoothed_white_density(
                rate, self.width, np.array([-1e-3, 0.0, 1e-3])
            )
            return float(values[0] - 2.0 * values[1] + values[2])

        smoothed = brentq(curvature, 0.15, 0.3, xtol=1.0e-7)
        assert abs(smoothed - (P_THRESHOLD + self.width**2)) < 5.0e-4

    def test_samples_locate_the_closed_form_threshold(self):
        generator = np.random.default_rng(11)
        uniforms = generator.random(self.count)
        signs = np.where(generator.random(self.count) < 0.5, -1.0, 1.0)
        grid = DensityGrid.box([-2.5], [2.5], [501])
        located = ModeCountThreshold(
            self._modes_at(uniforms, signs, grid),
            _bisection(1.0e-3),
        ).locate(0.05, 0.4)
        assert located is not None
        # Kernel smoothing delays the onset by h^2 exactly; requiring three standard
        # errors of persistence delays it further by about sqrt(3 se / C), with
        # C ~ 100 the curvature of the dip's depth in the parameter (measured on
        # the expectation above): ~0.015 here, 0.010-0.014 over six seeds. Both
        # vanish as n grows and h shrinks.
        smoothed = P_THRESHOLD + self.width**2
        assert smoothed < located < smoothed + 0.025

    def test_counts_on_either_side(self):
        generator = np.random.default_rng(12)
        uniforms = generator.random(self.count)
        signs = np.where(generator.random(self.count) < 0.5, -1.0, 1.0)
        grid = DensityGrid.box([-2.5], [2.5], [501])
        modes_at = self._modes_at(uniforms, signs, grid)
        assert modes_at(P_THRESHOLD - 0.06).count() == 1
        assert modes_at(P_THRESHOLD + 0.08).count() == 2
