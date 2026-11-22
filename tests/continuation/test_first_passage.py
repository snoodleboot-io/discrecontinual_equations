"""Mean first-passage and residence times: the grid against closed forms and ensembles.

The grid is checked against references that share no code with it: the exact
one-dimensional double integral by adaptive quadrature, and the Eyring-Kramers and
Bouchet-Reygner asymptotics, whose ingredients are worked out by hand for the
systems of :mod:`tests.continuation.first_passage_systems`. The ensemble is checked
against the grid within its Monte-Carlo error. Every random run is seeded, so each
assertion is a fixed computation, and every tolerance is stated against the
measurement it bounds.
"""

import itertools
import math
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.first_passage import (
    Ball,
    EnsembleFirstPassage,
    EnsembleResidence,
    HalfSpace,
    MeanFirstPassage,
    ResidenceSample,
    ResidenceTracker,
    TimeSample,
    eyring_kramers_time,
)
from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.stochastic import Ito
from discrecontinual_equations.solver.solver_config import StochasticConfig
from tests.continuation.first_passage_systems import (
    DoubleWellDrift,
    PlanarWell,
    exact_double_well_time,
    isotropic_noise,
)

_WALL = -3.0
_PLANAR_BOX = ([-2.2, -1.8], [2.2, 1.8])
_TARGET_RADIUS = 0.3


def _line_grid(cells: int) -> DensityGrid:
    """Cell centres on ``[-3, 1]`` with ``x = 1`` a centre and the wall a face.

    The quadrature puts the reflecting wall at ``a = -3`` and the target at
    ``b = 1``; the finite-volume grid has its wall half a cell outside the first
    centre, so the axis starts at ``-3 + h / 2`` and is spaced to land on ``1``.
    """
    spacing = (1.0 - _WALL) / (cells - 0.5)
    return DensityGrid([_WALL + spacing * (0.5 + np.arange(cells))])


def _line_time(variance: float, cells: int) -> float:
    grid = _line_grid(cells)
    problem = MeanFirstPassage(
        DoubleWellDrift(),
        isotropic_noise(math.sqrt(variance), 1),
        Ito(),
        grid,
    )
    return problem.solve(HalfSpace([1.0], 1.0 - 1.0e-9).on(grid)).at([-1.0])


def _planar(well: PlanarWell, variance: float, cells: int) -> MeanFirstPassage:
    return MeanFirstPassage(
        well.field(),
        isotropic_noise(math.sqrt(variance), 2),
        Ito(),
        DensityGrid.box(*_PLANAR_BOX, [cells, cells]),
    )


def _planar_time(well: PlanarWell, variance: float, cells: int) -> float:
    """Mean time from the left minimum to the ball about the right one."""
    problem = _planar(well, variance, cells)
    left, right = well.minima()
    target = Ball((right, 0.0), _TARGET_RADIUS).on(problem.grid)
    return problem.solve(target).at([left, 0.0])


def _kramers_line(variance: float) -> float:
    """Kramers' time for ``V = x^4/4 - x^2/2``: ``V'' = 2`` and ``-1``, ``dV = 1/4``."""
    return eyring_kramers_time(0.5 / variance, 1.0, [[-1.0]], [[2.0]])


def _planar_closed_form(well: PlanarWell, variance: float, nongibbs: float) -> float:
    """Eyring-Kramers / Bouchet-Reygner for the untilted planar well.

    ``U = 2 V / sigma^2``, ``dV = 1/4``; the Hessians of ``V`` are
    ``diag(2, 1 + k)`` at the minima and ``diag(-1, 1)`` at the saddle (the ratio
    of determinants is the same for ``U``); the unstable rate is ``1``, or
    ``sqrt(1 + gamma^2)`` for the divergence-free rotation.
    """
    rate = math.sqrt(1.0 + well.rotation**2) if well.divergence_free else 1.0
    return eyring_kramers_time(
        0.5 / variance,
        rate,
        [[-1.0, 0.0], [0.0, 1.0]],
        [[2.0, 0.0], [0.0, 1.0 + well.curvature]],
        nongibbs=nongibbs,
    )


class TestOneDimensionalExactTime(TestCase):
    """Oracle 1: the grid reproduces the exact double integral, to second order."""

    def test_grid_matches_the_exact_integral_at_second_order(self):
        # Measured relative errors at 101, 201, 401, 801 cells (variance 1/16):
        # 2.12e-3, 5.31e-4, 1.33e-4, 3.33e-5 - order 2.00 - and 2.8e-4 ... 4.4e-6
        # at variance 1/2. The error grows only linearly as the noise falls: the
        # exponential fitting resolves the saddle layer at any cell Peclet number.
        for variance in (0.5, 0.25, 0.125, 0.0625):
            exact = exact_double_well_time(-1.0, 1.0, _WALL, variance)
            errors = [
                abs(_line_time(variance, cells) / exact - 1.0)
                for cells in (101, 201, 401)
            ]
            assert errors[-1] < 2.0e-4, (variance, errors)
            for coarse, fine in itertools.pairwise(errors):
                assert 1.9 < math.log2(coarse / fine) < 2.1, (variance, errors)

    def test_exact_time_tends_to_kramers_as_the_noise_falls(self):
        # Kramers' law is the Laplace limit of the double integral. Measured
        # kramers / exact: 0.8000, 0.8755, 0.9460 at variance 1/4, 1/8, 1/16 (and
        # 0.846 at 1/2, before the asymptotic regime starts): the deficit halves
        # with the variance, as an O(sigma^2) = O(1 / dU) correction should.
        deficits = []
        for variance in (0.125, 0.0625, 0.03125):
            exact = exact_double_well_time(-1.0, 1.0, _WALL, variance)
            deficits.append(1.0 - _kramers_line(variance) / exact)
        assert deficits[-1] < 0.03, deficits
        for coarse, fine in itertools.pairwise(deficits):
            assert 1.7 < coarse / fine < 2.6, deficits

    def test_grid_tracks_the_exact_time_where_kramers_does_not(self):
        # At variance 1/4 Kramers is 20% short; the grid at 401 cells is within
        # 3.4e-5.
        exact = exact_double_well_time(-1.0, 1.0, _WALL, 0.25)
        assert abs(_kramers_line(0.25) / exact - 1.0) > 0.15
        assert abs(_line_time(0.25, 401) / exact - 1.0) < 1.0e-4


class TestGenerator(TestCase):
    """The backward generator is the adjoint of the density solve's operator."""

    def setUp(self):
        self.well = PlanarWell(curvature=1.0, rotation=1.0)
        self.problem = _planar(self.well, 0.25, 41)

    def test_rows_sum_to_zero_and_rates_are_non_negative(self):
        generator = self.problem.generator().toarray()
        scale = float(np.max(np.abs(generator)))
        assert float(np.max(np.abs(generator.sum(axis=1)))) < 1.0e-12 * scale
        off_diagonal = generator - np.diag(np.diag(generator))
        assert float(np.min(off_diagonal)) >= 0.0

    def test_stationary_density_is_the_null_vector_of_the_adjoint(self):
        # The guard that the two modules discretise the same operator: if the
        # density solve's stencil changes, this fails before any time is wrong.
        density = StationaryFokkerPlanck(
            self.well.field(),
            isotropic_noise(0.5, 2),
            Ito(),
            self.problem.grid,
        ).solve()
        values = density.values.ravel()
        residual = self.problem.generator().T @ values
        scale = abs(self.problem.generator()).T @ np.abs(values)
        # Relative to the largest flux: the residual is round-off (2.7e-14 of a
        # largest term of order one), and a different stencil leaves order one.
        assert float(np.max(np.abs(residual))) < 1.0e-12 * float(np.max(scale))

    def test_target_must_be_a_non_empty_grid_mask(self):
        grid = self.problem.grid
        with pytest.raises(ValueError, match="no grid cell"):
            self.problem.solve(np.zeros(grid.shape, dtype=bool))
        with pytest.raises(ValueError, match="cells"):
            self.problem.solve(np.ones(5, dtype=bool))


class TestPlanarClosedForms(TestCase):
    """The grid against Eyring-Kramers and Bouchet-Reygner, as the noise falls."""

    def test_gradient_eyring_kramers_limit(self):
        # k = 1: the minima's transverse curvature is 2, the saddle's 1, so the
        # prefactor is 2 pi / sqrt(2 * 2 / 1) - not the 1-D one. Measured
        # grid / Eyring-Kramers at 161 cells: 1.361, 1.215, 1.092, 1.042 at
        # variance 1/4 ... 1/32; the excess halves with the variance.
        well = PlanarWell(curvature=1.0)
        excess = [
            _planar_time(well, variance, 161) / _planar_closed_form(well, variance, 0.0)
            - 1.0
            for variance in (0.125, 0.0625, 0.03125)
        ]
        assert excess[-1] < 0.06, excess
        for coarse, fine in itertools.pairwise(excess):
            assert 1.7 < coarse / fine < 2.6, excess

    def test_non_gibbs_prefactor_of_bouchet_and_reygner(self):
        # l = gamma y J grad V keeps U = 2 V / sigma^2 but is not divergence-free,
        # so the non-Gibbs integral is +gamma leaving the left well and -gamma
        # leaving the right one (see first_passage_systems): the wells have the
        # same barrier and curvatures, and the times still differ by e^(2 gamma).
        # Measured grid / Bouchet-Reygner at 281 cells, gamma = 1/2, variance
        # 1/8, 1/16, 1/32, 1/48: 1.162, 1.073, 1.033, 1.022 (left to right) and
        # 1.290, 1.120, 1.054, 1.035 (right to left).
        well = PlanarWell(curvature=1.0, rotation=0.5)
        left, right = well.minima()
        excess = {+1: [], -1: []}
        for variance in (0.125, 0.0625, 0.03125):
            problem = _planar(well, variance, 201)
            to_right = problem.solve(
                Ball((right, 0.0), _TARGET_RADIUS).on(problem.grid),
            ).at([left, 0.0])
            to_left = problem.solve(
                Ball((left, 0.0), _TARGET_RADIUS).on(problem.grid),
            ).at([right, 0.0])
            for sign, time in ((+1, to_right), (-1, to_left)):
                closed = _planar_closed_form(well, variance, sign * well.rotation)
                excess[sign].append(time / closed - 1.0)
            # Without the non-Gibbs term the closed form is out by e^(+-1/2).
            gibbs = _planar_closed_form(well, variance, 0.0)
            assert to_right / gibbs > 1.3
            assert to_left / gibbs < 0.8
        for sign in (+1, -1):
            values = excess[sign]
            assert 0.0 < values[-1] < 0.07, (sign, values)
            assert values[0] > values[1] > values[2], (sign, values)

    def test_divergence_free_rotation_changes_only_the_unstable_rate(self):
        # l = gamma J grad V: no non-Gibbs term, and the unstable rate at the
        # saddle is sqrt(1 + gamma^2) = sqrt(2) rather than the Hessian's 1.
        # The rotation carries the flow across the cells obliquely, where the
        # scheme's error is first order and grows like |f| h / sigma^2, so this
        # case needs a finer grid at small noise. Measured grid / closed form at
        # variance 1/32: 0.816, 0.928, 0.983 at 161, 241, 361 cells; at 1/8 and
        # 1/16 with 281 cells, 1.157 and 1.055.
        well = PlanarWell(curvature=1.0, rotation=1.0, divergence_free=True)
        ratios = [
            _planar_time(well, variance, 241) / _planar_closed_form(well, variance, 0.0)
            for variance in (0.125, 0.0625)
        ]
        assert ratios[0] > ratios[1] > 1.0, ratios
        assert ratios[1] < 1.08, ratios
        # The gradient formula's rate of 1 would be out by sqrt(2).
        assert ratios[1] / math.sqrt(2.0) < 0.8

    def test_grid_converges_under_refinement(self):
        well = PlanarWell(curvature=1.0)
        coarse, fine = (_planar_time(well, 0.25, cells) for cells in (81, 161))
        assert abs(coarse / fine - 1.0) < 2.0e-3


class TestEnsembleAgainstGrid(TestCase):
    """Oracle 2: ensemble means against the grid's times, within Monte-Carlo error."""

    def test_first_passage_means_match_the_grid(self):
        # 2000 paths at dt = 0.0025; the grid at 161 cells is converged to 0.1%.
        well = PlanarWell(curvature=1.0)
        for variance, seed in ((0.5, 11), (0.25, 12)):
            grid_time = _planar_time(well, variance, 161)
            sample = EnsembleFirstPassage(
                StochasticConfig(step_size=0.0025, random_seed=seed),
            ).solve(
                well.equation(math.sqrt(variance)),
                [-1.0, 0.0],
                2000,
                Ball((1.0, 0.0), _TARGET_RADIUS),
                horizon=1.0e5,
            )
            assert sample.censored == 0
            low, high = sample.confidence_interval(3.0)
            assert low < grid_time < high, (variance, grid_time, sample.mean)


class TestResidences(TestCase):
    @staticmethod
    def _track(labels):
        """Run a one-dimensional path ``x = t`` with the given labels through."""
        labels = np.asarray(labels)
        tracker = ResidenceTracker(np.array([[0.0]]), labels[:1], 0.0)
        for time, label in enumerate(labels[1:], start=1):
            tracker.step(np.array([[float(time)]]), np.array([label]), float(time))
        return tracker.sample()

    def test_residences_are_cut_at_entry_into_the_other_set(self):
        # In A from t=0, out (-1) and back, enters B at t=4, leaves B, enters A
        # at t=7 and is still there at the end (open, not counted).
        sample = self._track([0, -1, 0, -1, 1, 1, -1, 0, 0])
        assert sample.first.times.tolist() == [4.0]
        assert sample.second.times.tolist() == [3.0]
        # Each residence began where the path entered: x = 0 in A, x = 4 in B.
        assert sample.first.starts.tolist() == [[0.0]]
        assert sample.second.starts.tolist() == [[4.0]]

    def test_a_path_that_starts_outside_is_counted_from_its_first_entry(self):
        sample = self._track([-1, -1, 1, -1, 0, 1])
        assert sample.second.times.tolist() == [2.0]
        assert sample.second.starts.tolist() == [[2.0]]
        assert sample.first.times.tolist() == [1.0]

    def test_a_path_that_never_reaches_a_set_contributes_nothing(self):
        sample = self._track([-1, -1, -1, -1, -1])
        assert sample.first.count == 0
        assert sample.second.count == 0
        assert sample.first.starts.shape == (0, 1)

    def test_interval_and_asymmetry(self):
        sample = TimeSample(np.array([1.0, 2.0, 3.0, 4.0]))
        assert sample.mean == pytest.approx(2.5)
        assert sample.standard_error == pytest.approx(np.std([1, 2, 3, 4], ddof=1) / 2)
        low, high = sample.confidence_interval()
        assert high - low == pytest.approx(
            2 * 1.959963984540054 * sample.standard_error
        )
        pair = ResidenceSample(sample, TimeSample(np.array([1.0, 1.0, 1.0, 1.0])))
        assert pair.asymmetry == pytest.approx(1.5)
        low, high = pair.asymmetry_interval()
        assert high - low == pytest.approx(
            2 * 1.959963984540054 * sample.standard_error,
        )


class TestSymmetryAndResponse(TestCase):
    """Oracle 3: equal times at zero field, an odd response, ensemble and grid."""

    def test_grid_times_are_equal_at_zero_field_and_odd_in_the_field(self):
        times = {}
        for tilt in (-0.05, 0.0, 0.05):
            well = PlanarWell(curvature=1.0, tilt=tilt)
            problem = _planar(well, 0.25, 81)
            left, right = well.minima()
            to_right = problem.solve(
                Ball((right, 0.0), _TARGET_RADIUS).on(problem.grid),
            ).at([left, 0.0])
            to_left = problem.solve(
                Ball((left, 0.0), _TARGET_RADIUS).on(problem.grid),
            ).at([right, 0.0])
            times[tilt] = to_right - to_left
        scale = abs(times[0.05])
        assert abs(times[0.0]) < 1.0e-8 * scale
        assert times[0.05] == pytest.approx(-times[-0.05], rel=1.0e-8)
        # A field to the right shortens the stay on the left.
        assert times[0.05] < 0.0

    def test_ensemble_residences_match_the_grid_in_both_wells(self):
        variance = 0.5
        for tilt, seed in ((0.0, 21), (0.05, 22)):
            well = PlanarWell(curvature=1.0, tilt=tilt)
            left, right = well.minima()
            problem = _planar(well, variance, 161)
            left_field = problem.solve(
                Ball((right, 0.0), _TARGET_RADIUS).on(problem.grid),
            )
            right_field = problem.solve(
                Ball((left, 0.0), _TARGET_RADIUS).on(problem.grid),
            )
            sample = EnsembleResidence(
                StochasticConfig(step_size=0.0025, random_seed=seed),
            ).solve(
                well.equation(math.sqrt(variance)),
                ((left, 0.0), (right, 0.0)),
                300,
                (
                    Ball((left, 0.0), _TARGET_RADIUS),
                    Ball((right, 0.0), _TARGET_RADIUS),
                ),
                horizon=300.0,
            )
            # By the strong Markov property a residence's mean is tau averaged
            # over where the residences began, which is what is compared. At
            # this noise tau is not flat in the well - the residences begin on
            # the side of the ball facing the saddle - and the minimum's own tau
            # (11.41 at zero tilt) is 3% above the ensemble's 11.08: outside its
            # 3-sigma interval, which is why it is not the reference.
            for estimate, grid_field in (
                (sample.first, left_field),
                (sample.second, right_field),
            ):
                expected = float(
                    np.mean([grid_field.at(start) for start in estimate.starts]),
                )
                low, high = estimate.confidence_interval(3.0)
                assert low < expected < high, (tilt, expected, estimate.mean)
            if tilt == 0.0:
                low, high = sample.asymmetry_interval(3.0)
                assert low < 0.0 < high


class TestClosedFormArguments(TestCase):
    def test_kramers_one_dimensional_value(self):
        # 2 pi / sqrt(2 * 1) * e^2
        assert eyring_kramers_time(2.0, 1.0, [[-1.0]], [[2.0]]) == pytest.approx(
            2.0 * math.pi / math.sqrt(2.0) * math.exp(2.0),
        )

    def test_nongibbs_multiplies_by_its_exponential(self):
        base = eyring_kramers_time(
            1.0, 2.0, [[-2.0, 0], [0, 3.0]], [[1.0, 0], [0, 4.0]]
        )
        assert eyring_kramers_time(
            1.0,
            2.0,
            [[-2.0, 0], [0, 3.0]],
            [[1.0, 0], [0, 4.0]],
            nongibbs=0.5,
        ) == pytest.approx(base * math.exp(0.5))
        assert base == pytest.approx(math.pi * math.sqrt(6.0 / 4.0) * math.e)

    def test_refuses_a_saddle_without_one_unstable_direction(self):
        with pytest.raises(ValueError, match="unstable direction"):
            eyring_kramers_time(1.0, 1.0, [[1.0]], [[1.0]])
        with pytest.raises(ValueError, match="unstable direction"):
            eyring_kramers_time(1.0, 1.0, [[-1.0]], [[-1.0]])
        with pytest.raises(ValueError, match="positive"):
            eyring_kramers_time(1.0, 0.0, [[-1.0]], [[1.0]])

    def test_first_passage_refuses_a_start_inside_the_target(self):
        well = PlanarWell()
        with pytest.raises(ValueError, match="already inside"):
            EnsembleFirstPassage(StochasticConfig(step_size=0.01)).solve(
                well.equation(0.5),
                [1.0, 0.0],
                10,
                Ball((1.0, 0.0), 0.3),
                horizon=1.0,
            )
