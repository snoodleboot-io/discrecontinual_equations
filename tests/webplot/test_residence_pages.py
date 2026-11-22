"""The residence-time pages' pieces that can be checked without the full sweep.

The sweep itself - a ``41^3`` grid solve per field and three long ensembles - is
what the atlas build runs; here the ring's symmetry, which the pages rely on to
serve both wells from one solve per field, is checked on a coarse grid, and the
histogram the density page draws is checked to be a density.
"""

from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.first_passage import (
    Ball,
    MeanFirstPassage,
)
from discrecontinual_equations.continuation.fokker_planck import DensityGrid
from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.continuation.stochastic import Ito
from examples.continuation import pages_residence as pages


class TestRingSymmetry(TestCase):
    def test_attractors_are_symmetric_states_mirrored_by_the_field(self):
        lower, upper = pages.attractors(0.05)
        mirrored_lower, mirrored_upper = pages.attractors(-0.05)
        assert np.allclose(lower, lower[0])
        assert np.allclose(upper, upper[0])
        assert np.allclose(mirrored_lower, -upper)
        assert np.allclose(mirrored_upper, -lower)
        zero_lower, zero_upper = pages.attractors(0.0)
        # s^2 = a - lambda at zero field.
        assert np.allclose(zero_upper, np.sqrt(1.5))
        assert np.allclose(zero_lower, -np.sqrt(1.5))

    def test_grid_time_from_the_upper_well_is_the_mirrored_lower_one(self):
        # tau_+(x; eps) = tau_-(-x; -eps): the identity the pages use to read
        # the upper well's time off the lower well's solve at the opposite field.
        # Checked against a direct solve into the lower well, on a coarse grid
        # (the identity is exact on any grid symmetric about the origin).
        field, points = 0.05, 13
        lower, upper = pages.attractors(field)
        grid = DensityGrid.box([-pages._BOX] * 3, [pages._BOX] * 3, [points] * 3)
        direct = MeanFirstPassage(
            pages._ring(field),
            NoiseMatrix([pages._NoiseColumn(index) for index in range(3)]),
            Ito(),
            grid,
        ).solve(Ball(lower, pages._READOUT_RADIUS).on(grid))
        mirrored = pages.grid_time_up(-field, points)
        assert direct.at(upper) == pytest.approx(mirrored.at(-upper), rel=1.0e-8)
        # The field favours the upper well: leaving it takes longer.
        assert direct.at(upper) > pages.grid_time_up(field, points).at(lower)


class TestHistogram(TestCase):
    def test_step_histogram_integrates_to_the_fraction_in_the_window(self):
        times = np.random.default_rng(1).exponential(10.0, 5000)
        points = pages._histogram(times, top=20.0)
        area = sum(
            (right[0] - left[0]) * left[1]
            for left, right in zip(points[::2], points[1::2])
        )
        assert abs(area - float(np.mean(times <= 20.0))) < 1.0e-9
