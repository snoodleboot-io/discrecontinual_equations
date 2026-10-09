"""The crater probe on a grid that straddles its centre.

A noise that vanishes at the reference state forbids a cell there - the
diffusion would be singular - so the grid has an even count and the centre
falls on a cell corner. The probe's offset steps the reference cell out from
the nearest one, and the sign it reads there is the curvature's.
"""

import math
from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    GridDensity,
)
from discrecontinual_equations.continuation.stochastic_threshold import RadialCrater


def _density(grid, profile):
    first, second = np.meshgrid(grid.axis(0), grid.axis(1), indexing="ij")
    raw = profile(np.hypot(first, second))
    return GridDensity(grid, raw / (raw.sum() * grid.cell_volume))


class TestRadialCraterOffset(TestCase):
    def test_the_offset_steps_the_reference_cell_out_along_the_axis(self):
        grid = DensityGrid.box([-2.0, -2.0], [2.0, 2.0], [40, 40])
        peaked = _density(grid, lambda r: np.exp(-(r**2) / 0.3))
        nearest = list(grid.nearest([0.0, 0.0]))
        for offset in (0, 1, 3):
            here = list(nearest)
            here[0] += offset
            there = list(here)
            there[0] += 1
            expected = math.log(peaked.at(there)) - math.log(peaked.at(here))
            found = RadialCrater([0.0, 0.0], offset=offset).value(peaked)
            assert abs(found - expected) < 1.0e-12

    def test_one_cell_out_the_sign_is_the_curvature_of_log_p(self):
        grid = DensityGrid.box([-2.0, -2.0], [2.0, 2.0], [40, 40])
        peaked = _density(grid, lambda r: np.exp(-(r**2) / 0.3))
        cratered = _density(grid, lambda r: np.exp(-((r - 0.8) ** 2) / 0.05))
        for offset in (1, 2, 3):
            assert RadialCrater([0.0, 0.0], offset=offset).value(peaked) < 0.0
            assert RadialCrater([0.0, 0.0], offset=offset).value(cratered) > 0.0

    def test_the_default_still_reads_the_centre_cell_on_an_odd_grid(self):
        grid = DensityGrid.box([-2.0, -2.0], [2.0, 2.0], [41, 41])
        peaked = _density(grid, lambda r: np.exp(-(r**2) / 0.3))
        assert RadialCrater([0.0, 0.0]).value(peaked) < 0.0
