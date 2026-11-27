"""The Fokker-Planck diffusion evaluated once and shared across drifts (DEQ-76).

A density scrubbed through a drift parameter at fixed noise solves the same ``G G^T``
on every frame. :class:`GridDiffusion` holds it; a problem given one must return the
density it would have computed itself, bit for bit, and under Ito must not evaluate
the noise at all.
"""

from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    GridDiffusion,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.continuation.stochastic import Ito, Stratonovich
from tests.continuation.fields import (
    Alpha,
    ConstantNoiseColumn,
    HopfField,
    IsotropicNoiseColumn,
    Kappa,
    Sigma,
    State,
)


class CountedColumn(IsotropicNoiseColumn):
    """An isotropic noise column that counts the points it is evaluated at."""

    calls = 0

    def eval(self, point, time=None):
        CountedColumn.calls += 1
        return super().eval(point, time)


def _states():
    return [State(), State()]


def _drift(mu):
    return HopfField(_states(), [Alpha(value=mu)], _states(), None)


def _noise():
    columns = [
        CountedColumn(
            index,
            _states(),
            [Alpha(value=0.0), Sigma(value=0.5), Kappa(value=0.4)],
            _states(),
            None,
        )
        for index in range(2)
    ]
    # A constant column overlapping both components puts D off the diagonal, so the
    # cross-diffusion terms read the cached cells too.
    columns.append(
        ConstantNoiseColumn(_states(), [Alpha(value=0.2)] * 2, _states(), None),
    )
    return NoiseMatrix(columns)


def _grid(side=11):
    return DensityGrid.box([-2.0, -2.5], [2.0, 2.5], [side, side + 2])


class TestGridDiffusion(TestCase):
    def test_a_shared_diffusion_changes_no_bit_of_any_density(self):
        grid = _grid()
        for convention in (Ito(), Stratonovich()):
            shared = GridDiffusion(_noise(), grid)
            for mu in (-0.4, 0.1, 0.7):
                own = StationaryFokkerPlanck(_drift(mu), _noise(), convention, grid)
                reused = StationaryFokkerPlanck(
                    _drift(mu),
                    _noise(),
                    convention,
                    grid,
                    shared,
                )
                np.testing.assert_array_equal(
                    reused.solve().values,
                    own.solve().values,
                )

    def test_the_cached_arrays_are_the_ones_a_solve_forms(self):
        grid = _grid()
        problem = StationaryFokkerPlanck(_drift(0.3), _noise(), Ito(), grid)
        shared = GridDiffusion(_noise(), grid)
        points = grid.points()
        np.testing.assert_array_equal(
            shared.cells,
            problem._diffusion(points),  # noqa: SLF001
        )
        indices = np.arange(grid.size).reshape(grid.shape)
        for axis in range(2):
            count = grid.shape[axis]
            low = np.take(indices, np.arange(count - 1), axis=axis).ravel()
            high = np.take(indices, np.arange(1, count), axis=axis).ravel()
            faces = 0.5 * (points[low] + points[high])
            _, diffusion = problem._face_coefficients(faces, axis)  # noqa: SLF001
            np.testing.assert_array_equal(shared.faces(axis), diffusion)
            assert shared.faces(axis).shape == (low.size,)

    def test_under_ito_a_shared_diffusion_evaluates_no_noise(self):
        grid = _grid()
        shared = GridDiffusion(_noise(), grid)
        CountedColumn.calls = 0
        StationaryFokkerPlanck(_drift(0.3), _noise(), Ito(), grid, shared).solve()
        assert CountedColumn.calls == 0
        StationaryFokkerPlanck(_drift(0.3), _noise(), Ito(), grid).solve()
        assert CountedColumn.calls > 0

    def test_under_stratonovich_the_noise_is_still_read_for_the_shift(self):
        grid = _grid()
        shared = GridDiffusion(_noise(), grid)
        CountedColumn.calls = 0
        StationaryFokkerPlanck(
            _drift(0.3),
            _noise(),
            Stratonovich(),
            grid,
            shared,
        ).solve()
        assert CountedColumn.calls > 0

    def test_a_diffusion_from_another_grid_is_refused(self):
        shared = GridDiffusion(_noise(), _grid(11))
        with pytest.raises(ValueError, match="different grid"):
            StationaryFokkerPlanck(_drift(0.3), _noise(), Ito(), _grid(9), shared)
        shifted = DensityGrid.box([-2.0, -2.0], [2.0, 2.0], [11, 13])
        with pytest.raises(ValueError, match="different grid"):
            StationaryFokkerPlanck(_drift(0.3), _noise(), Ito(), shifted, shared)

    def test_a_grid_changed_in_place_afterwards_is_refused(self):
        grid = _grid(11)
        shared = GridDiffusion(_noise(), grid)
        grid.axis(0)[:] = np.linspace(-1.0, 1.0, 11)
        with pytest.raises(ValueError, match="different grid"):
            StationaryFokkerPlanck(_drift(0.3), _noise(), Ito(), grid, shared)

    def test_an_equal_grid_built_again_is_accepted(self):
        shared = GridDiffusion(_noise(), _grid(11))
        problem = StationaryFokkerPlanck(
            _drift(0.3), _noise(), Ito(), _grid(11), shared
        )
        assert problem.solve().values.shape == (11, 13)
        assert shared.grid.shape == (11, 13)
