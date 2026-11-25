"""What the noisy Hopf film computes once and shares (DEQ-76).

The film's noise reads sigma alone, so ``G G^T`` on each grid is the same at every
mu and is evaluated once; and it draws the deterministic Hopf film's cycle branch,
which is traced once per process. Neither may change a bit of what is drawn.
"""

from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    StationaryFokkerPlanck,
)
from discrecontinual_equations.continuation.stochastic import Ito
from examples.continuation import stage_hopf as hopf
from examples.continuation import stage_noisy_hopf as film


class TestSharedDiffusion(TestCase):
    def test_every_mu_on_a_grid_shares_one_diffusion(self):
        grid = DensityGrid.box([-2.5, -2.5], [2.5, 2.5], [20, 20])
        first = film.density_at(0.3, grid)
        second = film.density_at(0.9, grid)
        assert first._known is second._known  # noqa: SLF001
        assert first._known is film._diffusion_on(grid)  # noqa: SLF001

    def test_the_shared_diffusion_changes_no_bit_of_the_density(self):
        grid = DensityGrid.box([-2.5, -2.5], [2.5, 2.5], [20, 20])
        for mu in (0.1, 0.49, 1.5):
            parameters = film._parameters(mu)  # noqa: SLF001
            own = StationaryFokkerPlanck(
                film._drift(parameters),  # noqa: SLF001
                film._noise(parameters),  # noqa: SLF001
                Ito(),
                grid,
            )
            np.testing.assert_array_equal(
                film.density_at(mu, grid).solve().values,
                own.solve().values,
            )


class TestSharedCycleBranch(TestCase):
    def test_the_branch_is_traced_once_and_handed_out_as_fresh_lists(self):
        first = hopf.cycle_branch()
        traced = hopf._traced_cycle_branch.cache_info().misses  # noqa: SLF001
        second = hopf.cycle_branch()
        assert hopf._traced_cycle_branch.cache_info().misses == traced  # noqa: SLF001
        assert first is not second
        assert len(first) == len(second) > 0
        assert all(a is b for a, b in zip(first, second, strict=True))
        first.clear()
        assert len(hopf.cycle_branch()) == len(second)
