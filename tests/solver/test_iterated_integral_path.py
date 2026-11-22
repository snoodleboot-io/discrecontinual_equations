"""IteratedIntegralPath reads any step's integrals off its cells by Chen's relation.

The reference composition here is written as a loop over cells, the relation in
its textbook form (Kloeden and Platen, Section 5.2), not as the path's cumulative
sums; and on cells that are Ito sums of a sub-grid walk, a reading must equal the
Ito sum over the whole step, computed directly.
"""

import numpy as np
import pytest

from discrecontinual_equations.solver.stochastic.iterated_integrals import (
    IteratedIntegralPath,
)


def _compose(first, second, length):
    """Chen: integrals over [s, u] from those over [s, r] and [r, u]."""
    dw1, dz1, d1, t1 = first
    dw2, dz2, d2, t2 = second
    return (
        dw1 + dw2,
        dz1 + dw1 * length + dz2,
        d1 + np.outer(dw1, dw2) + d2,
        t1 + np.einsum("jk,l->jkl", d1, dw2) + np.einsum("j,kl->jkl", dw1, d2) + t2,
    )


def _walk_cells(seed, cells, sub, channels):
    """A walk on ``cells * sub`` steps and each cell's Ito sums."""
    generator = np.random.default_rng(seed)
    fine = 1.0 / (cells * sub)
    steps = generator.standard_normal((cells, sub, channels)) * np.sqrt(fine)
    before = np.cumsum(steps, axis=1) - steps
    increments = np.einsum("csj,csk->csjk", before, steps)
    double_before = np.cumsum(increments, axis=1) - increments
    return steps, (
        steps.sum(axis=1),
        before.sum(axis=1) * fine,
        increments.sum(axis=1),
        np.einsum("csjk,csl->cjkl", double_before, steps),
    )


class TestChen:
    def test_reading_is_the_loop_composition_of_cells(self):
        path = IteratedIntegralPath.sample(0.0, 1.0, 1.0 / 16, 3, seed=2)
        cells = [path.iterated_integrals(i / 16, 1.0 / 16, 3) for i in range(16)]
        start = 4
        total = (
            cells[start].delta_w,
            cells[start].delta_z,
            cells[start].double,
            cells[start].triple,
        )
        for cell in cells[start + 1 : start + 8]:
            total = _compose(
                total,
                (cell.delta_w, cell.delta_z, cell.double, cell.triple),
                1.0 / 16,
            )
        reading = path.iterated_integrals(4 / 16, 8 / 16, 3)
        np.testing.assert_allclose(reading.delta_w, total[0], atol=1.0e-14)
        np.testing.assert_allclose(reading.delta_z, total[1], atol=1.0e-14)
        np.testing.assert_allclose(reading.double, total[2], atol=1.0e-14)
        np.testing.assert_allclose(reading.triple, total[3], atol=1.0e-14)

    def test_reading_of_ito_sum_cells_is_the_ito_sum_over_the_step(self):
        cells, sub, channels = 8, 16, 2
        steps, data = _walk_cells(3, cells, sub, channels)
        path = IteratedIntegralPath(0.0, 1.0 / cells, *data)
        reading = path.iterated_integrals(2 / cells, 4 / cells, channels)
        walk = steps[2:6].reshape(-1, channels)
        before = np.cumsum(walk, axis=0) - walk
        increments = np.einsum("sj,sk->sjk", before, walk)
        double_before = np.cumsum(increments, axis=0) - increments
        np.testing.assert_allclose(reading.delta_w, walk.sum(axis=0), atol=1.0e-14)
        np.testing.assert_allclose(
            reading.delta_z,
            before.sum(axis=0) / (cells * sub),
            atol=1.0e-14,
        )
        np.testing.assert_allclose(reading.double, increments.sum(axis=0), atol=1.0e-14)
        np.testing.assert_allclose(
            reading.triple,
            np.einsum("sjk,sl->jkl", double_before, walk),
            atol=1.0e-14,
        )

    def test_increments_and_levy_areas_agree_with_the_integrals(self):
        path = IteratedIntegralPath.sample(0.0, 1.0, 1.0 / 32, 2, seed=4)
        integrals = path.iterated_integrals(0.25, 0.25, 2)
        increments = path.increments(0.25, 0.25, 2, levy_area=True)
        np.testing.assert_array_equal(increments.delta_w, integrals.delta_w)
        np.testing.assert_array_equal(increments.delta_z, integrals.delta_z)
        np.testing.assert_allclose(
            increments.levy_area,
            0.5 * (integrals.double - integrals.double.T),
        )
        assert path.increments(0.25, 0.25, 2).levy_area is None

    def test_values_are_the_running_increments(self):
        path = IteratedIntegralPath.sample(1.0, 2.0, 0.25, 2, seed=5)
        values = path.values
        assert values.shape == (5, 2)
        np.testing.assert_allclose(
            values[-1],
            path.iterated_integrals(1.0, 1.0, 2).delta_w,
        )
        assert path.dimension == 2


class TestRefusals:
    def test_horizon_must_be_whole_cells(self):
        with pytest.raises(ValueError, match="whole number"):
            IteratedIntegralPath.sample(0.0, 1.0, 0.3, 2)

    def test_times_must_be_nodes(self):
        path = IteratedIntegralPath.sample(0.0, 1.0, 0.25, 2, seed=1)
        with pytest.raises(ValueError, match="not a grid node"):
            path.iterated_integrals(0.1, 0.25, 2)
        with pytest.raises(ValueError, match="not a grid node"):
            path.iterated_integrals(0.75, 0.5, 2)

    def test_channel_count_must_match(self):
        path = IteratedIntegralPath.sample(0.0, 1.0, 0.25, 2, seed=1)
        with pytest.raises(ValueError, match="2 channels"):
            path.iterated_integrals(0.0, 0.25, 3)

    def test_shapes_are_checked(self):
        with pytest.raises(ValueError, match="shapes"):
            IteratedIntegralPath(
                0.0,
                0.5,
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                np.zeros((2, 2, 2)),
                np.zeros((2, 2, 2)),
            )
