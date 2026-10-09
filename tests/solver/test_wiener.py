"""The Wiener functionals have the joint law the schemes were derived for.

The order 1.5 scheme's correctness rests on ``(dW, dZ)`` having ``Var dZ = h^3 /
3`` and ``Cov(dW, dZ) = h^2 / 2``; a wrong factor here would lower its order
while leaving every path plausible. The fixed path must read the same ``dW``
whatever the step, since that is what makes a strong-order measurement one.
"""

import numpy as np
import pytest

from discrecontinual_equations.solver.stochastic.wiener import (
    BrownianPath,
    GaussianWienerSource,
)

_STEP = 0.25
_DRAWS = 200000


class TestGaussianWienerSource:
    def test_joint_law_of_increment_and_time_integral(self):
        source = GaussianWienerSource(seed=5)
        samples = np.array(
            [source.increments(0.0, _STEP, 1) for _ in range(_DRAWS)],
            dtype=object,
        )
        dw = np.array([s.delta_w[0] for s in samples])
        dz = np.array([s.delta_z[0] for s in samples])
        tolerance = 0.03
        assert abs(np.mean(dw)) < tolerance * np.sqrt(_STEP)
        assert np.var(dw) == pytest.approx(_STEP, rel=tolerance)
        assert np.var(dz) == pytest.approx(_STEP**3 / 3, rel=tolerance)
        assert np.mean(dw * dz) == pytest.approx(_STEP**2 / 2, rel=tolerance)

    def test_channels_are_independent_and_seeds_repeat(self):
        first = GaussianWienerSource(seed=9).increments(0.0, _STEP, 3)
        second = GaussianWienerSource(seed=9).increments(0.0, _STEP, 3)
        assert np.array_equal(first.delta_w, second.delta_w)
        assert np.array_equal(first.delta_z, second.delta_z)
        assert len(set(first.delta_w.tolist())) == 3


class TestBrownianPath:
    def test_increments_sum_across_steps_and_agree_between_step_sizes(self):
        path = BrownianPath.sample(0.0, 1.0, 2.0**-6, 1, seed=2)
        coarse = path.increments(0.0, 0.5, 1).delta_w[0]
        fine = sum(path.increments(k * 0.125, 0.125, 1).delta_w[0] for k in range(4))
        assert coarse == pytest.approx(fine)
        assert path.value(0.5)[0] == pytest.approx(coarse)

    def test_time_integral_matches_the_trapezoid_of_the_path(self):
        path = BrownianPath.sample(0.0, 1.0, 0.125, 1, seed=3)
        dz = path.increments(0.0, 0.5, 1).delta_z[0]
        nodes = np.array([path.value(k * 0.125)[0] for k in range(5)])
        assert dz == pytest.approx(np.trapezoid(nodes, dx=0.125))

    def test_off_grid_times_and_wrong_dimension_are_refused(self):
        path = BrownianPath.sample(0.0, 1.0, 0.125, 2, seed=4)
        with pytest.raises(ValueError, match="grid node"):
            path.increments(0.0, 0.1, 2)
        with pytest.raises(ValueError, match="channels"):
            path.increments(0.0, 0.125, 1)
