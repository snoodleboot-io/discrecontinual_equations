"""The Stratonovich drift correction is added, component by component.

For ``dX = mu X dt + sigma X o dW`` the Ito drift is ``(mu + sigma^2 / 2) X``.
A subtracted correction converges, just as fast, to the wrong process, which is
why the sign is pinned here on its own and not only through a convergence test.
"""

import numpy as np
import pytest

from discrecontinual_equations.solver.stochastic.coefficients import ItoCoefficients
from tests.solver.geometric_brownian_motion import GeometricBrownianMotionCase


class TestItoCoefficients:
    def test_ito_drift_is_the_drift_as_written(self):
        case = GeometricBrownianMotionCase(mu=1.0, sigma=0.5)
        coefficients = ItoCoefficients(case.equation(), "ito")
        assert coefficients.drift(np.array([2.0]), 0.0)[0] == pytest.approx(2.0)
        assert coefficients.diffusion(np.array([2.0]), 0.0)[0] == pytest.approx(1.0)

    def test_stratonovich_drift_adds_half_b_b_prime(self):
        case = GeometricBrownianMotionCase(mu=1.0, sigma=0.5)
        coefficients = ItoCoefficients(case.equation(), "stratonovich")
        # (mu + sigma^2 / 2) x = (1 + 0.125) * 2
        assert coefficients.drift(np.array([2.0]), 0.0)[0] == pytest.approx(
            2.25,
            rel=1e-6,
        )

    def test_correction_uses_each_component_s_own_coefficient(self):
        case = GeometricBrownianMotionCase(mu=0.0, sigma=0.5, dimension=2)
        coefficients = ItoCoefficients(case.equation(), "stratonovich")
        drift = coefficients.drift(np.array([1.0, 4.0]), 0.0)
        assert drift == pytest.approx([0.125 * 1.0, 0.125 * 4.0], rel=1e-6)
