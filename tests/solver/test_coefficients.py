"""The Stratonovich drift correction is added, component by component.

For ``dX = mu X dt + sigma X o dW`` the Ito drift is ``(mu + sigma^2 / 2) X``.
A subtracted correction converges, just as fast, to the wrong process, which is
why the sign is pinned here on its own and not only through a convergence test.
"""

import numpy as np
import pytest

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.solver.stochastic.coefficients import ItoCoefficients
from discrecontinual_equations.variable import Variable
from tests.solver.conformal_geometric_brownian_motion import ConformalCase
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


class TestMatrixDiffusion:
    """The matrix form: ``m`` drivers, ``G @ dW``, and the column-sum correction."""

    @staticmethod
    def _conformal():
        return ConformalCase(a=0.0, sigma=0.5).equation()

    def test_vector_form_is_diagonal_with_one_driver_per_component(self):
        case = GeometricBrownianMotionCase(sigma=0.5, dimension=3)
        coefficients = ItoCoefficients(case.equation(), "ito")
        point = np.array([1.0, 2.0, 3.0])
        diffusion = coefficients.diffusion(point, 0.0)
        assert not ItoCoefficients.is_matrix(diffusion)
        assert coefficients.drivers(point, 0.0) == 3
        increments = np.array([0.1, -0.2, 0.3])
        assert coefficients.spread(diffusion, increments) == pytest.approx(
            diffusion * increments,
        )

    def test_matrix_form_counts_columns_and_applies_g_dw(self):
        coefficients = ItoCoefficients(self._conformal(), "ito")
        point = np.array([1.0, 2.0])
        diffusion = coefficients.diffusion(point, 0.0)
        assert ItoCoefficients.is_matrix(diffusion)
        np.testing.assert_allclose(diffusion, [[0.5, -1.0], [1.0, 0.5]])
        assert coefficients.drivers(point, 0.0) == 2
        increments = np.array([0.1, -0.2])
        assert coefficients.spread(diffusion, increments) == pytest.approx(
            [0.5 * 0.1 + 1.0 * 0.2, 1.0 * 0.1 - 0.5 * 0.2],
        )

    def test_column_jacobians_of_the_conformal_noise(self):
        coefficients = ItoCoefficients(self._conformal(), "ito")
        jacobians = coefficients.column_jacobians(np.array([1.0, 2.0]), 0.0)
        # G_1 = sigma (x, y): Jacobian sigma I. G_2 = sigma (-y, x): a rotation.
        assert jacobians[0] == pytest.approx(0.5 * np.eye(2), abs=1e-7)
        assert jacobians[1] == pytest.approx(
            0.5 * np.array([[0.0, -1.0], [1.0, 0.0]]),
            abs=1e-7,
        )

    def test_column_jacobians_of_a_vector_diffusion_sit_on_the_diagonal(self):
        case = GeometricBrownianMotionCase(sigma=0.5, dimension=2)
        coefficients = ItoCoefficients(case.equation(), "ito")
        jacobians = coefficients.column_jacobians(np.array([1.0, 4.0]), 0.0)
        # Column k is b_k e_k with b_k = sigma x_k: only entry (k, k, k) is nonzero.
        expected = np.zeros((2, 2, 2))
        expected[0, 0, 0] = 0.5
        expected[1, 1, 1] = 0.5
        assert jacobians == pytest.approx(expected, abs=1e-7)

    def test_stratonovich_correction_of_the_conformal_noise_vanishes(self):
        # (dG_1/dx) G_1 = sigma^2 (x, y) and (dG_2/dx) G_2 = sigma^2 (-x, -y): the
        # two columns cancel, so the Ito and Stratonovich equations coincide and
        # z0 exp(a t + sigma W_c) is the solution in both senses.
        coefficients = ItoCoefficients(self._conformal(), "stratonovich")
        assert coefficients.drift(np.array([1.0, 2.0]), 0.0) == pytest.approx(
            [0.0, 0.0],
            abs=1e-7,
        )

    def test_matrix_correction_reduces_to_the_vector_one_on_a_diagonal_matrix(self):
        class DiagonalAsMatrix(StochasticFunction):
            def eval(self, point, time=None):  # noqa: ARG002
                return [0.0, 0.0]

            def diffusion(self, point, time=None):  # noqa: ARG002
                return [[0.5 * point[0], 0.0], [0.0, 0.5 * point[1]]]

        variables = [
            Variable(name="x", discretization=[]),
            Variable(name="y", discretization=[]),
        ]
        time = Variable(name="t", discretization=[])
        results = [
            Variable(name="dx", discretization=[]),
            Variable(name="dy", discretization=[]),
        ]
        equation = DifferentialEquation(
            variables=variables,
            time=time,
            parameters=[],
            derivative=DiagonalAsMatrix(variables, [], results, time),
        )
        matrix = ItoCoefficients(equation, "stratonovich")
        vector = ItoCoefficients(
            GeometricBrownianMotionCase(mu=0.0, sigma=0.5, dimension=2).equation(),
            "stratonovich",
        )
        point = np.array([1.0, 4.0])
        assert matrix.drift(point, 0.0) == pytest.approx(
            vector.drift(point, 0.0),
            rel=1e-6,
        )
