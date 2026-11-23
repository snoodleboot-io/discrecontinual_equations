"""The Hopf test function, checked against the bialternate determinant it replaces.

The reference here is built independently of the code under test: the
bialternate product ``2A (.) I`` assembled entry by entry from Guckenheimer,
Myers & Sturmfels (1997), eq. (9), with the sign of its determinant taken from
``slogdet`` so that it cannot overflow.
"""

import math
from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.bialternate import bialternate_test_value
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

LOCATION_TOLERANCE = 1.0e-6
FREQUENCY_TOLERANCE = 1.0e-6
LARGE_DIMENSION = 32
STIFF_RATE = 5.0
SLOW_RATE = 0.01
TAIL_COUPLING = 0.5


class Mu(Parameter, name="Continuation parameter", abbreviation="mu"):
    pass


class X(Variable, name="State x", abbreviation="x"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


def reference_bialternate(matrix: np.ndarray) -> np.ndarray:
    """``2A (.) I`` from GMS (1997) eq. (9): rows ``pq`` (p > q), columns ``rs``."""
    dimension = matrix.shape[0]
    pairs = [(p, q) for p in range(1, dimension) for q in range(p)]
    result = np.zeros((len(pairs), len(pairs)))
    for row, (p, q) in enumerate(pairs):
        for column, (r, s) in enumerate(pairs):
            if r == q:
                result[row, column] = -matrix[p, s]
            elif r != p and s == q:
                result[row, column] = matrix[p, r]
            elif r == p and s == q:
                result[row, column] = matrix[p, p] + matrix[q, q]
            elif r == p:
                result[row, column] = matrix[q, s]
            elif s == p:
                result[row, column] = -matrix[q, r]
    return result


def reference_sign(matrix: np.ndarray) -> float:
    sign, _logarithm = np.linalg.slogdet(reference_bialternate(matrix))
    return float(sign)


class HopfWithTail(DeterministicFunction):
    """Planar Hopf normal form driving a chain ``x_i' = -k x_i + c x_{i-1}``.

    The Hopf point is at ``mu = 0`` with frequency 1; the tail adds ``n - 2``
    eigenvalues ``-k``, whose pairwise sums push the bialternate determinant past
    the floating-point range once ``n`` is a few dozen.
    """

    rate = STIFF_RATE

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        radius_squared = x * x + y * y
        tail = [
            -self.rate * point[index] + TAIL_COUPLING * point[index - 1]
            for index in range(2, len(point))
        ]
        return [
            mu * x - y - x * radius_squared,
            x + mu * y - y * radius_squared,
            *tail,
        ]


class StiffTail(HopfWithTail):
    rate = STIFF_RATE


class SlowTail(HopfWithTail):
    rate = SLOW_RATE


_ROTATION = np.linalg.qr(
    np.array([[1.0, 2.0, 0.5], [-0.3, 1.0, 2.0], [0.7, -1.2, 1.0]]),
)[0]


class RotatedHopf3D(DeterministicFunction):
    """A 3-D Hopf (``mu +- i`` and ``-1``) seen in rotated coordinates.

    The rotation fills the Jacobian, so every entry of the bialternate product
    is exercised; the Hopf point stays at ``mu = 0`` with frequency 1.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x, y, z = _ROTATION.T @ np.asarray(point, dtype=float)
        radius_squared = x * x + y * y
        local = np.array(
            [
                mu * x - y - x * radius_squared,
                x + mu * y - y * radius_squared,
                -z + x * y,
            ],
        )
        return list(_ROTATION @ local)


def build_equation(function_type, dimension: int) -> DifferentialEquation:
    mu = Mu(value=0.0)
    variables = [X() for _ in range(dimension)]
    derivative = function_type(
        variables=variables,
        parameters=[mu],
        results=[X() for _ in range(dimension)],
        time=None,
    )
    return DifferentialEquation(
        variables=variables,
        time=Time(),
        parameters=[mu],
        derivative=derivative,
    )


def hopf_points(function_type, dimension: int):
    equation = build_equation(function_type, dimension)
    config = ContinuationConfig(
        detectors=["hopf"],
        initial_parameter=-0.5,
        initial_step=0.05,
        maximum_step=0.15,
        parameter_lower_bound=-0.5,
        parameter_upper_bound=0.5,
    )
    continuer = ContinuerBuilder.build(config, equation)
    branch = continuer.solve(equation, initial_values=[0.0] * dimension)
    return [point for point in branch.special_points if point.kind == "hopf"]


class TestHopfDetectionAtLargeDimension(TestCase):
    def assert_single_hopf_at_origin(self, points) -> None:
        assert len(points) == 1
        assert abs(points[0].parameter) < LOCATION_TOLERANCE
        assert abs(points[0].frequency - 1.0) < FREQUENCY_TOLERANCE

    def test_hopf_found_when_the_determinant_would_underflow(self):
        # 435 tail factors of -0.02 underflow det(2A (.) I) to zero.
        self.assert_single_hopf_at_origin(hopf_points(SlowTail, LARGE_DIMENSION))

    def test_hopf_found_when_the_determinant_would_overflow(self):
        # 435 tail factors of -10 overflow det(2A (.) I) to infinity.
        self.assert_single_hopf_at_origin(hopf_points(StiffTail, LARGE_DIMENSION))

    def test_hopf_found_in_a_full_three_dimensional_jacobian(self):
        self.assert_single_hopf_at_origin(hopf_points(RotatedHopf3D, 3))


class TestBialternateTestValue(TestCase):
    def test_reference_matches_the_published_three_by_three_product(self):
        a = np.arange(1.0, 10.0).reshape(3, 3)
        expected = np.array(
            [
                [a[0, 0] + a[1, 1], a[1, 2], -a[0, 2]],
                [a[2, 1], a[0, 0] + a[2, 2], a[0, 1]],
                [-a[2, 0], a[1, 0], a[1, 1] + a[2, 2]],
            ],
        )
        np.testing.assert_array_equal(reference_bialternate(a), expected)

    def test_reference_eigenvalues_are_the_pairwise_sums(self):
        matrix = np.random.default_rng(5).standard_normal((6, 6))
        eigenvalues = np.linalg.eigvals(matrix)
        first, second = np.triu_indices(6, 1)
        expected = np.prod(eigenvalues[first] + eigenvalues[second]).real
        determinant = np.linalg.det(reference_bialternate(matrix))
        assert abs(determinant / expected - 1.0) < 1.0e-9

    def test_sign_matches_the_bialternate_determinant(self):
        rng = np.random.default_rng(0)
        for dimension in range(2, 13):
            for _ in range(20):
                matrix = rng.standard_normal((dimension, dimension))
                value = bialternate_test_value(matrix)
                assert math.copysign(1.0, value) == reference_sign(matrix), dimension

    def test_sign_matches_with_repeated_imaginary_parts(self):
        # Two complex pairs a1 +- i and a2 -+ i: the cross sums a1 + a2 are real
        # and come twice, so they must not flip the sign.
        block = np.array([[-1.0, -1.0], [1.0, -1.0]])
        matrix = np.zeros((5, 5))
        matrix[:2, :2] = block
        matrix[2:4, 2:4] = block + 3.0 * np.eye(2)
        matrix[4, 4] = 0.5
        value = bialternate_test_value(matrix)
        assert math.copysign(1.0, value) == reference_sign(matrix)

    def test_sign_matches_and_stays_finite_far_beyond_the_determinant_range(self):
        rng = np.random.default_rng(1)
        for scale in (1.0e-3, 1.0e3):
            matrix = scale * rng.standard_normal((40, 40))
            value = bialternate_test_value(matrix)
            assert math.isfinite(value), scale
            assert value != 0.0, scale
            assert math.copysign(1.0, value) == reference_sign(matrix), scale

    def test_planar_value_is_the_trace(self):
        matrix = np.array([[0.3, -2.0], [1.5, -0.7]])
        assert abs(bialternate_test_value(matrix) - (-0.4)) < 1.0e-14

    def test_magnitude_is_the_smallest_pairwise_sum(self):
        # Eigenvalues 0.1 +- 2i, -3, -5: sums 0.2, -8, -2.9 +- 2i, -4.9 +- 2i.
        matrix = np.zeros((4, 4))
        matrix[:2, :2] = [[0.1, -2.0], [2.0, 0.1]]
        matrix[2, 2] = -3.0
        matrix[3, 3] = -5.0
        # The real factors are 0.2 and -8, so the product is negative.
        assert abs(bialternate_test_value(matrix) - (-0.2)) < 1.0e-12

    def test_value_crosses_zero_linearly_at_a_hopf_point(self):
        def family(mu: float) -> np.ndarray:
            matrix = np.diag([mu, mu, -1.0, -2.0, -4.0])
            matrix[0, 1], matrix[1, 0] = -3.0, 3.0
            return matrix

        below = bialternate_test_value(family(-1.0e-3))
        above = bialternate_test_value(family(1.0e-3))
        assert below * above < 0.0
        assert abs(abs(below) - 2.0e-3) < 1.0e-12
        assert abs(abs(above) - 2.0e-3) < 1.0e-12

    def test_neutral_saddle_still_changes_sign(self):
        # Real eigenvalues 1 + mu and -1: their sum crosses zero, as with the
        # bialternate determinant (a neutral saddle, not a Hopf point).
        below = bialternate_test_value(np.diag([1.0 - 1.0e-3, -1.0, -3.0]))
        above = bialternate_test_value(np.diag([1.0 + 1.0e-3, -1.0, -3.0]))
        assert below * above < 0.0

    def test_one_dimensional_system_has_a_constant_value(self):
        assert bialternate_test_value(np.array([[-2.0]])) == 1.0
        assert bialternate_test_value(np.array([[3.0]])) == 1.0

    def test_non_finite_jacobian_gives_nan(self):
        matrix = np.array([[np.inf, 0.0], [0.0, 1.0]])
        assert math.isnan(bialternate_test_value(matrix))
