"""Cusp and branch-point coefficients on three-dimensional coupled systems.

An independent check of DEQ-102, with references from a graph center-manifold
reduction (``x = x0 + q y + S phi(y)``, ``S`` a basis of the range of ``A``, solved
order by order in sympy), not from Kuznetsov's closed formula. In ``y = (u, v, w)``,
mixed by the non-orthogonal ``M`` below (``x = M y``):

* ``Cusp3``: ``u' = b1 + b2 u - u^3 + u v + 2 u w``, ``v' = -v + u^2``,
  ``w' = -2 w + v + u^2``. Its fold curve is ``(b1, b2) = (4 s^3, -6 s^2)`` at
  ``u = s``, with one cusp, at ``s = 0``.
* ``Branch3``: ``u' = mu u + r u^2 + al u v + be u w + k u^3``, ``v' = -2 v + 3 u^2``,
  ``w' = -w + v - u^2`` at the origin. The center manifold is ``v = 3 u^2 / 2``,
  ``w = u^2 / 2``, so the cubic is ``3 al / 2 + be / 2 + k`` in ``u``, mostly from
  ``3 B(q, h2)`` when ``k`` is small; ``M e_u`` has length ``sqrt(1.4)``.
"""

import itertools
import math
from unittest import TestCase, mock

import numpy as np

from discrecontinual_equations.continuation.codim2 import Cusp
from discrecontinual_equations.continuation.continuation_state import (
    ContinuationState,
)
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.detection import (
    BranchPoint,
    DetectionContext,
    _branch_coefficients,
)
from discrecontinual_equations.continuation.jacobian import JacobianProvider
from discrecontinual_equations.continuation.normal_form import cusp_coefficient
from discrecontinual_equations.continuation.two_parameter_problem import (
    CurveDecoded,
    TwoParameterProblem,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

MIX = np.array([[1.0, 0.5, -0.3], [0.2, 1.0, 0.4], [-0.6, 0.1, 1.0]])
UNMIX = np.linalg.inv(MIX)
ALONG_U = MIX @ np.array([1.0, 0.0, 0.0])
SEEDS = range(6)
_SVD = np.linalg.svd

# (s, a) on the Cusp3 fold curve, q oriented along M e_u (graph reduction, exact).
FOLD_REFERENCE = [
    (-0.3, -4500.0 * math.sqrt(41.0) / 41041.0),
    (-0.1, -1500.0 * math.sqrt(4001.0) / 212053.0),
    (0.05, 6000.0 * math.sqrt(13961.0) / 2834083.0),
    (0.2, 375.0 * math.sqrt(1106.0) / 17143.0),
    (0.35, 6000.0 * math.sqrt(161.0) / 103753.0),
]
# (r, al, be, k) -> (|a|, c, label) in the library's scaling w' = a/2 w^2 + c/6 w^3.
BRANCH_REFERENCE = {
    (0.0, 1.0, -2.0, 0.1): (0.0, 18.0 / 7.0, "pitchfork"),
    (0.0, 1.0, -2.0, -0.5): (0.0, 0.0, "branch_point"),
    (0.0, -1.0, -1.0, 0.4): (0.0, -48.0 / 7.0, "pitchfork"),
    (1.0, 1.0, -2.0, 0.1): (2.0 / math.sqrt(1.4), 18.0 / 7.0, "transcritical"),
}


def _flipping_svd(seed: int):
    generator = np.random.default_rng(100 + seed)

    def svd(matrix, *args, **kwargs):
        left, values, right = _SVD(matrix, *args, **kwargs)
        signs = generator.choice([-1.0, 1.0], size=values.shape[-1])
        return left * signs, values, right * signs[:, None]

    return mock.patch.object(np.linalg, "svd", svd)


class First(Parameter, name="First parameter", abbreviation="b1"):
    pass


class Second(Parameter, name="Second parameter", abbreviation="b2"):
    pass


class State(Variable, name="State", abbreviation="x"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class Cusp3(DeterministicFunction):
    """``Cusp3`` of the module docstring, mixed by ``M``."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        u, v, w = (sum(UNMIX[i, j] * point[j] for j in range(3)) for i in range(3))
        rows = (
            b1 + b2 * u - u * u * u + u * v + 2.0 * u * w,
            -v + u * u,
            -2.0 * w + v + u * u,
        )
        return [sum(MIX[i, j] * rows[j] for j in range(3)) for i in range(3)]


class Rotated(DeterministicFunction):
    """``R F0(R^T x)`` with ``F0(y) = (y1^2, -y2)``; ``R`` turns by ``theta``.

    ``b2 = tan(theta / 2)``.

    At the origin ``q = R e1`` and ``a = +1`` for that ``q``, for every ``theta``.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        t = self.parameters[1].value
        cosine = (1.0 - t * t) / (1.0 + t * t)
        sine = 2.0 * t / (1.0 + t * t)
        first = cosine * point[0] + sine * point[1]
        second = -sine * point[0] + cosine * point[1]
        f1, f2 = first * first, -second
        return [cosine * f1 - sine * f2, sine * f1 + cosine * f2]


def _problem(function_type, size: int) -> TwoParameterProblem:
    parameters = [First(value=0.0), Second(value=0.0)]
    derivative = function_type(
        variables=[State() for _ in range(size)],
        parameters=parameters,
        results=[State() for _ in range(size)],
        time=None,
    )
    equation = DifferentialEquation(
        variables=[State() for _ in range(size)],
        time=Time(),
        parameters=parameters,
        derivative=derivative,
    )
    return TwoParameterProblem(equation, 0, 1, 0.0, AutomaticDifferentiation())


def _fold_point(s: float) -> tuple[np.ndarray, float, float]:
    return MIX @ np.array([s, s * s, s * s]), 4.0 * s**3, -6.0 * s**2


class _Branch3(JacobianProvider):
    """Closed-form Jacobian of ``Branch3`` in ``x``."""

    def __init__(self, coefficients: tuple[float, float, float, float]) -> None:
        self._coefficients = coefficients

    def state_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        r, al, be, k = self._coefficients
        u, v, w = UNMIX @ state
        inner = np.array(
            [
                [
                    parameter + 2.0 * r * u + al * v + be * w + 3.0 * k * u * u,
                    al * u,
                    be * u,
                ],
                [6.0 * u, -2.0, 0.0],
                [-2.0 * u, 1.0, -1.0],
            ],
        )
        return MIX @ inner @ UNMIX

    def parameter_derivative(self, state: np.ndarray, parameter: float) -> np.ndarray:
        _ = parameter
        return MIX @ np.array([(UNMIX @ state)[0], 0.0, 0.0])


def _branch_context(coefficients) -> DetectionContext:
    return DetectionContext(
        ContinuationState(np.zeros(3), 0.0, np.array([0.0, 0.0, 0.0, 1.0]), 0.0),
        _Branch3(coefficients),
    )


class TestThreeDimensionalCuspCoefficient(TestCase):
    def test_matches_the_graph_reduction_whatever_the_svd_signs(self):
        problem = _problem(Cusp3, 3)
        for s, reference in FOLD_REFERENCE:
            state, b1, b2 = _fold_point(s)
            for seed in SEEDS:
                with _flipping_svd(seed):
                    along = cusp_coefficient(
                        problem,
                        state,
                        b1,
                        b2,
                        orientation=ALONG_U,
                    )
                    against = cusp_coefficient(
                        problem,
                        state,
                        b1,
                        b2,
                        orientation=-ALONG_U,
                    )
                assert abs(along - reference) < 1.0e-12
                assert abs(against + reference) < 1.0e-12

    def test_cusp_test_changes_sign_once_along_the_fold_curve(self):
        problem = _problem(Cusp3, 3)
        for seed in SEEDS:
            detector = Cusp()
            values = []
            with _flipping_svd(seed):
                for s in np.linspace(-0.3, 0.35, 14):
                    state, b1, b2 = _fold_point(float(s))
                    decoded = CurveDecoded(state, b1, b2, None)
                    values.append(detector.value(problem, decoded))
            changes = sum(1 for x, y in itertools.pairwise(values) if x * y < 0.0)
            assert changes == 1


class TestContinuityFallback(TestCase):
    """Without a curve ``q``, ``q`` follows the previous point's, else the curve's."""

    @staticmethod
    def _values(degrees: list[float], *, with_curve_q: bool, seed: int) -> list[float]:
        problem = _problem(Rotated, 2)
        detector = Cusp()
        detector.reset()
        values = []
        with _flipping_svd(seed):
            for degree in degrees:
                theta = math.radians(degree)
                null_vector = 3.7 * np.array([math.cos(theta), math.sin(theta)])
                decoded = CurveDecoded(
                    np.zeros(2),
                    0.0,
                    math.tan(theta / 2.0),
                    None,
                    null_vector=null_vector if with_curve_q else None,
                )
                values.append(detector.value(problem, decoded))
        return values

    def test_turns_below_a_right_angle_keep_the_sign(self):
        for seed in SEEDS:
            values = self._values(
                [0.0, 85.0, 170.0, 255.0, 340.0],
                with_curve_q=False,
                seed=seed,
            )
            assert all(abs(value - 1.0) < 1.0e-12 for value in values)

    def test_curve_null_vector_survives_a_sharp_turn(self):
        # A 100-degree turn between points defeats continuity (q is a line, and
        # the nearer end is the wrong one); the Moore-Spence q does not care.
        for seed in SEEDS:
            values = self._values(
                [0.0, 10.0, 110.0, 120.0],
                with_curve_q=True,
                seed=seed,
            )
            assert all(abs(value - 1.0) < 1.0e-12 for value in values)


class TestThreeDimensionalBranchPoint(TestCase):
    def test_coefficients_and_labels_match_the_graph_reduction(self):
        for coefficients, (quadratic, cubic, kind) in BRANCH_REFERENCE.items():
            context = _branch_context(coefficients)
            for seed in SEEDS:
                with _flipping_svd(seed):
                    result = _branch_coefficients(context)
                    label = BranchPoint().refine_kind(context)
                assert result is not None
                a, c = result
                assert abs(abs(a) - quadratic) < 1.0e-7, coefficients
                assert abs(c - cubic) < 1.0e-5, (coefficients, c)
                assert label == kind, (coefficients, label)
