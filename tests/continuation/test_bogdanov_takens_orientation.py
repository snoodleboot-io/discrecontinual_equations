"""Degenerate Bogdanov-Takens test functions on a three-dimensional BT curve.

Every system is written in coordinates ``y = (u, v, w)``,

    ``u' = v``,  ``v' = b1 + b2 v + A(c) u^2 + B(c) u v + u w``,  ``w' = -w + u^2``,

and then turned by ``x = M(c) y`` with ``M(c) = R(theta(c)) M0``: ``R`` is a
rotation about the third axis by ``theta = 2.3 + 1.2 c`` and ``M0`` a fixed
non-orthogonal mixing. The Bogdanov-Takens curve is ``x = 0``, ``b1 = b2 = 0`` for
every ``c``. In ``y`` the Jordan chains are ``q0 = e_u``, ``q1 = e_v``, ``p1 = e_v``
and ``p0 = e_u``, so ``a = A(c)`` and ``b = B(c)``. With the normalisation
``|q0| = 1``, ``<q0, q1> = 0`` (Bosschaert and Kuznetsov 2021, eq. (19)),
``q0_x = M e_u / |M e_u|`` and, by the transformation rule worked out in this
module's derivation (``q -> lambda q`` and ``p -> p / lambda`` scale ``a`` and
``b`` by ``lambda``; the ``q1 -> q1 + t q0`` freedom leaves both unchanged),

    ``a_x = A(c) / |M0 e_u|``,  ``b_x = B(c) / |M0 e_u|``

for ``q0_x`` oriented along ``M(c) e_u``. None of this comes from the library.

``q0_x`` rotates with ``c``, and its largest component changes index (and sign)
near ``c = -0.15``: a scan that fixes the sign of ``q0`` by its largest component
sees ``a`` and ``b`` change sign there although neither vanishes.
"""

import math
from unittest import TestCase, mock

import numpy as np

from discrecontinual_equations.continuation.bogdanov_takens_curve import (
    BogdanovTakensCurveResidual,
)
from discrecontinual_equations.continuation.codim3 import (
    Codim3Analyzer,
    DegenerateBogdanovTakens,
    DegenerateBogdanovTakensB,
)
from discrecontinual_equations.continuation.codim3_builder import (
    Codim3Driver,
    Codim3Seed,
)
from discrecontinual_equations.continuation.codim3_config import Codim3Config
from discrecontinual_equations.continuation.cusp_curve import Codim3Decoded
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.normal_form import (
    bogdanov_takens_coefficient,
    bogdanov_takens_second_coefficient,
)
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

M0 = np.array([[1.0, 0.3, -0.2], [0.3, 1.0, 0.4], [0.2, -0.5, 1.0]])
SCALE = 1.0 / float(np.linalg.norm(M0[:, 0]))
SEEDS = range(8)
CURVE = np.linspace(-0.5, 0.5, 41)
QUADRATIC_ZERO = -0.35
CROSS_ZERO = 0.3
LOCATION_TOLERANCE = 1.0e-2
REFERENCE_TOLERANCE = 1.0e-9
_SVD = np.linalg.svd


def _flipping_svd(seed: int):
    """``np.linalg.svd`` with each singular pair's sign drawn at random."""
    generator = np.random.default_rng(seed)

    def svd(matrix, *args, **kwargs):
        result = _SVD(matrix, *args, **kwargs)
        if not kwargs.get("compute_uv", True):
            return result
        left, values, right = result
        left_signs = generator.choice([-1.0, 1.0], size=left.shape[-1])
        right_signs = generator.choice([-1.0, 1.0], size=right.shape[-2])
        shared = values.shape[-1]
        right_signs[:shared] = left_signs[:shared]
        return left * left_signs, values, right * right_signs[:, None]

    return svd


def _flipped(seed: int):
    return mock.patch.object(np.linalg, "svd", _flipping_svd(seed))


def _mixing(c: float) -> np.ndarray:
    angle = 2.3 + 1.2 * c
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    return rotation @ M0


def _direction(c: float) -> np.ndarray:
    """``M(c) e_u``: the reference orientation of ``q0``."""
    return _mixing(c)[:, 0]


class First(Parameter, name="BT first", abbreviation="b1"):
    pass


class Second(Parameter, name="BT second", abbreviation="b2"):
    pass


class Third(Parameter, name="BT third", abbreviation="c"):
    pass


class State(Variable, name="State", abbreviation="x"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class _TurnedBogdanovTakens(DeterministicFunction):
    """The mixed system of the module docstring, with ``A`` and ``B`` to fill in."""

    @staticmethod
    def quadratic(c: float) -> float:
        _ = c
        return 1.0

    @staticmethod
    def cross(c: float) -> float:
        _ = c
        return 1.0

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        c = self.parameters[2].value
        mixing = _mixing(c)
        unmixing = np.linalg.inv(mixing)
        u, v, w = (sum(unmixing[i, j] * point[j] for j in range(3)) for i in range(3))
        rows = (
            v,
            b1 + b2 * v + self.quadratic(c) * u * u + self.cross(c) * u * v + u * w,
            -w + u * u,
        )
        return [sum(mixing[i, j] * rows[j] for j in range(3)) for i in range(3)]


class GenericCurve(_TurnedBogdanovTakens):
    """``A = B = 1``: no degenerate Bogdanov-Takens point anywhere."""


class QuadraticZeroCurve(_TurnedBogdanovTakens):
    """``A = c + 0.35``: ``a = 0`` at ``c = -0.35``; ``b = 1``."""

    @staticmethod
    def quadratic(c: float) -> float:
        return c - QUADRATIC_ZERO


class CrossZeroCurve(_TurnedBogdanovTakens):
    """``B = c - 0.3``: ``b = 0`` at ``c = 0.3``; ``a = 1``."""

    @staticmethod
    def cross(c: float) -> float:
        return c - CROSS_ZERO


def _equation(function_type, c: float = 0.0) -> DifferentialEquation:
    parameters = [First(value=0.0), Second(value=0.0), Third(value=c)]
    variables = [State(), State(), State()]
    derivative = function_type(
        variables=variables,
        parameters=parameters,
        results=[State(), State(), State()],
        time=None,
    )
    return DifferentialEquation(
        variables=[State(), State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=derivative,
    )


def _problem(equation: DifferentialEquation) -> TwoParameterProblem:
    return TwoParameterProblem(equation, 0, 1, 0.0, AutomaticDifferentiation())


class _Point:
    def __init__(self, state: np.ndarray, parameter: float) -> None:
        self.state = list(state)
        self.parameter = parameter


class _Curve:
    """The exact BT curve as augmented points ``(x, q, q1, b1, b2)`` at ``c``.

    ``q`` is normalised by ``normalization . q = 1`` and ``q1`` by
    ``normalization . q1 = 0``, as :class:`BogdanovTakensCurveResidual` does.
    """

    def __init__(self, normalization: np.ndarray) -> None:
        self.points = []
        for c in CURVE:
            mixing = _mixing(float(c))
            kernel = mixing[:, 0] / (normalization @ mixing[:, 0])
            chain = mixing[:, 1] / (normalization @ mixing[:, 0])
            chain = chain - (normalization @ chain) * kernel
            state = np.concatenate([np.zeros(3), kernel, chain, [0.0, 0.0]])
            self.points.append(_Point(state, float(c)))


def _scan(function_type, normalization: np.ndarray, seed: int | None):
    equation = _equation(function_type)
    problem = _problem(equation)
    third = equation.derivative.parameters[2]
    residual = BogdanovTakensCurveResidual(problem, 3, normalization, third)
    analyzer = Codim3Analyzer(
        [DegenerateBogdanovTakens(), DegenerateBogdanovTakensB()],
        third,
    )
    if seed is None:
        return analyzer.analyze(_Curve(normalization), residual, problem)
    with _flipped(seed):
        return analyzer.analyze(_Curve(normalization), residual, problem)


def _normalizations():
    """The seed's null vector with either sign, as LAPACK might return it."""
    start = _direction(float(CURVE[0]))
    start = start / np.linalg.norm(start)
    return (start, -start)


class _Recorder(DegenerateBogdanovTakens):
    """The ``a`` test function, recording the orientation each scan starts from."""

    def __init__(self) -> None:
        super().__init__()
        self.starts: list[np.ndarray | None] = []

    def reset(self) -> None:
        super().reset()
        self.starts.append(self._previous)


class TestTheCurveTurnsItsNullVector(TestCase):
    def test_largest_component_of_q0_changes_index_along_the_curve(self):
        # The precondition of the bug: argmax |q0| is not constant along the curve.
        indices = {int(np.argmax(np.abs(_direction(float(c))))) for c in CURVE}
        assert len(indices) > 1


class TestBogdanovTakensCoefficientReference(TestCase):
    def _point(self, function_type, c: float):
        equation = _equation(function_type, c)
        return _problem(equation), np.zeros(3)

    def test_coefficients_match_the_closed_form_reference(self):
        for function_type in (GenericCurve, QuadraticZeroCurve, CrossZeroCurve):
            for c in (-0.45, -0.2, -0.1, 0.15, 0.4):
                problem, state = self._point(function_type, c)
                orientation = _direction(c)
                a = bogdanov_takens_coefficient(
                    problem,
                    state,
                    0.0,
                    0.0,
                    orientation=orientation,
                )
                b = bogdanov_takens_second_coefficient(
                    problem,
                    state,
                    0.0,
                    0.0,
                    orientation=orientation,
                )
                assert abs(a - SCALE * function_type.quadratic(c)) < REFERENCE_TOLERANCE
                assert abs(b - SCALE * function_type.cross(c)) < REFERENCE_TOLERANCE

    def test_orientation_decides_the_sign_whatever_the_svd_returns(self):
        problem, state = self._point(GenericCurve, -0.1)
        orientation = _direction(-0.1)
        for seed in SEEDS:
            with _flipped(seed):
                for sign in (1.0, -1.0):
                    a = bogdanov_takens_coefficient(
                        problem,
                        state,
                        0.0,
                        0.0,
                        orientation=sign * orientation,
                    )
                    b = bogdanov_takens_second_coefficient(
                        problem,
                        state,
                        0.0,
                        0.0,
                        orientation=sign * orientation,
                    )
                    assert abs(a - sign * SCALE) < REFERENCE_TOLERANCE
                    assert abs(b - sign * SCALE) < REFERENCE_TOLERANCE

    def test_without_orientation_the_value_does_not_follow_the_svd(self):
        problem, state = self._point(GenericCurve, 0.2)
        expected = bogdanov_takens_coefficient(problem, state, 0.0, 0.0)
        for seed in SEEDS:
            with _flipped(seed):
                value = bogdanov_takens_coefficient(problem, state, 0.0, 0.0)
            assert abs(value - expected) < REFERENCE_TOLERANCE


class TestDegenerateBogdanovTakensScan(TestCase):
    def test_bogdanov_takens_curve_decodes_its_null_vector(self):
        equation = _equation(GenericCurve)
        residual = BogdanovTakensCurveResidual(
            _problem(equation),
            3,
            np.array([1.0, 0.0, 0.0]),
            equation.derivative.parameters[2],
        )
        state = np.arange(11, dtype=float)
        decoded = residual.decode(state, 0.25)
        assert decoded.null_vector is not None
        assert np.array_equal(decoded.null_vector, [3.0, 4.0, 5.0])

    def test_generic_curve_reports_no_degenerate_point(self):
        for normalization in _normalizations():
            for seed in (None, *SEEDS):
                points = _scan(GenericCurve, normalization, seed)
                assert points == [], (seed, points)

    def test_quadratic_zero_is_found_once(self):
        for normalization in _normalizations():
            for seed in (None, *SEEDS):
                points = _scan(QuadraticZeroCurve, normalization, seed)
                assert [point.kind for point in points] == [
                    "degenerate_bogdanov_takens",
                ], (seed, points)
                assert abs(points[0].parameter_c - QUADRATIC_ZERO) < LOCATION_TOLERANCE

    def test_cross_zero_is_found_once(self):
        for normalization in _normalizations():
            for seed in (None, *SEEDS):
                points = _scan(CrossZeroCurve, normalization, seed)
                assert [point.kind for point in points] == [
                    "degenerate_bogdanov_takens_b",
                ], (seed, points)
                assert abs(points[0].parameter_c - CROSS_ZERO) < LOCATION_TOLERANCE

    def test_test_functions_follow_the_curve_null_vector(self):
        # Along a BT curve its own q (c . q = 1) fixes the orientation, whatever
        # the SVD returns and whatever the previous point said.
        equation = _equation(GenericCurve, -0.1)
        problem = _problem(equation)
        null_vector = _direction(-0.1)
        for seed in SEEDS:
            for detector in (DegenerateBogdanovTakens(), DegenerateBogdanovTakensB()):
                with _flipped(seed):
                    for sign in (1.0, -1.0, 1.0):
                        decoded = Codim3Decoded(
                            np.zeros(3),
                            0.0,
                            0.0,
                            -0.1,
                            null_vector=sign * null_vector,
                        )
                        value = detector.value(problem, decoded)
                        assert abs(value - sign * SCALE) < REFERENCE_TOLERANCE

    def test_test_functions_track_by_continuity_without_a_null_vector(self):
        equation = _equation(GenericCurve)
        problem = _problem(equation)
        third = equation.derivative.parameters[2]
        for seed in SEEDS:
            for detector in (DegenerateBogdanovTakens(), DegenerateBogdanovTakensB()):
                signs = set()
                with _flipped(seed):
                    for c in CURVE:
                        third.value = float(c)
                        decoded = Codim3Decoded(np.zeros(3), 0.0, 0.0, float(c))
                        value = detector.value(problem, decoded)
                        signs.add(math.copysign(1.0, value))
                assert len(signs) == 1, (seed, detector.kind)

    def test_each_scan_starts_without_an_orientation(self):
        recorder = _Recorder()
        equation = _equation(GenericCurve)
        problem = _problem(equation)
        third = equation.derivative.parameters[2]
        normalization = _normalizations()[0]
        residual = BogdanovTakensCurveResidual(problem, 3, normalization, third)
        analyzer = Codim3Analyzer([recorder], third)
        for _ in range(2):
            analyzer.analyze(_Curve(normalization), residual, problem)
        assert recorder.starts == [None, None]


class TestDegenerateBogdanovTakensContinuation(TestCase):
    """End to end: the driver continues the BT curve and scans it."""

    def _run(self, function_type):
        equation = _equation(function_type, -0.5)
        config = Codim3Config(
            continuation_parameter_index=0,
            second_parameter_index=1,
            third_parameter_index=2,
            codim3_curve="bogdanov_takens",
            codim3_detectors=[
                "degenerate_bogdanov_takens",
                "degenerate_bogdanov_takens_b",
            ],
            initial_parameter=-0.5,
            direction=1,
            parameter_lower_bound=-0.6,
            parameter_upper_bound=0.6,
            initial_step=0.02,
            maximum_step=0.05,
            maximum_points=200,
        )
        seed = Codim3Seed(state=[0.0, 0.0, 0.0], parameter_a=0.0, parameter_b=0.0)
        return Codim3Driver.run(config, equation, seed)

    def test_generic_curve_reports_no_degenerate_point(self):
        for seed in (None, 0, 1):
            if seed is None:
                result = self._run(GenericCurve)
            else:
                with _flipped(seed):
                    result = self._run(GenericCurve)
            third = [parameters[2] for parameters in result.curve_parameters]
            assert min(third) < -0.4
            assert max(third) > 0.4
            assert result.points == [], (seed, result.points)

    def test_quadratic_zero_is_found_once(self):
        with _flipped(2):
            result = self._run(QuadraticZeroCurve)
        assert [point.kind for point in result.points] == [
            "degenerate_bogdanov_takens",
        ]
        assert abs(result.points[0].parameter_c - QUADRATIC_ZERO) < LOCATION_TOLERANCE
