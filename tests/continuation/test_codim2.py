"""Two-parameter continuation and codim-2 detection.

Each system has an analytically known codim-2 point, so the detected location is
checked against the exact value. The augmented curve Jacobian and the normal-form
coefficients use derivatives from the automatic-differentiation provider, so they
are exact; the residual location tolerance reflects the continuation step size.
"""

from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.codim2_builder import (
    Codim2Driver,
    CurveSeed,
)
from discrecontinual_equations.continuation.codim2_config import Codim2Config
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.normal_form import (
    cubic_fold_coefficient,
    cusp_coefficient,
    first_lyapunov,
)
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

LOCATION_TOLERANCE = 2.0e-2


class First(Parameter, name="First parameter", abbreviation="p1"):
    pass


class Second(Parameter, name="Second parameter", abbreviation="p2"):
    pass


class State(Variable, name="State", abbreviation="s"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


def _variables(count: int) -> list[Variable]:
    return [State() for _ in range(count)]


def _equation(
    function_type: type[DeterministicFunction],
    count: int,
    first: float,
    second: float,
) -> DifferentialEquation:
    parameters = [First(value=first), Second(value=second)]
    derivative = function_type(
        variables=_variables(count),
        parameters=parameters,
        results=_variables(count),
        time=None,
    )
    return DifferentialEquation(
        variables=_variables(count),
        time=Time(),
        parameters=parameters,
        derivative=derivative,
    )


class BogdanovTakensNormalForm(DeterministicFunction):
    """x' = y, y' = b1 + b2 y + x^2 - x y; a Bogdanov-Takens point at (0, 0)."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        x, y = point[0], point[1]
        return [y, b1 + b2 * y + x * x - x * y]


class ZeroHopfSystem(DeterministicFunction):
    """A fold in x coupled to a Hopf in (y, z); zero-Hopf at mu1 = 0."""

    frequency = 1.0

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu1 = self.parameters[0].value
        mu2 = self.parameters[1].value
        x, y, z = point[0], point[1], point[2]
        radius = y * y + z * z
        return [
            mu1 - x * x,
            mu2 * y - self.frequency * z - y * radius,
            self.frequency * y + mu2 * z - z * radius,
        ]


class HopfHopfSystem(DeterministicFunction):
    """Two decoupled Hopf blocks; a Hopf-Hopf point at mu2 = 0."""

    first_frequency = 1.0
    second_frequency = 2.0

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu1 = self.parameters[0].value
        mu2 = self.parameters[1].value
        x, y, z, w = point[0], point[1], point[2], point[3]
        first = x * x + y * y
        second = z * z + w * w
        return [
            mu1 * x - self.first_frequency * y - x * first,
            self.first_frequency * x + mu1 * y - y * first,
            mu2 * z - self.second_frequency * w - z * second,
            self.second_frequency * z + mu2 * w - w * second,
        ]


class CuspNormalForm(DeterministicFunction):
    """x' = b1 + b2 x - x^3; a cusp at (b1, b2) = (0, 0), where x = 0."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        x = point[0]
        return [b1 + b2 * x - x * x * x]


class BautinSystem(DeterministicFunction):
    """Cubic Bautin form; Hopf at b1 = 0, first Lyapunov coefficient ~ b2."""

    frequency = 1.0

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        x, y = point[0], point[1]
        radius = x * x + y * y
        return [
            b1 * x - self.frequency * y + b2 * x * radius,
            self.frequency * x + b1 * y + b2 * y * radius,
        ]


class TestCodim2(TestCase):
    def test_hopf_curve_finds_bogdanov_takens(self):
        # The Hopf curve b1 = -b2^2 turns where omega -> 0, at the origin.
        equation = _equation(BogdanovTakensNormalForm, 2, first=-0.25, second=-0.5)
        seed = CurveSeed(state=[-0.5, 0.0], parameter_a=-0.25, frequency=1.0)
        config = Codim2Config(
            continuation_parameter_index=0,
            second_parameter_index=1,
            curve="hopf",
            codim2_detectors=["bogdanov_takens"],
            initial_parameter=-0.5,
            direction=1,
            parameter_lower_bound=-0.6,
            parameter_upper_bound=0.05,
        )
        result = Codim2Driver.run(config, equation, seed)
        points = [p for p in result.points if p.kind == "bogdanov_takens"]
        assert len(points) == 1
        assert abs(points[0].parameter_a) < LOCATION_TOLERANCE
        assert abs(points[0].parameter_b) < LOCATION_TOLERANCE
        assert abs(points[0].omega) < LOCATION_TOLERANCE

    def test_fold_curve_finds_bogdanov_takens(self):
        # The fold curve sits at b1 = 0; the second eigenvalue b2 vanishes at BT.
        equation = _equation(BogdanovTakensNormalForm, 2, first=0.0, second=-0.5)
        seed = CurveSeed(state=[0.0, 0.0], parameter_a=0.0)
        config = Codim2Config(
            continuation_parameter_index=0,
            second_parameter_index=1,
            curve="fold",
            codim2_detectors=["fold_bogdanov_takens"],
            initial_parameter=-0.5,
            direction=1,
            parameter_lower_bound=-0.6,
            parameter_upper_bound=0.5,
        )
        result = Codim2Driver.run(config, equation, seed)
        points = [p for p in result.points if p.kind == "bogdanov_takens"]
        assert len(points) == 1
        assert abs(points[0].parameter_a) < LOCATION_TOLERANCE
        assert abs(points[0].parameter_b) < LOCATION_TOLERANCE

    def test_finds_zero_hopf(self):
        # The Hopf curve lives at mu2 = 0; the fold in x gives zero-Hopf at mu1 = 0.
        equation = _equation(ZeroHopfSystem, 3, first=0.5, second=0.0)
        seed = CurveSeed(state=[0.5**0.5, 0.0, 0.0], parameter_a=0.0, frequency=1.0)
        config = Codim2Config(
            continuation_parameter_index=1,
            second_parameter_index=0,
            curve="hopf",
            codim2_detectors=["zero_hopf"],
            initial_parameter=0.5,
            direction=-1,
            parameter_lower_bound=-0.1,
            parameter_upper_bound=0.6,
        )
        result = Codim2Driver.run(config, equation, seed)
        points = [p for p in result.points if p.kind == "zero_hopf"]
        assert len(points) == 1
        assert abs(points[0].parameter_b) < LOCATION_TOLERANCE

    def test_finds_hopf_hopf(self):
        # Continuing the first Hopf curve, the second pair reaches the axis at mu2 = 0.
        equation = _equation(HopfHopfSystem, 4, first=0.0, second=0.5)
        seed = CurveSeed(
            state=[0.0, 0.0, 0.0, 0.0],
            parameter_a=0.0,
            frequency=1.0,
        )
        config = Codim2Config(
            continuation_parameter_index=0,
            second_parameter_index=1,
            curve="hopf",
            codim2_detectors=["hopf_hopf"],
            initial_parameter=0.5,
            direction=-1,
            parameter_lower_bound=-0.3,
            parameter_upper_bound=0.6,
        )
        result = Codim2Driver.run(config, equation, seed)
        points = [p for p in result.points if p.kind == "hopf_hopf"]
        assert len(points) == 1
        assert abs(points[0].parameter_b) < LOCATION_TOLERANCE

    def test_fold_curve_finds_cusp(self):
        # The fold curve (b1, b2) = (-2x^3, 3x^2) turns through the cusp at the
        # origin; the quadratic coefficient a = -3x changes sign as x crosses 0.
        equation = _equation(CuspNormalForm, 1, first=-0.25, second=0.75)
        seed = CurveSeed(state=[0.5], parameter_a=0.75)
        config = Codim2Config(
            continuation_parameter_index=1,
            second_parameter_index=0,
            curve="fold",
            codim2_detectors=["cusp"],
            initial_parameter=-0.25,
            direction=1,
            parameter_lower_bound=-0.35,
            parameter_upper_bound=0.35,
        )
        result = Codim2Driver.run(config, equation, seed)
        points = [p for p in result.points if p.kind == "cusp"]
        assert len(points) == 1
        assert abs(points[0].parameter_a) < LOCATION_TOLERANCE
        assert abs(points[0].parameter_b) < LOCATION_TOLERANCE

    def test_hopf_curve_finds_generalized_hopf(self):
        # The Hopf curve sits at b1 = 0; the first Lyapunov coefficient is
        # proportional to b2, so a generalized Hopf occurs at b2 = 0.
        equation = _equation(BautinSystem, 2, first=0.0, second=0.5)
        seed = CurveSeed(state=[0.0, 0.0], parameter_a=0.0, frequency=1.0)
        config = Codim2Config(
            continuation_parameter_index=0,
            second_parameter_index=1,
            curve="hopf",
            codim2_detectors=["generalized_hopf"],
            initial_parameter=0.5,
            direction=-1,
            parameter_lower_bound=-0.6,
            parameter_upper_bound=0.6,
        )
        result = Codim2Driver.run(config, equation, seed)
        points = [p for p in result.points if p.kind == "generalized_hopf"]
        assert len(points) == 1
        assert abs(points[0].parameter_b) < LOCATION_TOLERANCE


class TestNormalFormCoefficients(TestCase):
    """Lock the sign and value conventions of the normal-form coefficients."""

    def _problem(self, function_type, count, first, second):
        equation = _equation(function_type, count, first, second)
        return TwoParameterProblem(equation, 0, 1, 0.0, AutomaticDifferentiation())

    def test_first_lyapunov_sign(self):
        supercritical = self._problem(SupercriticalHopf, 2, 0.0, 0.0)
        subcritical = self._problem(SubcriticalHopf, 2, 0.0, 0.0)
        origin = np.array([0.0, 0.0])
        assert first_lyapunov(supercritical, origin, 0.0, 0.0, 1.0) < 0.0
        assert first_lyapunov(subcritical, origin, 0.0, 0.0, 1.0) > 0.0

    def test_cusp_and_cubic_coefficients(self):
        problem = self._problem(CuspNormalForm, 1, 0.0, 0.0)
        # a = (1/2) f_xx = -3x: zero at the cusp, -0.9 at x = 0.3.
        assert abs(cusp_coefficient(problem, np.array([0.0]), 0.0, 0.0)) < 1.0e-9
        assert abs(cusp_coefficient(problem, np.array([0.3]), 0.0, 0.0) + 0.9) < 1.0e-9
        # b = (1/6) f_xxx = -1 for x' = ... - x^3, everywhere.
        assert (
            abs(cubic_fold_coefficient(problem, np.array([0.0]), 0.0, 0.0) + 1.0)
            < 1.0e-9
        )


class SupercriticalHopf(DeterministicFunction):
    """z' = i z - z|z|^2; a supercritical Hopf (first Lyapunov coefficient < 0)."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        radius = x * x + y * y
        return [mu * x - y - x * radius, x + mu * y - y * radius]


class SubcriticalHopf(DeterministicFunction):
    """z' = i z + z|z|^2; a subcritical Hopf (first Lyapunov coefficient > 0)."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        radius = x * x + y * y
        return [mu * x - y + x * radius, x + mu * y + y * radius]
