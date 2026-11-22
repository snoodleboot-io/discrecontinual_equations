"""Two-parameter continuation and codim-2 detection.

Each system has an analytically known codim-2 point, so the detected location is
checked against the exact value. The augmented curve Jacobian and the normal-form
coefficients use derivatives from the automatic-differentiation provider, so they
are exact; the residual location tolerance reflects the continuation step size.
"""

import warnings
from unittest import TestCase

import numpy as np
from scipy.optimize import fsolve

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

    def test_first_lyapunov_van_der_pol_by_hand(self):
        # x' = y - x^3/3 + x, y' = eps (a - x) has a Hopf at a = 1, (1, -2/3), with
        # omega = sqrt(eps). In u = x - 1, Y = -(y + 2/3)/omega it is
        # u' = -omega Y - u^2 - u^3/3, Y' = omega u, so Guckenheimer-Holmes gives
        # a = -1/8; with the unit eigenvector (1, i sqrt(eps)) / sqrt(1 + eps) that
        # is l1 = -1 / (2 omega (1 + eps)). The Jacobian is normal only at eps = 1.
        state = np.array([1.0, -2.0 / 3.0])
        for epsilon in (0.05, 0.2, 1.0, 4.0):
            problem = self._problem(VanDerPolFastSlow, 2, 1.0, epsilon)
            omega = np.sqrt(epsilon)
            expected = -1.0 / (2.0 * omega * (1.0 + epsilon))
            l1 = first_lyapunov(problem, state, 1.0, epsilon, omega)
            assert abs(l1 - expected) < 1.0e-10 * abs(expected)

    def test_first_lyapunov_under_non_normal_similarity(self):
        # F(x) = S^-1 N(S x), with N the Hopf normal form y' = J y + b |y|^2 y. Its
        # l1 is 2 b / omega for the unit eigenvector (1, -i, 0...) / sqrt(2) of N,
        # and l1 scales with |q|^2, so with |q| = 1 in x it is
        # 2 b / (omega |S^-1 q_N|^2). The phase of <p, q> before normalisation
        # varies with S; the sign and size of l1 must not.
        generator = np.random.default_rng(96)
        checked = 0
        for dimension in (2, 2, 2, 2, 3, 3, 3, 3, 3, 3):
            matrix = generator.normal(size=(dimension, dimension))
            if abs(np.linalg.det(matrix)) < 0.2:
                continue
            SimilarHopf.matrix = matrix
            SimilarHopf.inverse = np.linalg.inv(matrix)
            problem = self._problem(SimilarHopf, dimension, 0.0, 0.0)
            normal_vector = np.zeros(dimension, dtype=complex)
            normal_vector[:2] = np.array([1.0, -1.0j]) / np.sqrt(2.0)
            scale = np.linalg.norm(np.linalg.solve(matrix, normal_vector)) ** 2
            expected = 2.0 * SimilarHopf.cubic / (SimilarHopf.frequency * scale)
            l1 = first_lyapunov(
                problem,
                np.zeros(dimension),
                0.0,
                0.0,
                SimilarHopf.frequency,
            )
            assert abs(l1 - expected) < 1.0e-9 * abs(expected)
            checked += 1
        assert checked >= 8

    def test_first_lyapunov_bykov_matches_matcont(self):
        # Kuznetsov and Neirynck, MatCont tutorial IV (2021): the Hopf of the
        # Bykov-Yablonskii-Kim model at K = 0.4 has first Lyapunov coefficient
        # 1.070259e01 as MatCont prints it, Re G21 / 2; the library's l1 is
        # Re G21 / (2 omega), so omega * l1 is compared. MatCont's printed Hopf
        # (Q2 = 1.051558) sits 5e-7 from the exact one and dl1/dQ2 is about 3.3e3
        # in that convention, hence the 2e-3 tolerance.
        problem = self._problem(Bykov, 3, 1.05, 0.4)

        def residual(unknowns):
            state, q2 = unknowns[:3], unknowns[3]
            jacobian = problem.state_jacobian(state, q2, 0.4)
            minors = sum(
                np.linalg.det(np.delete(np.delete(jacobian, i, 0), i, 1))
                for i in range(3)
            )
            hopf = np.trace(jacobian) * minors - np.linalg.det(jacobian)
            return [*problem.evaluate(state, q2, 0.4), hopf]

        with warnings.catch_warnings():
            # fsolve reports that it cannot beat xtol=1e-14; the residual decides.
            warnings.simplefilter("ignore", RuntimeWarning)
            point = fsolve(
                residual,
                [0.016357, 0.523973, 0.328336, 1.051558],
                xtol=1.0e-14,
            )
        assert np.linalg.norm(residual(point)) < 1.0e-13
        jacobian = problem.state_jacobian(point[:3], point[3], 0.4)
        omega = float(np.max(np.linalg.eigvals(jacobian).imag))
        l1 = first_lyapunov(problem, point[:3], point[3], 0.4, omega)
        assert abs(omega * l1 - 1.070259e01) < 2.0e-3

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


class VanDerPolFastSlow(DeterministicFunction):
    """x' = y - x^3/3 + x, y' = eps (a - x); a supercritical Hopf at a = 1."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        a = self.parameters[0].value
        epsilon = self.parameters[1].value
        x, y = point[0], point[1]
        return [y - x * x * x * (1.0 / 3.0) + x, epsilon * (a - x)]


class SimilarHopf(DeterministicFunction):
    """x' = S^-1 N(S x): a supercritical Hopf normal form in skewed coordinates.

    ``N`` rotates the first two coordinates at ``frequency`` with the cubic
    ``cubic |y|^2 y`` on them, and contracts any further coordinate linearly.
    """

    frequency = 1.3
    cubic = -1.0
    matrix = np.eye(2)
    inverse = np.eye(2)

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        dimension = len(point)
        image = [
            sum(float(self.matrix[i, j]) * point[j] for j in range(dimension))
            for i in range(dimension)
        ]
        radius = image[0] * image[0] + image[1] * image[1]
        flow = [
            -self.frequency * image[1] + self.cubic * radius * image[0],
            self.frequency * image[0] + self.cubic * radius * image[1],
        ]
        flow += [-2.0 * image[k] for k in range(2, dimension)]
        return [
            sum(float(self.inverse[i, j]) * flow[j] for j in range(dimension))
            for i in range(dimension)
        ]


class Bykov(DeterministicFunction):
    """Bykov-Yablonskii-Kim CO-oxidation model (MatCont tutorial IV); (Q2, K)."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        q2 = self.parameters[0].value
        k = self.parameters[1].value
        q1, q3, q4, q5, q6 = 2.5, 10.0, 0.0675, 1.0, 0.1
        x, y, s = point[0], point[1], point[2]
        z = 1.0 - x - y - s
        return [
            2.0 * q1 * z * z - 2.0 * q5 * x * x - q3 * x * y,
            q2 * z - q6 * y - q3 * x * y,
            q4 * z - k * q4 * s,
        ]
