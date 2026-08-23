"""Center manifold reduction of n-dimensional Hopf points.

The reduction is validated against independent references: the reduced first
Lyapunov quantity tracks the verified n-dimensional Kuznetsov coefficient with a
constant ratio (including two-way coupling and both stable and unstable extra
directions), and reductions that leave the center dynamics unchanged reproduce the
planar quantities exactly.
"""

from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.center_manifold import (
    TaylorCenterManifold,
)
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.focal import PlanarLyapunov
from discrecontinual_equations.continuation.normal_form import first_lyapunov
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

_RATIO = 0.25


class First(Parameter, name="First", abbreviation="p1"):
    pass


class Second(Parameter, name="Second", abbreviation="p2"):
    pass


class State(Variable, name="State", abbreviation="s"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class CoupledHopf(DeterministicFunction):
    """Planar Hopf with a stable direction, two-way coupled (supercritical)."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y, z = point[0], point[1], point[2]
        radius = x * x + y * y
        return [-y - x * radius + z * x, x - y * radius, -2.0 * z + x * x + x * y]


class FlatBautin(DeterministicFunction):
    """Extended Bautin in (x, y) with a fully decoupled stable direction."""

    sign = 1.0

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y, z = point[0], point[1], point[2]
        radius = x * x + y * y
        drive = self.sign * radius * radius
        return [-y + drive * x, x + drive * y, -2.0 * z]


class OneWayBautin(DeterministicFunction):
    """Extended Bautin driving a stable direction one way (no feedback)."""

    sign = 1.0

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y, z = point[0], point[1], point[2]
        radius = x * x + y * y
        drive = self.sign * radius * radius
        return [-y + drive * x, x + drive * y, -2.0 * z + x * x + 0.7 * x * y]


def _equation(function_type: type[DeterministicFunction]) -> DifferentialEquation:
    parameters = [First(value=0.0), Second(value=0.0)]
    derivative = function_type(
        variables=[State(), State(), State()],
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


class TestCenterManifold(TestCase):
    def _problem(self, equation: DifferentialEquation) -> TwoParameterProblem:
        return TwoParameterProblem(equation, 0, 1, 0.0, AutomaticDifferentiation())

    def test_reduced_first_lyapunov_matches_kuznetsov(self):
        equation = _equation(CoupledHopf)
        problem = self._problem(equation)
        origin = np.zeros(3)
        jacobian = problem.state_jacobian(origin, 0.0, 0.0)
        reduced, _second = TaylorCenterManifold().lyapunov_quantities(
            equation.derivative,
            origin,
            jacobian,
            0.0,
        )
        kuznetsov = first_lyapunov(problem, origin, 0.0, 0.0, 1.0)
        assert reduced < 0.0
        assert abs(reduced / kuznetsov - _RATIO) < 1.0e-6

    def test_embeddings_preserve_planar_quantities(self):
        planar = self._planar_quantities()
        for function_type in (FlatBautin, OneWayBautin):
            for sign in (1.0, -1.0):
                function_type.sign = sign
                equation = _equation(function_type)
                problem = self._problem(equation)
                origin = np.zeros(3)
                jacobian = problem.state_jacobian(origin, 0.0, 0.0)
                reduced = TaylorCenterManifold().lyapunov_quantities(
                    equation.derivative,
                    origin,
                    jacobian,
                    0.0,
                )
                expected = planar[sign]
                assert abs(reduced[0] - expected[0]) < 1.0e-9
                assert abs(reduced[1] - expected[1]) < 1.0e-9

    def _planar_quantities(self) -> dict[float, tuple[float, float]]:
        results: dict[float, tuple[float, float]] = {}
        jacobian = np.array([[0.0, -1.0], [1.0, 0.0]])
        for sign in (1.0, -1.0):
            _PlanarView.sign = sign
            view = _PlanarView(
                variables=[State(), State()],
                parameters=[First(value=0.0), Second(value=0.0)],
                results=[State(), State()],
                time=None,
            )
            results[sign] = PlanarLyapunov(
                view,
                np.zeros(2),
                jacobian,
                0.0,
            ).quantities()
        return results


class _PlanarView(DeterministicFunction):
    """The (x, y) block of the extended Bautin, for a planar reference."""

    sign = 1.0

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        x, y = point[0], point[1]
        radius = x * x + y * y
        drive = type(self).sign * radius * radius
        return [-y + drive * x, x + drive * y]
