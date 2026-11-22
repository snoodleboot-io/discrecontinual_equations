"""Three-parameter continuation and codim-3 detection.

The swallowtail normal form ``x' = b1 + b2 x + b3 x^2 - x^4`` has its cusp curve
parametrised by ``x = t`` as ``(b1, b2, b3) = (3 t^4, -8 t^3, 6 t^2)``, with cubic
coefficient ``b = -4 t``. Continuing the cusp curve through ``t = 0`` gives a
swallowtail at the origin, checked against that exact location. Exactness of the
high derivatives comes from the automatic-differentiation provider.

The coupled swallowtail adds a stable mode ``s`` slaved to ``s = x^2`` that feeds
back through ``g x s``. Its equilibria are those of ``Phi(x) = b1 + b2 x + b3 x^2 +
g x^3 - x^4`` (with ``s = x^2``), so its swallowtail is where the third derivative
of ``Phi`` vanishes: at ``x = g / 4``, not at the origin. Only the center
manifold's ``3 B(q, h2)`` share of the cubic coefficient moves it there (DEQ-83).
"""

from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.codim3_builder import (
    Codim3Driver,
    Codim3Seed,
)
from discrecontinual_equations.continuation.codim3_config import Codim3Config
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

LOCATION_TOLERANCE = 2.0e-2


class First(Parameter, name="First parameter", abbreviation="b1"):
    pass


class Second(Parameter, name="Second parameter", abbreviation="b2"):
    pass


class Third(Parameter, name="Third parameter", abbreviation="b3"):
    pass


class State(Variable, name="State", abbreviation="s"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class SwallowtailNormalForm(DeterministicFunction):
    """x' = b1 + b2 x + b3 x^2 - x^4; a swallowtail at the origin."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        b3 = self.parameters[2].value
        x = point[0]
        return [b1 + b2 * x + b3 * x * x - x**4]


class ExtendedBautin(DeterministicFunction):
    """z' = (b1 + i) z + b2 z|z|^2 + b3 z|z|^4 - z|z|^6.

    Hopf at b1 = 0; the first Lyapunov quantity ~ b2 (so the generalized-Hopf curve
    is b1 = b2 = 0) and the second ~ b3, giving a degenerate Bautin at b3 = 0.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        b3 = self.parameters[2].value
        x, y = point[0], point[1]
        radius = x * x + y * y
        coefficient = b1 + b2 * radius + b3 * radius * radius - radius**3
        return [coefficient * x - y, x + coefficient * y]


def _equation(
    function_type: type[DeterministicFunction],
    count: int,
    first: float,
    second: float,
    third: float,
) -> DifferentialEquation:
    parameters = [First(value=first), Second(value=second), Third(value=third)]
    derivative = function_type(
        variables=[State() for _ in range(count)],
        parameters=parameters,
        results=[State() for _ in range(count)],
        time=None,
    )
    return DifferentialEquation(
        variables=[State() for _ in range(count)],
        time=Time(),
        parameters=parameters,
        derivative=derivative,
    )


class CoupledSwallowtail(DeterministicFunction):
    """``x' = b1 + b2 x + b3 x^2 - x^4 + g x s``, ``s' = x^2 - s``, in ``X = T z``."""

    feedback = 1.0
    transform = np.array([[1.3, -0.4], [0.7, 0.9]])

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        transform = type(self).transform
        inverse = np.linalg.inv(transform)
        x, s = (
            sum(float(inverse[i, k]) * point[k] for k in range(2)) for i in range(2)
        )
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        b3 = self.parameters[2].value
        field = [
            b1 + b2 * x + b3 * x * x - x**4 + type(self).feedback * x * s,
            x * x - s,
        ]
        return [
            sum(float(transform[i, k]) * field[k] for k in range(2)) for i in range(2)
        ]


def _coupled_cusp_point(t: float, g: float) -> tuple[float, float, float]:
    """``(b1, b2, b3)`` where ``Phi``, ``Phi'`` and ``Phi''`` vanish at ``x = t``."""
    return 3.0 * t**4 - g * t**3, -8.0 * t**3 + 3.0 * g * t**2, 6.0 * t**2 - 3.0 * g * t


class CoupledExtendedBautin(DeterministicFunction):
    """The extended Bautin driving a coupled stable direction z.

    The planar (x, y) block is the extended Bautin; z is hyperbolic and driven by
    the center, so the reduced dynamics on the center manifold is the planar form
    and the degenerate Bautin still sits at b1 = b2 = b3 = 0. Detecting it in three
    dimensions exercises the center manifold reduction inside the curve residual.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        b1 = self.parameters[0].value
        b2 = self.parameters[1].value
        b3 = self.parameters[2].value
        x, y, z = point[0], point[1], point[2]
        radius = x * x + y * y
        coefficient = b1 + b2 * radius + b3 * radius * radius - radius**3
        return [
            coefficient * x - y,
            x + coefficient * y,
            -2.0 * z + x * x + 0.5 * x * y,
        ]


class TestCenterManifoldCodim3(TestCase):
    def test_degenerate_bautin_in_three_dimensions(self):
        equation = _equation(CoupledExtendedBautin, 3, 0.0, 0.0, 0.6)
        seed = Codim3Seed(
            state=[0.0, 0.0, 0.0],
            parameter_a=0.0,
            parameter_b=0.0,
            frequency=1.0,
        )
        config = Codim3Config(
            continuation_parameter_index=0,
            second_parameter_index=1,
            third_parameter_index=2,
            codim3_curve="generalized_hopf",
            codim3_detectors=["degenerate_bautin"],
            initial_parameter=0.6,
            direction=-1,
            parameter_lower_bound=-0.7,
            parameter_upper_bound=0.7,
            initial_step=0.03,
            maximum_step=0.06,
            maximum_points=300,
        )
        result = Codim3Driver.run(config, equation, seed)
        points = [p for p in result.points if p.kind == "degenerate_bautin"]
        assert len(points) == 1
        assert abs(points[0].parameter_a) < LOCATION_TOLERANCE
        assert abs(points[0].parameter_b) < LOCATION_TOLERANCE
        assert abs(points[0].parameter_c) < LOCATION_TOLERANCE


class TestCodim3(TestCase):
    def test_cusp_curve_finds_swallowtail(self):
        # Seed on the cusp curve at t = 0.5: x = 0.5, (b1, b2, b3) = (0.1875, -1, 1.5).
        equation = _equation(SwallowtailNormalForm, 1, 0.1875, -1.0, 1.5)
        seed = Codim3Seed(state=[0.5], parameter_a=0.1875, parameter_b=1.5)
        config = Codim3Config(
            continuation_parameter_index=0,
            second_parameter_index=2,
            third_parameter_index=1,
            curve="fold",
            codim3_curve="cusp",
            codim3_detectors=["swallowtail"],
            initial_parameter=-1.0,
            direction=1,
            parameter_lower_bound=-1.2,
            parameter_upper_bound=1.2,
            initial_step=0.01,
            maximum_step=0.02,
            maximum_points=500,
        )
        result = Codim3Driver.run(config, equation, seed)
        points = [p for p in result.points if p.kind == "swallowtail"]
        assert len(points) == 1
        assert abs(points[0].parameter_a) < LOCATION_TOLERANCE
        assert abs(points[0].parameter_b) < LOCATION_TOLERANCE
        assert abs(points[0].parameter_c) < LOCATION_TOLERANCE
        assert abs(points[0].state[0]) < LOCATION_TOLERANCE

    def test_coupled_swallowtail_sits_where_the_manifold_puts_it(self):
        g = CoupledSwallowtail.feedback
        start = 0.75
        b1, b2, b3 = _coupled_cusp_point(start, g)
        state = CoupledSwallowtail.transform @ np.array([start, start * start])
        equation = _equation(CoupledSwallowtail, 2, b1, b2, b3)
        seed = Codim3Seed(state=state.tolist(), parameter_a=b1, parameter_b=b3)
        config = Codim3Config(
            continuation_parameter_index=0,
            second_parameter_index=2,
            third_parameter_index=1,
            curve="fold",
            codim3_curve="cusp",
            codim3_detectors=["swallowtail"],
            initial_parameter=b2,
            direction=1,
            parameter_lower_bound=-3.0,
            parameter_upper_bound=3.0,
            initial_step=0.01,
            maximum_step=0.02,
            maximum_points=400,
        )
        result = Codim3Driver.run(config, equation, seed)
        points = [p for p in result.points if p.kind == "swallowtail"]
        expected = _coupled_cusp_point(g / 4.0, g)
        assert len(points) == 1
        x0 = (np.linalg.inv(CoupledSwallowtail.transform) @ points[0].state)[0]
        assert abs(points[0].parameter_a - expected[0]) < LOCATION_TOLERANCE
        assert abs(points[0].parameter_c - expected[1]) < LOCATION_TOLERANCE
        assert abs(points[0].parameter_b - expected[2]) < LOCATION_TOLERANCE
        assert abs(x0 - g / 4.0) < LOCATION_TOLERANCE

    def test_generalized_hopf_curve_finds_degenerate_bautin(self):
        # The generalized-Hopf curve is b1 = b2 = 0; the second Lyapunov quantity is
        # proportional to b3, giving a degenerate Bautin at b3 = 0.
        equation = _equation(ExtendedBautin, 2, 0.0, 0.0, 0.6)
        seed = Codim3Seed(
            state=[0.0, 0.0],
            parameter_a=0.0,
            parameter_b=0.0,
            frequency=1.0,
        )
        config = Codim3Config(
            continuation_parameter_index=0,
            second_parameter_index=1,
            third_parameter_index=2,
            codim3_curve="generalized_hopf",
            codim3_detectors=["degenerate_bautin"],
            initial_parameter=0.6,
            direction=-1,
            parameter_lower_bound=-0.7,
            parameter_upper_bound=0.7,
            initial_step=0.02,
            maximum_step=0.05,
            maximum_points=400,
        )
        result = Codim3Driver.run(config, equation, seed)
        points = [p for p in result.points if p.kind == "degenerate_bautin"]
        assert len(points) == 1
        assert abs(points[0].parameter_a) < LOCATION_TOLERANCE
        assert abs(points[0].parameter_b) < LOCATION_TOLERANCE
        assert abs(points[0].parameter_c) < LOCATION_TOLERANCE
