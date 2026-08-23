"""Codimension-three detection along a continued cusp curve.

Continuing the cusp curve in three parameters, a swallowtail is the point where
the cubic fold coefficient ``b`` changes sign. The analyzer scans the augmented
branch, sets the swept third parameter for each decoded point, evaluates the test
function, and brackets sign changes by linear interpolation - the same scheme the
codim-2 analyzer uses, extended to a third parameter.
"""

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.cusp_curve import Codim3Decoded
from discrecontinual_equations.continuation.normal_form import (
    bogdanov_takens_coefficient,
    bogdanov_takens_second_coefficient,
    cubic_fold_coefficient,
)
from discrecontinual_equations.continuation.residual import ResidualFunction
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
from discrecontinual_equations.parameter import Parameter


class Codim3Point(BaseModel):
    """A located codimension-three bifurcation."""

    model_config = ConfigDict(frozen=True)

    kind: str = Field(description="Codim-3 bifurcation kind")
    parameter_a: float = Field(description="First active parameter at the point")
    parameter_b: float = Field(description="Second active parameter at the point")
    parameter_c: float = Field(description="Third active parameter at the point")
    state: list[float] = Field(description="Equilibrium state at the point")
    coefficient: float = Field(description="Test-function value at the crossing")


class Codim3TestFunction(ABC):
    """A scalar whose sign change along the cusp curve marks a codim-3 point."""

    @property
    @abstractmethod
    def kind(self) -> str:
        """Registry-facing name of the detected bifurcation."""
        raise NotImplementedError

    @abstractmethod
    def value(self, problem: TwoParameterProblem, decoded: Codim3Decoded) -> float:
        """Evaluate the test function at a decoded cusp-curve point."""
        raise NotImplementedError


class Swallowtail(Codim3TestFunction):
    """On the cusp curve, the cubic coefficient ``b`` vanishes at a swallowtail."""

    @property
    def kind(self) -> str:
        return "swallowtail"

    def value(self, problem: TwoParameterProblem, decoded: Codim3Decoded) -> float:
        return cubic_fold_coefficient(
            problem,
            decoded.state.real,
            decoded.parameter_a,
            decoded.parameter_b,
        )


class DegenerateBautin(Codim3TestFunction):
    """On the generalized-Hopf curve, the second Lyapunov quantity vanishes."""

    @property
    def kind(self) -> str:
        return "degenerate_bautin"

    def value(self, problem: TwoParameterProblem, decoded: Codim3Decoded) -> float:
        _first, second = problem.lyapunov_quantities(
            decoded.state.real,
            decoded.parameter_a,
            decoded.parameter_b,
        )
        return second


class DegenerateBogdanovTakens(Codim3TestFunction):
    """On the Bogdanov-Takens curve, the quadratic coefficient ``a`` vanishes."""

    @property
    def kind(self) -> str:
        return "degenerate_bogdanov_takens"

    def value(self, problem: TwoParameterProblem, decoded: Codim3Decoded) -> float:
        return bogdanov_takens_coefficient(
            problem,
            decoded.state.real,
            decoded.parameter_a,
            decoded.parameter_b,
        )


class DegenerateBogdanovTakensB(Codim3TestFunction):
    """On the BT curve, the quadratic cross-coefficient ``b`` vanishes.

    Distinct from the ``a = 0`` degeneracy: here ``a != 0`` but the ``x y`` term of
    the Bogdanov-Takens normal form passes through zero, the codim-3 point at which
    the two topological types of the Bogdanov-Takens meet.
    """

    @property
    def kind(self) -> str:
        return "degenerate_bogdanov_takens_b"

    def value(self, problem: TwoParameterProblem, decoded: Codim3Decoded) -> float:
        return bogdanov_takens_second_coefficient(
            problem,
            decoded.state.real,
            decoded.parameter_a,
            decoded.parameter_b,
        )


class Codim3Analyzer:
    """Scan a continued cusp curve for sign changes of the test functions."""

    __slots__ = ["_test_functions", "_third"]

    def __init__(
        self,
        test_functions: list[Codim3TestFunction],
        third: Parameter,
    ) -> None:
        self._test_functions = test_functions
        self._third = third

    def analyze(
        self,
        branch: Branch,
        residual: ResidualFunction,
        problem: TwoParameterProblem,
    ) -> list[Codim3Point]:
        """Return the codim-3 points bracketed along the branch."""
        decoded = [
            residual.decode(np.asarray(point.state, dtype=float), point.parameter)
            for point in branch.points
        ]
        found: list[Codim3Point] = []
        for test_function in self._test_functions:
            found.extend(self._scan(test_function, decoded, problem))
        return found

    def _scan(
        self,
        test_function: Codim3TestFunction,
        decoded: list[Codim3Decoded],
        problem: TwoParameterProblem,
    ) -> list[Codim3Point]:
        found: list[Codim3Point] = []
        previous_value = None
        previous_point = None
        for current in decoded:
            self._third.value = current.parameter_c
            value = test_function.value(problem, current)
            if previous_value is not None and previous_value * value < 0:
                fraction = previous_value / (previous_value - value)
                found.append(
                    self._interpolate(
                        test_function.kind,
                        previous_point,
                        current,
                        fraction,
                    ),
                )
            previous_value = value
            previous_point = current
        return found

    def _interpolate(
        self,
        kind: str,
        lower: Codim3Decoded,
        upper: Codim3Decoded,
        fraction: float,
    ) -> Codim3Point:
        state = lower.state + fraction * (upper.state - lower.state)
        parameter_a = lower.parameter_a + fraction * (
            upper.parameter_a - lower.parameter_a
        )
        parameter_b = lower.parameter_b + fraction * (
            upper.parameter_b - lower.parameter_b
        )
        parameter_c = lower.parameter_c + fraction * (
            upper.parameter_c - lower.parameter_c
        )
        return Codim3Point(
            kind=kind,
            parameter_a=parameter_a,
            parameter_b=parameter_b,
            parameter_c=parameter_c,
            state=[float(component.real) for component in state],
            coefficient=0.0,
        )
