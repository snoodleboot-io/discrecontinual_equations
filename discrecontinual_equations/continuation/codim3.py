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
    oriented_null_vector,
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

    def reset(self) -> None:
        """Forget anything carried from point to point, before a new curve."""


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


class _OrientedBogdanovTakens(Codim3TestFunction, ABC):
    """A BT-curve test function that is odd in the null vector ``q0``.

    The SVD returns ``q0`` with either sign, and fixing it by its largest
    component flips it wherever that component changes index along the curve,
    which the scan would read as a sign change of the test function. ``q0`` is
    therefore oriented by continuity: by the BT curve's own ``q`` (normalised by
    ``c . q = 1``) when the decoded point carries one, and otherwise towards the
    previous point's ``q0``, as the codim-2 cusp test does.
    """

    __slots__ = ["_previous"]

    def __init__(self) -> None:
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def _orientation(
        self,
        problem: TwoParameterProblem,
        decoded: Codim3Decoded,
    ) -> np.ndarray:
        orientation = (
            self._previous if decoded.null_vector is None else decoded.null_vector
        )
        jacobian = problem.state_jacobian(
            decoded.state.real,
            decoded.parameter_a,
            decoded.parameter_b,
        )
        self._previous = oriented_null_vector(jacobian, orientation)
        return self._previous


class DegenerateBogdanovTakens(_OrientedBogdanovTakens):
    """On the Bogdanov-Takens curve, the quadratic coefficient ``a`` vanishes.

    ``a = (1/2) <p1, B(q0, q0)>`` is odd in ``q0``, which is oriented by
    continuity along the curve (see :class:`_OrientedBogdanovTakens`).

    References:
        [BosschaertKuznetsov2021] section 2.2.2 (``a`` and ``b``), and
        section 2.2.1: the chains are fixed "up to a plus or minus sign".
    """

    @property
    def kind(self) -> str:
        return "degenerate_bogdanov_takens"

    def value(self, problem: TwoParameterProblem, decoded: Codim3Decoded) -> float:
        return bogdanov_takens_coefficient(
            problem,
            decoded.state.real,
            decoded.parameter_a,
            decoded.parameter_b,
            orientation=self._orientation(problem, decoded),
        )


class DegenerateBogdanovTakensB(_OrientedBogdanovTakens):
    """On the BT curve, the quadratic cross-coefficient ``b`` vanishes.

    Distinct from the ``a = 0`` degeneracy: here ``a != 0`` but the ``x y`` term of
    the Bogdanov-Takens normal form passes through zero, the codim-3 point at which
    the two topological types of the Bogdanov-Takens meet. ``b`` is odd in ``q0``,
    which is oriented by continuity (see :class:`_OrientedBogdanovTakens`).

    References:
        [BosschaertKuznetsov2021] section 2.2.2 (``a`` and ``b``), and
        section 2.2.1: the chains are fixed "up to a plus or minus sign".
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
            orientation=self._orientation(problem, decoded),
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
        test_function.reset()
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
