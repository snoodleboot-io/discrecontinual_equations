"""Codimension-2 detection along a continued bifurcation curve.

Given a continued codim-1 curve (fold or Hopf), a codim-2 point is where a second
degeneracy appears. Each :class:`Codim2TestFunction` is a scalar over the original
system spectrum whose sign change brackets one codim-2 type:

- Bogdanov-Takens: on the Hopf curve, ``omega`` reaches zero.
- Zero-Hopf (fold-Hopf): ``det(f_u)`` changes sign (a real eigenvalue crosses the
  axis while the imaginary pair persists).
- Hopf-Hopf: a second conjugate pair reaches the imaginary axis.
- Bogdanov-Takens on the fold curve: the non-null eigenvalue nearest the axis
  reaches zero (the structural zero becomes a double zero).

New codim-2 types are added by registering another test function; the analyzer is
unchanged.
"""

import math
from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from discrecontinual_equations.continuation.branch import Branch
from discrecontinual_equations.continuation.normal_form import (
    cusp_coefficient,
    first_lyapunov,
)
from discrecontinual_equations.continuation.residual import ResidualFunction
from discrecontinual_equations.continuation.two_parameter_problem import (
    CurveDecoded,
    TwoParameterProblem,
)

_IMAGINARY_TOLERANCE = 1e-6
_OMEGA_FLOOR = 1.0e-4
_CONJUGATE_PAIR = 2


class Codim2Point(BaseModel):
    """A detected codimension-2 bifurcation on a continued curve."""

    kind: str = Field(description="Codim-2 classification, e.g. 'bogdanov_takens'")
    parameter_a: float = Field(description="First active parameter at the point")
    parameter_b: float = Field(description="Second active parameter at the point")
    state: list[float] = Field(description="Equilibrium state at the point")
    omega: float | None = Field(
        default=None,
        description="Hopf frequency at the point, if meaningful",
    )
    eigenvalues: list[tuple[float, float]] = Field(
        default_factory=list,
        description="Original-system eigenvalues as (real, imaginary) pairs",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Codim2TestFunction(ABC):
    """A scalar over the original spectrum whose zero marks a codim-2 point."""

    @property
    @abstractmethod
    def kind(self) -> str:
        """Codim-2 label recorded on a detected point."""
        raise NotImplementedError

    @abstractmethod
    def value(self, problem: TwoParameterProblem, decoded: CurveDecoded) -> float:
        """Scalar whose sign change brackets the bifurcation."""
        raise NotImplementedError


class BogdanovTakens(Codim2TestFunction):
    """On the Hopf curve, the frequency passes through zero."""

    @property
    def kind(self) -> str:
        return "bogdanov_takens"

    def value(self, problem: TwoParameterProblem, decoded: CurveDecoded) -> float:
        _ = problem
        return 0.0 if decoded.omega is None else decoded.omega


class ZeroHopf(Codim2TestFunction):
    """A real eigenvalue crosses the axis: det of the Jacobian changes sign."""

    @property
    def kind(self) -> str:
        return "zero_hopf"

    def value(self, problem: TwoParameterProblem, decoded: CurveDecoded) -> float:
        jacobian = problem.state_jacobian(
            decoded.state,
            decoded.parameter_a,
            decoded.parameter_b,
        )
        return float(np.linalg.det(jacobian))


class HopfHopf(Codim2TestFunction):
    """A second conjugate pair reaches the imaginary axis."""

    @property
    def kind(self) -> str:
        return "hopf_hopf"

    def value(self, problem: TwoParameterProblem, decoded: CurveDecoded) -> float:
        eigenvalues = problem.eigenvalues(
            decoded.state,
            decoded.parameter_a,
            decoded.parameter_b,
        )
        target = decoded.omega if decoded.omega is not None else 0.0
        remaining = _drop_critical_pair(eigenvalues, target)
        pairs = [value for value in remaining if abs(value.imag) > _IMAGINARY_TOLERANCE]
        if not pairs:
            return 1.0
        nearest = min(pairs, key=lambda value: abs(value.real))
        return float(nearest.real)


class FoldBogdanovTakens(Codim2TestFunction):
    """On the fold curve, the non-null eigenvalue nearest the axis reaches zero."""

    @property
    def kind(self) -> str:
        return "bogdanov_takens"

    def value(self, problem: TwoParameterProblem, decoded: CurveDecoded) -> float:
        eigenvalues = problem.eigenvalues(
            decoded.state,
            decoded.parameter_a,
            decoded.parameter_b,
        )
        ordered = sorted(eigenvalues, key=abs)
        # ordered[0] is the structural null eigenvalue of the fold; the next one
        # reaching zero is the Bogdanov-Takens condition.
        return float(ordered[1].real)


class Cusp(Codim2TestFunction):
    """On the fold curve, the quadratic coefficient ``a`` vanishes at a cusp."""

    @property
    def kind(self) -> str:
        return "cusp"

    def value(self, problem: TwoParameterProblem, decoded: CurveDecoded) -> float:
        return cusp_coefficient(
            problem,
            decoded.state.real,
            decoded.parameter_a,
            decoded.parameter_b,
        )


class GeneralizedHopf(Codim2TestFunction):
    """On the Hopf curve, the first Lyapunov coefficient vanishes (Bautin)."""

    @property
    def kind(self) -> str:
        return "generalized_hopf"

    def value(self, problem: TwoParameterProblem, decoded: CurveDecoded) -> float:
        if decoded.omega is None or abs(decoded.omega) < _OMEGA_FLOOR:
            return math.nan
        try:
            return first_lyapunov(
                problem,
                decoded.state.real,
                decoded.parameter_a,
                decoded.parameter_b,
                decoded.omega,
            )
        except np.linalg.LinAlgError:
            return math.nan


def _drop_critical_pair(eigenvalues: np.ndarray, omega: float) -> list[complex]:
    values = list(eigenvalues)
    for target in (1j * omega, -1j * omega):
        if not values:
            break
        index = min(
            range(len(values)),
            key=lambda position: abs(values[position] - target),
        )
        values.pop(index)
    return values


class Codim2Analyzer:
    """Scan a continued curve for sign changes of injected test functions."""

    __slots__ = ["_test_functions"]

    def __init__(self, test_functions: list[Codim2TestFunction]) -> None:
        self._test_functions = test_functions

    def analyze(
        self,
        branch: Branch,
        residual: ResidualFunction,
        problem: TwoParameterProblem,
    ) -> list[Codim2Point]:
        """Return the codim-2 points bracketed along the branch."""
        decoded = [
            residual.decode(np.asarray(point.state, dtype=float), point.parameter)
            for point in branch.points
        ]
        found: list[Codim2Point] = []
        for test_function in self._test_functions:
            found.extend(self._scan(test_function, decoded, problem))
        return found

    def _scan(
        self,
        test_function: Codim2TestFunction,
        decoded: list[CurveDecoded],
        problem: TwoParameterProblem,
    ) -> list[Codim2Point]:
        found: list[Codim2Point] = []
        previous_value = None
        previous_point = None
        for current in decoded:
            value = test_function.value(problem, current)
            if not math.isfinite(value):
                previous_value = None
                previous_point = None
                continue
            if previous_value is not None and previous_value * value < 0:
                fraction = previous_value / (previous_value - value)
                found.append(
                    self._interpolate(
                        test_function.kind,
                        previous_point,
                        current,
                        fraction,
                        problem,
                    ),
                )
            previous_value = value
            previous_point = current
        return found

    def _interpolate(
        self,
        kind: str,
        lower: CurveDecoded,
        upper: CurveDecoded,
        fraction: float,
        problem: TwoParameterProblem,
    ) -> Codim2Point:
        state = lower.state + fraction * (upper.state - lower.state)
        parameter_a = lower.parameter_a + fraction * (
            upper.parameter_a - lower.parameter_a
        )
        parameter_b = lower.parameter_b + fraction * (
            upper.parameter_b - lower.parameter_b
        )
        omega = None
        if lower.omega is not None and upper.omega is not None:
            omega = lower.omega + fraction * (upper.omega - lower.omega)
        eigenvalues = problem.eigenvalues(state, parameter_a, parameter_b)
        return Codim2Point(
            kind=kind,
            parameter_a=parameter_a,
            parameter_b=parameter_b,
            state=[float(component) for component in state],
            omega=omega,
            eigenvalues=[
                (float(value.real), float(value.imag)) for value in eigenvalues
            ],
        )
