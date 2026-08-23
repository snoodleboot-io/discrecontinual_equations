"""Augmented residual whose zero set is the cusp curve in three parameters.

A cusp is codimension two, so in three active parameters the set of cusps is a
one-dimensional curve. It is defined by the fold (Moore-Spence) system together
with the vanishing of the quadratic normal-form coefficient ``a``:

    f(u; p_a, p_b, p_c)      = 0
    f_u(u; p_a, p_b, p_c) q  = 0
    c . q - 1                = 0
    a(u; p_a, p_b, p_c)      = 0

with unknowns ``(u, q, p_a, p_b)`` and continuation coordinate ``p_c``. Continued
by the ordinary pseudo-arclength engine, its zero set is the cusp curve; a
swallowtail is where the next coefficient ``b`` vanishes along it.
"""

import numpy as np

from discrecontinual_equations.continuation.normal_form import cusp_coefficient
from discrecontinual_equations.continuation.residual import ResidualFunction
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
from discrecontinual_equations.parameter import Parameter

_NORMALIZATION_TARGET = 1.0


class Codim3Decoded:
    """Original-system quantities decoded from an augmented cusp-curve point."""

    __slots__ = ["parameter_a", "parameter_b", "parameter_c", "state"]

    def __init__(
        self,
        state: np.ndarray,
        parameter_a: float,
        parameter_b: float,
        parameter_c: float,
    ) -> None:
        self.state = state
        self.parameter_a = parameter_a
        self.parameter_b = parameter_b
        self.parameter_c = parameter_c


class CuspCurveResidual(ResidualFunction):
    """Fold system augmented with the cusp condition ``a = 0``."""

    __slots__ = ["_dimension", "_normalization", "_problem", "_third"]

    def __init__(
        self,
        problem: TwoParameterProblem,
        dimension: int,
        normalization: np.ndarray,
        third: Parameter,
    ) -> None:
        self._problem = problem
        self._dimension = dimension
        self._normalization = normalization
        self._third = third

    def evaluate(self, state: np.ndarray, parameter: float) -> np.ndarray:
        n = self._dimension
        equilibrium = state[:n]
        null_vector = state[n : 2 * n]
        parameter_a = float(state[2 * n])
        parameter_b = float(state[2 * n + 1])
        self._third.value = float(parameter)
        residual = self._problem.evaluate(equilibrium, parameter_a, parameter_b)
        singular = self._problem.directional(
            equilibrium,
            null_vector,
            parameter_a,
            parameter_b,
        )
        normalization = self._normalization @ null_vector - _NORMALIZATION_TARGET
        quadratic = cusp_coefficient(
            self._problem,
            equilibrium,
            parameter_a,
            parameter_b,
        )
        return np.concatenate([residual, singular, [normalization], [quadratic]])

    def decode(self, state: np.ndarray, parameter: float) -> Codim3Decoded:
        """Recover the original-system point from the augmented vector."""
        n = self._dimension
        self._third.value = float(parameter)
        return Codim3Decoded(
            state=state[:n],
            parameter_a=float(state[2 * n]),
            parameter_b=float(state[2 * n + 1]),
            parameter_c=float(parameter),
        )
