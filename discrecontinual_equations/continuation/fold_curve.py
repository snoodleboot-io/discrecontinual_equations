"""Augmented residual whose zero set is the fold (limit-point) curve.

Two-parameter continuation of a fold uses the Moore-Spence system

    f(u; p_a, p_b)      = 0
    f_u(u; p_a, p_b) q  = 0
    c . q - 1           = 0

with unknowns ``(u, q, p_a)`` and the continuation coordinate ``p_b``. Its zero
set is a one-dimensional curve of folds, continued by the ordinary
pseudo-arclength engine. ``q`` is the null vector of ``f_u`` along the curve.
"""

import numpy as np

from discrecontinual_equations.continuation.residual import ResidualFunction
from discrecontinual_equations.continuation.two_parameter_problem import (
    CurveDecoded,
    TwoParameterProblem,
)

_NORMALIZATION_TARGET = 1.0


class FoldCurveResidual(ResidualFunction):
    """Moore-Spence fold system as an injectable residual."""

    __slots__ = ["_dimension", "_normalization", "_problem"]

    def __init__(
        self,
        problem: TwoParameterProblem,
        dimension: int,
        normalization: np.ndarray,
    ) -> None:
        self._problem = problem
        self._dimension = dimension
        self._normalization = normalization

    def evaluate(self, state: np.ndarray, parameter: float) -> np.ndarray:
        n = self._dimension
        equilibrium = state[:n]
        null_vector = state[n : 2 * n]
        parameter_a = float(state[2 * n])
        parameter_b = float(parameter)
        residual = self._problem.evaluate(equilibrium, parameter_a, parameter_b)
        singular = self._problem.directional(
            equilibrium,
            null_vector,
            parameter_a,
            parameter_b,
        )
        normalization = self._normalization @ null_vector - _NORMALIZATION_TARGET
        return np.concatenate([residual, singular, [normalization]])

    def decode(self, state: np.ndarray, parameter: float) -> CurveDecoded:
        """Recover the original-system point from the augmented vector."""
        n = self._dimension
        return CurveDecoded(
            state=state[:n],
            parameter_a=float(state[2 * n]),
            parameter_b=float(parameter),
            omega=None,
        )
