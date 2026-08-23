"""Augmented residual whose zero set is the Hopf curve.

Two-parameter continuation of a Hopf point uses the real form of the complex
eigenproblem ``f_u q = i omega q`` with ``q = q_R + i q_I``:

    f(u; p_a, p_b)          = 0
    f_u q_R + omega q_I     = 0
    f_u q_I - omega q_R     = 0
    c . q_R - 1             = 0        (scale)
    c . q_I                 = 0        (phase)

Unknowns are ``(u, q_R, q_I, omega, p_a)`` with continuation coordinate ``p_b``.
Its zero set is a curve of Hopf points, along which ``omega`` is an explicit
coordinate - so a Bogdanov-Takens point is exactly where ``omega`` reaches zero.
"""

import numpy as np

from discrecontinual_equations.continuation.residual import ResidualFunction
from discrecontinual_equations.continuation.two_parameter_problem import (
    CurveDecoded,
    TwoParameterProblem,
)

_NORMALIZATION_TARGET = 1.0


class HopfCurveResidual(ResidualFunction):
    """Real Moore-Spence Hopf system as an injectable residual."""

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
        real_vector = state[n : 2 * n]
        imag_vector = state[2 * n : 3 * n]
        omega = float(state[3 * n])
        parameter_a = float(state[3 * n + 1])
        parameter_b = float(parameter)

        residual = self._problem.evaluate(equilibrium, parameter_a, parameter_b)
        real_image = self._problem.directional(
            equilibrium,
            real_vector,
            parameter_a,
            parameter_b,
        )
        imag_image = self._problem.directional(
            equilibrium,
            imag_vector,
            parameter_a,
            parameter_b,
        )
        eigen_real = real_image + omega * imag_vector
        eigen_imag = imag_image - omega * real_vector
        scale = self._normalization @ real_vector - _NORMALIZATION_TARGET
        phase = self._normalization @ imag_vector
        return np.concatenate([residual, eigen_real, eigen_imag, [scale, phase]])

    def decode(self, state: np.ndarray, parameter: float) -> CurveDecoded:
        """Recover the original-system point and ``omega`` from the vector."""
        n = self._dimension
        return CurveDecoded(
            state=state[:n],
            parameter_a=float(state[3 * n + 1]),
            parameter_b=float(parameter),
            omega=float(state[3 * n]),
        )
