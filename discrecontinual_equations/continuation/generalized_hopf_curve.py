"""Augmented residual whose zero set is the generalized-Hopf (Bautin) curve.

A generalized Hopf is codimension two, so in three active parameters its locus is a
one-dimensional curve. It is defined by the real Moore-Spence Hopf system together
with the vanishing of the first Lyapunov quantity ``l1``:

    f(u; p_a, p_b, p_c)      = 0
    f_u q_R + omega q_I      = 0
    f_u q_I - omega q_R      = 0
    c . q_R - 1              = 0
    c . q_I                  = 0
    l1(u; p_a, p_b, p_c)     = 0

with unknowns ``(u, q_R, q_I, omega, p_a, p_b)`` and continuation coordinate
``p_c``. A degenerate Bautin point is where the second Lyapunov quantity ``l2``
changes sign along this curve. The Lyapunov quantities are the planar focal values,
so this curve applies to planar systems.
"""

import numpy as np

from discrecontinual_equations.continuation.cusp_curve import Codim3Decoded
from discrecontinual_equations.continuation.residual import ResidualFunction
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
from discrecontinual_equations.parameter import Parameter

_NORMALIZATION_TARGET = 1.0


class GeneralizedHopfCurveResidual(ResidualFunction):
    """Hopf Moore-Spence system augmented with the condition ``l1 = 0``."""

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
        real_vector = state[n : 2 * n]
        imag_vector = state[2 * n : 3 * n]
        omega = float(state[3 * n])
        parameter_a = float(state[3 * n + 1])
        parameter_b = float(state[3 * n + 2])
        self._third.value = float(parameter)

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
        first, _second = self._problem.lyapunov_quantities(
            equilibrium,
            parameter_a,
            parameter_b,
        )
        return np.concatenate(
            [residual, eigen_real, eigen_imag, [scale, phase, first]],
        )

    def decode(self, state: np.ndarray, parameter: float) -> Codim3Decoded:
        """Recover the original-system point from the augmented vector."""
        n = self._dimension
        self._third.value = float(parameter)
        return Codim3Decoded(
            state=state[:n],
            parameter_a=float(state[3 * n + 1]),
            parameter_b=float(state[3 * n + 2]),
            parameter_c=float(parameter),
        )
