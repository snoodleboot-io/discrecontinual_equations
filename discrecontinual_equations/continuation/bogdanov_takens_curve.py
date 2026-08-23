"""Augmented residual whose zero set is the Bogdanov-Takens curve.

A Bogdanov-Takens point has a double-zero eigenvalue with a single eigenvector - a
Jordan block - so it is codimension two, and in three active parameters the set of
such points is a one-dimensional curve. Rather than track eigenvalues (which are
not smooth where they collide, exactly where the BT lives) the curve is defined by
the Jordan chain directly:

    f(u; p_a, p_b, p_c)          = 0
    f_u(u; p_a, p_b, p_c) q      = 0        (q spans the kernel)
    f_u(u; p_a, p_b, p_c) q1 - q = 0        (q1 completes the Jordan chain)
    c . q  - 1                   = 0        (fix the scale of q)
    c . q1                       = 0        (fix the freedom q1 -> q1 + t q)

with unknowns ``(u, q, q1, p_a, p_b)`` and continuation coordinate ``p_c``. The
chain ``A q = 0``, ``A q1 = q`` exists precisely when the zero eigenvalue is
non-semisimple, which is the defining property of the Bogdanov-Takens point; every
condition is smooth in the unknowns, so the ordinary pseudo-arclength engine
continues it without special handling.
"""

import numpy as np

from discrecontinual_equations.continuation.cusp_curve import Codim3Decoded
from discrecontinual_equations.continuation.residual import ResidualFunction
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
from discrecontinual_equations.parameter import Parameter

_NORMALIZATION_TARGET = 1.0


class BogdanovTakensCurveResidual(ResidualFunction):
    """Equilibrium and Jordan-chain conditions for the double-zero eigenvalue."""

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
        kernel = state[n : 2 * n]
        chain = state[2 * n : 3 * n]
        parameter_a = float(state[3 * n])
        parameter_b = float(state[3 * n + 1])
        self._third.value = float(parameter)
        residual = self._problem.evaluate(equilibrium, parameter_a, parameter_b)
        in_kernel = self._problem.directional(
            equilibrium,
            kernel,
            parameter_a,
            parameter_b,
        )
        jordan = (
            self._problem.directional(equilibrium, chain, parameter_a, parameter_b)
            - kernel
        )
        scale = self._normalization @ kernel - _NORMALIZATION_TARGET
        orthogonal = self._normalization @ chain
        return np.concatenate(
            [residual, in_kernel, jordan, [scale], [orthogonal]],
        )

    def decode(self, state: np.ndarray, parameter: float) -> Codim3Decoded:
        """Recover the original-system point from the augmented vector."""
        n = self._dimension
        self._third.value = float(parameter)
        return Codim3Decoded(
            state=state[:n],
            parameter_a=float(state[3 * n]),
            parameter_b=float(state[3 * n + 1]),
            parameter_c=float(parameter),
        )
