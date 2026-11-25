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

This is the *standard* (maximally extended) formulation: the eigenvector is an
unknown, so the system has ``3n + 2`` equations. MatCont and
:mod:`.generalized_hopf_curve` use the *minimally* augmented one instead, which
adds only two scalar equations through a bordered ``f_u^2 + kappa I``. The
standard system is simple and its Jacobian needs only directional derivatives of
``f_u``; it costs a ``(3n + 2)``-square solve per Newton step.

Limits: the scale row ``c . q_R = 1`` uses the fixed vector ``c`` from the seed, so
a curve along which ``q_R`` turns orthogonal to ``c`` becomes singular. At a
Bogdanov-Takens point ``q_R`` and ``q_I`` become dependent as ``omega -> 0``; the
curve can be located up to there, not continued through it as a Hopf curve. Past a
point where ``omega`` would become imaginary the curve continues as a curve of
neutral saddles only in the minimally augmented form (``kappa < 0``), not here.

References:
- Govaerts, W., Kuznetsov, Yu. A. and Sijnave, B. "Continuation of codimension-2
  equilibrium bifurcations in CONTENT", in *Numerical Methods for Bifurcation
  Problems and Large-Scale Dynamical Systems*, IMA Vol. Math. Appl., Springer
  (2000) 163-184, doi:10.1007/978-1-4612-1208-9_7, section 7.2, eq. (7.6): the
  maximally extended generalized-Hopf system, whose first rows are the eigenproblem
  ``A q - i omega q = 0`` with a normalisation row, written out as here.
- Govaerts, W., Kuznetsov, Yu. A., Meijer, H. G. E. et al. "MATCONT: Continuation
  toolbox for ODEs in Matlab", manual for MatCont 7.1 (August 2019), section 8.2: the
  minimally augmented alternative and the codim-2 test functions on a Hopf curve.
"""

import numpy as np

from discrecontinual_equations.continuation.residual import ResidualFunction
from discrecontinual_equations.continuation.two_parameter_problem import (
    CurveDecoded,
    TwoParameterProblem,
)

_NORMALIZATION_TARGET = 1.0


class HopfCurveResidual(ResidualFunction):
    """Real Moore-Spence Hopf system as an injectable residual.

    Args:
        problem: The two-parameter problem.
        dimension: The state dimension ``n``.
        normalization: The fixed vector ``c`` of the scale and phase rows.
    """

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
        """Return the ``3n + 2`` residual rows.

        Args:
            state: ``(u, q_R, q_I, omega, p_a)``.
            parameter: The continuation coordinate ``p_b``.

        Returns:
            The equilibrium, real and imaginary eigen, scale and phase residuals.
        """
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
        """Recover the original-system point and ``omega`` from the vector.

        Args:
            state: ``(u, q_R, q_I, omega, p_a)``.
            parameter: The continuation coordinate ``p_b``.

        Returns:
            The equilibrium, both parameters and ``omega``.
        """
        n = self._dimension
        return CurveDecoded(
            state=state[:n],
            parameter_a=float(state[3 * n + 1]),
            parameter_b=float(parameter),
            omega=float(state[3 * n]),
        )
