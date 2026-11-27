"""Minimally augmented residual whose zero set is the cusp curve in three parameters.

A cusp is codimension two, so in three active parameters the set of cusps is a
one-dimensional curve. It is defined by the minimally augmented fold system of
Govaerts (*Numerical Methods for Bifurcations of Dynamical Equilibria*, SIAM 2000,
section 4.1.2; MatCont's ``limitpoint.m``) together with the vanishing of the
quadratic normal-form coefficient:

    f(u; p_a, p_b, p_c)  = 0
    g(u; p_a, p_b, p_c)  = 0      [[f_u, w_bor], [v_bor^T, 0]] (v, g) = (0, 1)
    a(u; p_a, p_b, p_c)  = 0      a = (1/2) w^T B(v, v)

with unknowns ``(u, p_a, p_b)`` and continuation coordinate ``p_c``. Here ``w``
solves the transposed system ``[[f_u^T, v_bor], [w_bor^T, 0]] (w, g) = (0, 1)``, so
on the curve ``v`` and ``w`` are the right and left null vectors and ``a`` is
MatCont's cusp test function ``w^T f_uu[v, v]`` up to the factor one half.

Each row costs one LU of the bordered matrix. The state columns of the Jacobian
are analytic: ``g_u = -w^T B(v, e_k)`` (Govaerts' ``g_z = -w^T (f_u)_z v``), and
the derivative of ``a`` uses two more solves with the same factors in place of the
derivatives of ``v`` and ``w``. Only the three parameter columns are differenced,
centrally, at the relative step ``epsilon max(|p|, 1)`` and over the width actually
stepped (:func:`~discrecontinual_equations.finite_difference.central_column`,
DEQ-159), so the columns stay resolved on a curve whose parameters are of size
``1e12``; an absolute step there rounds away and returns zero columns.
The borders are frozen through a Newton solve and moved to the normalised ``v``
and ``w`` at every accepted point (MatCont's adaptation); the cusp curve itself
does not depend on them. A swallowtail is where the cubic coefficient vanishes
along the curve.
"""

import numpy as np

from discrecontinual_equations.continuation.bordered import BorderedMatrix
from discrecontinual_equations.continuation.normal_form import MultilinearForms
from discrecontinual_equations.continuation.residual import DifferentiableResidual
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
from discrecontinual_equations.finite_difference import central_column
from discrecontinual_equations.parameter import Parameter

_HALF = 0.5
_PARAMETER_COUNT = 2


class Codim3Decoded:
    """Original-system quantities decoded from an augmented cusp-curve point."""

    __slots__ = ["null_vector", "parameter_a", "parameter_b", "parameter_c", "state"]

    def __init__(
        self,
        state: np.ndarray,
        parameter_a: float,
        parameter_b: float,
        parameter_c: float,
        null_vector: np.ndarray | None = None,
    ) -> None:
        self.state = state
        self.parameter_a = parameter_a
        self.parameter_b = parameter_b
        self.parameter_c = parameter_c
        # The curve's own null vector, continuous along it (a BT curve's q0).
        self.null_vector = null_vector


class _CuspLinearization:
    """Everything the residual and its state columns share at one point."""

    __slots__ = ["bordered", "fold", "forms", "jacobian", "left", "residual", "right"]

    def __init__(
        self,
        problem: TwoParameterProblem,
        state: np.ndarray,
        borders: tuple[np.ndarray, np.ndarray],
    ) -> None:
        n = state.size - _PARAMETER_COUNT
        equilibrium = state[:n]
        parameter_a, parameter_b = float(state[n]), float(state[n + 1])
        right_border, left_border = borders
        self.jacobian = problem.state_jacobian(equilibrium, parameter_a, parameter_b)
        self.bordered = BorderedMatrix(self.jacobian, left_border, right_border)
        unit = np.zeros(n + 1)
        unit[n] = 1.0
        solution = self.bordered.solve(unit)
        self.right = solution[:n]
        self.fold = float(solution[n])
        self.left = self.bordered.solve_transposed(unit)[:n]
        self.forms = MultilinearForms(problem, equilibrium, parameter_a, parameter_b)
        self.residual = problem.evaluate(equilibrium, parameter_a, parameter_b)


class CuspCurveResidual(DifferentiableResidual):
    """Minimally augmented fold system with the cusp condition ``a = 0``."""

    __slots__ = [
        "_cache",
        "_dimension",
        "_left_border",
        "_problem",
        "_right_border",
        "_third",
    ]

    def __init__(
        self,
        problem: TwoParameterProblem,
        dimension: int,
        right_border: np.ndarray,
        left_border: np.ndarray,
        third: Parameter,
    ) -> None:
        self._problem = problem
        self._dimension = dimension
        self._right_border = right_border / np.linalg.norm(right_border)
        self._left_border = left_border / np.linalg.norm(left_border)
        self._third = third
        self._cache: tuple[bytes, _CuspLinearization] | None = None

    def _linearize(self, state: np.ndarray, parameter: float) -> _CuspLinearization:
        self._third.value = float(parameter)
        key = np.append(state, parameter).tobytes()
        if self._cache is None or self._cache[0] != key:
            borders = (self._right_border, self._left_border)
            self._cache = (key, _CuspLinearization(self._problem, state, borders))
        return self._cache[1]

    def evaluate(self, state: np.ndarray, parameter: float) -> np.ndarray:
        point = self._linearize(np.asarray(state, dtype=float), parameter)
        quadratic = point.forms.bilinear(point.right, point.right).real
        cusp = _HALF * float(point.left @ quadratic)
        return np.concatenate([point.residual, [point.fold], [cusp]])

    def extended_jacobian(
        self,
        state: np.ndarray,
        parameter: float,
        epsilon: float,
        typical: np.ndarray | None = None,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=float)
        n = self._dimension
        point = self._linearize(state, parameter)
        size = n + _PARAMETER_COUNT
        jacobian = np.zeros((size, size + 1))
        jacobian[:n, :n] = point.jacobian
        fold_row, cusp_row = self._state_rows(point)
        jacobian[n, :n] = fold_row
        jacobian[n + 1, :n] = cusp_row
        for column in range(n, size + 1):
            scale = 1.0 if typical is None else float(typical[column])
            jacobian[:, column] = self._difference(
                state,
                parameter,
                column,
                (epsilon, scale),
            )
        return jacobian

    def _state_rows(self, point: _CuspLinearization) -> tuple[np.ndarray, np.ndarray]:
        """Analytic ``d g / d u`` and ``d a / d u`` at frozen borders.

        With ``N = [[A, w_bor], [v_bor^T, 0]]``, ``N (v_k, g_k) = (-A_k v, 0)`` and
        ``N^T (w_k, g_k) = (-A_k^T w, 0)`` for ``A_k = B(., e_k)``. Then
        ``g_k = -w^T B(v, e_k)`` and

            a_k = (1/2) w_k^T B(v, v) + w^T B(v, v_k) + (1/2) w^T C(v, v, e_k)
                = -(1/2) w^T B(y, e_k) - t^T B(v, e_k) + (1/2) w^T C(v, v, e_k),

        where ``N (y, .) = (B(v, v), 0)`` and ``N^T (t, .) = (s, 0)`` with
        ``s_k = w^T B(v, e_k)``: two solves instead of ``2n``.
        """
        n = self._dimension
        right, left, forms = point.right, point.left, point.forms
        units = np.eye(n)
        along_right = np.column_stack(
            [forms.bilinear(right, units[k]).real for k in range(n)],
        )
        contracted = left @ along_right
        quadratic = forms.bilinear(right, right).real
        adjoint_square = point.bordered.solve(np.append(quadratic, 0.0))[:n]
        adjoint_contracted = point.bordered.solve_transposed(
            np.append(contracted, 0.0),
        )[:n]
        cusp_row = np.empty(n)
        for k in range(n):
            square_term = left @ forms.bilinear(adjoint_square, units[k]).real
            cubic_term = left @ forms.trilinear(right, right, units[k]).real
            cusp_row[k] = (
                -_HALF * square_term
                - adjoint_contracted @ along_right[:, k]
                + _HALF * cubic_term
            )
        return -contracted, cusp_row

    def _difference(
        self,
        state: np.ndarray,
        parameter: float,
        column: int,
        step: tuple[float, float],
    ) -> np.ndarray:
        """Column ``column`` at relative step and typical size ``step``."""
        epsilon, typical = step
        difference = central_column(
            self.evaluate,
            state,
            parameter,
            column,
            epsilon,
            typical,
        )
        self._third.value = float(parameter)
        return difference

    def adapt(self, state: np.ndarray, parameter: float) -> None:
        """Move the borders to the normalised null vectors at an accepted point."""
        point = self._linearize(np.asarray(state, dtype=float), parameter)
        self._right_border = point.right / np.linalg.norm(point.right)
        self._left_border = point.left / np.linalg.norm(point.left)
        self._cache = None

    def decode(self, state: np.ndarray, parameter: float) -> Codim3Decoded:
        """Recover the original-system point from the augmented vector."""
        n = self._dimension
        self._third.value = float(parameter)
        return Codim3Decoded(
            state=state[:n],
            parameter_a=float(state[n]),
            parameter_b=float(state[n + 1]),
            parameter_c=float(parameter),
        )
