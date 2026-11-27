"""A bordered matrix factorised once and solved against many right-hand sides.

Minimally augmented defining systems (Govaerts, *Numerical Methods for
Bifurcations of Dynamical Equilibria*, SIAM 2000; MatCont's ``limitpoint.m`` and
``hopf.m``) border a nearly singular matrix ``L`` with ``k`` columns ``W`` and
``k`` rows ``V^T``,

    [[L, W], [V^T, 0]],

so that the bordered matrix is nonsingular even where ``L`` is not. One LU gives
the test functions, the null vectors, and every derivative solve, with the matrix
or its transpose.

For a fold, ``L = F_u``, ``k = 1``: solving ``[[L, p0], [q0^T, 0]] (w, g) = (0, 1)``
gives a scalar ``g`` that vanishes exactly where ``L`` is singular, and the matrix
stays nonsingular there as long as ``q0`` and ``p0`` are near the null vectors of
``L`` and ``L^T`` (Kuznetsov (1998), eqs. (10.68)-(10.69)); for Hopf the same with
``L = 2 A (.) I_n`` the bialternate product (eq. (10.71)). The transposed solve gives
the left null vector ``v``, and ``g_z = -v^T L_z w`` its derivatives (Kuznetsov
(1998), eq. (10.70) and the line after it); by Cramer's rule
``g = det L / det M``. Govaerts (2000, Chapter 3) develops the bordered matrices
these systems rest on; minimally augmented systems go back to Griewank and Reddien
(fold) and Govaerts, Guckenheimer and Khibnik (Hopf), as Kuznetsov's notes record.

Limits: the border must be refreshed as the curve moves (see
:meth:`~.residual.ResidualFunction.adapt`), or the bordered matrix can itself
become singular; and a single border (``k = 1``) cannot remove a kernel of
dimension two, so the matrix is singular wherever ``L`` has one - for the
bialternate system that is every Hopf-Hopf point, whatever the borders
(Kuznetsov's footnote to (10.71)).

Used by the minimally augmented two-parameter curves, which are documented with
the bifurcation-curve pages.

References:
- Govaerts, W. J. F. "Numerical Methods for Bifurcations of Dynamical Equilibria",
  SIAM, 2000, doi:10.1137/1.9780898719543, Chapter 3 ("Bordered Matrices").
- Kuznetsov, Yu. A. "Elements of Applied Bifurcation Theory", 2nd ed., Applied
  Mathematical Sciences 112, Springer, 1998, doi:10.1007/b98848, Section 10.3.1,
  eqs. (10.68)-(10.71), and Section 10.8 (bibliographical notes).
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve


class BorderedMatrix:
    """``[[L, W], [V^T, 0]]`` with an LU factorisation for repeated solves.

    Args:
        matrix: The ``(n, n)`` matrix ``L``.
        column: The border columns ``W``, shape ``(n,)`` or ``(n, k)``.
        row: The border rows ``V`` (as columns), shape ``(n,)`` or ``(n, k)``.

    Raises:
        ValueError: If ``column`` or ``row`` cannot be reshaped to ``(n, k)``
            (raised by numpy).

    References:
    - Govaerts (2000), Chapter 3.
    - Kuznetsov (1998), eqs. (10.69) and (10.71).
    """

    __slots__ = ["_factors", "_size"]

    def __init__(self, matrix: np.ndarray, column: np.ndarray, row: np.ndarray) -> None:
        size = matrix.shape[0]
        column = column.reshape(size, -1)
        row = row.reshape(size, -1)
        border = column.shape[1]
        bordered = np.zeros((size + border, size + border), dtype=matrix.dtype)
        bordered[:size, :size] = matrix
        bordered[:size, size:] = column
        bordered[size:, :size] = row.T
        self._factors = lu_factor(bordered)
        self._size = size

    @property
    def size(self) -> int:
        """Dimension of the unbordered matrix ``L``."""
        return self._size

    def solve(self, right: np.ndarray) -> np.ndarray:
        """Solve ``[[L, W], [V^T, 0]] x = right`` (one or many columns)."""
        return lu_solve(self._factors, right)

    def solve_transposed(self, right: np.ndarray) -> np.ndarray:
        """Solve ``[[L^T, V], [W^T, 0]] x = right`` (one or many columns)."""
        return lu_solve(self._factors, right, trans=1)
