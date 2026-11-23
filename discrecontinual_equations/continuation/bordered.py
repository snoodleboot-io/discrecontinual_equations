"""A bordered matrix factorised once and solved against many right-hand sides.

Minimally augmented defining systems (Govaerts, *Numerical Methods for
Bifurcations of Dynamical Equilibria*, SIAM 2000; MatCont's ``limitpoint.m`` and
``hopf.m``) border a nearly singular matrix ``L`` with ``k`` columns ``W`` and
``k`` rows ``V^T``,

    [[L, W], [V^T, 0]],

so that the bordered matrix is nonsingular even where ``L`` is not. One LU gives
the test functions, the null vectors, and every derivative solve, with the matrix
or its transpose.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve


class BorderedMatrix:
    """``[[L, W], [V^T, 0]]`` with an LU factorisation for repeated solves."""

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
