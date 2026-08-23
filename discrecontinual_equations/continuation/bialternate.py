"""Bialternate product ``2A (.) I`` used as the Hopf test function.

The determinant of the bialternate product vanishes exactly when the matrix has a
pair of eigenvalues summing to zero, in particular a purely imaginary pair, which
is the defining condition of a Hopf point.
"""

import numpy as np

_MINIMUM_DIMENSION = 2


def _offdiagonal(
    matrix: np.ndarray,
    row_pair: tuple[int, int],
    column_pair: tuple[int, int],
) -> float:
    """Return one entry of the bialternate product (Kuznetsov's rules)."""
    first, second = row_pair
    third, fourth = column_pair
    if third == second:
        value = -matrix[first, fourth]
    elif third != first and fourth == first:
        value = matrix[second, third]
    elif third == first and fourth == second:
        value = matrix[first, first] + matrix[second, second]
    elif third == first:
        value = matrix[second, fourth]
    elif fourth == second:
        value = matrix[first, third]
    else:
        value = 0.0
    return float(value)


def bialternate_matrix(matrix: np.ndarray) -> np.ndarray:
    """Build the bialternate product of ``matrix`` of shape ``(m, m)``."""
    dimension = matrix.shape[0]
    pairs = [(p, q) for p in range(1, dimension) for q in range(p)]
    size = len(pairs)
    result = np.zeros((size, size))
    for row, row_pair in enumerate(pairs):
        for column, column_pair in enumerate(pairs):
            result[row, column] = _offdiagonal(matrix, row_pair, column_pair)
    return result


def bialternate_determinant(matrix: np.ndarray) -> float:
    """Return the Hopf test value; constant when the system is one-dimensional."""
    if matrix.shape[0] < _MINIMUM_DIMENSION:
        return 1.0
    return float(np.linalg.det(bialternate_matrix(matrix)))
