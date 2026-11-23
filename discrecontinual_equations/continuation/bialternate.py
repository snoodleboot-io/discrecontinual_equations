"""Hopf test function: the bialternate product ``2A (.) I`` read via its eigenvalues.

The bialternate product ``2A (.) I`` of an ``m x m`` matrix ``A`` has the
``m(m - 1)/2`` pairwise sums ``lambda_i + lambda_j`` (``i < j``) of the eigenvalues of
``A`` as its eigenvalues (Guckenheimer, Myers & Sturmfels 1997, Theorem 3.1), so its
determinant ``prod_{i<j} (lambda_i + lambda_j)`` vanishes exactly when ``A`` has a
pair of eigenvalues summing to zero: a purely imaginary pair (a Hopf point) or a
real pair ``+-r`` (a neutral saddle).

Forming that ``m(m - 1)/2``-square matrix and taking its determinant costs ``O(m^6)``
and overflows (or underflows) once ``m`` reaches a few dozen, and an infinite or zero
test value cannot change sign. Here the same product is read from one eigenvalue
solve in ``O(m^3)``: its sign is counted from the signs of the factors (which
cannot overflow), and its magnitude is replaced by the smallest factor
``min |lambda_i + lambda_j|``. The result changes sign exactly where the determinant
does, is continuous, vanishes linearly at a simple crossing, and is measured in the
units of the eigenvalues, so the localizer's value tolerance means the same thing at
every dimension. For a planar system it equals the trace, as the determinant did.
"""

import numpy as np

_MINIMUM_DIMENSION = 2


def bialternate_test_value(matrix: np.ndarray) -> float:
    """Return the Hopf test value; constant when the system is one-dimensional.

    The sign is that of ``det(2A (.) I) = prod_{i<j} (lambda_i + lambda_j)``. Factors
    from a complex-conjugate pair ``lambda, conj(lambda)`` and from two real
    eigenvalues are real. Every other factor comes with its conjugate, whose real
    part is the same and whose product with it is positive, so counting the
    factors with a negative real part counts each such conjugate pair twice and
    leaves the parity of the negative real factors.
    """
    dimension = matrix.shape[0]
    if dimension < _MINIMUM_DIMENSION:
        return 1.0
    if not np.all(np.isfinite(matrix)):
        return float("nan")
    eigenvalues = np.linalg.eigvals(matrix).astype(complex)
    first, second = np.triu_indices(dimension, 1)
    sums = eigenvalues[first] + eigenvalues[second]
    negative = np.count_nonzero(sums.real < 0.0)
    sign = -1.0 if negative % 2 else 1.0
    return float(sign * np.min(np.abs(sums)))
