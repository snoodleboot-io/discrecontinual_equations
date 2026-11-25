"""Linear-system solving abstraction used by the corrector and tangent computer.

Every Newton correction and every tangent solves one square system with the
*augmented Jacobian* ``[[F_y], [t^T]]``: the ``n x (n + 1)`` Jacobian of the
residual bordered by a row ``t^T``. Allgower and Georg (1990, Section 2.1, after
eq. (2.1.5)) show it is nonsingular all along a regular curve - folds included,
where ``F_u`` alone is singular - because ``t`` is not orthogonal to the kernel of
``F_y``. It *is* singular at a branch point, where ``F_y`` loses rank, which is
exactly where continuation wants to pass.

:class:`LeastSquaresFallbackSolver` solves by LU and, only when LAPACK reports an
exactly singular matrix, returns the minimum-norm least-squares solution instead.
It does not detect *near* singularity: an ill-conditioned matrix is solved by LU
as usual, and its error is amplified by the condition number.

References:
- Allgower, E. L. and Georg, K. "Numerical Continuation Methods: An Introduction",
  Springer, 1990, doi:10.1007/978-3-642-61257-2, Section 2.1, eq. (2.1.5) and
  Lemma (3.2.3).
"""

from abc import ABC, abstractmethod

import numpy as np


class LinearSolver(ABC):
    """Solve ``matrix @ x = vector`` for ``x``."""

    @abstractmethod
    def solve(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Return the solution vector ``x``.

        Args:
            matrix: Square coefficient matrix.
            vector: Right-hand side.

        Returns:
            The solution ``x`` (a least-squares one where an implementation allows).
        """
        raise NotImplementedError


class LeastSquaresFallbackSolver(LinearSolver):
    """Direct solve, falling back to least squares when the matrix is singular.

    Continuation regularly forms Jacobians that become singular exactly at the
    bifurcations of interest, so a graceful least-squares fallback keeps the
    corrector well behaved there.

    The fallback is :func:`numpy.linalg.lstsq` with the default cutoff, which
    gives the minimum-norm solution; for a system ``[[A], [t^T]] x = [b; 0]`` that
    is the Moore-Penrose step ``A^+ b`` of Allgower and Georg's Lemma (3.2.3).

    References:
    - Allgower and Georg (1990), Lemma (3.2.3) and Definition (3.2.2).
    """

    def solve(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.solve(matrix, vector)
        except np.linalg.LinAlgError:
            solution, _residuals, _rank, _singular = np.linalg.lstsq(
                matrix,
                vector,
                rcond=None,
            )
            return solution
