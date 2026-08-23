"""Linear-system solving abstraction used by the corrector and tangent computer."""

from abc import ABC, abstractmethod

import numpy as np


class LinearSolver(ABC):
    """Solve ``matrix @ x = vector`` for ``x``."""

    @abstractmethod
    def solve(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Return the solution vector ``x``."""
        raise NotImplementedError


class LeastSquaresFallbackSolver(LinearSolver):
    """Direct solve, falling back to least squares when the matrix is singular.

    Continuation regularly forms Jacobians that become singular exactly at the
    bifurcations of interest, so a graceful least-squares fallback keeps the
    corrector well behaved there.
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
