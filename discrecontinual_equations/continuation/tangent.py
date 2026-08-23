"""Computation of the oriented branch tangent."""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.continuation.jacobian import JacobianProvider
from discrecontinual_equations.continuation.linear_solver import LinearSolver

_ORIENTATION_SEED = 1.0


class TangentComputer(ABC):
    """Compute the unit tangent to the branch at a point."""

    @abstractmethod
    def compute(
        self,
        state: np.ndarray,
        parameter: float,
        previous_tangent: np.ndarray | None,
    ) -> np.ndarray:
        """Return the oriented unit tangent in augmented space."""
        raise NotImplementedError


class BorderedTangentComputer(TangentComputer):
    """Tangent as the null vector of the extended Jacobian via a bordered solve.

    Orientation is inherited from ``previous_tangent`` for continuity; at the
    seed it is set by ``direction``.
    """

    __slots__ = ["_direction", "_jacobian", "_linear_solver"]

    def __init__(
        self,
        jacobian: JacobianProvider,
        linear_solver: LinearSolver,
        direction: int,
    ) -> None:
        self._jacobian = jacobian
        self._linear_solver = linear_solver
        self._direction = direction

    def compute(
        self,
        state: np.ndarray,
        parameter: float,
        previous_tangent: np.ndarray | None,
    ) -> np.ndarray:
        extended = self._jacobian.extended_jacobian(state, parameter)
        dimension = extended.shape[0]
        border = self._border(dimension, previous_tangent)
        matrix = np.vstack([extended, border])
        right_hand = np.zeros(dimension + 1)
        right_hand[-1] = _ORIENTATION_SEED
        tangent = self._linear_solver.solve(matrix, right_hand)
        tangent = tangent / np.linalg.norm(tangent)
        return self._orient(tangent, previous_tangent)

    def _border(
        self,
        dimension: int,
        previous_tangent: np.ndarray | None,
    ) -> np.ndarray:
        if previous_tangent is not None:
            return previous_tangent
        border = np.zeros(dimension + 1)
        border[-1] = _ORIENTATION_SEED
        return border

    def _orient(
        self,
        tangent: np.ndarray,
        previous_tangent: np.ndarray | None,
    ) -> np.ndarray:
        if previous_tangent is None:
            return self._direction * tangent
        if tangent @ previous_tangent < 0:
            return -tangent
        return tangent
