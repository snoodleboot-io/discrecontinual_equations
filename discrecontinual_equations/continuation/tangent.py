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
    seed it is set by ``direction``. With a ``state_scale`` ``S`` other than 1
    the tangent is a unit vector, and is oriented, in the weighted inner product
    ``<a, b>_W = a_u . b_u / S^2 + a_lambda b_lambda`` of
    :class:`~.parameterization.PseudoArclength`, as AUTO normalises its tangent
    with the weights ``THU`` (Doedel and Oldeman 2021, ``src/ae.f90``,
    subroutine ``STPRAE``).

    Args:
        jacobian: Source of the extended Jacobian.
        linear_solver: Solves the bordered system.
        direction: ``+1`` or ``-1``, the seed's orientation in the parameter.
        state_scale: The state size ``S`` that one unit of arclength stands for.
    """

    __slots__ = ["_direction", "_jacobian", "_linear_solver", "_state_weight"]

    def __init__(
        self,
        jacobian: JacobianProvider,
        linear_solver: LinearSolver,
        direction: int,
        state_scale: float = 1.0,
    ) -> None:
        self._jacobian = jacobian
        self._linear_solver = linear_solver
        self._direction = direction
        self._state_weight = 1.0 / (state_scale * state_scale)

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
        if self._state_weight == 1.0:
            tangent = tangent / np.linalg.norm(tangent)
        else:
            tangent = tangent / np.sqrt(self._inner(tangent, tangent))
        return self._orient(tangent, previous_tangent)

    def _inner(self, first: np.ndarray, second: np.ndarray) -> float:
        state_part = self._state_weight * float(first[:-1] @ second[:-1])
        return state_part + float(first[-1] * second[-1])

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
        alignment = (
            tangent @ previous_tangent
            if self._state_weight == 1.0
            else self._inner(tangent, previous_tangent)
        )
        if alignment < 0:
            return -tangent
        return tangent
