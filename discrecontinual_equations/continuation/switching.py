"""Seeding of secondary branches at detected branch points."""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.continuation.continuation_point import ContinuationPoint
from discrecontinual_equations.continuation.jacobian import JacobianProvider

_TRANSVERSE_COMPONENT = 0


class BranchSwitcher(ABC):
    """Produce a seed on the branch that crosses at a branch point."""

    @abstractmethod
    def secondary_seed(
        self,
        branch_point: ContinuationPoint,
    ) -> tuple[np.ndarray, float]:
        """Return a ``(state, parameter)`` seed on the bifurcating branch."""
        raise NotImplementedError


class NullSpaceBranchSwitcher(BranchSwitcher):
    """Switch along the second null direction of the extended Jacobian.

    Where both branches share a tangent (a pitchfork), the transverse component
    vanishes and the switcher falls back to a state kick transverse to the
    trivial branch.
    """

    __slots__ = ["_jacobian", "_kick", "_singular_tolerance"]

    def __init__(
        self,
        jacobian: JacobianProvider,
        kick: float,
        singular_tolerance: float,
    ) -> None:
        self._jacobian = jacobian
        self._kick = kick
        self._singular_tolerance = singular_tolerance

    def secondary_seed(
        self,
        branch_point: ContinuationPoint,
    ) -> tuple[np.ndarray, float]:
        state = np.asarray(branch_point.state, dtype=float)
        parameter = float(branch_point.parameter)
        extended = self._jacobian.extended_jacobian(state, parameter)
        _left, _singular, right_vectors = np.linalg.svd(extended)
        primary = np.asarray(branch_point.tangent, dtype=float)
        primary = primary / np.linalg.norm(primary)
        candidate = right_vectors[-1]
        secondary = candidate - (candidate @ primary) * primary
        if np.linalg.norm(secondary) < self._singular_tolerance:
            return self._transverse_kick(state, parameter)
        secondary = secondary / np.linalg.norm(secondary)
        seed = np.concatenate([state, [parameter]]) + self._kick * secondary
        return seed[:-1], float(seed[-1])

    def _transverse_kick(
        self,
        state: np.ndarray,
        parameter: float,
    ) -> tuple[np.ndarray, float]:
        kicked = state.copy()
        kicked[_TRANSVERSE_COMPONENT] += self._kick
        return kicked, parameter + self._kick
