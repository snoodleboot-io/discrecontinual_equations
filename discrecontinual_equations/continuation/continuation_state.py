"""Immutable working state for a single point during continuation."""

import numpy as np


class ContinuationState:
    """A point on the branch while it is being traced.

    Distinct from :class:`~.continuation_point.ContinuationPoint`, which is the
    finished, serializable record. This carries only what the driver needs to
    take the next step: the augmented coordinate and its tangent.
    """

    __slots__ = ["_arclength", "_parameter", "_state", "_tangent"]

    def __init__(
        self,
        state: np.ndarray,
        parameter: float,
        tangent: np.ndarray,
        arclength: float,
    ) -> None:
        self._state = state
        self._parameter = parameter
        self._tangent = tangent
        self._arclength = arclength

    @property
    def state(self) -> np.ndarray:
        """Equilibrium state vector ``u``."""
        return self._state

    @property
    def parameter(self) -> float:
        """Continuation parameter ``lambda``."""
        return self._parameter

    @property
    def tangent(self) -> np.ndarray:
        """Unit tangent in augmented ``(u, lambda)`` space."""
        return self._tangent

    @property
    def arclength(self) -> float:
        """Cumulative arclength along the branch."""
        return self._arclength
