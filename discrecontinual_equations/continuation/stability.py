"""Stability classification from the state Jacobian spectrum."""

from abc import ABC, abstractmethod

import numpy as np


class StabilityResult:
    """Eigenvalues and the stability classification of an equilibrium."""

    __slots__ = ["_eigenvalues", "_label", "_unstable_dimension"]

    def __init__(
        self,
        eigenvalues: list[tuple[float, float]],
        unstable_dimension: int,
        label: str,
    ) -> None:
        self._eigenvalues = eigenvalues
        self._unstable_dimension = unstable_dimension
        self._label = label

    @property
    def eigenvalues(self) -> list[tuple[float, float]]:
        """Eigenvalues as ``(real, imaginary)`` pairs."""
        return self._eigenvalues

    @property
    def unstable_dimension(self) -> int:
        """Number of eigenvalues with positive real part."""
        return self._unstable_dimension

    @property
    def label(self) -> str:
        """One of ``stable``, ``unstable`` or ``saddle``."""
        return self._label


class StabilityAnalyzer(ABC):
    """Classify an equilibrium from its state Jacobian."""

    @abstractmethod
    def analyze(self, state_jacobian: np.ndarray) -> StabilityResult:
        """Return the stability result for the given Jacobian."""
        raise NotImplementedError


class EigenvalueStabilityAnalyzer(StabilityAnalyzer):
    """Classify by counting eigenvalues with positive real part."""

    __slots__ = ["_tolerance"]

    def __init__(self, tolerance: float) -> None:
        self._tolerance = tolerance

    def analyze(self, state_jacobian: np.ndarray) -> StabilityResult:
        eigenvalues = np.linalg.eigvals(state_jacobian)
        unstable = int(np.sum(eigenvalues.real > self._tolerance))
        pairs = [(float(value.real), float(value.imag)) for value in eigenvalues]
        return StabilityResult(pairs, unstable, self._label(unstable, eigenvalues.size))

    def _label(self, unstable: int, total: int) -> str:
        if unstable == 0:
            return "stable"
        if unstable == total:
            return "unstable"
        return "saddle"
