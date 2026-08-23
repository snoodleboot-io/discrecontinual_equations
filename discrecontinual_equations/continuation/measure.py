"""Scalar measures plotted against the parameter on a bifurcation diagram."""

from abc import ABC, abstractmethod

import numpy as np


class Measure(ABC):
    """Reduce a state vector to a single scalar for diagrams."""

    @abstractmethod
    def of(self, state: np.ndarray) -> float:
        """Return the scalar measure of ``state``."""
        raise NotImplementedError


class Norm(Measure):
    """Euclidean norm of the state vector."""

    def of(self, state: np.ndarray) -> float:
        return float(np.linalg.norm(state))


class Component(Measure):
    """A single chosen component of the state vector."""

    __slots__ = ["_index"]

    def __init__(self, index: int) -> None:
        self._index = index

    def of(self, state: np.ndarray) -> float:
        return float(state[self._index])
