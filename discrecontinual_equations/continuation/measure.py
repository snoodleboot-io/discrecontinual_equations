"""Scalar measures plotted against the parameter on a bifurcation diagram.

A one-parameter bifurcation diagram plots one number per equilibrium against the
parameter. :class:`Norm` uses ``||u||``, which separates branches but cannot tell
``u`` from ``-u`` (the two halves of a pitchfork overlap); :class:`Component` uses
one coordinate, which keeps the sign.
"""

from abc import ABC, abstractmethod

import numpy as np


class Measure(ABC):
    """Reduce a state vector to a single scalar for diagrams."""

    @abstractmethod
    def of(self, state: np.ndarray) -> float:
        """Return the scalar measure of ``state``.

        Args:
            state: The equilibrium state.

        Returns:
            The value plotted on the diagram.
        """
        raise NotImplementedError


class Norm(Measure):
    """Euclidean norm of the state vector."""

    def of(self, state: np.ndarray) -> float:
        return float(np.linalg.norm(state))


class Component(Measure):
    """A single chosen component of the state vector.

    Args:
        index: The component plotted.
    """

    __slots__ = ["_index"]

    def __init__(self, index: int) -> None:
        self._index = index

    def of(self, state: np.ndarray) -> float:
        return float(state[self._index])
