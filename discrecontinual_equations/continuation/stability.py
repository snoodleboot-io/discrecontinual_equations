"""Stability classification from the state Jacobian spectrum.

An equilibrium ``x*`` of ``x' = f(x)`` is hyperbolic when no eigenvalue of the
state Jacobian ``f_x(x*)`` lies on the imaginary axis; then the flow near ``x*`` is
the linear flow's, and the eigenvalues alone decide its type. This module counts
the eigenvalues with positive real part (the dimension of the unstable manifold) and
names the equilibrium ``stable`` (none), ``unstable`` (all) or ``saddle`` (some).

Limits: the count is only meaningful for a hyperbolic point. An eigenvalue whose
real part is within ``tolerance`` of zero is counted as stable, so at a fold, a Hopf
point or any other bifurcation the label is that of one side and says nothing about
the nonlinear terms that actually decide stability there; that is what the normal
forms in :mod:`.normal_form` and :mod:`.codim2_normal_form` are for. A centre
(purely imaginary pair, no unstable eigenvalue) is labelled ``stable``.
"""

from abc import ABC, abstractmethod

import numpy as np


class StabilityResult:
    """Eigenvalues and the stability classification of an equilibrium.

    Args:
        eigenvalues: The Jacobian's eigenvalues as ``(real, imaginary)`` pairs.
        unstable_dimension: How many have positive real part.
        label: ``stable``, ``unstable`` or ``saddle``.
    """

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
        """Return the stability result for the given Jacobian.

        Args:
            state_jacobian: The square Jacobian ``f_x`` at the equilibrium.

        Returns:
            The eigenvalues, the unstable dimension and the label.
        """
        raise NotImplementedError


class EigenvalueStabilityAnalyzer(StabilityAnalyzer):
    """Classify by counting eigenvalues with positive real part.

    Args:
        tolerance: An eigenvalue counts as unstable only when its real part exceeds
            this; anything closer to the axis is treated as stable.
    """

    __slots__ = ["_tolerance"]

    def __init__(self, tolerance: float) -> None:
        self._tolerance = tolerance

    def analyze(self, state_jacobian: np.ndarray) -> StabilityResult:
        """Count the eigenvalues of ``state_jacobian`` right of ``tolerance``.

        Args:
            state_jacobian: The square Jacobian ``f_x`` at the equilibrium.

        Returns:
            The eigenvalues as ``(real, imaginary)`` pairs, the number with real
            part above ``tolerance`` and the label it implies.
        """
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
