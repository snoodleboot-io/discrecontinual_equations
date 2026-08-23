"""Symmetry reduction for cyclically equivariant systems (rings of cells).

A ring of identical cells is equivariant under the cyclic group: rotating the cells
commutes with the dynamics, ``f(S x) = S f(x)`` for the shift ``S``. At the fully
symmetric equilibrium the Jacobian is therefore *circulant*, and the Fourier matrix
block-diagonalises it into one-dimensional isotypic components - the Fourier modes.
The eigenvalue of mode ``k`` is ``a + b e^{2 pi i k / N}`` for nearest-neighbour
coupling, so the spectrum is available in closed form and, more usefully, the
*first mode to lose stability* names the emerging pattern and its type: a real mode
(``k = 0`` or ``k = N/2``) gives a steady symmetry-breaking bifurcation, a complex
mode a rotating wave through a Hopf bifurcation.

:func:`fourier_reduce` performs the reduction and reports the critical mode;
:func:`equivariance_defect` checks the symmetry itself.
"""

import numpy as np

from discrecontinual_equations.function.function import Function

_CIRCULANT_TOLERANCE = 1.0e-9


def cyclic_action(size: int, step: int = 1) -> np.ndarray:
    """The shift matrix ``S`` with ``(S x)[i] = x[(i + step) % size]``."""
    action = np.zeros((size, size))
    for i in range(size):
        action[i, (i + step) % size] = 1.0
    return action


def equivariance_defect(
    function: Function,
    action: np.ndarray,
    point: np.ndarray,
) -> float:
    """Return ``||f(A x) - A f(x)||``; zero when ``f`` is equivariant under ``A``."""
    state = np.asarray(point, dtype=float)
    transformed = np.array(function.eval(point=list(action @ state), time=None))
    original = np.array(function.eval(point=list(state), time=None))
    return float(np.linalg.norm(transformed - action @ original))


class FourierReduction:
    """The spectrum of a circulant Jacobian, resolved by Fourier mode."""

    __slots__ = ["_eigenvalues", "_residual"]

    def __init__(self, eigenvalues: np.ndarray, residual: float) -> None:
        self._eigenvalues = eigenvalues
        self._residual = residual

    @property
    def eigenvalues(self) -> np.ndarray:
        """Complex eigenvalue of each Fourier mode ``k = 0 .. N - 1``."""
        return self._eigenvalues

    @property
    def is_circulant(self) -> bool:
        """Whether the Fourier basis actually diagonalised the matrix."""
        return self._residual < _CIRCULANT_TOLERANCE

    @property
    def critical_mode(self) -> int:
        """Index of the mode with the largest real part (first to destabilise)."""
        return int(np.argmax(self._eigenvalues.real))

    @property
    def critical_eigenvalue(self) -> complex:
        """Eigenvalue of the critical mode."""
        return complex(self._eigenvalues[self.critical_mode])

    @property
    def is_oscillatory(self) -> bool:
        """Whether the critical mode is complex: a rotating wave (Hopf), not steady."""
        return abs(self.critical_eigenvalue.imag) > _CIRCULANT_TOLERANCE


def fourier_reduce(jacobian: np.ndarray) -> FourierReduction:
    """Block-diagonalise a circulant Jacobian in the Fourier basis.

    Returns the per-mode eigenvalues and the off-diagonal residual, which is zero
    exactly when the matrix is circulant (i.e. the system is cyclically symmetric).
    """
    matrix = np.asarray(jacobian, dtype=float)
    size = matrix.shape[0]
    indices = np.arange(size)
    root = np.exp(2j * np.pi * np.outer(indices, indices) / size)
    fourier = root / np.sqrt(size)
    reduced = fourier.conj().T @ matrix @ fourier
    diagonal = np.diag(reduced).copy()
    residual = float(np.linalg.norm(reduced - np.diag(diagonal)))
    return FourierReduction(diagonal, residual)
