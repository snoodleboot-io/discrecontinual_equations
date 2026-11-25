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

The linear algebra is Gray's Theorem 3.1: every circulant matrix ``C`` with
``C[k, j] = c_((j - k) mod n)`` is ``U Psi U^*`` with ``U`` the unitary Fourier
matrix, the same for every circulant, and ``psi_m = sum_k c_k e^(-2 pi i m k / n)``
(his eq. (3.7)). Here the Fourier matrix is built with ``e^(+2 pi i j k / N)``, so
this module's mode ``k`` is Gray's mode ``N - k``: its eigenvalue is
``sum_j c_j e^(+2 pi i j k / N)``, which for ``J = a I + b S`` is the
``a + b e^(2 pi i k / N)`` above. For a real matrix modes ``k`` and ``N - k`` are
complex conjugates, so the labelling changes neither the spectrum nor which real
part is largest - only which index of a conjugate pair is reported.

Limits: one cell per site (``N x N`` Jacobian, scalar cells). A ring of
``d``-dimensional cells is block-circulant and needs ``d x d`` blocks per mode,
which this reduction does not do - it reports ``is_circulant = False``. The
interpretation of the critical mode (steady pattern for a real one, rotating wave
for a complex one) is the standard reading of a symmetry-breaking bifurcation, not
a computation: it does not check nondegeneracy or the direction of branching.

References:
- Gray, R. M. "Toeplitz and Circulant Matrices: A Review", Foundations and Trends
  in Communications and Information Theory 2(3), 155-239, 2006,
  doi:10.1561/0100000006, Chapter 3, eqs. (3.1), (3.7), (3.12) and Theorem 3.1.
"""

import numpy as np

from discrecontinual_equations.function.function import Function

_CIRCULANT_TOLERANCE = 1.0e-9


def cyclic_action(size: int, step: int = 1) -> np.ndarray:
    """The shift matrix ``S`` with ``(S x)[i] = x[(i + step) % size]``.

    Args:
        size: Number of cells ``N``.
        step: Shift, in cells.

    Returns:
        The ``(N, N)`` permutation matrix.
    """
    action = np.zeros((size, size))
    for i in range(size):
        action[i, (i + step) % size] = 1.0
    return action


def equivariance_defect(
    function: Function,
    action: np.ndarray,
    point: np.ndarray,
) -> float:
    """Return ``||f(A x) - A f(x)||``; zero when ``f`` is equivariant under ``A``.

    Args:
        function: The vector field ``f`` (evaluated at ``time=None``).
        action: The group element ``A`` as a matrix.
        point: The state ``x`` to test at.

    Returns:
        The defect's Euclidean norm. One point proves nothing in general; test a
        few generic points.
    """
    state = np.asarray(point, dtype=float)
    transformed = np.array(function.eval(point=list(action @ state), time=None))
    original = np.array(function.eval(point=list(state), time=None))
    return float(np.linalg.norm(transformed - action @ original))


class FourierReduction:
    """The spectrum of a circulant Jacobian, resolved by Fourier mode.

    Args:
        eigenvalues: The diagonal of the Fourier-transformed matrix, one per mode.
        residual: Norm of its off-diagonal part.
    """

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

    Args:
        jacobian: A square real matrix, typically the Jacobian at the symmetric
            equilibrium.

    Returns:
        The per-mode eigenvalues and the residual; ``is_circulant`` tests the
        residual against an absolute ``1e-9``, so scale large matrices first.

    References:
    - Gray (2006), Theorem 3.1 and eq. (3.13).
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
