"""Saddle-focus equilibria and the Shilnikov saddle index.

A saddle-focus in three dimensions has one real eigenvalue and a complex-conjugate
pair whose real part has the opposite sign: the flow spirals in (or out) on a plane
while expanding (or contracting) along a transverse line. Shilnikov's theorem states
that a homoclinic orbit to such a point produces chaos - a countable set of
horseshoes - precisely when the *saddle index*

    delta = |Re(complex pair)| / |real eigenvalue|

is less than one, i.e. the one-dimensional direction dominates the spiral. When
``delta > 1`` the homoclinic loop is tame.

:func:`classify_saddle_focus` reads this geometry off a Jacobian and returns a
:class:`SaddleFocus` carrying the index and the chaos criterion. It detects the
linear condition; whether a homoclinic connection actually exists, and the resulting
chaos, can be confirmed with the flow's top Lyapunov exponent
(:func:`discrecontinual_equations.continuation.lyapunov.lyapunov_spectrum`).
"""

import numpy as np

_IMAGINARY_TOLERANCE = 1.0e-9
_CHAOS_THRESHOLD = 1.0


class SaddleFocus:
    """The eigenstructure of a three-dimensional saddle-focus equilibrium."""

    __slots__ = ["_real_eigenvalue", "_spiral_imaginary", "_spiral_real"]

    def __init__(
        self,
        real_eigenvalue: float,
        spiral_real: float,
        spiral_imaginary: float,
    ) -> None:
        self._real_eigenvalue = real_eigenvalue
        self._spiral_real = spiral_real
        self._spiral_imaginary = spiral_imaginary

    @property
    def real_eigenvalue(self) -> float:
        """The lone real eigenvalue (the one-dimensional direction)."""
        return self._real_eigenvalue

    @property
    def spiral_real(self) -> float:
        """Real part of the complex pair (the spiral's growth or decay rate)."""
        return self._spiral_real

    @property
    def spiral_frequency(self) -> float:
        """Imaginary part of the complex pair (the spiral's rotation rate)."""
        return self._spiral_imaginary

    @property
    def saddle_index(self) -> float:
        """Shilnikov index ``|Re(complex)| / |real eigenvalue|``."""
        return abs(self._spiral_real) / abs(self._real_eigenvalue)

    @property
    def satisfies_shilnikov_criterion(self) -> bool:
        """Whether a homoclinic loop here would be chaotic (``saddle_index < 1``)."""
        return self.saddle_index < _CHAOS_THRESHOLD


def classify_saddle_focus(jacobian: np.ndarray) -> SaddleFocus | None:
    """Return the :class:`SaddleFocus` of a Jacobian, or ``None`` if it is not one.

    The Jacobian must have exactly one real eigenvalue and one complex-conjugate
    pair whose real part has the opposite sign to the real eigenvalue.
    """
    eigenvalues = np.linalg.eigvals(np.asarray(jacobian, dtype=float))
    reals = [
        value.real for value in eigenvalues if abs(value.imag) < _IMAGINARY_TOLERANCE
    ]
    pairs = [value for value in eigenvalues if value.imag > _IMAGINARY_TOLERANCE]
    if len(reals) != 1 or len(pairs) != 1:
        return None
    real_eigenvalue = reals[0]
    spiral = pairs[0]
    if real_eigenvalue * spiral.real >= 0.0:
        return None
    return SaddleFocus(real_eigenvalue, spiral.real, spiral.imag)
