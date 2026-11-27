"""Scalar root-finding on a bracketed test function, as an injected strategy.

Localizing a bifurcation reduces to finding where a scalar test function crosses
zero along the segment between two accepted points. A bifurcation is *detected*
when a test function ``psi`` changes sign between consecutive points,
``psi(y_k) psi(y_(k+1)) < 0`` (Kuznetsov (1998), eq. (10.35)), and *located* by a
one-dimensional root-finder along the curve - Kuznetsov names the secant method for
the case where Newton on the augmented system is unavailable (Section 10.2.2), and
Allgower and Georg (1990, Section 9.2, eq. (9.2.1)) give the Newton/secant
steplength iteration for a zero of ``f(c(s))``. The methods are interchangeable,
so the method is injected into the localizer rather than hardcoded. Two are
provided:

* :class:`Bisection` halves a bracket ``[low, high]`` on which the function changes
  sign: one evaluation per iteration, the bracket shrinks by two each time, and it
  cannot leave the bracket. Linear convergence, but certain.
* :class:`Secant` replaces the function by its chord through the last two iterates:
  superlinear (order ``(1 + sqrt 5) / 2``) near a simple root, but not bracketing,
  so the iterate is clamped to ``[low, high]`` and a flat chord stops it.

A Brent-type hybrid is not implemented.

The value test ``|f(x)| < value_tolerance`` is absolute by default, which suits a
function whose values have a known scale (a Lyapunov exponent, a probability). A
test function of a continuation has units instead: a bordered determinant of an
``n``-dimensional system scales as ``s^n`` when time is rescaled by ``s``, so an
absolute tolerance of ``1e-8`` accepts the first midpoint of a slow system and
cannot be met by a fast one. With ``relative=True`` the test reads
``|f(x)| < value_tolerance * max(|f(low)|, |f(high)|)``, which is unchanged when
``f`` is multiplied by any constant: near a simple root, where ``f`` is close to
linear along the bracket, it locates the root to about ``value_tolerance`` of the
bracket's width whatever the units. The secant's flat-chord test is scaled the
same way. The relative stop is this library's own choice, justified only by that
invariance; it is not taken from the references below.

Limits: both assume one sign change in the bracket. A test function that touches
zero without changing sign, or two crossings inside one continuation step, are not
seen; and a test function that jumps (for instance when the corrector fails and the
localizer substitutes a fallback value) can make bisection converge to the jump.

References:
- Kuznetsov, Yu. A. "Elements of Applied Bifurcation Theory", 2nd ed., Applied
  Mathematical Sciences 112, Springer, 1998, doi:10.1007/b98848, Section 10.2.2,
  eqs. (10.35)-(10.36).
- Allgower, E. L. and Georg, K. "Numerical Continuation Methods: An Introduction",
  Springer, 1990, doi:10.1007/978-3-642-61257-2, Section 9.2, eq. (9.2.1).
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

_MIDPOINT = 0.5
_DEGENERATE_SLOPE = 1e-14


class RootFinderSettings:
    """Stopping criteria shared by scalar root-finders.

    Args:
        value_tolerance: Stop once ``|function(x)|`` is below this.
        fraction_tolerance: Stop once the bracket is narrower than this (used by
            bisection).
        max_iterations: Hard cap on iterations.
        relative: Read ``value_tolerance`` as a fraction of the larger of
            ``|function(low)|`` and ``|function(high)|``, so the stop does not
            depend on the units of the function (see the module docstring).
    """

    __slots__ = [
        "_fraction_tolerance",
        "_max_iterations",
        "_relative",
        "_value_tolerance",
    ]

    def __init__(
        self,
        value_tolerance: float,
        fraction_tolerance: float,
        max_iterations: int,
        *,
        relative: bool = False,
    ) -> None:
        self._value_tolerance = value_tolerance
        self._fraction_tolerance = fraction_tolerance
        self._max_iterations = max_iterations
        self._relative = relative

    @property
    def value_tolerance(self) -> float:
        """Stop when ``|function(x)|`` falls below this."""
        return self._value_tolerance

    @property
    def fraction_tolerance(self) -> float:
        """Stop when the bracket is narrower than this."""
        return self._fraction_tolerance

    @property
    def max_iterations(self) -> int:
        """Maximum iterations."""
        return self._max_iterations

    @property
    def relative(self) -> bool:
        """Whether ``value_tolerance`` is relative to the bracket's end values."""
        return self._relative

    def value_scale(self, lower_value: float, upper_value: float) -> float:
        """The size ``value_tolerance`` is measured against for this bracket.

        Args:
            lower_value: ``function(low)``.
            upper_value: ``function(high)``.

        Returns:
            ``max(|lower_value|, |upper_value|)`` when :attr:`relative`, else 1.
        """
        if not self._relative:
            return 1.0
        return max(abs(lower_value), abs(upper_value))


class ScalarRootFinder(ABC):
    """Find ``x`` in ``[low, high]`` where ``function(x) = 0``."""

    @abstractmethod
    def locate(
        self,
        function: Callable[[float], float],
        low: float,
        high: float,
    ) -> float | None:
        """Return a root location, or ``None`` if none could be found.

        Args:
            function: The scalar function, defined on ``[low, high]``.
            low: Left end of the bracket.
            high: Right end of the bracket.

        Returns:
            The located root, or ``None``. The provided finders always return
            their last iterate, even when the iteration cap was reached first.
        """
        raise NotImplementedError


class Bisection(ScalarRootFinder):
    """Robust bracketing bisection.

    Keeps the half whose ends differ in sign; returns the midpoint once its value
    is below ``value_tolerance`` (times the larger end value when the settings
    are relative) or the bracket is narrower than ``fraction_tolerance``. A
    relative stop costs one more evaluation, at ``high``. If the ends do not
    differ in sign the bracket slides to ``high`` and the result is not a root.

    References:
    - Kuznetsov (1998), Section 10.2.2.
    """

    __slots__ = ["_settings"]

    def __init__(self, settings: RootFinderSettings) -> None:
        self._settings = settings

    def locate(
        self,
        function: Callable[[float], float],
        low: float,
        high: float,
    ) -> float | None:
        lower_value = function(low)
        upper_value = function(high) if self._settings.relative else 0.0
        tolerance = self._settings.value_tolerance * self._settings.value_scale(
            lower_value,
            upper_value,
        )
        mid = _MIDPOINT * (low + high)
        for _ in range(self._settings.max_iterations):
            mid = _MIDPOINT * (low + high)
            value = function(mid)
            narrow = (high - low) < self._settings.fraction_tolerance
            if abs(value) < tolerance or narrow:
                return mid
            if lower_value * value < 0:
                high = mid
            else:
                low = mid
                lower_value = value
        return mid


class Secant(ScalarRootFinder):
    """Faster secant iteration, clamped to the initial bracket for safety.

    ``x <- x_1 - f_1 (x_1 - x_0) / (f_1 - f_0)``, clamped to ``[low, high]``; stops
    on ``|f| < value_tolerance`` or a chord flatter than ``1e-14`` (both times the
    larger end value when the settings are relative). Not bracketing:
    it can settle on the clamp instead of a root.

    References:
    - Kuznetsov (1998), Section 10.2.2 (secant location along the curve).
    - Allgower and Georg (1990), Section 9.2, eq. (9.2.1).
    """

    __slots__ = ["_settings"]

    def __init__(self, settings: RootFinderSettings) -> None:
        self._settings = settings

    def locate(
        self,
        function: Callable[[float], float],
        low: float,
        high: float,
    ) -> float | None:
        left, right = low, high
        value_left, value_right = function(left), function(right)
        scale = self._settings.value_scale(value_left, value_right)
        tolerance = self._settings.value_tolerance * scale
        current = right
        for _ in range(self._settings.max_iterations):
            if abs(value_right - value_left) < _DEGENERATE_SLOPE * scale:
                break
            current = right - value_right * (right - left) / (value_right - value_left)
            current = min(max(current, low), high)
            value = function(current)
            if abs(value) < tolerance:
                return current
            left, value_left = right, value_right
            right, value_right = current, value
        return current
