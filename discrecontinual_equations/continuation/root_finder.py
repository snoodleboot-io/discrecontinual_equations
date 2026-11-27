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
cannot be met by a fast one.

With ``relative=True`` the stop is on the location, not on the value:
``value_tolerance`` is read as a fraction of the initial bracket ``[low, high]``,
and a finder returns ``x`` only once a sign change is known to lie within
``value_tolerance (high - low)`` of it. That is unchanged when ``f`` is multiplied
by any constant, and it holds for any continuous ``f``. A stop on the value
measured against the end values, ``|f(x)| < value_tolerance max(|f(low)|,
|f(high)|)`` (DEQ-146), is not sound: it is the location tolerance only where
``f`` is close to linear across the whole bracket. On ``sinh(40 (x - 0.37))`` over
``[0, 1]`` the ends are ``-1.3e6`` and ``4.4e10``, the first midpoint's value of 90
passes, and bisection stopped 0.13 from the root (DEQ-179). AUTO stops its search
for a special point on the arclength correction relative to the step
(``EPSS``, Doedel and Oldeman 2021) for the same reason. Bisection meets the
location stop by halving its bracket; the secant method, which keeps no bracket,
confirms a converged iterate by a sign change across ``x -/+ value_tolerance
(high - low)`` and falls back to bisection when it cannot.

The ends the finder evaluates must be finite. An infinite or undefined end value
has no sign to bracket with and no size to measure against (with the DEQ-146
value stop a ``f(high)`` of ``inf`` passed every midpoint), so it is refused with
a :class:`ValueError`.

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
- Doedel, E. J. and Oldeman, B. E. "AUTO-07P: Continuation and Bifurcation
  Software for Ordinary Differential Equations", manual, Concordia University,
  August 2021 (https://github.com/auto-07p/auto-07p, ``doc/auto.tex``), section
  "Tolerances", constants ``EPSS`` ("relative arclength convergence criterion for
  the detection of special solutions") and ``ITMX``; the stop
  ``|ds| / (1 + sqrt(|DS DSMAX|)) < EPSS`` is subroutine ``LCSPAE`` of
  ``src/ae.f90``.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable

_MIDPOINT = 0.5
_TWO_SIDED = 2.0
_DEGENERATE_SLOPE = 1e-14


class RootFinderSettings:
    """Stopping criteria shared by scalar root-finders.

    Args:
        value_tolerance: Stop once ``|function(x)|`` is below this.
        fraction_tolerance: Stop once the bracket is narrower than this (used by
            bisection).
        max_iterations: Hard cap on iterations.
        relative: Stop on the location instead of the value: return ``x`` once a
            sign change is known to lie within ``value_tolerance`` times the
            initial bracket's width of it, so the stop does not depend on the
            units of the function (see the module docstring).
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
        """Whether ``value_tolerance`` is a fraction of the bracket's width."""
        return self._relative

    def value_scale(self, lower_value: float, upper_value: float) -> float:
        """The size the secant's flat-chord test is measured against.

        Args:
            lower_value: ``function(low)``.
            upper_value: ``function(high)``.

        Returns:
            ``max(|lower_value|, |upper_value|)`` when :attr:`relative`, else 1.
        """
        if not self._relative:
            return 1.0
        return max(abs(lower_value), abs(upper_value))

    def location_tolerance(self, low: float, high: float) -> float:
        """How close to a sign change a relative stop returns its point.

        Args:
            low: Left end of the initial bracket.
            high: Right end of the initial bracket.

        Returns:
            ``value_tolerance (high - low)``.
        """
        return self._value_tolerance * (high - low)


def _finite_end(value: float, end: float) -> float:
    """Return ``value``, or refuse an end value that is infinite or undefined."""
    if not math.isfinite(value):
        message = (
            f"the root-finder's end value at {end!r} is {value!r}: a bracket needs "
            "finite end values (DEQ-179)"
        )
        raise ValueError(message)
    return value


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

    Keeps the half whose ends differ in sign. With absolute settings it returns
    the midpoint once its value is below ``value_tolerance`` or the bracket is
    narrower than ``fraction_tolerance``. With relative settings the value is
    used only for its sign: the midpoint is returned once the bracket is narrower
    than ``2 value_tolerance (high - low)``, so it lies within ``value_tolerance
    (high - low)`` of the sign change, or when its value is exactly zero. A
    relative stop costs one more evaluation, at ``high``. If the ends do not
    differ in sign the bracket slides to ``high`` and the result is not a root.

    Raises:
        ValueError: If an evaluated end value is not finite.

    References:
    - Kuznetsov (1998), Section 10.2.2.
    - Doedel and Oldeman (2021), AUTO-07P manual, ``EPSS`` (a relative stop on
      the location rather than the value).
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
        settings = self._settings
        lower_value = _finite_end(function(low), low)
        if settings.relative:
            _finite_end(function(high), high)
            width = _TWO_SIDED * settings.location_tolerance(low, high)
            tolerance = 0.0
        else:
            width = 0.0
            tolerance = settings.value_tolerance
        mid = _MIDPOINT * (low + high)
        for _ in range(settings.max_iterations):
            mid = _MIDPOINT * (low + high)
            value = function(mid)
            narrow = (high - low) < settings.fraction_tolerance
            # A relative stop waits for the bracket itself: a small value says
            # nothing about the distance to the root unless f is linear (DEQ-179).
            located = (high - low) <= width or value == 0.0
            if abs(value) < tolerance or narrow or located:
                return mid
            if lower_value * value < 0:
                high = mid
            else:
                low = mid
                lower_value = value
        return mid


class Secant(ScalarRootFinder):
    """Faster secant iteration, clamped to the initial bracket for safety.

    ``x <- x_1 - f_1 (x_1 - x_0) / (f_1 - f_0)``, clamped to ``[low, high]``. With
    absolute settings it stops on ``|f| < value_tolerance`` or a chord flatter
    than ``1e-14``, and, not bracketing, it can settle on the clamp instead of a
    root. With relative settings a step shorter than ``value_tolerance (high -
    low)`` only proposes a stop: it is accepted once ``f`` changes sign within
    that distance of the iterate (one or two more evaluations), and if no
    iterate is confirmed - the chord went flat, the iterate stuck on the clamp,
    or the iterations ran out - the bracket is handed to :class:`Bisection`.

    Raises:
        ValueError: If an end value is not finite.

    References:
    - Kuznetsov (1998), Section 10.2.2 (secant location along the curve).
    - Allgower and Georg (1990), Section 9.2, eq. (9.2.1).
    - Doedel and Oldeman (2021), AUTO-07P manual, ``EPSS`` (a relative stop on
      the location rather than the value).
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
        settings = self._settings
        left, right = low, high
        value_left = _finite_end(function(left), left)
        value_right = _finite_end(function(right), right)
        scale = settings.value_scale(value_left, value_right)
        tolerance = 0.0 if settings.relative else settings.value_tolerance * scale
        distance = settings.location_tolerance(low, high)
        current = right
        for _ in range(settings.max_iterations):
            if abs(value_right - value_left) < _DEGENERATE_SLOPE * scale:
                break
            current = right - value_right * (right - left) / (value_right - value_left)
            current = min(max(current, low), high)
            value = function(current)
            if abs(value) < tolerance or (settings.relative and value == 0.0):
                return current
            if settings.relative and abs(current - right) <= distance:
                # A short step is not a root until f changes sign nearby (DEQ-179).
                if _changes_sign_near(function, current, value, (low, high), distance):
                    return current
            left, value_left = right, value_right
            right, value_right = current, value
        if settings.relative:
            return Bisection(settings).locate(function, low, high)
        return current


def _changes_sign_near(
    function: Callable[[float], float],
    point: float,
    value: float,
    bracket: tuple[float, float],
    distance: float,
) -> bool:
    """Whether ``function`` changes sign within ``distance`` of ``point``."""
    low, high = bracket
    for neighbour in (min(point + distance, high), max(point - distance, low)):
        if neighbour != point and value * function(neighbour) <= 0.0:
            return True
    return False
