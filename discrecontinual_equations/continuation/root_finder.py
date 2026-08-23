"""Scalar root-finding on a bracketed test function, as an injected strategy.

Localizing a bifurcation reduces to finding where a scalar test function crosses
zero along the segment between two accepted points. Bisection, secant, and Brent
are interchangeable ways to do that, so the method is injected into the localizer
rather than hardcoded.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

_MIDPOINT = 0.5
_DEGENERATE_SLOPE = 1e-14


class RootFinderSettings:
    """Stopping criteria shared by scalar root-finders."""

    __slots__ = ["_fraction_tolerance", "_max_iterations", "_value_tolerance"]

    def __init__(
        self,
        value_tolerance: float,
        fraction_tolerance: float,
        max_iterations: int,
    ) -> None:
        self._value_tolerance = value_tolerance
        self._fraction_tolerance = fraction_tolerance
        self._max_iterations = max_iterations

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


class ScalarRootFinder(ABC):
    """Find ``x`` in ``[low, high]`` where ``function(x) = 0``."""

    @abstractmethod
    def locate(
        self,
        function: Callable[[float], float],
        low: float,
        high: float,
    ) -> float | None:
        """Return a root location, or ``None`` if none could be found."""
        raise NotImplementedError


class Bisection(ScalarRootFinder):
    """Robust bracketing bisection."""

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
        mid = _MIDPOINT * (low + high)
        for _ in range(self._settings.max_iterations):
            mid = _MIDPOINT * (low + high)
            value = function(mid)
            narrow = (high - low) < self._settings.fraction_tolerance
            if abs(value) < self._settings.value_tolerance or narrow:
                return mid
            if lower_value * value < 0:
                high = mid
            else:
                low = mid
                lower_value = value
        return mid


class Secant(ScalarRootFinder):
    """Faster secant iteration, clamped to the initial bracket for safety."""

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
        current = right
        for _ in range(self._settings.max_iterations):
            if abs(value_right - value_left) < _DEGENERATE_SLOPE:
                break
            current = right - value_right * (right - left) / (value_right - value_left)
            current = min(max(current, low), high)
            value = function(current)
            if abs(value) < self._settings.value_tolerance:
                return current
            left, value_left = right, value_right
            right, value_right = current, value
        return current
