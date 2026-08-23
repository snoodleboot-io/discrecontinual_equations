"""Stochastic bifurcation of one-dimensional systems.

For ``dx = f(x) dt + g(x) dW`` there are two distinct notions of bifurcation, and
this module treats them as different objects:

* **Phenomenological (P) bifurcation** - a qualitative change in the stationary
  probability density ``p_s(x)`` (for example a mode moving off the origin). The
  stationary Fokker-Planck density in one dimension is known in closed form,
  ``p_s(x) = (C / g^2) exp(integral 2 f_ito / g^2 dx)`` on a grid via
  :class:`StationaryDensity`,
  evaluates it on a grid and :class:`DensityModes` reports its shape.
* **Dynamical (D) bifurcation** - a sign change of the top Lyapunov exponent of
  the linearised flow about a reference state. In one dimension that exponent is
  ``f'(x0) - (1/2) g'(x0)^2`` (Ito); :class:`LyapunovExponent` returns it.

The Ito/Stratonovich distinction is explicit through :class:`NoiseConvention`,
since it shifts the effective drift and therefore both thresholds.
"""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.function.function import Function

_STRATONOVICH_FACTOR = 0.5
_SLOPE_STEP = 1.0e-6


def _value(function: Function, x: float) -> float:
    return float(function.eval(point=[x], time=None)[0])


def _slope(function: Function, x: float) -> float:
    forward = _value(function, x + _SLOPE_STEP)
    backward = _value(function, x - _SLOPE_STEP)
    return (forward - backward) / (2.0 * _SLOPE_STEP)


class NoiseConvention(ABC):
    """Map a stochastic differential equation to its Ito drift."""

    @abstractmethod
    def drift_correction(self, diffusion: float, diffusion_slope: float) -> float:
        """Amount added to the drift to obtain the equivalent Ito drift."""
        raise NotImplementedError


class Ito(NoiseConvention):
    """The Ito interpretation; the drift is already the Ito drift."""

    def drift_correction(self, diffusion: float, diffusion_slope: float) -> float:  # noqa: ARG002
        return 0.0


class Stratonovich(NoiseConvention):
    """The Stratonovich interpretation; adds the spurious drift ``g g' / 2``."""

    def drift_correction(self, diffusion: float, diffusion_slope: float) -> float:
        return _STRATONOVICH_FACTOR * diffusion * diffusion_slope


class StationaryDensity:
    """The stationary Fokker-Planck density of a one-dimensional system."""

    __slots__ = ["_convention", "_diffusion", "_drift", "_grid"]

    def __init__(
        self,
        drift: Function,
        diffusion: Function,
        convention: NoiseConvention,
        grid: np.ndarray,
    ) -> None:
        self._drift = drift
        self._diffusion = diffusion
        self._convention = convention
        self._grid = grid

    def evaluate(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the grid and the normalised stationary density on it."""
        integrand = np.array([self._integrand(x) for x in self._grid])
        potential = np.concatenate(
            [[0.0], np.cumsum(_trapezoid(self._grid, integrand))],
        )
        squared = np.array([self._diffusion_squared(x) for x in self._grid])
        density = np.exp(potential - potential.max()) / squared
        area = float(np.sum(_trapezoid(self._grid, density)))
        return self._grid, density / area

    def _integrand(self, x: float) -> float:
        drift = _value(self._drift, x)
        diffusion = _value(self._diffusion, x)
        correction = self._convention.drift_correction(
            diffusion,
            _slope(self._diffusion, x),
        )
        return 2.0 * (drift + correction) / (diffusion * diffusion)

    def _diffusion_squared(self, x: float) -> float:
        diffusion = _value(self._diffusion, x)
        return diffusion * diffusion


class DensityModes:
    """Describe the shape of a stationary density: its interior local maxima."""

    __slots__ = ["_density", "_grid"]

    def __init__(self, grid: np.ndarray, density: np.ndarray) -> None:
        self._grid = grid
        self._density = density

    def interior_maxima(self) -> list[float]:
        """Locations of interior local maxima of the density."""
        return [
            float(self._grid[i])
            for i in range(1, len(self._density) - 1)
            if self._density[i] > self._density[i - 1]
            and self._density[i] >= self._density[i + 1]
        ]

    def dominant_mode(self) -> float:
        """Location of the global maximum of the density."""
        return float(self._grid[int(np.argmax(self._density))])


class LyapunovExponent:
    """The top Lyapunov exponent of the linearised one-dimensional flow."""

    __slots__ = ["_convention", "_diffusion", "_drift"]

    def __init__(
        self,
        drift: Function,
        diffusion: Function,
        convention: NoiseConvention,
    ) -> None:
        self._drift = drift
        self._diffusion = diffusion
        self._convention = convention

    def at(self, reference: float) -> float:
        """Return ``f'(x0) - g'(x0)^2 / 2`` (Ito) about the reference state."""
        diffusion_slope = _slope(self._diffusion, reference)
        ito_drift_slope = _slope(self._drift, reference) + self._correction_slope(
            reference,
        )
        return (
            ito_drift_slope - _STRATONOVICH_FACTOR * diffusion_slope * diffusion_slope
        )

    def _correction_slope(self, reference: float) -> float:
        forward = self._convention.drift_correction(
            _value(self._diffusion, reference + _SLOPE_STEP),
            _slope(self._diffusion, reference + _SLOPE_STEP),
        )
        backward = self._convention.drift_correction(
            _value(self._diffusion, reference - _SLOPE_STEP),
            _slope(self._diffusion, reference - _SLOPE_STEP),
        )
        return (forward - backward) / (2.0 * _SLOPE_STEP)


def _trapezoid(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    widths = np.diff(grid)
    return widths * 0.5 * (values[:-1] + values[1:])
