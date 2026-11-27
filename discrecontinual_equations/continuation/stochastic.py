"""Stochastic bifurcation of one-dimensional systems, in closed form.

For ``dx = f(x) dt + g(x) dW`` there are two distinct notions of bifurcation, and
this module treats them as different objects:

* **Phenomenological (P) bifurcation** - a qualitative change in the stationary
  probability density ``p_s(x)`` (for example a mode moving off the origin). The
  stationary Fokker-Planck density in one dimension is known in closed form,
  ``p_s(x) = (C / g^2) exp(integral 2 f_ito / g^2 dx)``;
  :class:`StationaryDensity` evaluates it on a grid and :class:`DensityModes`
  reports its shape.
* **Dynamical (D) bifurcation** - a sign change of the top Lyapunov exponent of
  the linearised flow about a reference state. In one dimension that exponent is
  ``f'(x0) - (1/2) g'(x0)^2`` (Ito); :class:`LyapunovExponent` returns it.

The Ito/Stratonovich distinction is explicit through :class:`NoiseConvention`,
since it shifts the effective drift and therefore both thresholds.

Both closed forms stop at one dimension, and for different reasons. The density
formula is an antiderivative, and a vector field has one only when it is a gradient,
which a planar drift carrying a rotation is not; :mod:`.fokker_planck` therefore
solves the stationary equation on a grid instead. The exponent formula is the
one-dimensional case of a spectrum, which above one dimension has to be accumulated
along an orbit; :mod:`.lyapunov` does that. Locating either threshold in the parameter
is :mod:`.stochastic_threshold`. :class:`NoiseConvention` is reused unchanged by all
three, because the shift it describes is the same shift in any dimension.

Example: the pitchfork with multiplicative noise, ``f = a x - x^3``, ``g = s x``.
Under Stratonovich the exponent at the origin is ``a`` and the density
``p_s ~ x^(2a/s^2 - 1) exp(-x^2/s^2)`` leaves the origin at ``a = s^2 / 2``; under
Ito the exponent is ``a - s^2 / 2`` and the density leaves the origin at
``a = s^2``. In both conventions D comes first and P a distance ``s^2 / 2`` later,
so the two notions genuinely differ. (A linear SDE ``dx = a x dt + s x dW`` is
geometric Brownian motion, ``x(t) = x(0) exp((a - s^2/2) t + s W(t))`` under Ito,
which is where ``a - s^2 / 2`` comes from.)

Limits. Derivatives are central differences with a fixed absolute step of
``1e-6``, so a drift or noise that varies on a much smaller scale, or a reference
state of very large magnitude, is out of reach. The density is integrated by the
trapezoid rule on the caller's grid and normalised over that grid only: mass
outside it is lost, and a grid that ends where ``g = 0`` makes the density
singular, which is why the examples start the grid just off the origin. The
exponent is the linearisation about a *fixed* reference state, which must be an
equilibrium of the noisy flow (``f(x0) = g(x0) = 0``) for it to be the top
Lyapunov exponent of the invariant measure there.

References:
    - [Arnold1998] Arnold, L. "Random Dynamical Systems". Springer Monographs in
      Mathematics, Springer, 1998. doi:10.1007/978-3-662-12878-7. Ch. 9,
      "Bifurcation Theory" (pp. 465-531): phenomenological (P) and dynamical (D)
      bifurcation, and the explicitly solvable one-dimensional pitchfork.
    - [KloedenPlaten1992] Kloeden, P. E. and Platen, E. "Numerical Solution of
      Stochastic Differential Equations". Springer, 1992.
      doi:10.1007/978-3-662-12616-5. Ch. 4, "Stochastic Differential
      Equations" (pp. 103-160): Stratonovich equations and the drift correction
      ``(1/2) g g'`` to their Ito form. Chapter checked against the publisher's
      record only; no section or equation number is cited.
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
    """Map a stochastic differential equation to its Ito drift.

    The Ito and Stratonovich readings of ``dx = f dt + g dW`` are different
    processes; the Stratonovich one equals the Ito one with drift
    ``f + (1/2) g g'``. Every threshold in this package moves with that shift.

    References:
        - [KloedenPlaten1992] Springer, 1992, Ch. 4.
    """

    @abstractmethod
    def drift_correction(self, diffusion: float, diffusion_slope: float) -> float:
        """Amount added to the drift to obtain the equivalent Ito drift.

        Args:
            diffusion: ``g(x)``.
            diffusion_slope: ``g'(x)``.

        Returns:
            The drift correction at ``x``.
        """
        raise NotImplementedError


class Ito(NoiseConvention):
    """The Ito interpretation; the drift is already the Ito drift."""

    def drift_correction(
        self, diffusion: float, diffusion_slope: float  # noqa: ARG002
    ) -> float:
        return 0.0


class Stratonovich(NoiseConvention):
    """The Stratonovich interpretation; adds the spurious drift ``g g' / 2``."""

    def drift_correction(self, diffusion: float, diffusion_slope: float) -> float:
        return _STRATONOVICH_FACTOR * diffusion * diffusion_slope


class StationaryDensity:
    """The stationary Fokker-Planck density of a one-dimensional system.

    In one dimension the zero-flux stationary density is
    ``p_s(x) = (C / g^2) exp(integral 2 f_ito / g^2 dx)``, with ``f_ito`` the drift
    after the convention's correction. The integral is a cumulative trapezoid sum
    over ``grid`` and ``C`` normalises the density over the grid.

    Args:
        drift: ``f``.
        diffusion: ``g``; must not vanish on the grid.
        convention: How to read ``g dW``.
        grid: Increasing points on which to evaluate the density.

    References:
        - [Arnold1998] Springer, 1998, Ch. 9: the stationary density whose change
          of shape is the P-bifurcation.
    """

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
        """Return the grid and the normalised stationary density on it.

        Returns:
            The grid and the density on it, normalised to unit trapezoid mass.
        """
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
    """The top Lyapunov exponent of the linearised one-dimensional flow.

    About an equilibrium ``x0`` the linearisation is ``dv = f_ito'(x0) v dt +
    g'(x0) v dW``, geometric Brownian motion, whose exponent is
    ``f_ito'(x0) - g'(x0)^2 / 2``. A sign change of it in a parameter is the
    D-bifurcation.

    Args:
        drift: ``f``.
        diffusion: ``g``.
        convention: How to read ``g dW``.

    References:
        - [Arnold1998] Springer, 1998, Ch. 9: the D-bifurcation.
    """

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
        """Return ``f'(x0) - g'(x0)^2 / 2`` (Ito) about the reference state.

        Args:
            reference: The state ``x0``.

        Returns:
            The exponent, with ``f'`` the Ito drift's slope.
        """
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
