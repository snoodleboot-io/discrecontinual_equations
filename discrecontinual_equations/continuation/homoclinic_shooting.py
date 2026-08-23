"""Locate a homoclinic orbit by shooting the unstable manifold.

A homoclinic orbit is a codimension-one phenomenon: it exists only at isolated
parameter values. To find one, integrate the saddle's one-dimensional unstable
manifold forward and measure how far the returning trajectory misses the saddle,
projected onto the unstable direction (a signed *return gap*). The gap changes sign
as a parameter sweeps through the homoclinic value, so a bracketing search locates it.

The trajectory this produces is a natural seed for the projection boundary-value
problem in :mod:`.connecting_orbit`, which refines it to high accuracy: shooting
finds the connection, the boundary-value problem sharpens it.
"""

import numpy as np

from discrecontinual_equations.continuation.region_analysis import (
    _jacobian,
    find_equilibria,
)
from discrecontinual_equations.function.function import Function

_DEFAULT_DT = 2.0e-3
_DEFAULT_HORIZON = 60.0
_DEFAULT_DEPARTURE = 0.5
_DEFAULT_OFFSET = 1.0e-5
_DIVERGENCE = 1.0e3
_RETURN_BAND = 0.4
_DEFAULT_BISECTIONS = 32


class ReturnSettings:
    """Controls for integrating the unstable manifold and detecting its return."""

    __slots__ = ["departure_radius", "dt", "horizon", "offset"]

    def __init__(
        self,
        dt: float = _DEFAULT_DT,
        horizon: float = _DEFAULT_HORIZON,
        departure_radius: float = _DEFAULT_DEPARTURE,
        offset: float = _DEFAULT_OFFSET,
    ) -> None:
        self.dt = dt
        self.horizon = horizon
        self.departure_radius = departure_radius
        self.offset = offset


class Departure:
    """Where and in which sense to leave the saddle: a seed and a direction hint."""

    __slots__ = ["direction", "equilibrium_seed"]

    def __init__(self, equilibrium_seed: np.ndarray, direction: np.ndarray) -> None:
        self.equilibrium_seed = equilibrium_seed
        self.direction = direction


class HomoclinicShooting:
    """Find the parameter at which a saddle has a homoclinic orbit."""

    __slots__ = ["_departure", "_function", "_parameter_index", "_settings"]

    def __init__(
        self,
        function: Function,
        parameter_index: int,
        departure: Departure,
        settings: ReturnSettings | None = None,
    ) -> None:
        self._function = function
        self._parameter_index = parameter_index
        self._departure = departure
        self._settings = settings if settings is not None else ReturnSettings()

    def _field(self, state: np.ndarray) -> np.ndarray:
        return np.array(self._function.eval(point=list(state), time=None), dtype=float)

    def _step(self, state: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._field(state)
        k2 = self._field(state + 0.5 * dt * k1)
        k3 = self._field(state + 0.5 * dt * k2)
        k4 = self._field(state + dt * k3)
        return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _unstable_direction(self, jacobian: np.ndarray) -> np.ndarray:
        values, vectors = np.linalg.eig(jacobian)
        index = int(np.argmax(values.real))
        direction = vectors[:, index].real
        direction = direction / np.linalg.norm(direction)
        if direction @ self._departure.direction < 0.0:
            direction = -direction
        return direction

    def gap(self, parameter: float) -> float:
        """Signed return miss of the unstable manifold at ``parameter``.

        Returns NaN if the equilibrium cannot be found or the trajectory diverges
        before returning.
        """
        self._function.parameters[self._parameter_index].value = parameter
        equilibria = find_equilibria(self._function, [self._departure.equilibrium_seed])
        if not equilibria:
            return float("nan")
        equilibrium = equilibria[0]
        unstable = self._unstable_direction(_jacobian(self._function, equilibrium))
        state = equilibrium + self._settings.offset * unstable
        left, closest, best = False, None, np.inf
        steps = int(self._settings.horizon / self._settings.dt)
        for _ in range(steps):
            state = self._step(state, self._settings.dt)
            if np.linalg.norm(state) > _DIVERGENCE:
                break  # trajectory escaped; use the closest return already recorded
            distance = float(np.linalg.norm(state - equilibrium))
            if distance > self._settings.departure_radius:
                left = True
            if left and distance < _RETURN_BAND and distance < best:
                best, closest = distance, state.copy()
        if closest is None:
            return float("nan")
        return float((closest - equilibrium) @ unstable)

    def locate(self, lower: float, upper: float) -> float:
        """Bracketing search for the homoclinic parameter in ``[lower, upper]``."""
        low_gap = self.gap(lower)
        for _ in range(_DEFAULT_BISECTIONS):
            middle = 0.5 * (lower + upper)
            middle_gap = self.gap(middle)
            if low_gap * middle_gap <= 0.0:
                upper = middle
            else:
                lower, low_gap = middle, middle_gap
        return 0.5 * (lower + upper)

    def trace(
        self,
        control_index: int,
        control_values: list[float],
        guess: float,
        half_width: float,
    ) -> list[tuple[float, float]]:
        """Trace the homoclinic locus in a second parameter (predictor-corrector).

        For each value of the ``control_index`` parameter, locate the primary
        parameter in a bracket centred on the previous result. Returns the
        ``(control_value, located_parameter)`` pairs of the homoclinic curve.
        """
        centre = guess
        curve: list[tuple[float, float]] = []
        for value in control_values:
            self._function.parameters[control_index].value = value
            centre = self.locate(centre - half_width, centre + half_width)
            curve.append((value, centre))
        return curve
