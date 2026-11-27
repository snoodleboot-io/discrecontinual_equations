"""Locate a homoclinic orbit by shooting the unstable manifold.

A homoclinic orbit is a codimension-one phenomenon: it exists only at isolated
parameter values. To find one, integrate the saddle's one-dimensional unstable
manifold forward and measure how far the returning trajectory misses the saddle,
projected onto the unstable direction (a signed *return gap*). The gap changes sign
as a parameter sweeps through the homoclinic value, so a bracketing search locates it.

The trajectory this produces is a natural seed for the projection boundary-value
problem in :mod:`.connecting_orbit`, which refines it to high accuracy: shooting
finds the connection, the boundary-value problem sharpens it.

The method, step by step, at one parameter value:

1. refine the saddle from ``Departure.equilibrium_seed`` (Newton, via
   :func:`.region_analysis.find_equilibria`);
2. take the eigenvector of the eigenvalue with the largest real part, oriented to
   agree with ``Departure.direction``, and start at
   ``saddle + offset * direction`` - the explicit start ``u(tau_-) = xi + eps v``
   of Doedel et al. (2008, eq. 6);
3. integrate with fixed-step classical RK4 for ``horizon`` time units;
4. once the orbit has been further than ``departure_radius`` from the saddle,
   keep its closest approach inside ``0.4`` of it, and return that point's
   component along the unstable direction.

Limits:

* the saddle must have a one-dimensional unstable manifold with a real leading
  eigenvalue. Nothing checks this; with a complex leading pair the real part of
  its eigenvector is used and the gap has no meaning;
* ``locate`` assumes the gap changes sign in ``[lower, upper]``. It does not
  check, and always returns a point of the bracket after 32 halvings. The gap
  is NaN where there is no equilibrium or no return within ``horizon`` - which,
  away from the homoclinic value, is common: the manifold settles on another
  attractor or escapes. A NaN never counts as a sign change: when the gap at
  ``lower`` or at a midpoint is NaN, ``lower`` moves to that point and then on to
  the next midpoint without a test. If that skips the root, the bracket no longer
  changes sign and the result is ``upper``. Check ``gap`` at both ends first;
* the radii are absolute distances, so a system whose orbit is much smaller or
  larger than order one needs its own ``ReturnSettings``;
* ``gap`` and ``trace`` set parameter values on the function and leave them set.

References:
- Doedel, E. J., Kooi, B. W., Kuznetsov, Yu. A. and van Voorn, G. A. K.
  "Continuation of connecting orbits in 3D-ODEs: (I) Point-to-cycle
  connections", Int. J. Bifurcation and Chaos 18 (2008) 1889-1903,
  arXiv:0706.1688, doi:10.1142/S0218127408021439. Eq. (6), the start along the
  unstable eigenvector.
- Beyn, W.-J. "The numerical computation of connecting orbits in dynamical
  systems", IMA J. Numer. Anal. 10 (1990) 379-405, doi:10.1093/imanum/10.3.379.
  The boundary-value problem the located orbit seeds.
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
# Subintervals ``trace`` samples its bracket on, to find a sign change of finite
# gaps for ``locate``.
_BRACKET_SAMPLES = 8


class ReturnSettings:
    """Controls for integrating the unstable manifold and detecting its return.

    Args:
        dt: The fixed RK4 step.
        horizon: How long to integrate before giving up on a return.
        departure_radius: The distance from the saddle the orbit must exceed
            before a close approach counts as a return.
        offset: The initial displacement along the unstable eigenvector.
    """

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
    """Where and in which sense to leave the saddle: a seed and a direction hint.

    Args:
        equilibrium_seed: A guess for the saddle, refined at every parameter value.
        direction: Any vector on the side of the unstable manifold to follow; the
            unstable eigenvector is flipped to have a positive component along it.
    """

    __slots__ = ["direction", "equilibrium_seed"]

    def __init__(self, equilibrium_seed: np.ndarray, direction: np.ndarray) -> None:
        self.equilibrium_seed = equilibrium_seed
        self.direction = direction


class HomoclinicShooting:
    """Find the parameter at which a saddle has a homoclinic orbit.

    Args:
        function: The vector field; its parameters are changed in place.
        parameter_index: Which parameter to search over.
        departure: The saddle seed and the branch of its unstable manifold.
        settings: Integration and return-detection controls.

    References:
        Doedel et al. (2008), eq. (6); Beyn (1990).
    """

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

        Args:
            parameter: The value to set the searched parameter to.

        Returns:
            The component, along the unstable eigenvector, of the orbit's closest
            return to the saddle; zero at a homoclinic orbit.
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
        """Bisect for the homoclinic parameter in ``[lower, upper]``.

        The return gap changes sign through the homoclinic value, so bisection
        on its sign finds it, provided the bracket holds a sign change of
        *finite* gaps. ``gap`` is NaN wherever the manifold does not return to
        the saddle's neighbourhood, and NaN compares false with everything. A
        sign test on it once moved the bracket silently: a root at 0.3 came
        back as 1.0 (DEQ-135). So both ends must give finite gaps of opposite
        sign, and a midpoint whose gap is not finite stops the search. It
        cannot say which half holds the root.

        Args:
            lower: One end of the bracket.
            upper: The other end.

        Returns:
            The midpoint of the final bracket, ``_DEFAULT_BISECTIONS`` halvings
            in; an end whose gap is exactly zero is returned as it is.

        Raises:
            ValueError: An end's gap is not finite, both ends' gaps have the
                same sign, or a midpoint's gap is not finite.

        References:
            R. L. Burden and J. D. Faires, *Numerical Analysis*, 10th ed.,
            Cengage, 2016, section 2.1, the bisection method. It needs a
            continuous function whose values at the two ends have opposite
            signs; the intermediate value theorem then puts a root between
            them. Read through the Wikipedia article "Bisection method",
            sections "The method" and "Iteration process", which cites it. The
            book itself was not fetched. See docs/references.md,
            ``Burden2016``.
        """
        low_gap = self._finite_gap(lower, "lower")
        high_gap = self._finite_gap(upper, "upper")
        if low_gap == 0.0:
            return lower
        if high_gap == 0.0:
            return upper
        if (low_gap > 0.0) == (high_gap > 0.0):
            message = (
                f"the return gap has the same sign at both ends of [{lower:g}, "
                f"{upper:g}] ({low_gap:.3g} and {high_gap:.3g}), so the bracket "
                f"holds no homoclinic parameter, or an even number of them"
            )
            raise ValueError(message)
        for _ in range(_DEFAULT_BISECTIONS):
            middle = 0.5 * (lower + upper)
            middle_gap = self.gap(middle)
            if not np.isfinite(middle_gap):
                message = (
                    f"the return gap is not finite at {middle:g}, inside the "
                    f"bracket [{lower:g}, {upper:g}]: the unstable manifold does "
                    f"not return there, so the sign of the gap cannot say which "
                    f"half holds the homoclinic parameter; narrow the bracket "
                    f"to where it returns"
                )
                raise ValueError(message)
            if (low_gap > 0.0) == (middle_gap > 0.0):
                lower, low_gap = middle, middle_gap
            else:
                upper = middle
        return 0.5 * (lower + upper)

    def _finite_gap(self, parameter: float, end: str) -> float:
        """The gap at one end of a bracket, refused if it is not finite."""
        value = self.gap(parameter)
        if not np.isfinite(value):
            message = (
                f"the return gap is not finite at the {end} end of the bracket, "
                f"{parameter:g}: the unstable manifold does not return to the "
                f"saddle there, so the end cannot anchor a sign change"
            )
            raise ValueError(message)
        return value

    def trace(
        self,
        control_index: int,
        control_values: list[float],
        guess: float,
        half_width: float,
    ) -> list[tuple[float, float]]:
        """Trace the homoclinic locus in a second parameter.

        For each value of the ``control_index`` parameter, the primary parameter
        is sought within ``half_width`` of the previous result. That window
        need not have a sign change at its ends, which can fall where the
        manifold does not return (NaN), or past a second root. So the window is
        sampled at ``_BRACKET_SAMPLES + 1`` points. The adjacent pair of finite
        gaps with opposite signs that lies nearest the previous result is the
        bracket handed to :meth:`locate`. That pair is the continuation of the
        branch being followed.

        Args:
            control_index: The parameter swept along the curve.
            control_values: Its values, in order.
            guess: The searched parameter at the first control value.
            half_width: Half the bracket around the previous result; it must
                contain the next point.

        Returns:
            The ``(control_value, located_parameter)`` pairs of the curve.

        Raises:
            ValueError: No adjacent pair of samples in a window holds a sign
                change of finite gaps. The curve has then left ``half_width``
                of the previous point, or entered a region where the manifold
                does not return.
        """
        centre = guess
        curve: list[tuple[float, float]] = []
        for value in control_values:
            self._function.parameters[control_index].value = value
            lower, upper = self._bracket(centre, half_width)
            centre = self.locate(lower, upper)
            curve.append((value, centre))
        return curve

    def _bracket(self, centre: float, half_width: float) -> tuple[float, float]:
        """The sampled sign change of finite gaps nearest ``centre``."""
        points = np.linspace(
            centre - half_width,
            centre + half_width,
            _BRACKET_SAMPLES + 1,
        )
        gaps = [self.gap(float(point)) for point in points]
        pairs = [
            (abs(0.5 * (points[k] + points[k + 1]) - centre), k)
            for k in range(_BRACKET_SAMPLES)
            if np.isfinite(gaps[k])
            and np.isfinite(gaps[k + 1])
            and (gaps[k] > 0.0) != (gaps[k + 1] > 0.0)
        ]
        if not pairs:
            sampled = ", ".join(f"{gap:.3g}" for gap in gaps)
            message = (
                f"no sign change of finite return gaps within {half_width:g} of "
                f"{centre:g} (sampled: {sampled}); the curve has left the window "
                f"or the manifold does not return there"
            )
            raise ValueError(message)
        _, nearest = min(pairs)
        return float(points[nearest]), float(points[nearest + 1])
