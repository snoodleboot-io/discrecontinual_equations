"""The Ito drift and diffusion of an equation, however the equation was written.

Every scheme in the ``srk*`` packages is derived from an Ito-Taylor expansion, so
it integrates an Ito equation ``dX = a(X, t) dt + b(X, t) dW``. An equation the
user wrote in the Stratonovich sense, ``dX = a dt + b o dW``, is the Ito equation
with drift ``a + (1/2) b b'``: the Stratonovich integral sees half the quadratic
variation that the Ito integral attributes to the drift. The sign matters and is
easy to get backwards - for Stratonovich geometric Brownian motion
``dX = mu X dt + sigma X o dW`` the exact solution is ``X0 exp(mu t + sigma W)``,
and only the Ito drift ``(mu + sigma^2 / 2) X`` reproduces it. (This is the
convention :mod:`discrecontinual_equations.continuation.stochastic` uses as well.)

With the diagonal noise this library's interface implies (one diffusion
coefficient per component, each with its own Wiener process) the correction for
component ``i`` involves only its own coefficient, ``(1/2) b_i d b_i / d x_i``,
since the cross derivatives multiply Wiener processes that are independent and
so contribute no quadratic variation. The derivative is taken by a central
difference, which is what the rest of the package does rather than ask the user
for one more function.
"""

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation


class ItoCoefficients:
    """Evaluate ``a`` and ``b`` of the Ito form of an equation at a point."""

    __slots__ = ["_equation", "_slope_step", "_stratonovich"]

    def __init__(
        self,
        equation: DifferentialEquation,
        calculus: str,
        slope_step: float = 1.0e-6,
    ) -> None:
        self._equation = equation
        self._stratonovich = calculus == "stratonovich"
        self._slope_step = slope_step

    def diffusion(self, point: np.ndarray, time: float) -> np.ndarray:
        """``b(point, time)``, one coefficient per component."""
        return np.asarray(
            self._equation.derivative.diffusion(point=point.tolist(), time=time),
            dtype=float,
        )

    def drift(self, point: np.ndarray, time: float) -> np.ndarray:
        """``a(point, time)`` in the Ito sense."""
        drift = np.asarray(
            self._equation.derivative.eval(point=point.tolist(), time=time),
            dtype=float,
        )
        if not self._stratonovich:
            return drift
        return drift + 0.5 * self.diffusion(point, time) * self._diffusion_slope(
            point,
            time,
        )

    def _diffusion_slope(self, point: np.ndarray, time: float) -> np.ndarray:
        """``d b_i / d x_i`` for every component, by central differences."""
        slope = np.empty(len(point))
        for index in range(len(point)):
            forward = point.copy()
            backward = point.copy()
            forward[index] += self._slope_step
            backward[index] -= self._slope_step
            slope[index] = (
                self.diffusion(forward, time)[index]
                - self.diffusion(backward, time)[index]
            ) / (2.0 * self._slope_step)
        return slope
