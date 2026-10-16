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

The diffusion comes in two shapes, and the shape is the declaration (DEQ-28). A
vector of ``n`` coefficients is diagonal noise: component ``i`` is driven by its
own Wiener process, and the increments act on the state as ``b * dW``. An
``n x m`` matrix ``G`` is general noise with ``m`` independent drivers, and the
increments act as ``G @ dW`` with ``dW`` of length ``m`` - a rotation, a shear,
one driver shared by several components, or more drivers than components. The
vector form is the ``m = n``, ``G = diag(b)`` special case, and it is kept as its
own code path rather than routed through the matrix one, so that a seeded run
gives the same numbers it always has. :meth:`ItoCoefficients.drivers` is the
channel count a Wiener source must supply, and :meth:`ItoCoefficients.spread`
applies whichever form the equation returned.

With diagonal noise the correction for component ``i`` involves only its own
coefficient, ``(1/2) b_i d b_i / d x_i``, since the cross derivatives multiply
Wiener processes that are independent and so contribute no quadratic variation.
With a matrix it is ``(1/2) sum_k (dG_k / dx) G_k`` over the columns - the same
algebra :mod:`discrecontinual_equations.continuation.noise` uses on the analysis
side - and that reduces to the diagonal form when ``G = diag(b)``. The derivatives
are taken by central differences, which is what the rest of the package does
rather than ask the user for one more function; :meth:`ItoCoefficients.column_jacobians`
is the same differences arranged as the Milstein scheme needs them.
"""

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation

_MATRIX = 2


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
        """``b`` as a vector (diagonal noise) or ``G`` as an ``n x m`` matrix."""
        return np.asarray(
            self._equation.derivative.diffusion(point=point.tolist(), time=time),
            dtype=float,
        )

    @staticmethod
    def is_matrix(diffusion: np.ndarray) -> bool:
        """Whether a value of :meth:`diffusion` is the matrix form."""
        return diffusion.ndim == _MATRIX

    def drivers(self, point: np.ndarray, time: float) -> int:
        """How many independent Wiener processes drive the equation: ``m``."""
        diffusion = self.diffusion(point, time)
        return diffusion.shape[1] if self.is_matrix(diffusion) else diffusion.size

    @staticmethod
    def spread(diffusion: np.ndarray, increments: np.ndarray) -> np.ndarray:
        """What the Wiener increments do to the state: ``b * dW`` or ``G @ dW``."""
        if ItoCoefficients.is_matrix(diffusion):
            return diffusion @ increments
        return diffusion * increments

    def written_drift(self, point: np.ndarray, time: float) -> np.ndarray:
        """``a(point, time)`` as the equation wrote it, in whichever calculus."""
        return np.asarray(
            self._equation.derivative.eval(point=point.tolist(), time=time),
            dtype=float,
        )

    def drift(self, point: np.ndarray, time: float) -> np.ndarray:
        """``a(point, time)`` in the Ito sense."""
        drift = self.written_drift(point, time)
        if not self._stratonovich:
            return drift
        diffusion = self.diffusion(point, time)
        if self.is_matrix(diffusion):
            # (1/2) sum_k (dG_k/dx) G_k: jacobians[k, i, j] = d G[i, k] / d x_j.
            jacobians = self.column_jacobians(point, time)
            return drift + 0.5 * np.einsum("kij,jk->i", jacobians, diffusion)
        return drift + 0.5 * diffusion * self._diffusion_slope(point, time)

    def column_jacobians(self, point: np.ndarray, time: float) -> np.ndarray:
        """``dG_k / dx`` for every driver ``k``, as an ``(m, n, n)`` array.

        For the vector form the columns are ``b_k e_k``, so column ``k``'s Jacobian
        has one nonzero row, ``d b_k / dx`` in row ``k``. Central differences in
        every state direction, ``2 n`` diffusion evaluations.
        """
        diffusion = self.diffusion(point, time)
        dimension = point.size
        matrix = self.is_matrix(diffusion)
        drivers = diffusion.shape[1] if matrix else diffusion.size
        jacobians = np.zeros((drivers, dimension, dimension))
        for j in range(dimension):
            forward = point.copy()
            backward = point.copy()
            forward[j] += self._slope_step
            backward[j] -= self._slope_step
            difference = (
                self.diffusion(forward, time) - self.diffusion(backward, time)
            ) / (2.0 * self._slope_step)
            if matrix:
                jacobians[:, :, j] = difference.T
            else:
                jacobians[np.arange(drivers), np.arange(drivers), j] = difference
        return jacobians

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
