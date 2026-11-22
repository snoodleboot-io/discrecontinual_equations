from typing import ClassVar

import numpy as np

from discrecontinual_equations.function.function import Function


class StochasticFunction(Function):
    """Function class for stochastic differential equations.

    ``eval`` is the drift and ``diffusion`` the noise amplitude, one point at a
    time. ``diffusion`` may return a vector of one coefficient per component
    (diagonal noise) or an ``n x m`` nested list (a full matrix with ``m``
    independent drivers); the solvers read the shape (DEQ-28).

    The two ``*_batch`` methods evaluate many points at once for the ensemble
    solver (DEQ-32). Their defaults loop over the points through ``eval`` and
    ``diffusion``, so every existing function works in an ensemble unchanged;
    a function that overrides them with array arithmetic is what makes an
    ensemble of ten thousand paths cost what one path used to.

    ``diagonal_noise`` is a promise a subclass may make by setting it to True:
    that ``diffusion`` returns the vector form and that ``b_i`` depends on
    ``x_i`` (and ``t``) alone - Kloeden and Platen's diagonal noise (Section
    10.3). The solvers then read every ``d b_i / d x_i`` from two evaluations,
    moving all components at once, instead of ``2 n`` moving one at a time, so
    a Milstein step costs ``O(C_b)`` rather than ``O(n C_b)`` (DEQ-48). The
    promise is checked once per solve, at the first state it is used (the
    initial state): the full Jacobian of ``b`` there, ``2 n`` evaluations. An
    off-diagonal slope beyond the commutativity tolerance of the largest one
    withdraws it with a ``UserWarning`` naming the function, and that solve
    runs as if nothing had been declared. Only that one state is looked at, so
    a ``b_i`` whose dependence on another component vanishes there (``x_0 *
    x_1`` started at ``x_0 = 0``, say) still makes a wrong step, not an error.
    Without the declaration the solvers find commuting diagonal noise
    themselves.
    """

    diagonal_noise: ClassVar[bool] = False

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        """Evaluate the drift term μ(x, t) for SDEs."""
        raise NotImplementedError

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,
    ) -> list[float] | list[list[float]]:
        """Evaluate the diffusion term σ(x, t) for SDEs."""
        raise NotImplementedError

    def eval_batch(self, points: np.ndarray, time: float | None = None) -> np.ndarray:
        """The drift at every row of ``points`` (``(paths, n)``), as ``(paths, n)``."""
        return np.array(
            [self.eval(point=list(point), time=time) for point in points],
            dtype=float,
        )

    def diffusion_batch(
        self,
        points: np.ndarray,
        time: float | None = None,
    ) -> np.ndarray:
        """The diffusion at every row of ``points``.

        ``(paths, n)`` for the vector form, ``(paths, n, m)`` for the matrix form:
        one more leading axis than ``diffusion`` either way.
        """
        return np.array(
            [self.diffusion(point=list(point), time=time) for point in points],
            dtype=float,
        )
