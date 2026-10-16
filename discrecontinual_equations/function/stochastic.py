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
    """

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        """Evaluate the drift term μ(x, t) for SDEs."""
        raise NotImplementedError

    def diffusion(self, point: list[float], time: float | None = None) -> list[float]:
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
