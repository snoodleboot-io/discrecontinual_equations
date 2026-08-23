"""Find distinct roots of a vector field by deflated Newton iteration.

Ordinary Newton, restarted from many guesses, keeps rediscovering the same root.
*Deflation* multiplies the residual by a factor that blows up at every root already
found, ``M(x) = prod_i (||x - r_i||^-2 + 1)``, so Newton is repelled from known
solutions and driven toward new ones - even from a single starting point. This is
how disconnected equilibrium branches are discovered without knowing they exist.

The Newton correction is solved either directly (a finite-difference Jacobian and a
dense solve) or matrix-free (Newton-Krylov: GMRES on Jacobian-vector products, which
never forms the Jacobian and scales to large systems). The two agree on the same
roots; the matrix-free path is the one that survives growing dimension.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from discrecontinual_equations.function.function import Function

_TOLERANCE = 1.0e-9
_ITERATIONS = 100
_STEP = 1.0e-7
_DIVERGENCE = 1.0e6
_MERGE = 1.0e-5
_TINY = 1.0e-30
_ROOT_RESIDUAL = 1.0e-6
_KRYLOV_TOLERANCE = 1.0e-8


class DeflationSettings:
    """Controls for the Newton iteration."""

    __slots__ = ["iterations", "step", "tolerance"]

    def __init__(
        self,
        tolerance: float = _TOLERANCE,
        iterations: int = _ITERATIONS,
        step: float = _STEP,
    ) -> None:
        self.tolerance = tolerance
        self.iterations = iterations
        self.step = step


class DeflatedSolver:
    """Locate distinct roots of a vector field by deflated Newton iteration."""

    __slots__ = ["_function", "_matrix_free", "_settings"]

    def __init__(
        self,
        function: Function,
        settings: DeflationSettings | None = None,
        *,
        matrix_free: bool = False,
    ) -> None:
        self._function = function
        self._settings = settings if settings is not None else DeflationSettings()
        self._matrix_free = matrix_free

    def _field(self, state: np.ndarray) -> np.ndarray:
        return np.array(self._function.eval(point=list(state), time=None), dtype=float)

    def _deflation(self, state: np.ndarray, roots: list[np.ndarray]) -> float:
        factor = 1.0
        for root in roots:
            gap = state - root
            factor *= 1.0 / max(float(gap @ gap), _TINY) + 1.0
        return factor

    def _residual(self, state: np.ndarray, roots: list[np.ndarray]) -> np.ndarray:
        return self._deflation(state, roots) * self._field(state)

    def _correction(
        self,
        state: np.ndarray,
        base: np.ndarray,
        roots: list[np.ndarray],
    ) -> np.ndarray | None:
        step = self._settings.step

        def apply(vector: np.ndarray) -> np.ndarray:
            return (self._residual(state + step * vector, roots) - base) / step

        if self._matrix_free:
            size = state.size
            operator = LinearOperator((size, size), matvec=apply)
            correction, _info = gmres(operator, -base, rtol=_KRYLOV_TOLERANCE, atol=0.0)
            # Inexact Newton: a finite-difference matvec cannot reach machine tol, so
            # accept the returned iterate whenever it is finite rather than requiring
            # full GMRES convergence.
            return correction if np.all(np.isfinite(correction)) else None
        columns = [apply(unit) for unit in np.eye(state.size)]
        jacobian = np.column_stack(columns)
        try:
            return np.linalg.solve(jacobian, -base)
        except np.linalg.LinAlgError:
            return None

    def _newton(
        self,
        guess: np.ndarray,
        roots: list[np.ndarray],
    ) -> np.ndarray | None:
        state = np.asarray(guess, dtype=float).copy()
        for _ in range(self._settings.iterations):
            base = self._residual(state, roots)
            if np.linalg.norm(base) < self._settings.tolerance:
                return state
            correction = self._correction(state, base, roots)
            if correction is None:
                return None
            state = state + correction
            if np.linalg.norm(state) > _DIVERGENCE:
                return None
        return None

    def find(self, seeds: list[np.ndarray]) -> list[np.ndarray]:
        """Return the distinct roots reachable by deflating from each seed."""
        roots: list[np.ndarray] = []
        for seed in seeds:
            while True:
                candidate = self._newton(seed, roots)
                if candidate is None:
                    break
                is_root = np.linalg.norm(self._field(candidate)) < _ROOT_RESIDUAL
                is_new = all(
                    np.linalg.norm(candidate - root) > _MERGE for root in roots
                )
                if is_root and is_new:
                    roots.append(candidate)
                else:
                    break
        return roots
