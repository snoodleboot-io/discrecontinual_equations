"""Refine a raw seed onto the solution set before continuation begins.

Continuation needs a first point on the curve. A user's guess ``u_0`` is polished
at the fixed starting parameter ``lambda_0`` by plain Newton on the state,

    u <- u - F_u(u, lambda_0)^(-1) F(u, lambda_0),

the equilibrium location of Kuznetsov (1998, Section 10.1.1). Newton converges
quadratically from a close enough guess at a regular equilibrium. It fails, or
crawls, at a fold or branch point, where ``F_u`` is singular (the linear solver then
takes a least-squares step), and from a poor guess it may converge to a different
equilibrium or not at all. The refiner returns its last iterate either way: check
the residual of the first branch point when the seed is doubtful.

References:
- Kuznetsov, Yu. A. "Elements of Applied Bifurcation Theory", 2nd ed., Applied
  Mathematical Sciences 112, Springer, 1998, doi:10.1007/b98848, Section 10.1.1.
"""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.continuation.jacobian import JacobianProvider
from discrecontinual_equations.continuation.linear_solver import LinearSolver
from discrecontinual_equations.continuation.residual import ResidualFunction


class SeedRefiner(ABC):
    """Project an approximate seed onto an equilibrium at a fixed parameter."""

    @abstractmethod
    def refine(self, state: np.ndarray, parameter: float) -> np.ndarray:
        """Return a state satisfying ``f(state, parameter) = 0`` (approximately).

        Args:
            state: The initial guess.
            parameter: The fixed parameter value.

        Returns:
            The refined state.
        """
        raise NotImplementedError


class NewtonSeedRefiner(SeedRefiner):
    """Fixed-parameter Newton iteration on the state alone.

    Args:
        residual: The equilibrium residual ``F``.
        jacobian: Source of ``F_u``.
        linear_solver: Solver for the Newton step.
        iterations: Maximum Newton iterations.
        tolerance: Stop once ``||F|| < tolerance``.

    References:
    - Kuznetsov (1998), Section 10.1.1.
    """

    __slots__ = [
        "_iterations",
        "_jacobian",
        "_linear_solver",
        "_residual",
        "_tolerance",
    ]

    def __init__(
        self,
        residual: ResidualFunction,
        jacobian: JacobianProvider,
        linear_solver: LinearSolver,
        iterations: int,
        tolerance: float,
    ) -> None:
        self._residual = residual
        self._jacobian = jacobian
        self._linear_solver = linear_solver
        self._iterations = iterations
        self._tolerance = tolerance

    def refine(self, state: np.ndarray, parameter: float) -> np.ndarray:
        current = np.asarray(state, dtype=float).copy()
        for _ in range(self._iterations):
            residual = self._residual.evaluate(current, parameter)
            if np.linalg.norm(residual) < self._tolerance:
                break
            jacobian = self._jacobian.state_jacobian(current, parameter)
            current = current + self._linear_solver.solve(jacobian, -residual)
        return current
