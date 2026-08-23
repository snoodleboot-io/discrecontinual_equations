"""Refine a raw seed onto the solution set before continuation begins."""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.continuation.jacobian import JacobianProvider
from discrecontinual_equations.continuation.linear_solver import LinearSolver
from discrecontinual_equations.continuation.residual import ResidualFunction


class SeedRefiner(ABC):
    """Project an approximate seed onto an equilibrium at a fixed parameter."""

    @abstractmethod
    def refine(self, state: np.ndarray, parameter: float) -> np.ndarray:
        """Return a state satisfying ``f(state, parameter) = 0`` (approximately)."""
        raise NotImplementedError


class NewtonSeedRefiner(SeedRefiner):
    """Fixed-parameter Newton iteration on the state alone."""

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
