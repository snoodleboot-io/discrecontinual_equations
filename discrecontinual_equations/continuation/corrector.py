"""Corrector step: a generic Newton loop over an injected parameterization.

The corrector no longer knows which continuation constraint it enforces; that is
supplied as a :class:`~.parameterization.Parameterization`. The same class serves
natural-parameter, pseudo-arclength, or any future constraint.
"""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.continuation.jacobian import JacobianProvider
from discrecontinual_equations.continuation.linear_solver import LinearSolver
from discrecontinual_equations.continuation.parameterization import (
    CorrectionStep,
    Parameterization,
)
from discrecontinual_equations.continuation.residual import ResidualFunction

_RESIDUAL_SAFETY_FACTOR = 100.0


class CorrectionResult:
    """Outcome of a corrector call."""

    __slots__ = ["_converged", "_iterations", "_parameter", "_state"]

    def __init__(
        self,
        state: np.ndarray,
        parameter: float,
        iterations: int,
        *,
        converged: bool,
    ) -> None:
        self._state = state
        self._parameter = parameter
        self._iterations = iterations
        self._converged = converged

    @property
    def state(self) -> np.ndarray:
        """Corrected state vector."""
        return self._state

    @property
    def parameter(self) -> float:
        """Corrected parameter value."""
        return self._parameter

    @property
    def iterations(self) -> int:
        """Number of Newton iterations used."""
        return self._iterations

    @property
    def converged(self) -> bool:
        """Whether the corrector met its tolerance."""
        return self._converged


class NewtonSettings:
    """Convergence criteria for the Newton corrector."""

    __slots__ = ["_max_iterations", "_tolerance"]

    def __init__(self, tolerance: float, max_iterations: int) -> None:
        self._tolerance = tolerance
        self._max_iterations = max_iterations

    @property
    def tolerance(self) -> float:
        """Residual-norm tolerance."""
        return self._tolerance

    @property
    def max_iterations(self) -> int:
        """Maximum Newton iterations per corrector call."""
        return self._max_iterations


class Corrector(ABC):
    """Project a predicted point back onto the solution set."""

    @abstractmethod
    def correct(
        self,
        predicted_state: np.ndarray,
        predicted_parameter: float,
        tangent: np.ndarray,
    ) -> CorrectionResult:
        """Return the corrected point and convergence information."""
        raise NotImplementedError


class NewtonCorrector(Corrector):
    """Newton on the bordered system with an injected constraint ``g``."""

    __slots__ = [
        "_jacobian",
        "_linear_solver",
        "_parameterization",
        "_residual",
        "_settings",
    ]

    def __init__(
        self,
        residual: ResidualFunction,
        jacobian: JacobianProvider,
        linear_solver: LinearSolver,
        parameterization: Parameterization,
        settings: NewtonSettings,
    ) -> None:
        self._residual = residual
        self._jacobian = jacobian
        self._linear_solver = linear_solver
        self._parameterization = parameterization
        self._settings = settings

    def correct(
        self,
        predicted_state: np.ndarray,
        predicted_parameter: float,
        tangent: np.ndarray,
    ) -> CorrectionResult:
        state = predicted_state.copy()
        parameter = float(predicted_parameter)
        for iteration in range(self._settings.max_iterations):
            step = CorrectionStep(
                state,
                parameter,
                predicted_state,
                predicted_parameter,
                tangent,
            )
            augmented = self._augmented_residual(state, parameter, step)
            if np.linalg.norm(augmented) < self._settings.tolerance:
                return CorrectionResult(state, parameter, iteration, converged=True)
            bordered = np.vstack(
                [
                    self._jacobian.extended_jacobian(state, parameter),
                    self._parameterization.border(step),
                ],
            )
            delta = self._linear_solver.solve(bordered, -augmented)
            state = state + delta[:-1]
            parameter = parameter + float(delta[-1])
        return CorrectionResult(
            state,
            parameter,
            self._settings.max_iterations,
            converged=self._within_relaxed_tolerance(state, parameter),
        )

    def _augmented_residual(
        self,
        state: np.ndarray,
        parameter: float,
        step: CorrectionStep,
    ) -> np.ndarray:
        residual = self._residual.evaluate(state, parameter)
        constraint = self._parameterization.constraint(step)
        return np.concatenate([residual, [constraint]])

    def _within_relaxed_tolerance(self, state: np.ndarray, parameter: float) -> bool:
        residual = self._residual.evaluate(state, parameter)
        limit = self._settings.tolerance * _RESIDUAL_SAFETY_FACTOR
        return bool(np.linalg.norm(residual) < limit)
