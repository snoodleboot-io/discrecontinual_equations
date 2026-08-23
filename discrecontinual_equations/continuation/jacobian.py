"""Jacobian abstraction, decoupled from how derivatives are obtained."""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.continuation.residual import ResidualFunction

_CENTRAL_DIFFERENCE_POINTS = 2.0


class JacobianProvider(ABC):
    """Supply the derivatives of the residual needed for continuation."""

    @abstractmethod
    def state_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        """Return ``df/du`` of shape ``(n, n)``."""
        raise NotImplementedError

    @abstractmethod
    def parameter_derivative(self, state: np.ndarray, parameter: float) -> np.ndarray:
        """Return ``df/dlambda`` of shape ``(n,)``."""
        raise NotImplementedError

    def extended_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        """Return the augmented ``[df/du | df/dlambda]`` of shape ``(n, n + 1)``."""
        state_part = self.state_jacobian(state, parameter)
        parameter_part = self.parameter_derivative(state, parameter).reshape(-1, 1)
        return np.hstack([state_part, parameter_part])


class FiniteDifferenceJacobian(JacobianProvider):
    """Central finite-difference Jacobian; needs no analytic derivatives."""

    __slots__ = ["_epsilon", "_residual"]

    def __init__(self, residual: ResidualFunction, epsilon: float) -> None:
        self._residual = residual
        self._epsilon = epsilon

    def state_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        dimension = state.size
        jacobian = np.zeros((dimension, dimension))
        for column in range(dimension):
            step = np.zeros(dimension)
            step[column] = self._epsilon
            forward = self._residual.evaluate(state + step, parameter)
            backward = self._residual.evaluate(state - step, parameter)
            jacobian[:, column] = (forward - backward) / (
                _CENTRAL_DIFFERENCE_POINTS * self._epsilon
            )
        return jacobian

    def parameter_derivative(self, state: np.ndarray, parameter: float) -> np.ndarray:
        forward = self._residual.evaluate(state, parameter + self._epsilon)
        backward = self._residual.evaluate(state, parameter - self._epsilon)
        return (forward - backward) / (_CENTRAL_DIFFERENCE_POINTS * self._epsilon)
