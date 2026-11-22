"""Jacobian abstraction, decoupled from how derivatives are obtained."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np

from discrecontinual_equations.continuation.residual import (
    DifferentiableResidual,
    ResidualFunction,
)

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

    @contextmanager
    def retained(self) -> Iterator[None]:
        """Scope within which a repeated request at one point may be remembered.

        The base provider remembers nothing. :class:`MemoizedJacobian` overrides this
        so that, inside the scope, the converged point's Jacobian is built once and
        shared by the tangent, the detectors and the point's stability.
        """
        yield


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


class AnalyticJacobian(JacobianProvider):
    """The extended Jacobian the residual computes itself.

    For augmented systems whose rows are bordered test functions, differencing
    every column would redo the whole bordered reduction per column; the residual
    instead differentiates its rows analytically (see
    :class:`~.residual.DifferentiableResidual`).
    """

    __slots__ = ["_epsilon", "_residual"]

    def __init__(self, residual: ResidualFunction, epsilon: float) -> None:
        if not isinstance(residual, DifferentiableResidual):
            message = "AnalyticJacobian needs a DifferentiableResidual"
            raise TypeError(message)
        self._residual = residual
        self._epsilon = epsilon

    def extended_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        return self._residual.extended_jacobian(state, parameter, self._epsilon)

    def state_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        return self.extended_jacobian(state, parameter)[:, :-1]

    def parameter_derivative(self, state: np.ndarray, parameter: float) -> np.ndarray:
        return self.extended_jacobian(state, parameter)[:, -1]


class MemoizedJacobian(JacobianProvider):
    """Remember the last few points' derivatives, but only inside :meth:`retained`.

    One accepted continuation point asks for the Jacobian at the converged state
    four times - the tangent, the branch-point determinant, the Hopf test and the
    stability - and the localizer asks again at the bracket's lower end. Inside a
    retained scope each of those after the first is a copy of the remembered array,
    so a point costs ``I + 1`` Jacobians instead of ``I + 4`` and every value is
    bit-identical to recomputing it. Points are keyed by the exact bytes of the
    state and the parameter.

    The memory lives only for the scope (one :meth:`Continuer.solve`): the residual
    reads the equation's other parameters live, so a Jacobian remembered across
    solves could be stale after the caller changes one of them.
    """

    __slots__ = ["_active", "_memory", "_provider"]

    capacity = 4

    def __init__(self, provider: JacobianProvider) -> None:
        self._provider = provider
        self._active = False
        self._memory: OrderedDict[tuple[bytes, bytes], dict[str, np.ndarray]] = (
            OrderedDict()
        )

    def state_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        return self._remembered(
            "state",
            state,
            parameter,
            self._provider.state_jacobian,
        )

    def parameter_derivative(self, state: np.ndarray, parameter: float) -> np.ndarray:
        return self._remembered(
            "parameter",
            state,
            parameter,
            self._provider.parameter_derivative,
        )

    @contextmanager
    def retained(self) -> Iterator[None]:
        self._active = True
        try:
            yield
        finally:
            self._active = False
            self._memory.clear()

    def _remembered(
        self,
        part: str,
        state: np.ndarray,
        parameter: float,
        compute: Callable[[np.ndarray, float], np.ndarray],
    ) -> np.ndarray:
        if not self._active:
            return compute(state, parameter)
        key = (
            np.ascontiguousarray(state, dtype=float).tobytes(),
            np.float64(parameter).tobytes(),
        )
        parts = self._memory.get(key)
        if parts is None:
            parts = {}
            self._memory[key] = parts
            if len(self._memory) > self.capacity:
                self._memory.popitem(last=False)
        else:
            self._memory.move_to_end(key)
        if part not in parts:
            parts[part] = np.array(compute(state, parameter), dtype=float)
        return parts[part].copy()
