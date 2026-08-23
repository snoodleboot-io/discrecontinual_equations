"""Providers of state derivatives of a vector field at a point.

A :class:`DerivativeProvider` returns the directional Taylor coefficients of the
field along a direction (coefficient ``m`` is ``D**m f(direction, ...) / m!``) and
the state Jacobian. Two implementations share one interface:

- :class:`AutomaticDifferentiation` evaluates the field on Taylor jets, giving
  exact derivatives to any order over real or complex directions. It is the enabler
  for the derivative-hungry detectors (cusp, generalized Hopf) and codim-3.
- :class:`FiniteDifferenceDerivatives` uses central differences: exact only to
  first order in practice, real directions only, and included as a fallback.
"""

from abc import ABC, abstractmethod
from math import factorial

import numpy as np

from discrecontinual_equations.continuation.jet import Jet
from discrecontinual_equations.function.function import Function

_DEFAULT_EPSILON = 1e-6
_SECOND_ORDER = 2
_THIRD_ORDER = 3
_FOURTH_ORDER = 4
_CENTRAL_PAIR = 2.0


class DerivativeProvider(ABC):
    """State derivatives of a vector field at a point."""

    @abstractmethod
    def taylor(
        self,
        function: Function,
        point: np.ndarray,
        direction: np.ndarray,
        order: int,
        time: float,
    ) -> list[np.ndarray]:
        """Return Taylor coefficients ``a_0 .. a_order`` of ``f(point + s dir)``."""
        raise NotImplementedError

    def jacobian(
        self,
        function: Function,
        point: np.ndarray,
        time: float,
    ) -> np.ndarray:
        """Return the state Jacobian, column by column from unit directions."""
        dimension = point.size
        columns = []
        for index in range(dimension):
            unit = np.zeros(dimension)
            unit[index] = 1.0
            columns.append(self.taylor(function, point, unit, 1, time)[1].real)
        return np.column_stack(columns)


class AutomaticDifferentiation(DerivativeProvider):
    """Exact derivatives by evaluating the field on Taylor jets."""

    def taylor(
        self,
        function: Function,
        point: np.ndarray,
        direction: np.ndarray,
        order: int,
        time: float,
    ) -> list[np.ndarray]:
        seeds = [
            Jet.seed(complex(point[i]), complex(direction[i]), order)
            for i in range(point.size)
        ]
        raw = function.eval(point=seeds, time=time)
        coefficients: list[np.ndarray] = []
        for degree in range(order + 1):
            row = [_coefficient(component, degree) for component in raw]
            coefficients.append(np.array(row, dtype=complex))
        return coefficients


class FiniteDifferenceDerivatives(DerivativeProvider):
    """Central-difference derivatives; a fallback for non-analytic fields."""

    __slots__ = ["_epsilon"]

    def __init__(self, epsilon: float = _DEFAULT_EPSILON) -> None:
        self._epsilon = epsilon

    def taylor(
        self,
        function: Function,
        point: np.ndarray,
        direction: np.ndarray,
        order: int,
        time: float,
    ) -> list[np.ndarray]:
        coefficients = [self._evaluate(function, point, time).astype(complex)]
        for degree in range(1, order + 1):
            derivative = self._derivative(function, point, direction, degree, time)
            coefficients.append(derivative / factorial(degree))
        return coefficients

    def _evaluate(
        self,
        function: Function,
        point: np.ndarray,
        time: float,
    ) -> np.ndarray:
        return np.asarray(
            function.eval(point=[float(value) for value in point], time=time),
            dtype=float,
        )

    def _sample(
        self,
        function: Function,
        point: np.ndarray,
        direction: np.ndarray,
        steps: int,
        time: float,
    ) -> np.ndarray:
        return self._evaluate(function, point + steps * self._epsilon * direction, time)

    def _derivative(
        self,
        function: Function,
        point: np.ndarray,
        direction: np.ndarray,
        degree: int,
        time: float,
    ) -> np.ndarray:
        step = self._epsilon
        sample = self._sample
        if degree == 1:
            return (
                sample(function, point, direction, 1, time)
                - sample(function, point, direction, -1, time)
            ) / (_CENTRAL_PAIR * step)
        if degree == _SECOND_ORDER:
            return (
                sample(function, point, direction, 1, time)
                - 2.0 * sample(function, point, direction, 0, time)
                + sample(function, point, direction, -1, time)
            ) / step**2
        if degree == _THIRD_ORDER:
            return (
                sample(function, point, direction, 2, time)
                - 2.0 * sample(function, point, direction, 1, time)
                + 2.0 * sample(function, point, direction, -1, time)
                - sample(function, point, direction, -2, time)
            ) / (_CENTRAL_PAIR * step**3)
        if degree == _FOURTH_ORDER:
            return (
                sample(function, point, direction, 2, time)
                - 4.0 * sample(function, point, direction, 1, time)
                + 6.0 * sample(function, point, direction, 0, time)
                - 4.0 * sample(function, point, direction, -1, time)
                + sample(function, point, direction, -2, time)
            ) / step**4
        message = "finite differences support order <= 4; use automatic differentiation"
        raise NotImplementedError(message)


def _coefficient(component: object, degree: int) -> complex:
    if isinstance(component, Jet):
        coefficients = component.coefficients
        return complex(coefficients[degree]) if degree < len(coefficients) else 0.0
    return complex(component) if degree == 0 else 0.0
