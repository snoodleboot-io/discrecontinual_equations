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
from discrecontinual_equations.continuation.sparsity import SparsityPattern
from discrecontinual_equations.finite_difference import step_scales
from discrecontinual_equations.function.function import Function

_DEFAULT_EPSILON = 1e-6
_SECOND_ORDER = 2
_THIRD_ORDER = 3
_FOURTH_ORDER = 4
_CENTRAL_PAIR = 2.0
_UNIT_LENGTH = 1.0


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
        sparsity: SparsityPattern | None = None,
    ) -> np.ndarray:
        """Return the state Jacobian, column by column from unit directions.

        With a ``sparsity`` pattern, one first-order pass along each group's 0/1
        direction gives that group's compressed column (Curtis-Powell-Reid
        seeding), so the Jacobian costs one pass per group instead of one per
        column; each entry reads only the states its row depends on, so the
        result equals the column-by-column one.
        """
        if sparsity is not None:
            compressed = np.column_stack(
                [
                    self.taylor(function, point, direction, 1, time)[1].real
                    for direction in sparsity.directions()
                ],
            )
            return sparsity.expand(compressed)
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
    """Central-difference derivatives; a fallback for non-analytic fields.

    Coefficient ``m`` along a direction ``d`` is the central difference of order
    ``m`` of ``s -> f(x + s d)`` at ``s = 0``, over the stencil ``s = k t``.
    ``epsilon`` is a *relative* step (DEQ-118). With ``s_j = max(|x_j|,
    typical_j)`` the spacing is

        ``t = epsilon min(1, max_j |d_j|) / max_j(|d_j| / s_j)``,

    so no component moves more than its own step, ``|t d_j| <= epsilon s_j``,
    however long ``d`` is (DEQ-163): a spacing that grew with ``max |d|`` moved
    ``x = d = 1e6`` by ``1e6`` and got the first coefficient of ``x^3`` wrong by
    a third. Along a unit direction ``e_j`` the spacing is Dennis and
    Schnabel's per-component step ``epsilon s_j``, so the Jacobian is
    differenced column by column at each component's scale. A direction no
    longer than one (``max |d| <= 1``, every eigenvector-like direction the
    library passes) takes ``epsilon max|d| / max_j(|d_j| / s_j)``, and where
    every component it moves is no larger than its typical size (one by
    default) that is ``epsilon`` itself, bit for bit, as it was when the step
    was absolute. A longer direction at unit scale takes ``epsilon / max |d|``
    instead, which is what keeps its move to ``epsilon``.

    A component of size ``1e12`` would round an absolute ``1e-6`` away and
    return zero; one of size ``1e-12`` needs ``typical`` - one per component,
    or one for all - or a step floored at one swamps it. Exact only to first
    order in practice, and the stencil is not divided by the width actually
    stepped, which along a general direction differs per component; the
    rounding that leaves is the same order as the field's own.

    References:
        Dennis and Schnabel (1983), Sec. 5.4 and Algorithm A5.4.1: the step
        scaled per component by ``max(|x_j|, typx_j)``.
        SciPy 1.14.1, ``scipy/optimize/_nonlin.py``, ``KrylovJacobian.matvec``:
        a directional difference whose spacing is divided by the direction's
        length, ``omega / ||v||``, so the move is the same for any length. We
        depart from it on purpose for directions shorter than one, which keep
        the spacing of a unit one, so that results at unit scale are unchanged.
        See ``docs/references.md``.
    """

    __slots__ = ["_epsilon", "_typical"]

    def __init__(
        self,
        epsilon: float = _DEFAULT_EPSILON,
        typical: float | np.ndarray | None = None,
    ) -> None:
        self._epsilon = epsilon
        self._typical = typical

    def taylor(
        self,
        function: Function,
        point: np.ndarray,
        direction: np.ndarray,
        order: int,
        time: float,
    ) -> list[np.ndarray]:
        coefficients = [self._evaluate(function, point, time).astype(complex)]
        step = self.step(point, direction)
        for degree in range(1, order + 1):
            derivative = self._derivative(
                function,
                point,
                step * direction,
                (degree, step),
                time,
            )
            coefficients.append(derivative / factorial(degree))
        return coefficients

    def jacobian(
        self,
        function: Function,
        point: np.ndarray,
        time: float,
        sparsity: SparsityPattern | None = None,
    ) -> np.ndarray:
        """The state Jacobian, each column at its own component's step.

        Column by column this is the base class's. With a ``sparsity`` pattern a
        group's columns are moved together, but each by the step its own unit
        direction would take, ``epsilon max(|x_j|, typical_j)``, and divided by
        it: one shared step along the group's 0/1 direction would move a column
        of size ``1e12`` by the step of one of size one, which rounds away. Every
        entry then equals the column-by-column one as long as the pattern covers
        the field's.
        """
        if sparsity is None:
            return super().jacobian(function, point, time)
        units = np.eye(point.size)
        steps = np.array([self.step(point, unit) for unit in units])
        compressed = []
        for direction in sparsity.directions():
            offset = steps * direction
            compressed.append(
                self._evaluate(function, point + offset, time)
                - self._evaluate(function, point - offset, time),
            )
        return sparsity.expand(np.column_stack(compressed)) / (_CENTRAL_PAIR * steps)

    def step(self, point: np.ndarray, direction: np.ndarray) -> float:
        """The stencil spacing ``t`` along ``direction``, relative to ``point``.

        ``t = epsilon min(1, max_j |d_j|) / max_j(|d_j| / s_j)``, so that no
        component moves more than its own step, ``|t d_j| <= epsilon s_j``.
        """
        size = np.abs(direction)
        scales = step_scales(np.asarray(point, dtype=float), self._typical)
        weight = float(np.max(size / scales))
        if weight == 0.0:
            return self._epsilon
        return self._epsilon * (min(_UNIT_LENGTH, float(np.max(size))) / weight)

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
        offset: np.ndarray,
        steps: int,
        time: float,
    ) -> np.ndarray:
        return self._evaluate(function, point + steps * offset, time)

    def _derivative(
        self,
        function: Function,
        point: np.ndarray,
        direction: np.ndarray,
        order: tuple[int, float],
        time: float,
    ) -> np.ndarray:
        """Derivative of order ``degree`` along ``direction``, one ``step`` apart.

        ``direction`` is the stencil's offset, already ``step`` times the
        caller's direction.
        """
        degree, step = order
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
