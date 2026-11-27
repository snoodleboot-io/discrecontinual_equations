"""Providers of state derivatives of a vector field at a point.

A :class:`DerivativeProvider` returns the directional Taylor coefficients of the
field along a direction (coefficient ``m`` is ``D**m f(direction, ...) / m!``) and
the state Jacobian. Two implementations share one interface:

- :class:`AutomaticDifferentiation` evaluates the field on Taylor jets, giving
  exact derivatives to any order over real or complex directions. It is the enabler
  for the derivative-hungry detectors (cusp, generalized Hopf) and codim-3.
- :class:`FiniteDifferenceDerivatives` uses central differences, each order at
  its own relative step ``eps^(1/(m+2))``, over real or complex directions (a
  complex one is split into real ones); orders up to four, accurate to about
  ``1e-6`` relative, and included as a fallback for fields that cannot take jets.
"""

import cmath
import math
from abc import ABC, abstractmethod
from math import factorial

import numpy as np
from sweet_tea.abstract_factory import AbstractFactory

from discrecontinual_equations.continuation.jet import Jet
from discrecontinual_equations.continuation.sparsity import SparsityPattern
from discrecontinual_equations.finite_difference import order_step, step_scales
from discrecontinual_equations.function.function import Function

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
    """Exact derivatives by evaluating the field on Taylor jets.

    Complexity:
        ``taylor`` is one field evaluation on jets, ``O(C_f)`` jet operations of
        ``O(k)`` to ``O(k^2)`` each for order ``k``; ``jacobian`` is ``n``
        order-one passes, ``O(n C_f)``, or one pass per column group, ``O(g C_f)``,
        when a sparsity pattern is given.
    """

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
    ``m`` of ``s -> f(x + s d)`` at ``s = 0``, over the stencil ``s = k t``,
    divided by ``m!``. Each order takes its own relative step ``r_m``: by
    default ``eps^(1/(m+2))``
    (:func:`~discrecontinual_equations.finite_difference.order_step`), which
    balances the stencil's ``O(t^2)`` truncation against its ``eps /
    t^m`` rounding - ``6.1e-6`` for the slope, ``1.2e-4``, ``7.4e-4`` and
    ``2.5e-3`` for orders two to four. One step for every order is what broke
    the third differences: at ``1e-6`` their rounding is ``eps / 1e-18``, and
    van der Pol's ``l1`` came back ``+21`` instead of ``-0.93`` (DEQ-167). A
    given ``epsilon`` is used for every order instead, as before.

    With ``s_j = max(|x_j|, typical_j)`` the spacing for order ``m`` is

        ``t = r_m min(1, max_j |d_j|) / max_j(|d_j| / s_j)``,

    so no component moves more than its own step, ``|t d_j| <= r_m s_j``,
    however long ``d`` is (DEQ-163): a spacing that grew with ``max |d|`` moved
    ``x = d = 1e6`` by ``1e6`` and got the first coefficient of ``x^3`` wrong by
    a third. Along a unit direction ``e_j`` the spacing is Dennis and
    Schnabel's per-component step ``r_m s_j``, so the Jacobian is differenced
    column by column at each component's scale. A direction no longer than
    one (``max |d| <= 1``) takes ``r_m max|d| / max_j(|d_j| / s_j)``, which is
    ``r_m`` itself where every component it moves is no larger than its
    typical size (one by default). A longer direction at unit scale takes
    ``r_m / max |d|``, which keeps its move to ``r_m``.

    A complex direction ``d = u + i v`` - an eigenvector at a Hopf point - is
    split into real ones. ``P_m(d) = D^m f(d, ..., d) / m!`` is homogeneous of
    degree ``m``, and along ``d_k = cos(theta_k) u + sin(theta_k) v`` with
    ``theta_k = pi k / (m + 1)``, ``k = 0 .. m``,

        ``P_m(u + i v) = 2^m / (m + 1) sum_k exp(i m theta_k) P_m(d_k)``

    exactly: writing ``d_k = (exp(-i theta_k) d + exp(i theta_k) conj(d)) / 2``
    and expanding, the sum over ``k`` keeps only the ``d^m`` term. So order
    ``m`` along a complex direction costs ``m + 1`` real ones, each with its
    own spacing, and the field is only ever evaluated at real points. Before
    DEQ-167 the imaginary part was cast away with a ``ComplexWarning`` and
    ``P_m(u)`` was returned. A point with a nonzero imaginary part is refused.

    The library's detectors pass directions up to ``max |d| = 2.74`` (the
    ``MultilinearForms`` polarization sums), so they take the shorter spacing
    above. This provider is used only when chosen through the opt-in
    ``derivative_provider`` registry key (``Codim2Config.derivative_epsilon``
    sets its ``epsilon``); the default is :class:`AutomaticDifferentiation`.
    On the codim-2 scenarios of the test suite and on van der Pol every
    normal-form coefficient it gives agrees with automatic differentiation to
    ``2.4e-7`` of its record's largest coefficient (``l1`` of van der Pol:
    ``2.5e-7`` relative; it was ``+21`` against ``-0.93``) (DEQ-167).

    A component of size ``1e12`` would round an absolute step away and return
    zero; one of size ``1e-12`` needs ``typical`` - one per component, or one
    for all - or a step floored at one swamps it. The stencil is not divided
    by the width actually stepped, which along a general direction differs per
    component. Orders above four are refused: use automatic differentiation.

    Limit: one spacing serves every component, so the components that bind
    move ``r_m s_j`` and the others move less, in proportion to ``|d_j| /
    s_j``. A component whose ratio is ``R`` times smaller than the binding
    one moves ``r_m s_j / R``, and its share of the slope errs by about ``eps
    R / r_1`` relative: a few per cent once ``R`` passes about ``1e8`` at
    ``r_1 = 1e-6``, and the move rounds away entirely, losing that share, near
    ``R = r_1 / eps``. At ``x = (1e9, 1e-3)``, ``d = (1, 1e6)``, ``epsilon =
    1e-6`` the spacing is ``1e-12`` and ``x_0`` does not move: the identity
    field's first coefficient comes back ``(0, 1e6)``, not ``(1, 1e6)``.
    Difference such a direction component by component (the Jacobian)
    instead.

    Args:
        epsilon: One relative step for every order, or ``None`` (the default)
            for ``eps^(1/(m+2))`` at order ``m``.
        typical: The typical size of each state component, or one for all;
            ``None`` for one.

    References:
        Dennis and Schnabel (1983), Sec. 5.4 and Algorithm A5.4.1: the step
        scaled per component by ``max(|x_j|, typx_j)``, and ``eps^(1/3)`` for a
        central slope. Mikulski and Uhl (2026), arXiv:2608.11020, Sec. 3.2 and
        Appendix B: the minimiser ``C_k eps^(1/(k+2))`` of a second-order
        central stencil's error for the ``k``-th derivative (we take ``C_k =
        1``). Fornberg (1988): the central stencils of accuracy two, as
        tabulated in Wikipedia's "Finite difference coefficient". SciPy 1.14.1,
        ``scipy/optimize/_nonlin.py``, ``KrylovJacobian.matvec``: a directional
        difference whose spacing is divided by the direction's length, ``omega /
        ||v||``, so the move is the same for any length. We depart from it on
        purpose for directions shorter than one, which keep the spacing of a
        unit one, so that results at unit scale are unchanged. The
        complex-direction identity is the discrete Fourier inversion of a
        homogeneous polynomial; it is derived above, not taken from a source.
        See ``docs/references.md``.
    """

    __slots__ = ["_epsilon", "_typical"]

    def __init__(
        self,
        epsilon: float | None = None,
        typical: float | np.ndarray | None = None,
    ) -> None:
        if epsilon is not None and not epsilon > 0.0:
            message = f"epsilon must be positive or None, got {epsilon}"
            raise ValueError(message)
        self._epsilon = epsilon
        self._typical = typical

    def relative_step(self, order: int) -> float:
        """The relative step ``r_m`` of an order-``order`` difference.

        The configured ``epsilon`` if one was given, else ``eps^(1/(m+2))``.
        """
        if self._epsilon is not None:
            return self._epsilon
        return order_step(order)

    def taylor(
        self,
        function: Function,
        point: np.ndarray,
        direction: np.ndarray,
        order: int,
        time: float,
    ) -> list[np.ndarray]:
        """Return ``a_0 .. a_order`` of ``f(point + s direction)``.

        ``direction`` may be complex; ``point`` must be real (a complex array
        with a zero imaginary part is accepted).

        Raises:
            ValueError: If ``point`` has a nonzero imaginary part.
            NotImplementedError: If ``order`` is above four.
        """
        real_point = _real_point(point)
        coefficients = [self._evaluate(function, real_point, time).astype(complex)]
        vector = np.asarray(direction)
        along = np.asarray(vector.real, dtype=float)
        across = np.asarray(vector.imag, dtype=float)
        for degree in range(1, order + 1):
            if np.any(across):
                value = self._complex_power(
                    function,
                    real_point,
                    (along, across),
                    degree,
                    time,
                )
            else:
                value = self._power(function, real_point, along, degree, time).astype(
                    complex,
                )
            coefficients.append(value)
        return coefficients

    def _power(
        self,
        function: Function,
        point: np.ndarray,
        direction: np.ndarray,
        degree: int,
        time: float,
    ) -> np.ndarray:
        """``P_m(d) = D^m f(d, ..., d) / m!`` along a real direction."""
        step = self.step(point, direction, degree)
        derivative = self._derivative(
            function,
            point,
            step * direction,
            (degree, step),
            time,
        )
        return derivative / factorial(degree)

    def _complex_power(
        self,
        function: Function,
        point: np.ndarray,
        direction: tuple[np.ndarray, np.ndarray],
        degree: int,
        time: float,
    ) -> np.ndarray:
        """``P_m(u + i v)`` from ``m + 1`` real directions (see the class)."""
        along, across = direction
        count = degree + 1
        total: np.ndarray = np.zeros(1, dtype=complex)
        for index in range(count):
            angle = math.pi * index / count
            rotated = math.cos(angle) * along + math.sin(angle) * across
            power = self._power(function, point, rotated, degree, time)
            total = total + cmath.exp(1j * degree * angle) * power
        return total * (_CENTRAL_PAIR**degree / count)

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
        real_point = _real_point(point)
        units = np.eye(real_point.size)
        steps = np.array([self.step(real_point, unit) for unit in units])
        compressed = []
        for direction in sparsity.directions():
            offset = steps * direction
            compressed.append(
                self._evaluate(function, real_point + offset, time)
                - self._evaluate(function, real_point - offset, time),
            )
        return sparsity.expand(np.column_stack(compressed)) / (_CENTRAL_PAIR * steps)

    def step(
        self,
        point: np.ndarray,
        direction: np.ndarray,
        order: int = 1,
    ) -> float:
        """The stencil spacing ``t`` along ``direction``, relative to ``point``.

        ``t = r min(1, max_j |d_j|) / max_j(|d_j| / s_j)``, with ``r`` the
        relative step of an order-``order`` difference, so that no component
        moves more than its own step, ``|t d_j| <= r s_j``.
        """
        relative = self.relative_step(order)
        size = np.abs(direction)
        scales = step_scales(_real_point(point), self._typical)
        weight = float(np.max(size / scales))
        if weight == 0.0:
            return relative
        return relative * (min(_UNIT_LENGTH, float(np.max(size))) / weight)

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


def configured_provider(key: str, epsilon: float | None) -> DerivativeProvider:
    """The registered provider ``key``, given ``epsilon`` only when one is set.

    ``Codim2Config.derivative_epsilon`` reaches the finite-difference provider
    this way; left unset, every provider is built with no configuration, as
    before, and the finite-difference one steps each order at
    ``eps^(1/(m+2))``.

    Raises:
        sweet_tea.sweet_tea_error.SweetTeaError: If ``key`` is not registered.
    """
    if epsilon is None:
        return AbstractFactory[DerivativeProvider].create(key)
    return AbstractFactory[DerivativeProvider].create(
        key,
        configuration={"epsilon": epsilon},
    )


def _real_point(point: np.ndarray) -> np.ndarray:
    """``point`` as a real array; a nonzero imaginary part is refused."""
    values = np.asarray(point)
    if np.iscomplexobj(values):
        if np.any(values.imag):
            message = (
                "finite differences evaluate the field at real points only; "
                "this point has a nonzero imaginary part. Use automatic "
                "differentiation"
            )
            raise ValueError(message)
        values = values.real
    return np.asarray(values, dtype=float)


def _coefficient(component: object, degree: int) -> complex:
    if isinstance(component, Jet):
        coefficients = component.coefficients
        return complex(coefficients[degree]) if degree < len(coefficients) else 0.0
    return complex(component) if degree == 0 else 0.0
