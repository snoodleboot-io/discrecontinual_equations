"""A vector field, and its state Jacobian, at many states in one evaluation.

The cycle continuation needs ``f`` and ``df/dx`` at every node of a mesh - and
the monodromy at nine points of every interval - on every Newton step. Called a
state at a time, a field written in plain Python costs one interpreter pass per
state, and its Jacobian by Taylor jets one pass per state *and* per direction.
Most fields are written in ``+ - * /`` and integer powers, and those evaluate
just as well with every coordinate an array of all the states at once: one pass
for the lot.

:func:`field_values` hands the field arrays. :func:`field_jacobians` hands it
:class:`_Dual` numbers, first-order jets whose value is an array over the states
and whose derivative carries every direction at once, so one pass gives every
column of every Jacobian. Each operation of :class:`_Dual` repeats the
arithmetic of :class:`~.jet.Jet` at first order, in the same order.

Both agree with the state-at-a-time evaluation *to rounding*, not to the last
bit on every machine. An array power ``x**3`` may run NumPy's vectorised loop
where a scalar one calls libm's ``pow``, and the two differ by an ulp on some
CPUs (CI's runners: 2 of 200 entries, one ulp each); a platform that fuses a
multiply and an add can do the same to the jets' complex products. On the
machine the films are built on the two were identical over every row of five
films. Through a forward difference, as :meth:`~.periodic_orbit.PeriodicOrbit.
_jacobians` takes, an ulp of ``f`` becomes about ``ulp / step`` of ``df/dx``.

Neither can tell in advance whether a field accepts arrays. A field that
branches on its state (``if radius < floor``), calls :mod:`math`, or takes a
fractional power raises, and the caller is told so by ``None`` and evaluates a
state at a time as before. A field that silently mis-evaluates on arrays - a
row-vector product, say - is caught by comparing the first and last states
against the caller's own state-at-a-time evaluation: a difference beyond
rounding (:data:`_SPOT_ULPS` units of the row's largest entry) returns ``None``
too. Two states are a spot check, not a proof; a field wrong on arrays is wrong
by far more than rounding, so the bound costs it nothing.
"""

from collections.abc import Callable
from numbers import Number

import numpy as np

from discrecontinual_equations.function.function import Function

# What a field that cannot take arrays or duals raises on them: a comparison
# or ``if`` on an array, a ``math`` call, a fractional power, a missing method.
_REFUSALS = (TypeError, ValueError, AttributeError, IndexError)
# How far a spot-checked row may sit from the state-at-a-time one, in units of
# the row's largest entry times machine epsilon: a few ulps of rounding in a
# vectorised loop, and of cancellation after it, but nothing a wrong field gives.
_SPOT_ULPS = 16.0
# Fewer states than this are evaluated one at a time: a one-element array
# passes an ``if`` that a longer one would refuse, so it proves nothing.
_MINIMUM_BATCH = 2


class _Dual:
    """A first-order jet over a batch: values ``(n,)``, derivatives ``(n, d)``.

    Column ``k`` of the derivative is the coefficient a :class:`~.jet.Jet`
    seeded along unit direction ``k`` would carry at first order. Every
    operation forms it exactly as the jet does, term by term and in the same
    order, so the two agree to rounding (bit for bit where each operation is
    rounded on its own); the jet's skipped zero terms add a zero, which changes
    nothing but the sign of a zero result. Like the jet,
    it has no comparisons and only integer powers.
    """

    __slots__ = ["derivative", "value"]
    # Make NumPy scalars defer to the reflected operators below.
    __array_ufunc__ = None

    def __init__(self, value: np.ndarray, derivative: np.ndarray) -> None:
        self.value = value
        self.derivative = derivative

    def __add__(self, other: object) -> "_Dual":
        if isinstance(other, _Dual):
            return _Dual(self.value + other.value, self.derivative + other.derivative)
        if isinstance(other, Number):
            return _Dual(self.value + other, self.derivative)  # type: ignore[operator]
        return NotImplemented

    __radd__ = __add__

    def __neg__(self) -> "_Dual":
        return _Dual(-self.value, -self.derivative)

    def __sub__(self, other: object) -> "_Dual":
        if isinstance(other, _Dual):
            return _Dual(self.value - other.value, self.derivative - other.derivative)
        if isinstance(other, Number):
            return _Dual(self.value - other, self.derivative)  # type: ignore[operator]
        return NotImplemented

    def __rsub__(self, other: object) -> "_Dual":
        if isinstance(other, Number):
            # The jet negates, then adds: -a + n, the same number as n - a.
            return _Dual(-self.value + other, -self.derivative)  # type: ignore[operator]
        return NotImplemented

    def __mul__(self, other: object) -> "_Dual":
        if isinstance(other, Number):
            return _Dual(self.value * other, self.derivative * other)  # type: ignore[operator]
        if not isinstance(other, _Dual):
            return NotImplemented
        return _Dual(
            self.value * other.value,
            self.value[:, None] * other.derivative
            + self.derivative * other.value[:, None],
        )

    __rmul__ = __mul__

    def _reciprocal(self) -> "_Dual":
        inverse = 1.0 / self.value
        return _Dual(
            inverse,
            -(self.derivative * inverse[:, None]) / self.value[:, None],
        )

    def __truediv__(self, other: object) -> "_Dual":
        if isinstance(other, Number):
            return _Dual(self.value / other, self.derivative / other)  # type: ignore[operator]
        if not isinstance(other, _Dual):
            return NotImplemented
        return self * other._reciprocal()

    def __rtruediv__(self, other: object) -> "_Dual":
        if not isinstance(other, Number):
            return NotImplemented
        inverse = self._reciprocal()
        return _Dual(other * inverse.value, other * inverse.derivative)  # type: ignore[operator]

    def __pow__(self, exponent: object) -> "_Dual":
        if not isinstance(exponent, int) or isinstance(exponent, bool):
            return NotImplemented
        if exponent < 0:
            return (self ** (-exponent))._reciprocal()
        result = _Dual(np.ones_like(self.value), np.zeros_like(self.derivative))
        base = self
        power = exponent
        while power:
            if power & 1:
                result = result * base
            base = base * base
            power >>= 1
        return result


def _agrees(batched: np.ndarray, single: np.ndarray) -> bool:
    """Whether a batched row is the state-at-a-time one, to rounding."""
    batched = np.asarray(batched, dtype=float)
    single = np.asarray(single, dtype=float)
    if batched.shape != single.shape:
        return False
    finite = np.isfinite(batched) & np.isfinite(single)
    # NaNs and infinities must sit in the same places, and match exactly.
    if not np.array_equal(batched[~finite], single[~finite], equal_nan=True):
        return False
    if not finite.any():
        return True
    scale = max(np.max(np.abs(batched[finite])), np.max(np.abs(single[finite])))
    bound = _SPOT_ULPS * np.finfo(float).eps * scale
    return bool(np.all(np.abs(batched[finite] - single[finite]) <= bound))


def _stacked(raw: object, count: int) -> np.ndarray:
    """The field's components as ``(count, components)`` floats."""
    components = [
        np.broadcast_to(np.asarray(component, dtype=float), (count,))
        for component in raw  # type: ignore[attr-defined]
    ]
    return np.stack(components, axis=1)


def field_values(
    function: Function,
    states: np.ndarray,
    single: Callable[[np.ndarray], np.ndarray],
    time: float | None = None,
) -> np.ndarray | None:
    """``f`` at every row of ``states`` in one call, or ``None`` if it cannot be.

    ``single`` is the caller's own evaluation at one state, against which the
    first and last rows are checked to rounding (:func:`_agrees`).
    """
    count = states.shape[0]
    if count < _MINIMUM_BATCH:
        return None
    try:
        coordinates = [states[:, i] for i in range(states.shape[1])]
        raw = function.eval(point=coordinates, time=time)  # type: ignore[arg-type]
        values = _stacked(raw, count)
    except _REFUSALS:
        return None
    for row in (0, count - 1):
        if not _agrees(values[row], single(states[row])):
            return None
    return values


def field_jacobians(
    function: Function,
    states: np.ndarray,
    single: Callable[[np.ndarray], np.ndarray],
    time: float = 0.0,
) -> np.ndarray | None:
    """``df/dx`` at every row of ``states``, shape ``(n, d, d)``, or ``None``.

    The field is evaluated once on :class:`_Dual` coordinates seeded along
    every unit direction together. ``single`` is the caller's own Jacobian at
    one state, against which the first and last are checked to rounding.
    """
    count, dimension = states.shape
    if count < _MINIMUM_BATCH:
        return None
    seeds = [
        _Dual(
            np.array(states[:, i], dtype=float),
            np.broadcast_to(np.eye(dimension)[i], (count, dimension)),
        )
        for i in range(dimension)
    ]
    try:
        raw = function.eval(point=seeds, time=time)  # type: ignore[arg-type]
        rows = []
        for component in raw:  # type: ignore[attr-defined]
            if isinstance(component, _Dual):
                rows.append(np.broadcast_to(component.derivative, (count, dimension)))
            elif isinstance(component, Number):
                rows.append(np.zeros((count, dimension)))
            else:
                return None
        jacobians = np.stack(rows, axis=1).astype(float)
    except _REFUSALS:
        return None
    for row in (0, count - 1):
        if not _agrees(jacobians[row], single(states[row])):
            return None
    return jacobians
