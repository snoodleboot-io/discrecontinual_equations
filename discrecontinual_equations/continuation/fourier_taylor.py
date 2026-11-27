"""Truncated Taylor series in one variable, sampled on a grid of phases.

A :class:`GridSeries` holds the Taylor coefficients in ``sigma`` of a scalar
quantity at every point of a grid in the phase ``theta`` at once: entry
``[n, j]`` is the coefficient of ``sigma**n`` at the ``j``-th phase. Evaluating
a vector field with each state entry replaced by such a series gives the
Fourier-Taylor expansion of the field along a parameterized manifold of a
cycle, every phase in one pass of numpy arithmetic. Products are Cauchy
products in ``sigma`` and pointwise in ``theta``, which is the pseudo-spectral
form of the Fourier convolutions in Castelli, Lessard and Mireles James (2015).

Like :class:`~.jet.Jet`, the series supports ``+ - * /`` and powers (integer or
real), which covers every polynomial or rational vector field and fields with a
radius ``(x * x + y * y) ** 0.5``. A field written with ``math`` functions
rejects it; :func:`evaluate_field` says so.
"""

from numbers import Number

import numpy as np

from discrecontinual_equations.function.function import Function


class GridSeries:
    """A truncated power series in ``sigma`` at every phase of a grid."""

    __slots__ = ["_coefficients"]

    def __init__(self, coefficients: np.ndarray) -> None:
        self._coefficients = coefficients

    @property
    def coefficients(self) -> np.ndarray:
        """Coefficients, ``[order + 1, phases]``: row ``n`` multiplies ``sigma**n``."""
        return self._coefficients

    @property
    def order(self) -> int:
        """Highest retained power of ``sigma``."""
        return self._coefficients.shape[0] - 1

    def _coerce(self, other: object) -> "GridSeries":
        if isinstance(other, GridSeries):
            return other
        if isinstance(other, Number):
            constant = np.zeros_like(self._coefficients)
            constant[0] = other
            return GridSeries(constant)
        return NotImplemented

    def __add__(self, other: object) -> "GridSeries":
        right = self._coerce(other)
        if right is NotImplemented:
            return NotImplemented
        return GridSeries(self._coefficients + right._coefficients)

    __radd__ = __add__

    def __neg__(self) -> "GridSeries":
        return GridSeries(-self._coefficients)

    def __pos__(self) -> "GridSeries":
        return self

    def __sub__(self, other: object) -> "GridSeries":
        right = self._coerce(other)
        if right is NotImplemented:
            return NotImplemented
        return GridSeries(self._coefficients - right._coefficients)

    def __rsub__(self, other: object) -> "GridSeries":
        right = self._coerce(other)
        if right is NotImplemented:
            return NotImplemented
        return GridSeries(right._coefficients - self._coefficients)

    def __mul__(self, other: object) -> "GridSeries":
        if isinstance(other, Number):
            return GridSeries(self._coefficients * other)
        right = self._coerce(other)
        if right is NotImplemented:
            return NotImplemented
        a = self._coefficients
        b = right._coefficients
        product = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=a.dtype)
        for n in range(product.shape[0]):
            # The Cauchy product: c_n = sum_k a_k b_(n-k), phase by phase.
            product[n] = np.einsum("kj,kj->j", a[: n + 1], b[n::-1])
        return GridSeries(product)

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> "GridSeries":
        if isinstance(other, Number):
            return GridSeries(self._coefficients / other)
        right = self._coerce(other)
        if right is NotImplemented:
            return NotImplemented
        return self * right.reciprocal()

    def __rtruediv__(self, other: object) -> "GridSeries":
        left = self._coerce(other)
        if left is NotImplemented:
            return NotImplemented
        return left * self.reciprocal()

    def reciprocal(self) -> "GridSeries":
        """``1 / self``: ``q_0 = 1 / a_0``, ``q_n = -(sum_k a_k q_(n-k)) / a_0``."""
        a = self._coefficients
        inverse = np.zeros_like(a)
        inverse[0] = 1.0 / a[0]
        for n in range(1, a.shape[0]):
            inverse[n] = -np.einsum("kj,kj->j", a[1 : n + 1], inverse[n - 1 :: -1])
            inverse[n] = inverse[n] / a[0]
        return GridSeries(inverse)

    def __pow__(self, exponent: object) -> "GridSeries":
        if isinstance(exponent, int) and exponent >= 0:
            result = GridSeries(self._coerce(1.0)._coefficients)
            base = self
            power = exponent
            while power:
                if power & 1:
                    result = result * base
                base = base * base
                power >>= 1
            return result
        if isinstance(exponent, (int, float)):
            return self._real_power(float(exponent))
        return NotImplemented

    def _real_power(self, exponent: float) -> "GridSeries":
        """``a**p`` by the recurrence from ``a c' = p a' c`` (Knuth, TAOCP 4.7).

        ``c_0 = a_0**p`` and ``n a_0 c_n = sum_(k=1..n) (p k - (n - k)) a_k c_(n-k)``.
        """
        a = self._coefficients
        power = np.zeros_like(a)
        power[0] = a[0] ** exponent
        for n in range(1, a.shape[0]):
            weights = np.array([exponent * k - (n - k) for k in range(1, n + 1)])
            total = np.einsum("k,kj,kj->j", weights, a[1 : n + 1], power[n - 1 :: -1])
            power[n] = total / (n * a[0])
        return GridSeries(power)


def evaluate_field(
    function: Function,
    coefficients: np.ndarray,
) -> np.ndarray:
    """The Taylor coefficients of ``f(P)`` for ``P`` given by ``coefficients``.

    ``coefficients`` is ``[order + 1, phases, dimension]``, the Taylor
    coefficients of each state component at each phase; the result has the
    same shape. A component the field returns as a plain number is a constant.
    """
    series = [
        GridSeries(coefficients[:, :, index]) for index in range(coefficients.shape[2])
    ]
    try:
        # Fields are written against floats and run on series by duck typing,
        # as they are on the scalar Jet; the annotation names only the floats.
        raw = function.eval(point=series, time=None)  # type: ignore[arg-type]
    except TypeError as error:
        message = (
            "the field cannot be evaluated on Taylor series; write it with "
            "arithmetic and powers only (no math functions)"
        )
        raise TypeError(message) from error
    result = np.zeros_like(coefficients)
    for index, component in enumerate(raw):
        if isinstance(component, GridSeries):
            result[:, :, index] = component.coefficients
        else:
            result[0, :, index] = component
    return result


def field_jacobians(function: Function, states: np.ndarray) -> np.ndarray:
    """``Df`` at each of ``states`` (``[points, d]``), as ``[points, d, d]``.

    Column ``c`` is the first-order coefficient of ``f(x + sigma e_c)``, exact
    to rounding, and fields with a radius are differentiated like any other.
    Every column comes from one evaluation: the grid holds ``d`` copies of the
    states, copy ``c`` seeded along ``e_c``.
    """
    points, dimension = states.shape
    seeded = np.zeros((2, dimension * points, dimension))
    seeded[0] = np.tile(states, (dimension, 1))
    for column in range(dimension):
        seeded[1, column * points : (column + 1) * points, column] = 1.0
    derivatives = evaluate_field(function, seeded)[1]
    # derivatives[c * points + j, i] is d f_i / d x_c at state j.
    return derivatives.reshape(dimension, points, dimension).transpose(1, 2, 0)
