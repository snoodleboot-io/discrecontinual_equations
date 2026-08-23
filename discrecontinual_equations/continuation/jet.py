"""Forward-mode automatic differentiation by truncated Taylor jets.

A :class:`Jet` carries the Taylor coefficients of a scalar quantity along a single
direction ``s``: entry ``i`` is ``f**(i)(0) / i!``. Evaluating a vector field with
each state entry replaced by ``Jet.seed(x0[i], v[i], order)`` yields, per output
component, the coefficients of ``f(x0 + s v)`` - so coefficient ``m`` equals
``D**m f(v, ..., v) / m!``. Directional derivatives to any order follow exactly,
with no step size and no finite-difference cancellation.

Jets support arithmetic (``+ - * / **``) over real or complex coefficients, which
covers every polynomial or rational vector field. Fields built from transcendental
functions need a jet-aware elementary library or the finite-difference provider.
"""

from numbers import Number

_Scalar = complex | float | int


class Jet:
    """A truncated Taylor series in one variable with real or complex coefficients."""

    __slots__ = ["_coefficients"]

    def __init__(self, coefficients: list[_Scalar]) -> None:
        self._coefficients = list(coefficients)

    @property
    def coefficients(self) -> list[_Scalar]:
        """Taylor coefficients ``[f, f', f''/2!, ...]`` along the seed direction."""
        return self._coefficients

    @property
    def order(self) -> int:
        """Highest retained Taylor order."""
        return len(self._coefficients) - 1

    @classmethod
    def constant(cls, value: _Scalar, order: int) -> "Jet":
        """A jet with no dependence on the seed direction."""
        return cls([value] + [0.0] * order)

    @classmethod
    def seed(cls, value: _Scalar, direction: _Scalar, order: int) -> "Jet":
        """A jet seeded so its first-order coefficient is ``direction``."""
        return cls([value, direction] + [0.0] * (order - 1))

    def _coerce(self, other: object) -> "Jet":
        if isinstance(other, Jet):
            return other
        if isinstance(other, Number):
            return Jet.constant(other, self.order)  # type: ignore[arg-type]
        return NotImplemented

    def __add__(self, other: object) -> "Jet":
        if isinstance(other, Jet):
            return Jet(
                [
                    a + b
                    for a, b in zip(
                        self._coefficients,
                        other._coefficients,
                        strict=True,
                    )
                ],
            )
        if isinstance(other, Number):
            coefficients = list(self._coefficients)
            coefficients[0] = coefficients[0] + other
            return Jet(coefficients)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self) -> "Jet":
        return Jet([-a for a in self._coefficients])

    def __sub__(self, other: object) -> "Jet":
        if isinstance(other, Jet):
            return Jet(
                [
                    a - b
                    for a, b in zip(
                        self._coefficients,
                        other._coefficients,
                        strict=True,
                    )
                ],
            )
        if isinstance(other, Number):
            coefficients = list(self._coefficients)
            coefficients[0] = coefficients[0] - other
            return Jet(coefficients)
        return NotImplemented

    def __rsub__(self, other: object) -> "Jet":
        if isinstance(other, Number):
            coefficients = [-a for a in self._coefficients]
            coefficients[0] = coefficients[0] + other
            return Jet(coefficients)
        right = self._coerce(other)
        if right is NotImplemented:
            return NotImplemented
        return right + (-self)

    def __mul__(self, other: object) -> "Jet":
        if isinstance(other, Number):
            return Jet([a * other for a in self._coefficients])
        if not isinstance(other, Jet):
            return NotImplemented
        order = self.order
        left = self._coefficients
        other_coefficients = other._coefficients
        product: list[_Scalar] = [0.0] * (order + 1)
        for i in range(order + 1):
            left_i = left[i]
            if left_i == 0.0:
                continue
            for j in range(order + 1 - i):
                product[i + j] += left_i * other_coefficients[j]
        return Jet(product)

    __rmul__ = __mul__

    def _reciprocal(self) -> "Jet":
        order = self.order
        coefficients = self._coefficients
        leading = coefficients[0]
        inverse: list[_Scalar] = [0.0] * (order + 1)
        inverse[0] = 1.0 / leading
        for n in range(1, order + 1):
            total = sum(coefficients[k] * inverse[n - k] for k in range(1, n + 1))
            inverse[n] = -total / leading
        return Jet(inverse)

    def __truediv__(self, other: object) -> "Jet":
        if isinstance(other, Number):
            return Jet([a / other for a in self._coefficients])
        if not isinstance(other, Jet):
            return NotImplemented
        return self * other._reciprocal()

    def __rtruediv__(self, other: object) -> "Jet":
        right = self._coerce(other)
        if right is NotImplemented:
            return NotImplemented
        return right * self._reciprocal()

    def __pow__(self, exponent: int) -> "Jet":
        if not isinstance(exponent, int):
            return NotImplemented
        if exponent < 0:
            return (self**-exponent)._reciprocal()
        result = Jet.constant(1.0, self.order)
        base = self
        power = exponent
        while power:
            if power & 1:
                result = result * base
            base = base * base
            power >>= 1
        return result
