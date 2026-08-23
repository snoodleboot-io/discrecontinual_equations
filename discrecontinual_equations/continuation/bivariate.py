"""Two-variable truncated-Taylor series for planar normal-form computations.

A :class:`Bivariate` holds the coefficients of a truncated power series in two
variables ``(s, t)`` up to a fixed total degree. Evaluating a planar vector field
with each coordinate replaced by a seeded :class:`Bivariate` gives the full local
Taylor expansion of the field - every mixed monomial, not just the symmetric
directional contractions - which the Lyapunov focal-value recursion needs.

Only arithmetic is provided (``+ - * **``), which covers polynomial and rational
planar fields; that is the setting for the classical focal-value computation.
"""

from numbers import Number

_Scalar = complex | float | int


class Bivariate:
    """A truncated power series in two variables to a fixed total degree."""

    __slots__ = ["_coefficients", "_order"]

    def __init__(
        self,
        coefficients: dict[tuple[int, int], _Scalar],
        order: int,
    ) -> None:
        self._order = order
        self._coefficients = {
            key: value
            for key, value in coefficients.items()
            if sum(key) <= order and value != 0
        }

    @property
    def order(self) -> int:
        """Maximum retained total degree."""
        return self._order

    @property
    def coefficients(self) -> dict[tuple[int, int], _Scalar]:
        """Mapping from exponent pair ``(a, b)`` to the coefficient of ``s^a t^b``."""
        return self._coefficients

    def coefficient(self, first: int, second: int) -> _Scalar:
        """Return the coefficient of ``s^first t^second`` (zero if absent)."""
        return self._coefficients.get((first, second), 0.0)

    @classmethod
    def constant(cls, value: _Scalar, order: int) -> "Bivariate":
        """A series with only a constant term."""
        return cls({(0, 0): value}, order)

    @classmethod
    def seed(cls, value: _Scalar, axis: int, order: int) -> "Bivariate":
        """A series ``value + s`` (axis 0) or ``value + t`` (axis 1)."""
        key = (1, 0) if axis == 0 else (0, 1)
        return cls({(0, 0): value, key: 1.0}, order)

    def _combine(self, other: object) -> "Bivariate":
        if isinstance(other, Bivariate):
            return other
        if isinstance(other, Number):
            return Bivariate.constant(other, self._order)  # type: ignore[arg-type]
        return NotImplemented

    def __add__(self, other: object) -> "Bivariate":
        right = self._combine(other)
        if right is NotImplemented:
            return NotImplemented
        result = dict(self._coefficients)
        for key, value in right._coefficients.items():
            result[key] = result.get(key, 0.0) + value
        return Bivariate(result, self._order)

    __radd__ = __add__

    def __neg__(self) -> "Bivariate":
        return Bivariate(
            {key: -value for key, value in self._coefficients.items()},
            self._order,
        )

    def __sub__(self, other: object) -> "Bivariate":
        right = self._combine(other)
        if right is NotImplemented:
            return NotImplemented
        return self + (-right)

    def __rsub__(self, other: object) -> "Bivariate":
        right = self._combine(other)
        if right is NotImplemented:
            return NotImplemented
        return right + (-self)

    def __mul__(self, other: object) -> "Bivariate":
        right = self._combine(other)
        if right is NotImplemented:
            return NotImplemented
        order = self._order
        result: dict[tuple[int, int], _Scalar] = {}
        for (a, b), left_value in self._coefficients.items():
            for (c, d), right_value in right._coefficients.items():
                if a + b + c + d > order:
                    continue
                key = (a + c, b + d)
                result[key] = result.get(key, 0.0) + left_value * right_value
        return Bivariate(result, order)

    __rmul__ = __mul__

    def __pow__(self, exponent: int) -> "Bivariate":
        if not isinstance(exponent, int) or exponent < 0:
            return NotImplemented
        result = Bivariate.constant(1.0, self._order)
        base = self
        power = exponent
        while power:
            if power & 1:
                result = result * base
            base = base * base
            power >>= 1
        return result

    def derivative(self, axis: int) -> "Bivariate":
        """Partial derivative with respect to ``s`` (axis 0) or ``t`` (axis 1)."""
        result: dict[tuple[int, int], _Scalar] = {}
        for (a, b), value in self._coefficients.items():
            if axis == 0 and a > 0:
                result[(a - 1, b)] = value * a
            elif axis == 1 and b > 0:
                result[(a, b - 1)] = value * b
        return Bivariate(result, self._order)

    def homogeneous(self, degree: int) -> dict[tuple[int, int], _Scalar]:
        """Coefficients of the terms of exactly the given total degree."""
        return {
            key: value
            for key, value in self._coefficients.items()
            if sum(key) == degree
        }
