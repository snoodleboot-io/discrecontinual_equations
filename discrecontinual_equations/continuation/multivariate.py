"""n-variable truncated-Taylor series for center manifold reduction.

A :class:`Multivariate` holds the coefficients of a truncated power series in
``dimension`` variables up to a fixed total degree. Evaluating a vector field on
seeded multivariate series gives its full local Taylor expansion in the split
(center, hyperbolic) coordinates; substituting the center manifold back in - by
evaluating that expansion at :class:`~.bivariate.Bivariate` arguments - yields the
reduced field on the center coordinates.
"""

from numbers import Number

from discrecontinual_equations.continuation.bivariate import Bivariate

_Scalar = complex | float | int
_LINEAR_DEGREE = 2


class Multivariate:
    """A truncated power series in several variables to a fixed total degree."""

    __slots__ = ["_coefficients", "_dimension", "_order"]

    def __init__(
        self,
        coefficients: dict[tuple[int, ...], _Scalar],
        order: int,
        dimension: int,
    ) -> None:
        self._order = order
        self._dimension = dimension
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
    def dimension(self) -> int:
        """Number of variables."""
        return self._dimension

    @property
    def coefficients(self) -> dict[tuple[int, ...], _Scalar]:
        """Mapping from exponent tuple to coefficient."""
        return self._coefficients

    @classmethod
    def constant(cls, value: _Scalar, order: int, dimension: int) -> "Multivariate":
        """A series with only a constant term."""
        return cls({(0,) * dimension: value}, order, dimension)

    @classmethod
    def variable(
        cls,
        axis: int,
        value: _Scalar,
        order: int,
        dimension: int,
    ) -> "Multivariate":
        """The series ``value + x_axis``."""
        key = tuple(1 if i == axis else 0 for i in range(dimension))
        return cls({(0,) * dimension: value, key: 1.0}, order, dimension)

    def _combine(self, other: object) -> "Multivariate":
        if isinstance(other, Multivariate):
            return other
        if isinstance(other, Number):
            return Multivariate.constant(other, self._order, self._dimension)  # type: ignore[arg-type]
        return NotImplemented

    def __add__(self, other: object) -> "Multivariate":
        right = self._combine(other)
        if right is NotImplemented:
            return NotImplemented
        result = dict(self._coefficients)
        for key, value in right._coefficients.items():
            result[key] = result.get(key, 0.0) + value
        return Multivariate(result, self._order, self._dimension)

    __radd__ = __add__

    def __neg__(self) -> "Multivariate":
        return Multivariate(
            {key: -value for key, value in self._coefficients.items()},
            self._order,
            self._dimension,
        )

    def __sub__(self, other: object) -> "Multivariate":
        right = self._combine(other)
        if right is NotImplemented:
            return NotImplemented
        return self + (-right)

    def __rsub__(self, other: object) -> "Multivariate":
        right = self._combine(other)
        if right is NotImplemented:
            return NotImplemented
        return right + (-self)

    def __mul__(self, other: object) -> "Multivariate":
        right = self._combine(other)
        if right is NotImplemented:
            return NotImplemented
        order = self._order
        result: dict[tuple[int, ...], _Scalar] = {}
        for left_key, left_value in self._coefficients.items():
            left_degree = sum(left_key)
            for right_key, right_value in right._coefficients.items():
                if left_degree + sum(right_key) > order:
                    continue
                key = tuple(a + b for a, b in zip(left_key, right_key, strict=True))
                result[key] = result.get(key, 0.0) + left_value * right_value
        return Multivariate(result, order, self._dimension)

    __rmul__ = __mul__

    def __pow__(self, exponent: int) -> "Multivariate":
        if not isinstance(exponent, int) or exponent < 0:
            return NotImplemented
        result = Multivariate.constant(1.0, self._order, self._dimension)
        base = self
        power = exponent
        while power:
            if power & 1:
                result = result * base
            base = base * base
            power >>= 1
        return result

    def linear_coefficient(self, axis: int) -> _Scalar:
        """Coefficient of the degree-one term in the given variable."""
        key = tuple(1 if i == axis else 0 for i in range(self._dimension))
        return self._coefficients.get(key, 0.0)

    def nonlinear(self) -> "Multivariate":
        """The series with its constant and linear terms removed."""
        return Multivariate(
            {
                key: value
                for key, value in self._coefficients.items()
                if sum(key) >= _LINEAR_DEGREE
            },
            self._order,
            self._dimension,
        )

    def compose(self, arguments: list[Bivariate], order: int) -> Bivariate:
        """Evaluate the series with each variable replaced by a Bivariate."""
        result = Bivariate({}, order)
        for key, value in self._coefficients.items():
            term = Bivariate.constant(value, order)
            for axis, power in enumerate(key):
                if power:
                    term = term * (arguments[axis] ** power)
            result = result + term
        return result
