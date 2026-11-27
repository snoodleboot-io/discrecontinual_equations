"""n-variable truncated-Taylor series for center manifold reduction.

A :class:`Multivariate` holds the coefficients of a truncated power series in
``dimension`` variables up to a fixed total degree. Evaluating a vector field on
seeded multivariate series gives its full local Taylor expansion in the split
(center, hyperbolic) coordinates; substituting the center manifold back in - by
evaluating that expansion at :class:`~.bivariate.Bivariate` arguments - yields the
reduced field on the center coordinates.

The arithmetic is Taylor-mode algorithmic differentiation (Griewank and Walther
(2008), Chapter 13) carried as a full multivariate polynomial: each product is the
convolution of the two coefficient tables with every term above the total degree
``order`` dropped. The number of monomials of degree at most ``order`` in ``d``
variables is ``C(order + d, d)``, and a product touches every pair, so the cost
grows like ``C(order + d, d)^2``; this suits the low-dimensional, low-order
expansions of a center manifold reduction, not large systems. As in
:mod:`~.bivariate`, the arithmetic is ``+ - * /`` and ``**`` to non-negative
integer powers. Division by a number scales the coefficients; division by a
series with a non-zero constant term is the truncated quotient of
:func:`~.bivariate.truncated_quotient` (Neidinger 2010, Sections 6 and 8), and a
zero constant term raises ``ZeroDivisionError``, since the quotient then has no
power series at the expansion point.

References:
- Griewank, A. and Walther, A. "Evaluating Derivatives: Principles and
  Techniques of Algorithmic Differentiation", 2nd ed., SIAM, 2008,
  doi:10.1137/1.9780898717761, Chapter 13 ("Taylor and Tensor Coefficients").
- Neidinger, R. D. "Introduction to Automatic Differentiation and MATLAB
  Object-Oriented Programming", SIAM Review 52(3), 545-563, 2010,
  doi:10.1137/080743627, Sections 6 and 8 (division of series).
"""

from numbers import Number
from typing import cast

from discrecontinual_equations.continuation.bivariate import (
    Bivariate,
    truncated_quotient,
)

_Scalar = complex | float | int
_LINEAR_DEGREE = 2


class Multivariate:
    """A truncated power series in several variables to a fixed total degree.

    Args:
        coefficients: Map from exponent tuple to coefficient; terms above
            ``order`` and zero terms are dropped.
        order: Maximum retained total degree.
        dimension: Number of variables.

    References:
    - Griewank and Walther (2008), Chapter 13.
    """

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
        """The series ``value + x_axis``.

        Args:
            axis: Which variable to seed.
            value: The constant term.
            order: Maximum retained total degree.
            dimension: Number of variables.

        Returns:
            The seeded independent variable.
        """
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

    def __truediv__(self, other: object) -> "Multivariate":
        """Divide by a number, or by a series with a non-zero constant term.

        A series quotient is truncated at this series' ``order``; a zero
        constant term in the denominator raises ``ZeroDivisionError`` (see
        :func:`~.bivariate.truncated_quotient`).
        """
        if isinstance(other, Number):
            scale = cast("_Scalar", other)
            return Multivariate(
                {key: value / scale for key, value in self._coefficients.items()},
                self._order,
                self._dimension,
            )
        if not isinstance(other, Multivariate):
            return NotImplemented
        quotient = truncated_quotient(
            self._coefficients,
            other._coefficients,
            self._order,
            self._dimension,
        )
        return Multivariate(quotient, self._order, self._dimension)

    def __rtruediv__(self, other: object) -> "Multivariate":
        right = self._combine(other)
        if right is NotImplemented:
            return NotImplemented
        return right / self

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
        """Evaluate the series with each variable replaced by a Bivariate.

        Args:
            arguments: One :class:`~.bivariate.Bivariate` per variable.
            order: Truncation degree of the result.

        Returns:
            The composed two-variable series.
        """
        result = Bivariate({}, order)
        for key, value in self._coefficients.items():
            term = Bivariate.constant(value, order)
            for axis, power in enumerate(key):
                if power:
                    term = term * (arguments[axis] ** power)
            result = result + term
        return result
