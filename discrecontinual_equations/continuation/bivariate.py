"""Two-variable truncated-Taylor series for planar normal-form computations.

A :class:`Bivariate` holds the coefficients of a truncated power series in two
variables ``(s, t)`` up to a fixed total degree. Evaluating a planar vector field
with each coordinate replaced by a seeded :class:`Bivariate` gives the full local
Taylor expansion of the field - every mixed monomial, not just the symmetric
directional contractions - which the Lyapunov focal-value recursion needs.

Only arithmetic is provided (``+ - * / **``), which covers polynomial and rational
planar fields; that is the setting for the classical focal-value computation.

This is Taylor-mode algorithmic differentiation in two variables: each arithmetic
operation is applied to truncated series, products by the Cauchy convolution of
coefficients with every term above the total degree ``order`` dropped, which is
the propagation of Taylor coefficients of Griewank and Walther (2008, Chapter 13),
here carried as a full multivariate polynomial rather than through univariate
directional propagation. A product costs ``O(order^4)`` coefficient multiplications.
Division by a number scales every coefficient. Division by a series ``w`` whose
constant term ``w_0`` is non-zero is the truncated quotient: from ``u = h w``,
taken degree by degree, ``h_k = (u_k - sum_{j<k} h_j w_{k-j}) / w_0``, where
``u_k``, ``w_k`` and ``h_k`` are the parts of exact total degree ``k`` - the
univariate recurrence of Neidinger (2010, Section 6) applied to homogeneous
parts, as his Section 8 suggests for multivariable series. A denominator whose
constant term is zero vanishes at the expansion point, so the quotient is not a
power series there; it raises ``ZeroDivisionError`` instead.

References:
- Griewank, A. and Walther, A. "Evaluating Derivatives: Principles and
  Techniques of Algorithmic Differentiation", 2nd ed., SIAM, 2008,
  doi:10.1137/1.9780898717761, Chapter 13 ("Taylor and Tensor Coefficients").
- Neidinger, R. D. "Introduction to Automatic Differentiation and MATLAB
  Object-Oriented Programming", SIAM Review 52(3), 545-563, 2010,
  doi:10.1137/080743627, Section 6 (division of series, from the
  multiplication theorem (6.2)) and Section 8 (multivariable series).
"""

from numbers import Number
from typing import TypeVar, cast

_Scalar = complex | float | int
_Key = TypeVar("_Key", bound=tuple[int, ...])


def truncated_quotient(
    numerator: dict[_Key, _Scalar],
    denominator: dict[_Key, _Scalar],
    order: int,
    dimension: int,
) -> dict[_Key, _Scalar]:
    """Coefficients of ``numerator / denominator`` to total degree ``order``.

    The idea: ``u = h w`` holds degree by degree, so the part of ``h`` of degree
    ``k`` is ``(u_k - sum_{j<k} h_j w_{k-j}) / w_0``, each term a product of
    homogeneous parts. Every coefficient of degree at most ``order`` is exact
    (up to rounding); nothing above it is formed.

    Args:
        numerator: Exponent tuple to coefficient of ``u``.
        denominator: Exponent tuple to coefficient of ``w``.
        order: Maximum retained total degree.
        dimension: Number of variables (the length of every exponent tuple).

    Returns:
        Exponent tuple to coefficient of the quotient ``h``.

    Raises:
        ZeroDivisionError: If the constant term ``w_0`` is zero: then ``w``
            vanishes at the expansion point and ``u / w`` has no power series.

    References:
    - Neidinger (2010), Section 6, the division formula derived from (6.2);
      Section 8 for its extension to multivariable series.
    """
    leading = denominator.get(cast(_Key, (0,) * dimension), 0)
    if leading == 0:
        message = (
            "cannot divide by a series whose constant term is zero: it vanishes "
            "at the expansion point, so the quotient is not a power series there; "
            "expand about a point where the denominator is non-zero"
        )
        raise ZeroDivisionError(message)
    denominator_parts = _by_degree(denominator, order)
    numerator_parts = _by_degree(numerator, order)
    quotient_parts: list[dict[_Key, _Scalar]] = []
    for degree in range(order + 1):
        part = dict(numerator_parts[degree])
        for lower in range(degree):
            for left_key, left_value in quotient_parts[lower].items():
                for right_key, right_value in denominator_parts[degree - lower].items():
                    key = cast(
                        _Key,
                        tuple(a + b for a, b in zip(left_key, right_key, strict=True)),
                    )
                    part[key] = part.get(key, 0.0) - left_value * right_value
        quotient_parts.append({key: value / leading for key, value in part.items()})
    return {key: value for part in quotient_parts for key, value in part.items()}


def _by_degree(
    coefficients: dict[_Key, _Scalar],
    order: int,
) -> list[dict[_Key, _Scalar]]:
    parts: list[dict[_Key, _Scalar]] = [{} for _ in range(order + 1)]
    for key, value in coefficients.items():
        degree = sum(key)
        if degree <= order:
            parts[degree][key] = value
    return parts


class Bivariate:
    """A truncated power series in two variables to a fixed total degree.

    Args:
        coefficients: Map from exponent pair ``(a, b)`` to the coefficient of
            ``s^a t^b``; terms above ``order`` and zero terms are dropped.
        order: Maximum retained total degree.

    References:
    - Griewank and Walther (2008), Chapter 13.
    """

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
        """A series ``value + s`` (axis 0) or ``value + t`` (axis 1).

        Args:
            value: The constant term (the point's coordinate).
            axis: 0 for ``s``, 1 for ``t``.
            order: Maximum retained total degree.

        Returns:
            The seeded independent variable.
        """
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

    def __truediv__(self, other: object) -> "Bivariate":
        """Divide by a number, or by a series with a non-zero constant term.

        A series quotient is truncated at this series' ``order``; a zero
        constant term in the denominator raises ``ZeroDivisionError`` (see
        :func:`truncated_quotient`).
        """
        if isinstance(other, Number):
            scale = cast(_Scalar, other)
            return Bivariate(
                {key: value / scale for key, value in self._coefficients.items()},
                self._order,
            )
        if not isinstance(other, Bivariate):
            return NotImplemented
        quotient = truncated_quotient(
            self._coefficients,
            other._coefficients,
            self._order,
            2,
        )
        return Bivariate(quotient, self._order)

    def __rtruediv__(self, other: object) -> "Bivariate":
        right = self._combine(other)
        if right is NotImplemented:
            return NotImplemented
        return right / self

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
