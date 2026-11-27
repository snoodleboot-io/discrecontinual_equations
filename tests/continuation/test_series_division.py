"""Division of truncated multivariate series (DEQ-143).

``Bivariate`` and ``Multivariate`` divide by a number, and by a series whose
constant term is non-zero, through the truncated quotient recurrence; a series
with a zero constant term is refused. The references here are built without the
code under test: closed-form coefficients of ``1 / (1 - s - t)`` and of
``1 / (1 + s)^2``, and the focal values of a normal form whose orbits a positive
factor does not change.
"""

from math import comb, factorial
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.bivariate import Bivariate
from discrecontinual_equations.continuation.focal import PlanarLyapunov, recurse
from discrecontinual_equations.continuation.multivariate import Multivariate
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.variable import Variable

_ORDER = 6


def _s() -> Bivariate:
    return Bivariate({(1, 0): 1.0}, _ORDER)


def _t() -> Bivariate:
    return Bivariate({(0, 1): 1.0}, _ORDER)


class TestDivisionByNumber(TestCase):
    def test_bivariate_divided_by_two_halves_every_coefficient(self):
        x = Bivariate({(0, 0): 3.0, (1, 0): 1.0, (1, 2): -4.0}, _ORDER)
        half = x / 2
        assert half.coefficients == {(0, 0): 1.5, (1, 0): 0.5, (1, 2): -2.0}
        assert half.order == _ORDER

    def test_multivariate_divided_by_a_complex_number(self):
        x = Multivariate({(0, 0, 0): 2.0, (0, 1, 1): 4.0j}, _ORDER, 3)
        quotient = x / 2j
        assert quotient.coefficients == {(0, 0, 0): -1.0j, (0, 1, 1): 2.0}
        assert quotient.dimension == 3

    def test_division_by_zero_number_raises(self):
        with pytest.raises(ZeroDivisionError):
            _ = _s() / 0
        with pytest.raises(ZeroDivisionError):
            _ = Multivariate.variable(0, 1.0, _ORDER, 2) / 0.0

    def test_division_by_something_else_is_not_implemented(self):
        with pytest.raises(TypeError):
            _ = _s() / "two"


class TestDivisionBySeries(TestCase):
    def test_reciprocal_of_one_minus_s_minus_t_has_binomial_coefficients(self):
        # 1 / (1 - s - t) = sum (s + t)^n: the coefficient of s^a t^b is C(a+b, a).
        quotient = 1 / (1 - _s() - _t())
        for a in range(_ORDER + 1):
            for b in range(_ORDER + 1 - a):
                assert quotient.coefficient(a, b) == pytest.approx(comb(a + b, a))

    def test_inverse_square_has_alternating_linear_coefficients(self):
        # 1 / (1 + s)^2 = sum (-1)^a (a + 1) s^a, and nothing in t.
        quotient = Bivariate.constant(1.0, _ORDER) / (1 + _s()) ** 2
        for a in range(_ORDER + 1):
            assert quotient.coefficient(a, 0) == pytest.approx((-1) ** a * (a + 1))
        assert all(b == 0 for (_, b) in quotient.coefficients)

    def test_terms_of_the_top_degree_enter_the_quotient(self):
        # e^s / (1 + t) = sum s^a / a! * sum (-t)^b: the coefficient of s^a t^b is
        # (-1)^b / a!. The numerator's s^order term must reach the quotient, and so
        # must the denominator's top-degree part (here through 1 / (1 + t^order)).
        exp_s = Bivariate(
            {(a, 0): 1.0 / factorial(a) for a in range(_ORDER + 1)},
            _ORDER,
        )
        quotient = exp_s / (1 + _t())
        for a in range(_ORDER + 1):
            for b in range(_ORDER + 1 - a):
                assert quotient.coefficient(a, b) == pytest.approx(
                    (-1) ** b / factorial(a),
                )
        top = 1 / (1 + _t() ** _ORDER)
        assert top.coefficient(0, _ORDER) == pytest.approx(-1.0)

    def test_quotient_times_denominator_recovers_numerator(self):
        numerator = 2 + 3 * _s() - _t() * _s() + _t() ** 4
        denominator = 0.5 - _s() + 2 * _t() + _s() * _t()
        product = (numerator / denominator) * denominator
        for key in set(product.coefficients) | set(numerator.coefficients):
            assert product.coefficient(*key) == pytest.approx(
                numerator.coefficient(*key),
                abs=1e-12,
            )

    def test_multivariate_quotient_matches_geometric_series(self):
        # 1 / (1 - x - y - z): the coefficient of x^a y^b z^c is the multinomial.
        dimension = 3
        x, y, z = (
            Multivariate.variable(axis, 0.0, 4, dimension) for axis in range(dimension)
        )
        quotient = 1 / (1 - x - y - z)
        for key, value in quotient.coefficients.items():
            a, b, c = key
            multinomial = comb(a + b + c, a) * comb(b + c, b)
            assert value == pytest.approx(multinomial)
        assert len(quotient.coefficients) == comb(4 + dimension, dimension)

    def test_zero_constant_term_is_refused_with_a_reason(self):
        with pytest.raises(ZeroDivisionError, match="constant term"):
            _ = 1 / _s()
        y = Multivariate.variable(1, 0.0, _ORDER, 2)
        with pytest.raises(ZeroDivisionError, match="constant term"):
            _ = y / (y * y)


class TestRationalFieldFocalValues(TestCase):
    """A positive factor equal to 1 at the focus keeps the orbits, so the focal value.

    The normal form ``(-t + a s r^2 + b s r^4, s + a t r^2 + b t r^4)`` has
    ``eta4 = a`` and, at ``a = 0``, ``eta6 = b``. Dividing it by ``1 + s + t^2``
    reparametrises time without moving an orbit; the Poincare map, and with it the
    first non-zero focal value, is unchanged.
    """

    @staticmethod
    def _field(a: float, b: float) -> list[Bivariate]:
        s, t = _s(), _t()
        r2 = s * s + t * t
        speed = 1 + s + t * t
        return [
            (-1.0 * t + a * s * r2 + b * s * r2 * r2) / speed,
            (s + a * t * r2 + b * t * r2 * r2) / speed,
        ]

    def test_first_focal_value_survives_the_division(self):
        eta4, _ = recurse(self._field(-0.3, 0.7))
        assert eta4 == pytest.approx(-0.3, abs=1e-12)

    def test_second_focal_value_survives_at_a_bautin_point(self):
        eta4, eta6 = recurse(self._field(0.0, 0.7))
        assert eta4 == pytest.approx(0.0, abs=1e-12)
        assert eta6 == pytest.approx(0.7, abs=1e-12)

    def test_planar_lyapunov_on_a_rational_field_off_the_origin(self):
        # The normal form centred at (1, -2), and the same divided by 2 + (x - 1):
        # the quotient rotates at frequency 1/2, which PlanarLyapunov scales out.
        # Its eigenvector normalisation scales eta4 (the value is -0.15, not -0.3,
        # for both), so the reference is the undivided field through the same path.
        values = []
        for speed, frequency in ((None, 1.0), (2.0, 0.5)):
            _ShiftedRational.speed = speed
            lyapunov = PlanarLyapunov(
                _ShiftedRational(
                    [Variable(), Variable()], [], [Variable(), Variable()]
                ),
                np.array(_CENTRE),
                np.array([[0.0, -frequency], [frequency, 0.0]]),
                0.0,
            )
            values.append(lyapunov.quantities()[0])
        assert values[0] < 0
        assert values[1] == pytest.approx(values[0], abs=1e-12)


_CENTRE = (1.0, -2.0)


class _ShiftedRational(DeterministicFunction):
    speed: float | None = None

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        s, t = point[0] - _CENTRE[0], point[1] - _CENTRE[1]
        r2 = s * s + t * t
        drift = [-1.0 * t - 0.3 * s * r2, s - 0.3 * t * r2]
        if self.speed is None:
            return drift
        return [component / (self.speed + s) for component in drift]
