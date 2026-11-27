"""Series of different orders combine at the lower order (DEQ-173).

A series truncated at degree ``p`` says nothing about its terms above ``p``, so
the sum, product or quotient of it and a higher-order series is known only to
degree ``p``. The reference is the exact polynomial: with ``x = 1 + s`` known to
degree 1 and ``y = 1 + s + s^2 + s^3`` known to degree 3, every operation's
degree-1 part is exact and nothing above degree 1 may be kept. The left operand's
order was kept before, so ``x * y`` carried ``s^2 + s^3`` terms that the true
``x`` (whose ``s^2`` coefficient is unknown) does not determine.
"""

import operator

import pytest

from discrecontinual_equations.continuation.bivariate import Bivariate
from discrecontinual_equations.continuation.multivariate import Multivariate

_OPERATIONS = [operator.add, operator.sub, operator.mul, operator.truediv]
# Degree-0 and degree-1 coefficients of x op y, x = 1 + s, y = 1 + s + ...:
# sum 2 + 2s, difference 0, product 1 + 2s, quotient 1 + 0s.
_EXACT = {
    operator.add: {(0, 0): 2.0, (1, 0): 2.0},
    operator.sub: {},
    operator.mul: {(0, 0): 1.0, (1, 0): 2.0},
    operator.truediv: {(0, 0): 1.0},
}


def _pair_bivariate() -> tuple[Bivariate, Bivariate]:
    low = Bivariate({(0, 0): 1.0, (1, 0): 1.0}, 1)
    high = Bivariate({(0, 0): 1.0, (1, 0): 1.0, (2, 0): 1.0, (3, 0): 1.0}, 3)
    return low, high


def _pair_multivariate() -> tuple[Multivariate, Multivariate]:
    low = Multivariate({(0, 0): 1.0, (1, 0): 1.0}, 1, 2)
    high = Multivariate({(0, 0): 1.0, (1, 0): 1.0, (2, 0): 1.0, (3, 0): 1.0}, 3, 2)
    return low, high


@pytest.mark.parametrize("pair", [_pair_bivariate, _pair_multivariate])
@pytest.mark.parametrize("operation", _OPERATIONS, ids=lambda op: op.__name__)
@pytest.mark.parametrize("low_on_left", [True, False], ids=["low-left", "high-left"])
def test_the_result_has_the_lower_order(pair, operation, low_on_left):
    low, high = pair()
    result = operation(low, high) if low_on_left else operation(high, low)
    assert result.order == 1
    assert all(sum(key) <= 1 for key in result.coefficients)
    if low_on_left:
        assert result.coefficients == pytest.approx(_EXACT[operation])


def test_multivariate_series_of_different_dimensions_are_refused():
    two = Multivariate({(1, 0): 1.0}, 2, 2)
    three = Multivariate({(1, 0, 0): 1.0}, 2, 3)
    for operation in _OPERATIONS:
        with pytest.raises(ValueError, match="2 and 3 variables"):
            operation(two, three)
