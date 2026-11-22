"""Periodic normal form coefficients against exact and independent values (DEQ-35)."""

import math
from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.cycle_normal_form import CycleNormalForm
from tests.continuation.cycle_oracles import (
    Coordinate,
    ExtendedBautin,
    First,
    FoldTorus,
    RotatedBautin,
    RotatedTakens,
    Second,
)


def _field(kind: type, values: list[float], dimension: int):
    return kind(
        variables=[Coordinate() for _ in range(dimension)],
        parameters=[First(value=values[0]), Second(value=values[1])],
        results=[Coordinate() for _ in range(dimension)],
        time=None,
    )


def _circle(radius: float, omega: float, intervals: int, extra: int) -> tuple:
    period = 2.0 * math.pi / omega
    time = np.linspace(0.0, period, intervals + 1)
    columns = [radius * np.cos(omega * time), radius * np.sin(omega * time)]
    columns += [np.zeros_like(time)] * extra
    return np.column_stack(columns), period


def _bautin_fold(rho: float, intervals: int) -> tuple[float, float]:
    c, e = ExtendedBautin.C, ExtendedBautin.E
    m2 = 3.0 * rho * rho - 2.0 * c * rho
    m1 = -2.0 * rho**3 + c * rho * rho
    states, period = _circle(math.sqrt(rho), 1.0 + e * rho, intervals, 0)
    b = CycleNormalForm(_field(ExtendedBautin, [m1, m2], 2), states, period).fold()
    exact = -rho * (2.0 * c - 6.0 * rho) * (1.0 + e * rho) / e
    return b, exact


class TestFold(TestCase):
    def test_b_on_the_extended_bautin_fold_curve(self) -> None:
        # Both signs: outside the cusp (rho > c/3) and inside it.
        for rho in (0.6, 0.45, 0.2):
            b, exact = _bautin_fold(rho, 40)
            assert abs(b - exact) < 2e-5 * abs(exact), (rho, b, exact)

    def test_b_converges_at_fourth_order(self) -> None:
        # Measured 1.7e-3, 1.05e-4, 6.5e-6, 4.1e-7 relative from 10 to 80.
        coarse = abs(np.subtract(*_bautin_fold(0.6, 20)))
        fine = abs(np.subtract(*_bautin_fold(0.6, 40)))
        assert coarse / fine > 12.0

    def test_b_where_the_fold_is_in_rho_alone(self) -> None:
        # rho' = 2 rho (b1 + u^2): b = -P''(1) / (2 Omega') = -4 / 0.6.
        states, period = _circle(1.0, 1.0, 40, 2)
        b = CycleNormalForm(_field(FoldTorus, [0.0, -0.1], 4), states, period).fold()
        assert abs(b + 4.0 / (2.0 * FoldTorus.E)) < 1e-4


class TestTorus(TestCase):
    def _rotated_bautin(self, b2: float) -> complex:
        states, period = _circle(1.0, 1.0, 40, 1)
        normal = CycleNormalForm(_field(RotatedBautin, [0.0, b2], 3), states, period)
        return normal.torus(RotatedBautin.W0 * period)

    def test_re_d_on_the_rotated_bautin_form(self) -> None:
        # Re d = 4 |c|^2 b2, |c|^2 fixed by int_0^T |v|^2 = 1 over the lifted
        # eigenvector, phase drift (Omega' = E) included.
        w0, e = RotatedBautin.W0, RotatedBautin.E
        weight = 0.25 + 1.0 + e * e / (w0 * w0)
        exact = 4.0 * 0.1 / (2.0 * math.pi * weight)
        d = self._rotated_bautin(0.1)
        assert abs(d.real - exact) < 1e-5 * exact
        assert abs(self._rotated_bautin(-0.1).real + exact) < 1e-5 * exact
        assert abs(self._rotated_bautin(0.0).real) < 1e-6

    def test_re_d_against_an_independent_return_map(self) -> None:
        # The reduced Hopf at b2 = 0.05: a DOP853 return map of the (u, z)
        # system (rtol 1e-13) gives the amplitude gain 2.257e-9 at u-amplitude
        # 5e-4 and 1.809e-8 at 1e-3 over 2 pi / omega, which with the lifted
        # eigenvector's normalisation reads Re d = 1.0257 and 1.0276; their
        # extrapolation to zero amplitude is 1.0238.
        u = -0.05
        rho = 1.0 + u
        omega = math.sqrt(-4.0 * rho * u)
        states, period = _circle(math.sqrt(rho), 1.0 + RotatedTakens.E * u, 40, 1)
        normal = CycleNormalForm(
            _field(RotatedTakens, [-u * u, -u], 3),
            states,
            period,
        )
        d = normal.torus(omega * period)
        assert abs(d.real - 1.0238) < 2e-3
