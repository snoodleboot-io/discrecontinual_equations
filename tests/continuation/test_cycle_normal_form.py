"""Periodic normal form coefficients against exact and independent values (DEQ-35)."""

import math
from unittest import TestCase

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.cycle_normal_form import CycleNormalForm
from tests.continuation.cycle_oracles import (
    Coordinate,
    ExtendedBautin,
    First,
    FoldTorus,
    RotatedBautin,
    RotatedTakens,
    Second,
    ShearedBautin,
    ShearedTakens,
    ThetaFlip,
    ThetaTorus,
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

    def test_b_is_the_same_in_sheared_coordinates(self) -> None:
        c, e, shear = ExtendedBautin.C, ExtendedBautin.E, ShearedBautin.S
        for rho in (0.6, 0.2):
            m2 = 3.0 * rho * rho - 2.0 * c * rho
            m1 = -2.0 * rho**3 + c * rho * rho
            states, period = _circle(math.sqrt(rho), 1.0 + e * rho, 40, 0)
            states[:, 0] += shear * states[:, 1] ** 2
            field = _field(ShearedBautin, [m1, m2], 2)
            b = CycleNormalForm(field, states, period).fold()
            exact = -rho * (2.0 * c - 6.0 * rho) * (1.0 + e * rho) / e
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
        # The same cycle in sheared coordinates. d is normalised by
        # int |v|^2 = 1, which the shear changes: Re d scales by 1 / N^2 with
        # N^2 = int |D phi v|^2 = 1.19220 for the oracle's analytic eigenvector
        # (lifted (du, dz, dtheta), quadrature on 2e5 points), so 1.0238 / N^2.
        states[:, 0] += ShearedTakens.S * states[:, 1] ** 2
        sheared = CycleNormalForm(
            _field(ShearedTakens, [-u * u, -u], 3),
            states,
            period,
        ).torus(omega * period)
        assert abs(sheared.real - 1.0238 / 1.19220) < 2e-3


def _theta_cycle(field, start: tuple[float, float], period: float, intervals: int):
    """The cycle through ``u = start[0], z = start[1]`` on ``theta = 0``, by DOP853."""
    initial = [math.sqrt(1.0 + start[0]), 0.0, start[1]]
    run = solve_ivp(
        lambda _t, state: field.eval(list(state)),
        (0.0, period),
        initial,
        method="DOP853",
        rtol=1e-13,
        atol=1e-15,
        dense_output=True,
    )
    states = run.sol(np.linspace(0.0, period, intervals + 1)).T
    states[-1] = states[0]
    return states


class TestIndependentReturnMap(TestCase):
    """c and d where every term counts, against a return map built without the code.

    The references are the verifier's (outputs/verification/verify-deq35):
    the theta-flow return map by DOP853 at rtol 1e-13, its Taylor coefficients
    by Cauchy integrals over complex initial points, its centre-manifold
    reduction (Kuznetsov, Elements, flip and Neimark-Sacker of maps), and the
    cycle's eigenfunction normalised as the code normalises it
    (``int_0^T |v|^2 = 1``, from DOP853 variational equations): then
    ``c = -C_map / T`` and ``Re d = Re(conj(lambda) c_1) / T``. The same
    pipeline gives the extended Bautin's exact ``b`` to ten figures. Neither
    oracle has a symmetry, and both have ``a != 0``: the flip's ``-6 a A v``
    term is 16% of ``c``, and ``-a int <v*, A v>`` turns ``Re d`` from
    ``+0.0022`` to ``-0.0539``.
    """

    def test_c_on_the_theta_flip(self) -> None:
        m, u, z, period = (
            -0.27992881012102044,
            -0.06618157052528172,
            -0.018106357412197554,
            6.493169908041482,
        )
        field = _field(ThetaFlip, [m, -0.3], 3)
        states = _theta_cycle(field, (u, z), period, 80)
        c = CycleNormalForm(field, states, period).flip()
        assert abs(c + 0.0322799670) < 2e-5 * 0.0322799670, c

    def test_d_on_the_theta_torus(self) -> None:
        m, u, z, period = (
            0.001959370032366524,
            -0.056907808081685746,
            -0.0013562076192046326,
            6.436030265065528,
        )
        field = _field(ThetaTorus, [m, -0.5], 3)
        states = _theta_cycle(field, (u, z), period, 80)
        # The critical pair's angle, from the same DOP853 monodromy.
        d = CycleNormalForm(field, states, period).torus(2.3299304702)
        assert abs(d.real + 0.0538821123) < 2e-5 * 0.0538821123, d
