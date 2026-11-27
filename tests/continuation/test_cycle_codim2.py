"""Fold, flip and torus curves of cycles in two parameters (DEQ-35)."""

import math
from unittest import TestCase

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.cycle_codim2 import CycleCurveContinuation
from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CyclePoint,
    CycleSeed,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from tests.continuation.cycle_oracles import (
    Adaptive,
    Coordinate,
    ExtendedBautin,
    First,
    FoldTorus,
    PredatorPrey,
    Rossler,
    RotatedBautin,
    RotatedTakens,
    Second,
    Third,
)


def _equation(kind: type, values: list[float], dimension: int) -> DifferentialEquation:
    parameters = [
        cls(value=value)
        for cls, value in zip((First, Second, Third), values, strict=False)
    ]
    return DifferentialEquation(
        variables=[Coordinate() for _ in range(dimension)],
        time=Coordinate(),
        parameters=parameters,
        derivative=kind(
            variables=[Coordinate() for _ in range(dimension)],
            parameters=parameters,
            results=[Coordinate() for _ in range(dimension)],
            time=None,
        ),
    )


def _circle(radius: float, omega: float, intervals: int, extra: int) -> CycleSeed:
    period = 2.0 * math.pi / omega
    time = np.linspace(0.0, period, intervals + 1)
    columns = [radius * np.cos(omega * time), radius * np.sin(omega * time)]
    columns += [np.zeros_like(time)] * extra
    return np.column_stack(columns), period


def _start(  # noqa: PLR0913 (a one-parameter trace, spelled out)
    equation: DifferentialEquation,
    seed: CycleSeed,
    intervals: int,
    kind: str,
    trace: tuple[float, int, float],
    phase: tuple[int, float] = (1, 0.0),
) -> CyclePoint:
    """The traced point nearest the first ``kind`` crossing of a one-parameter trace."""
    continuation = CycleContinuation(
        equation,
        0,
        intervals,
        phase_index=phase[0],
        phase_value=phase[1],
    )
    points, bifurcations = continuation.trace(seed, trace[0], trace[1], trace[2])
    crossing = next(b for b in bifurcations if b.kind == kind)
    return min(points, key=lambda point: abs(point.parameter - crossing.parameter))


def _special(curve, kind: str) -> tuple[float, float]:
    found = [point for point in curve.special if point.kind == kind]
    assert len(found) == 1, [point.kind for point in curve.special]
    return found[0].parameters


class TestFoldCurve(TestCase):
    """The extended Bautin oracle: the fold curve and its cusp in closed form."""

    @classmethod
    def setUpClass(cls) -> None:
        c = ExtendedBautin.C
        m2 = -0.12
        rho = 0.8
        m1 = -(m2 * rho + c * rho * rho - rho**3)
        equation = _equation(ExtendedBautin, [m1, m2], 2)
        states, period = _circle(math.sqrt(rho), 1.0 + ExtendedBautin.E * rho, 40, 0)
        start = _start(
            equation,
            CycleSeed(states, period, m1),
            40,
            "fold_of_cycles",
            (0.02, 80, -1.0),
        )
        cls.curve = CycleCurveContinuation(
            equation,
            (0, 1),
            40,
            phase_index=1,
            phase_value=0.0,
        ).trace(start, "fold_of_cycles", 0.02, 24, direction=-1.0)
        cls.equation = equation

    def test_the_curve_is_the_double_root_locus(self) -> None:
        c, e = ExtendedBautin.C, ExtendedBautin.E
        assert len(self.curve.points) == 25
        for point in self.curve.points:
            m1, m2 = point.parameters
            # The double root from m2 = 3 rho^2 - 2 c rho: whichever of its two
            # roots puts m1 on the curve.
            root = math.sqrt(max(4.0 * c * c + 12.0 * m2, 0.0))
            rho = min(
                ((2.0 * c + root) / 6.0, (2.0 * c - root) / 6.0),
                key=lambda r: abs(m1 - (-2.0 * r**3 + c * r * r)),
            )
            assert abs(m1 - (-2.0 * rho**3 + c * rho * rho)) < 1e-9
            exact = -rho * (2.0 * c - 6.0 * rho) * (1.0 + e * rho) / e
            assert abs(point.coefficient - exact) < 1e-4 * max(1.0, abs(exact))

    def test_the_cusp_is_where_b_vanishes(self) -> None:
        c = ExtendedBautin.C
        m1, m2 = _special(self.curve, "cusp_of_cycles")
        assert abs(m1 - c**3 / 27.0) < 1e-8
        assert abs(m2 + c * c / 3.0) < 1e-8

    def test_the_second_parameter_is_handed_back(self) -> None:
        assert self.equation.derivative.parameters[1].value == pytest.approx(-0.12)


class TestTorusCurve(TestCase):
    """The rotated Takens oracle: the torus curve, its resonances, the 1:1 end."""

    @classmethod
    def setUpClass(cls) -> None:
        b1, b2 = -0.003, 0.05
        equation = _equation(RotatedTakens, [b1, b2], 3)
        u = -math.sqrt(-b1)
        states, period = _circle(math.sqrt(1 + u), 1 + RotatedTakens.E * u, 40, 1)
        start = _start(
            equation,
            CycleSeed(states, period, b1),
            40,
            "neimark_sacker",
            (0.0005, 40, 1.0),
        )
        cls.curve = CycleCurveContinuation(
            equation,
            (0, 1),
            40,
            phase_index=1,
            phase_value=0.0,
        ).trace(start, "neimark_sacker", 0.01, 8, direction=-1.0)

    @staticmethod
    def _angle(u: float) -> float:
        omega = math.sqrt(-4.0 * (1.0 + u) * u)
        return omega * 2.0 * math.pi / (1.0 + RotatedTakens.E * u)

    @pytest.mark.slow
    def test_the_curve_and_kappa_are_the_reduced_hopf(self) -> None:
        for point in self.curve.points:
            b1, b2 = point.parameters
            assert abs(b1 + b2 * b2) < 1e-8
            if b2 > 0:  # a torus: kappa = cos(omega T)
                assert abs(point.kappa - math.cos(self._angle(-b2))) < 1e-5
            else:  # a neutral saddle past the 1:1 resonance
                assert point.kappa > 1.0

    def test_the_strong_resonances(self) -> None:
        for name, angle in (
            ("strong_resonance_1_3", 2.0 * math.pi / 3.0),
            ("strong_resonance_1_4", math.pi / 2.0),
        ):
            u = _bisect(lambda v, target=angle: self._angle(v) - target, -0.2, -1e-9)
            b1, b2 = _special(self.curve, name)
            assert abs(b2 + u) < 1e-6, name
            assert abs(b1 + u * u) < 1e-7, name
        b1, b2 = _special(self.curve, "strong_resonance_1_1")
        assert abs(b2) < 1e-3
        assert abs(b1) < 1e-6


class TestChenciner(TestCase):
    def test_re_d_vanishes_at_the_bautin_point(self) -> None:
        equation = _equation(RotatedBautin, [-0.02, 0.1], 3)
        states, period = _circle(1.0, 1.0, 30, 1)
        start = _start(
            equation,
            CycleSeed(states, period, -0.02),
            30,
            "neimark_sacker",
            (0.005, 12, 1.0),
        )
        curve = CycleCurveContinuation(
            equation,
            (0, 1),
            30,
            phase_index=1,
            phase_value=0.0,
        ).trace(start, "neimark_sacker", 0.05, 4, direction=-1.0)
        b1, b2 = _special(curve, "chenciner")
        assert abs(b1) < 1e-8
        assert abs(b2) < 1e-6


class TestFoldTorus(TestCase):
    def test_the_fold_curve_meets_the_torus(self) -> None:
        equation = _equation(FoldTorus, [-0.01, -0.1], 4)
        u = 0.1
        states, period = _circle(math.sqrt(1 + u), 1 + FoldTorus.E * u, 30, 2)
        start = _start(
            equation,
            CycleSeed(states, period, -0.01),
            30,
            "fold_of_cycles",
            (0.02, 60, 1.0),
        )
        curve = CycleCurveContinuation(
            equation,
            (0, 1),
            30,
            phase_index=1,
            phase_value=0.0,
        ).trace(start, "fold_of_cycles", 0.05, 4, direction=1.0)
        b1, b2 = _special(curve, "fold_neimark_sacker")
        assert abs(b1) < 1e-8
        assert abs(b2) < 1e-8


class TestFlipCurve(TestCase):
    def _rossler(self, intervals: int) -> tuple[DifferentialEquation, CyclePoint]:
        a, b, c = 0.25, 0.4, 4.5
        equation = _equation(Rossler, [a, b, c], 3)

        def field(_t, u):
            return [-u[1] - u[2], u[0] + a * u[1], b * u[0] - c * u[2] + u[0] * u[2]]

        settle = solve_ivp(field, (0, 300), [0.0, -3.0, 0.1], rtol=1e-10, atol=1e-12)
        orbit = solve_ivp(
            field,
            (0, 30),
            settle.y[:, -1],
            method="DOP853",
            rtol=1e-11,
            atol=1e-12,
            dense_output=True,
            events=_upward_x,
        )
        start, end = orbit.t_events[0][-2:]
        states = orbit.sol(np.linspace(start, end, intervals + 1)).T
        point = _start(
            equation,
            CycleSeed(states, end - start, a),
            intervals,
            "period_doubling",
            (0.05, 60, 1.0),
            phase=(0, float(states[0, 0])),
        )
        return equation, point

    def test_the_first_flip_is_matcont_s(self) -> None:
        # MATCONT tutorial III (Kuznetsov and Neirynck, 2021): A = 0.27191773,
        # T = 6.148967 at B = 0.4, C = 4.5.
        equation, start = self._rossler(60)
        curve = CycleCurveContinuation(
            equation,
            (0, 1),
            60,
            phase_index=0,
            phase_value=float(start.solution.states[0, 0]),
        ).trace(start, "period_doubling", 0.05, 3, direction=1.0)
        first = curve.points[0]
        assert abs(first.parameters[0] - 0.27191773) < 2e-7
        assert abs(first.period - 6.148967) < 2e-6
        # The flip curve's own tests read the multiplier that is neither the
        # trivial one nor the flip's: 1:2 resonance is mu + 1, fold-flip mu - 1.
        for point in curve.points:
            ordered = sorted(point.multipliers, key=lambda m: abs(m - 1.0))
            other = min(ordered[1:], key=lambda m: -abs(m + 1.0)).real
            assert point.tests["strong_resonance_1_2"] == pytest.approx(other + 1.0)
            assert point.tests["fold_flip"] == pytest.approx(other - 1.0)
        # Every point has the multiplier -1 for an independent DOP853 shooting.
        for point in curve.points[1:]:
            a, b = point.parameters
            assert abs(_shooting_flip(point, a, b) + 1.0) < 1e-5

    def test_c_against_the_doubled_cycle(self) -> None:
        # adaptx at b = 1: c = -0.30512 from DOP853 doubled cycles at distances
        # 4e-3 .. 5e-4 below the flip (rtol 1e-12), -beta / xi^2 with xi the
        # L2 half-difference of the doubled cycle's two laps, extrapolated.
        equation = _equation(Adaptive, [0.7, 1.0], 3)

        def field(_t, u):
            return [u[1], u[2], -0.7 * u[2] - u[1] - u[0] + u[0] ** 2]

        orbit = solve_ivp(
            field,
            (0, 400),
            [0.1, 0.0, 0.0],
            method="DOP853",
            rtol=1e-11,
            atol=1e-12,
            dense_output=True,
            events=_upward_y,
        )
        start, end = orbit.t_events[0][-2:]
        states = orbit.sol(np.linspace(start, end, 41)).T
        point = _start(
            equation,
            CycleSeed(states, end - start, 0.7),
            40,
            "period_doubling",
            (0.02, 60, -1.0),
        )
        curve = CycleCurveContinuation(
            equation,
            (0, 1),
            40,
            phase_index=1,
            phase_value=0.0,
        ).trace(point, "period_doubling", 0.02, 0)
        flip = curve.points[0]
        # MATCONT's manual (2019, section 7.7) puts this flip at 0.6303020.
        assert abs(flip.parameters[0] - 0.630302) < 3e-6
        assert abs(flip.coefficient + 0.30512) < 1e-3


class TestPublishedFlipTorus(TestCase):
    @pytest.mark.slow
    def test_the_predator_prey_flip_neimark_sacker_point(self) -> None:
        # De Witte et al. (2013), section 4.2.1: PDNS at (b2, eps) = (0.277, 0.530).
        eps, b2 = 0.50, 0.277
        equation = _equation(PredatorPrey, [eps, b2], 6)
        function = equation.derivative
        settle = solve_ivp(
            lambda _t, u: function.eval(list(u)),
            (0, 3000),
            [0.3, 0.3, 0.3, 0.3, 1.0, 0.0],
            method="DOP853",
            rtol=1e-10,
            atol=1e-12,
            dense_output=True,
            events=_upward_v2,
        )
        start = settle.t_events[0][-2]
        states = settle.sol(np.linspace(start, start + 2 * math.pi, 41)).T
        point = _start(
            equation,
            CycleSeed(states, 2 * math.pi, eps),
            40,
            "period_doubling",
            (0.02, 10, 1.0),
            phase=(5, 0.0),
        )
        curve = CycleCurveContinuation(
            equation,
            (0, 1),
            40,
            phase_index=5,
            phase_value=0.0,
            normal_forms=False,
        ).trace(point, "period_doubling", 0.01, 3, direction=-1.0)
        eps_found, b2_found = _special(curve, "flip_neimark_sacker")
        assert abs(eps_found - 0.530) < 1e-3
        assert abs(b2_found - 0.277) < 1e-3


class TestRefusals(TestCase):
    def test_kind_scheme_and_start(self) -> None:
        equation = _equation(RotatedTakens, [-0.003, 0.05], 3)
        with pytest.raises(ValueError, match="one-point-per-interval"):
            CycleCurveContinuation(equation, (0, 1), 10, collocation="gauss_legendre")
        continuation = CycleCurveContinuation(equation, (0, 1), 10)
        states, period = _circle(1.0, 1.0, 10, 1)
        start = CyclePoint(
            -0.003,
            _solution(states, period),
            np.array([1.0, 0.5, 0.2]),
        )
        with pytest.raises(ValueError, match="kind"):
            continuation.trace(start, "homoclinic", 0.01, 1)
        with pytest.raises(ValueError, match="complex multiplier pair"):
            continuation.trace(start, "neimark_sacker", 0.01, 1)


def _solution(states: np.ndarray, period: float):
    from discrecontinual_equations.continuation.periodic_orbit import (  # noqa: PLC0415
        PeriodicOrbitSolution,
    )

    return PeriodicOrbitSolution(states, period)


def _bisect(function, low: float, high: float) -> float:
    f_low = function(low)
    for _ in range(200):
        middle = 0.5 * (low + high)
        value = function(middle)
        if value * f_low > 0:
            low, f_low = middle, value
        else:
            high = middle
    return 0.5 * (low + high)


def _upward_x(_t, u):
    return u[0]


_upward_x.direction = 1  # type: ignore[attr-defined]


def _upward_y(_t, u):
    return u[1]


_upward_y.direction = 1  # type: ignore[attr-defined]


def _upward_v2(_t, u):
    return u[5]


_upward_v2.direction = 1  # type: ignore[attr-defined]


def _shooting_flip(point, a: float, b: float) -> float:
    """The multiplier nearest -1 of the Rossler cycle by DOP853 shooting.

    Newton on the return to the section ``x = x_0`` with the variational
    equation alongside; no collocation anywhere.
    """
    c = 4.5

    def field(u):
        return np.array(
            [-u[1] - u[2], u[0] + a * u[1], b * u[0] - c * u[2] + u[0] * u[2]]
        )

    def jacobian(u):
        return np.array([[0, -1, -1], [1, a, 0], [b + u[2], 0, -c + u[0]]])

    def flow(start, period):
        def rhs(_t, w):
            u, y = w[:3], w[3:].reshape(3, 3)
            return np.concatenate([field(u), (jacobian(u) @ y).ravel()])

        run = solve_ivp(
            rhs,
            (0, period),
            np.concatenate([start, np.eye(3).ravel()]),
            method="DOP853",
            rtol=1e-12,
            atol=1e-13,
        )
        return run.y[:3, -1], run.y[3:, -1].reshape(3, 3)

    state = np.array(point.states[0], dtype=float)
    period = float(point.period)
    for _ in range(20):
        end, monodromy = flow(state, period)
        residual = np.concatenate([end[1:] - state[1:], [end[0] - state[0]]])
        if np.linalg.norm(residual) < 1e-12:
            break
        # Unknowns (y, z, T); x stays on the section.
        matrix = np.zeros((3, 3))
        matrix[:2, :2] = monodromy[1:, 1:] - np.eye(2)
        matrix[2, :2] = monodromy[0, 1:]
        matrix[:2, 2] = field(end)[1:]
        matrix[2, 2] = field(end)[0]
        update = np.linalg.solve(matrix, -residual)
        state[1:] += update[:2]
        period += update[2]
    multipliers = np.linalg.eigvals(flow(state, period)[1])
    return float(multipliers[np.argmin(np.abs(multipliers + 1.0))].real)
