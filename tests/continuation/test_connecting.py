"""Connecting orbits: homoclinic, heteroclinic, and the shooting that finds them.

The slowest tests in the suite live here. Locating a homoclinic means
integrating to a horizon of 100 at dt = 0.002 - fifty thousand steps per
shot - and then shooting repeatedly to follow the curve, so these are
expensive by construction rather than by defect.
"""

import itertools
import math
from unittest import TestCase

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from discrecontinual_equations.continuation.connecting_orbit import (
    HeteroclinicOrbit,
    HomoclinicOrbit,
    MeshSpec,
    NotConvergedError,
    Terminus,
    _projection_conditions,
    _split_eigenspaces,
)
from discrecontinual_equations.continuation.homoclinic_curve import HomoclinicCurve
from discrecontinual_equations.continuation.homoclinic_shooting import (
    Departure,
    HomoclinicShooting,
    ReturnSettings,
)
from discrecontinual_equations.continuation.lyapunov import (
    LyapunovSettings,
    lyapunov_spectrum,
)
from discrecontinual_equations.continuation.region_analysis import (
    _jacobian,
)
from discrecontinual_equations.continuation.shilnikov import classify_saddle_focus
from discrecontinual_equations.continuation.symmetry import (
    cyclic_action,
    equivariance_defect,
    fourier_reduce,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.systems.coupled_ring import (
    FerroelectricRing,
    FluxgateRing,
)
from tests.continuation.fields import (
    _TILT,
    Alpha,
    BetaOne,
    BetaTwo,
    BogdanovTakensField,
    DampedWellField,
    HeteroclinicField,
    JerkField,
    MelnikovField,
    State,
    TiltedHomoclinicField,
    Time,
    TwoParameterJerkField,
    _saddle,
)

_SADDLE_JACOBIAN = np.array([[0.0, 1.0], [1.0, 0.0]])


_SADDLE_JACOBIAN = np.array([[0.0, 1.0], [1.0, 0.0]])


class TestHomoclinicOrbit(TestCase):
    def test_converges_to_exact_homoclinic(self):
        half, intervals = 10.0, 80
        mesh = MeshSpec(half, intervals, phase_index=1, phase_value=0.0)
        times = np.linspace(-half, half, intervals + 1)
        seed = np.zeros((intervals + 1, 2))
        for k, t in enumerate(times):
            seed[k, 0] = 1.2 / math.cosh(0.45 * t) ** 2
            seed[k, 1] = -0.9 * math.tanh(0.45 * t) / math.cosh(0.45 * t) ** 2
        solution = HomoclinicOrbit(
            _saddle(),
            np.zeros(2),
            _SADDLE_JACOBIAN,
            mesh,
        ).solve(
            seed,
        )
        exact = 1.5 / np.cosh(times / 2.0) ** 2
        assert np.max(np.abs(solution.component(0) - exact)) < 5.0e-2
        assert abs(solution.component(0)[intervals // 2] - 1.5) < 5.0e-2


class TestHomoclinicCurve(TestCase):
    """The BT homoclinic curve obeys b1 = -(49/25) b2^2 (the 5/7 asymptotic)."""

    def test_bogdanov_takens_homoclinic_asymptotic(self):
        parameters = [BetaOne(value=0.0), BetaTwo(value=0.0)]
        equation = DifferentialEquation(
            variables=[State(), State()],
            time=Time(),
            parameters=parameters,
            derivative=BogdanovTakensField(
                variables=[State(), State()],
                parameters=parameters,
                results=[State(), State()],
                time=None,
            ),
        )
        epsilon, nodes, span = 0.25, 101, 5.0
        half = span * math.sqrt(2.0) / epsilon
        times = np.linspace(-half, half, nodes)
        scaled = epsilon * times / math.sqrt(2.0)
        profile = 1.0 / np.cosh(scaled) ** 2
        orbit = np.column_stack(
            [
                epsilon**2 * (1.0 - 3.0 * profile),
                epsilon**3 * 3.0 * math.sqrt(2.0) * profile * np.tanh(scaled),
            ],
        )
        mesh = MeshSpec(half, nodes - 1, phase_index=1, phase_value=0.0)
        curve = HomoclinicCurve(equation, 0, 1, mesh)
        sweep = (5.0 / 7.0) * epsilon**2
        point = curve.solve_point(
            sweep,
            orbit,
            np.array([epsilon**2, 0.0]),
            -(epsilon**4),
        )
        assert point is not None
        oracle = -(49.0 / 25.0) * sweep**2
        assert point.continuation < 0.0
        assert abs(point.continuation / oracle - 1.0) < 1.5e-2


class TestShilnikovContinuation(TestCase):
    """Two-parameter continuation of a saddle-focus homoclinic through a Belyakov point.

    The jerk system ``x' = y, y' = z, z' = -a z - b y + x - x^2`` has a
    saddle-focus at the origin: one unstable eigenvalue ``gamma`` and a stable
    pair ``-rho +- i omega``. Its saddle index ``nu = rho / gamma`` crosses one -
    the Belyakov transition from Shilnikov chaos (``nu < 1``) to a tame
    homoclinic - on the curve between ``b = 0.85`` and ``b = 0.825``.

    The curve is continued by the boundary-value problem (:class:`HomoclinicCurve`
    with the saddle-focus projection rows), not by shooting: the shooter's return
    gap jumps across zero near the curve (at ``b = 0.85`` within ``7e-5`` of it),
    so bisection there lands on a jump (DEQ-174). Every traced point is checked
    against :func:`_independent_gap`, which shares no code with the library.
    """

    def _jerk(self, a: float, b: float) -> TwoParameterJerkField:
        return TwoParameterJerkField(
            variables=[State(), State(), State()],
            parameters=[Alpha(value=a), BetaOne(value=b)],
            results=[State(), State(), State()],
            time=None,
        )

    def _jacobian(self, a: float, b: float) -> np.ndarray:
        return np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -b, -a]])

    def _trace(self, a: float) -> list:
        """The BVP curve over ``_BELYAKOV_SWEEP``, seeded at ``(a, 0.9)``."""
        field = self._jerk(a, _BELYAKOV_SWEEP[0])
        equation = DifferentialEquation(
            variables=field.variables,
            time=Time(),
            parameters=field.parameters,
            derivative=field,
        )
        mesh = MeshSpec(_BELYAKOV_HALF, _BELYAKOV_INTERVALS, 1, 0.0)
        curve = HomoclinicCurve(equation, 0, 1, mesh)
        seed = _manifold_seed(a, _BELYAKOV_SWEEP[0], mesh)
        return curve.trace(_BELYAKOV_SWEEP, seed, np.zeros(3), a)

    def _assert_one_curve_through_belyakov(self, points: list) -> None:
        assert [point.sweep for point in points] == _BELYAKOV_SWEEP
        located = [point.continuation for point in points]
        # each point brackets the independent root within the pin, with the
        # same orientation everywhere: one root, followed
        for point in points:
            a, b = point.continuation, point.sweep
            assert _independent_gap(a - _BELYAKOV_PIN, b) < 0.0, (a, b)
            assert _independent_gap(a + _BELYAKOV_PIN, b) > 0.0, (a, b)
        # smooth, monotone locus: a increases as b decreases, at a slope that
        # changes by less than 10% from one step to the next (measured: 0.2% to
        # 4%; a jump to a neighbouring root 1e-2 away would change it by 30%)
        slopes = [
            (q.continuation - p.continuation) / (q.sweep - p.sweep)
            for p, q in itertools.pairwise(points)
        ]
        assert all(slope < 0.0 for slope in slopes)
        for earlier, later in itertools.pairwise(slopes):
            assert abs(later / earlier - 1.0) < _BELYAKOV_SLOPE_CHANGE, slopes
        # neighbouring orbits differ by a step's worth (0.11 to 0.15 per 0.025
        # of b, measured), not by another loop round the saddle (order one)
        for p, q in itertools.pairwise(points):
            assert np.max(np.abs(q.orbit.states - p.orbit.states)) < 0.2
        # every point is a genuine saddle-focus, and nu crosses one along it
        for point in points:
            focus = classify_saddle_focus(
                self._jacobian(point.continuation, point.sweep),
            )
            assert focus is not None
        top = classify_saddle_focus(self._jacobian(located[0], _BELYAKOV_SWEEP[0]))
        bottom = classify_saddle_focus(
            self._jacobian(located[-1], _BELYAKOV_SWEEP[-1]),
        )
        assert top.saddle_index < 1.0
        assert bottom.saddle_index > 1.0

    def test_the_bvp_follows_one_homoclinic_through_belyakov(self):
        """Seeded at the independent root, the BVP traces all nine values."""
        self._assert_one_curve_through_belyakov(self._trace(_BELYAKOV_SEED))

    @pytest.mark.slow
    def test_traces_saddle_focus_homoclinic_through_belyakov(self):
        """Shooting seeds the curve; the boundary-value problem continues it.

        Once ``xfail``: the old trace returned its bracket's upper end at every
        step (DEQ-135), and followed honestly the shooter loses the curve near
        ``a = 0.591`` (DEQ-174).
        """
        shooter = HomoclinicShooting(
            self._jerk(0.5, _BELYAKOV_SWEEP[0]),
            0,
            Departure(np.zeros(3), np.array([1.0, 0.0, 0.0])),
            ReturnSettings(dt=0.002, horizon=100.0, departure_radius=0.6),
        )
        # -0.0149 at 0.519 and +0.0571 at 0.520 (horizon 100)
        seed = shooter.locate(0.519, 0.520)
        assert abs(seed - _BELYAKOV_SEED) < 1.0e-6
        self._assert_one_curve_through_belyakov(self._trace(seed))

    @pytest.mark.slow
    def test_shooting_refuses_the_jump_next_to_the_curve(self):
        """At ``b = 0.85`` the gap jumps from ``-0.0123`` to ``+0.0120`` at 0.59113.

        That is ``7e-5`` from the curve (0.591061), and bisection converged on
        it: the ``|gap|`` it left, ``1.2e-2``, is what refuses it now.
        """
        shooter = HomoclinicShooting(
            self._jerk(0.5, 0.85),
            0,
            Departure(np.zeros(3), np.array([1.0, 0.0, 0.0])),
            ReturnSettings(dt=0.002, horizon=100.0, departure_radius=0.6),
        )
        with pytest.raises(ValueError, match="jump of the gap") as caught:
            shooter.locate(0.5911168, 0.5911368)
        assert "0.5911" in str(caught.value)


def _manifold_seed(a: float, b: float, mesh: MeshSpec) -> np.ndarray:
    """The jerk system's unstable manifold at ``(a, b)``, on ``mesh``, by SciPy.

    Started ``1e-9`` along the unstable eigenvector and integrated by DOP853, it
    is shifted so the peak of ``x`` on its first excursion sits at ``t = 0``,
    where the phase condition ``y = 0`` holds.
    """
    jacobian = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -b, -a]])
    values, vectors = np.linalg.eig(jacobian)
    unstable = vectors[:, int(np.argmax(values.real))].real
    unstable *= np.sign(unstable[0]) / np.linalg.norm(unstable)
    solution = solve_ivp(
        lambda _t, s: [s[1], s[2], -a * s[2] - b * s[1] + s[0] - s[0] * s[0]],
        (0.0, 80.0),
        1.0e-9 * unstable,
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-15,
        dense_output=True,
    )
    times = np.linspace(0.0, 80.0, 80_001)
    states = solution.sol(times)
    departed = int(np.argmax(np.linalg.norm(states, axis=0) > 0.6))
    falling = np.flatnonzero(np.diff(states[0, departed:]) < 0.0)
    peak = times[departed + int(falling[0])]
    nodes = np.linspace(-mesh.half_length, mesh.half_length, mesh.intervals + 1)
    return solution.sol(nodes + peak).T


# The traced curve: b from 0.9 down to 0.7, seeded at the independent root at
# b = 0.9. The mesh is h = 0.05 on [-12, 12]; there the BVP's a lies 1.4e-4 to
# 2.4e-4 below the independent root at every b (h^2 convergence: 7.5e-5 at
# h = 0.025, 3.4e-5 at 0.0125; the rest is truncation, 2e-5 to 7e-5), so the pin
# is twice the largest. Neighbouring roots of the shooter's gap lie 1e-2 apart.
# outputs/verification/deq-172/belyakov_crosscheck.out (DEQ-174).
_BELYAKOV_SWEEP = [0.9, 0.875, 0.85, 0.825, 0.8, 0.775, 0.75, 0.725, 0.7]
_BELYAKOV_SEED = 0.5190497227
_BELYAKOV_HALF = 12.0
_BELYAKOV_INTERVALS = 480
_BELYAKOV_PIN = 5.0e-4
_BELYAKOV_SLOPE_CHANGE = 0.1


# The primary homoclinic of the jerk system near a = 0.465, computed without the
# library by _independent_homoclinic below (DOP853 at rtol 1e-12, left-eigenvector
# gap, brentq): 0.465109935774 to 0.465109935777 for offsets 1e-6 to 1e-8 and
# rtol 1e-11 to 1e-13. The shooter (RK4 at dt 0.002, offset 1e-5) lands 3e-12
# from it at horizon 100; at horizon 60 it finds no return and gives the bracket
# end 0.49999999, 0.035 away (DEQ-161).
_SHILNIKOV_HOMOCLINIC = 0.465109935777
_LOCATE_TOLERANCE = 1.0e-8


def _independent_gap(
    a: float,
    b: float = 1.0,
    offset: float = 1.0e-7,
    rtol: float = 1.0e-12,
) -> float:
    """The jerk system's return gap at ``(a, b)``, by SciPy alone.

    The unstable manifold of the origin, started ``offset`` along the unstable
    eigenvector and integrated by DOP853, makes its first return into the ball of
    radius 0.4 after leaving radius 0.6; its gap is the closest point's component
    along the left unstable eigenvector, the coordinate that vanishes on the
    stable eigenspace (the shooter projects on the right one instead). Near the
    saddle that coordinate keeps its sign through the passage, so the gap changes
    sign where the first return becomes a homoclinic orbit. Nothing here comes
    from the library.
    """
    jacobian = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -b, -a]])
    values, right = np.linalg.eig(jacobian)
    unstable = right[:, int(np.argmax(values.real))].real
    unstable *= np.sign(unstable[0]) / np.linalg.norm(unstable)
    values, left = np.linalg.eig(jacobian.T)
    covector = left[:, int(np.argmax(values.real))].real
    covector /= covector @ unstable
    solution = solve_ivp(
        lambda _t, s: [s[1], s[2], -a * s[2] - b * s[1] + s[0] - s[0] * s[0]],
        (0.0, 100.0),
        offset * unstable,
        method="DOP853",
        rtol=rtol,
        atol=1.0e-3 * rtol,
        dense_output=True,
    )
    times = np.linspace(0.0, 100.0, 200_001)
    # past its return the manifold may escape to infinity: inf radii are fine
    with np.errstate(over="ignore", invalid="ignore"):
        radius = np.linalg.norm(solution.sol(times).T, axis=1)
    band = (np.cumsum(radius > 0.6) > 0) & (radius < 0.4)
    first = int(np.flatnonzero(band)[0])
    leaves = np.flatnonzero(~band[first:])
    last = first + int(leaves[0]) if leaves.size else band.size
    closest = first + int(np.argmin(radius[first:last]))
    fine = np.linspace(times[closest - 1], times[closest + 1], 2001)
    states = solution.sol(fine).T
    return float(covector @ states[np.argmin(np.linalg.norm(states, axis=1))])


def _independent_homoclinic(offset: float = 1.0e-7, rtol: float = 1.0e-12) -> float:
    """The jerk system's homoclinic parameter in [0.465, 0.4675] at ``b = 1``.

    brentq on :func:`_independent_gap`; nothing here comes from the library.
    """
    return float(
        brentq(
            lambda a: _independent_gap(a, 1.0, offset, rtol),
            0.465,
            0.4675,
            xtol=1.0e-13,
        ),
    )


class TestShilnikovHomoclinic(TestCase):
    """The full pipeline: locate a saddle-focus homoclinic and confirm its chaos."""

    def _jerk(self, a: float) -> JerkField:
        return JerkField(
            variables=[State(), State(), State()],
            parameters=[Alpha(value=a)],
            results=[State(), State(), State()],
            time=None,
        )

    def _jacobian(self, a: float) -> np.ndarray:
        return np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -1.0, -a]])

    @pytest.mark.slow
    def test_locates_saddle_focus_homoclinic(self):
        departure = Departure(np.zeros(3), np.array([1.0, 0.0, 0.0]))
        shooter = HomoclinicShooting(
            self._jerk(0.48),
            0,
            departure,
            ReturnSettings(dt=0.002, horizon=100.0, departure_radius=0.6),
        )
        # The gap is +0.200 at 0.46 and -0.130 at 0.47 (horizon 100). At 0.50 it
        # is +0.128, so [0.46, 0.50] holds two roots and no sign change; the
        # old bisection found 0.4651 only through its midpoints (DEQ-135).
        located = shooter.locate(0.46, 0.47)
        # A search that ends at a bracket end has not located anything (DEQ-161).
        assert min(located - 0.46, 0.47 - located) > 1.0e-3, located
        assert abs(located - _SHILNIKOV_HOMOCLINIC) < _LOCATE_TOLERANCE, located
        focus = classify_saddle_focus(self._jacobian(located))
        assert focus is not None
        assert focus.satisfies_shilnikov_criterion

    @pytest.mark.slow
    def test_the_reference_homoclinic_is_settled(self):
        # The reference above, recomputed, and unmoved by a tenfold change of
        # offset or tolerance: its digits are the system's, not the method's.
        assert abs(_independent_homoclinic() - _SHILNIKOV_HOMOCLINIC) < 1.0e-10
        coarse = _independent_homoclinic(offset=1.0e-6, rtol=1.0e-11)
        assert abs(coarse - _SHILNIKOV_HOMOCLINIC) < 1.0e-10

    @pytest.mark.slow
    def test_chaos_tracks_the_saddle_index(self):
        settings = LyapunovSettings(dt=0.005, horizon=18000, transient=5000)
        chaotic = lyapunov_spectrum(
            self._jerk(0.50),
            np.array([0.1, 0.0, 0.0]),
            count=1,
            settings=settings,
        ).top
        tame = lyapunov_spectrum(
            self._jerk(0.70),
            np.array([0.1, 0.0, 0.0]),
            count=1,
            settings=settings,
        ).top
        assert chaotic > 0.05
        assert tame < 0.02
        assert classify_saddle_focus(self._jacobian(0.50)).saddle_index < 1.0
        assert classify_saddle_focus(self._jacobian(0.70)).saddle_index > 1.0


class TestHomoclinicShooting(TestCase):
    """Locate a homoclinic orbit by shooting, and seed the boundary-value problem."""

    def _field(self, mu: float) -> DampedWellField:
        return DampedWellField(
            variables=[State(), State()],
            parameters=[Alpha(value=mu)],
            results=[State(), State()],
            time=None,
        )

    def _shooter(self) -> HomoclinicShooting:
        # The well is symmetric, so the root is mu = 0 at any step: dt 0.01 over 30
        # time units locates the same 1.2e-11 as the default dt 0.002 over 60 (DEQ-144)
        departure = Departure(np.array([0.05, 0.05]), np.array([1.0, 1.0]))
        return HomoclinicShooting(
            self._field(0.0),
            0,
            departure,
            ReturnSettings(dt=1.0e-2, horizon=30.0),
        )

    def test_return_gap_changes_sign_at_homoclinic(self):
        shooter = self._shooter()
        assert shooter.gap(-0.03) > 0.0
        assert shooter.gap(0.03) < 0.0

    def test_locates_homoclinic_parameter(self):
        assert abs(self._shooter().locate(-0.05, 0.05)) < 1.0e-3

    def test_locate_refuses_a_jump_and_returns_a_root(self):
        """Bisection converges on any sign change; only a root leaves ``gap ~ 0``.

        The gap here is ``sqrt(d)`` either side of a root at ``0.3``, as the
        saddle-focus gap falls (``d**0.46``, measured), and steps from
        ``-0.012`` to ``+0.012`` at ``0.7``, as it did at ``b = 0.85``.
        """

        class _Scripted(HomoclinicShooting):
            def gap(self, parameter: float) -> float:
                if parameter < 0.5:
                    distance = parameter - 0.3
                    return math.copysign(abs(distance) ** 0.5, distance)
                return 0.012 if parameter > 0.7 else -0.012

        shooter = _Scripted(
            self._field(0.0),
            0,
            Departure(np.zeros(2), np.ones(2)),
        )
        assert abs(shooter.locate(0.1, 0.45) - 0.3) < 1.0e-9
        with pytest.raises(ValueError, match="jump of the gap"):
            shooter.locate(0.6, 0.8)
        lenient = _Scripted(
            self._field(0.0),
            0,
            Departure(np.zeros(2), np.ones(2)),
            ReturnSettings(jump_tolerance=0.02),
        )
        assert abs(lenient.locate(0.6, 0.8) - 0.7) < 1.0e-9

    def test_traces_homoclinic_curve_along_melnikov_line(self):
        grid = np.linspace(-40.0, 40.0, 200001)
        loop_x = 1.5 / np.cosh(grid / 2.0) ** 2
        loop_y = -1.5 / np.cosh(grid / 2.0) ** 2 * np.tanh(grid / 2.0)
        first = float(np.trapezoid(loop_y * loop_y, grid))
        second = float(np.trapezoid(loop_x * loop_y * loop_y, grid))
        slope = -second / first
        field = MelnikovField(
            variables=[State(), State()],
            parameters=[Alpha(value=0.0), BetaOne(value=0.0)],
            results=[State(), State()],
            time=None,
        )
        shooter = HomoclinicShooting(
            field,
            0,
            Departure(np.array([0.05, 0.05]), np.array([1.0, 1.0])),
            # dt 0.01 over 40 time units misses the line by 1e-5, as dt 0.002 over 120
            # and dt 0.005 over 60 do; three points still fit the slope (DEQ-144)
            ReturnSettings(dt=0.01, horizon=40.0, departure_radius=0.6),
        )
        curve = shooter.trace(1, [0.0, 0.05, 0.10], 0.0, 0.06)
        for control, located in curve:
            assert abs(located - slope * control) < 5.0e-3

    def test_shooting_seed_refines_to_exact_loop(self):
        """Pinned where the orbit crosses ``y = 0``, the solve converges.

        Pinned at the peak ``x = 1.5`` instead, as this test once was, the phase
        condition fixes the time shift only to second order (``x' = 0`` there),
        the Newton matrix is ill-conditioned along the shift (``sigma_min /
        sigma_max`` 5e-6) and Gauss-Newton never converges: the best iterate's
        correction is 0.36 of the state. That iterate was returned, and is now
        refused (DEQ-120), though it lies within 1e-3 of the loop.
        """
        mu = self._shooter().locate(-0.05, 0.05)
        field = self._field(mu)
        half_length, intervals = 12.0, 240
        times = np.linspace(-half_length, half_length, intervals + 1)
        exact = np.column_stack(
            [
                1.5 / np.cosh(times / 2.0) ** 2,
                -1.5 / np.cosh(times / 2.0) ** 2 * np.tanh(times / 2.0),
            ],
        )
        seed = np.column_stack(
            [1.5 / np.cosh(times / 1.8) ** 2, np.zeros_like(times)],
        )
        jacobian = np.array([[0.0, 1.0], [1.0, mu]])
        solution = HomoclinicOrbit(
            field,
            np.zeros(2),
            jacobian,
            MeshSpec(
                half_length=half_length,
                intervals=intervals,
                phase_index=1,
                phase_value=0.0,
            ),
        ).solve(seed)
        assert np.max(np.linalg.norm(solution.states - exact, axis=1)) < 2.0e-2
        assert solution.residual.is_trajectory
        peak = HomoclinicOrbit(
            field,
            np.zeros(2),
            jacobian,
            MeshSpec(
                half_length=half_length,
                intervals=intervals,
                phase_index=0,
                phase_value=1.5,
            ),
        )
        with pytest.raises(NotConvergedError) as caught:
            peak.solve(seed)
        best = caught.value.solution.states
        assert np.max(np.linalg.norm(best - exact, axis=1)) < 2.0e-2


class TestThreeDimensionalConnection(TestCase):
    """Complex-eigenspace projection and a three-dimensional homoclinic orbit."""

    def test_complex_projection_constrains_the_spiral(self):
        jacobian = np.array(
            [[-0.3, -2.0, 0.0], [2.0, -0.3, 0.0], [0.0, 0.0, 1.0]],
        )
        values, left, stable, _ = _split_eigenspaces(jacobian)
        in_unstable = _projection_conditions(
            left,
            values,
            stable,
            np.array([0.0, 0.0, 1.0]),
        )
        assert len(in_unstable) == 2
        assert max(abs(value) for value in in_unstable) < 1.0e-9
        mixed = _projection_conditions(
            left,
            values,
            stable,
            np.array([0.5, 0.2, 1.0]),
        )
        assert max(abs(value) for value in mixed) > 1.0e-2

    def test_three_dimensional_homoclinic_matches_exact(self):
        half_length, intervals = 12.0, 240
        times = np.linspace(-half_length, half_length, intervals + 1)
        base = np.column_stack(
            [
                1.5 / np.cosh(times / 2.0) ** 2,
                -1.5 / np.cosh(times / 2.0) ** 2 * np.tanh(times / 2.0),
                np.zeros_like(times),
            ],
        )
        exact = base @ _TILT.T
        seed = (
            np.column_stack(
                [
                    1.2 / np.cosh(times / 2.5) ** 2,
                    np.zeros_like(times),
                    np.zeros_like(times),
                ],
            )
            @ _TILT.T
        )
        jacobian = (
            _TILT
            @ np.array(
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -2.0]],
            )
            @ _TILT.T
        )
        field = TiltedHomoclinicField(
            variables=[State(), State(), State()],
            parameters=[],
            results=[State(), State(), State()],
            time=None,
        )
        orbit = HomoclinicOrbit(
            field,
            np.zeros(3),
            jacobian,
            MeshSpec(
                half_length=half_length,
                intervals=intervals,
                phase_index=0,
                phase_value=float(exact[intervals // 2, 0]),
            ),
        )
        solution = orbit.solve(seed)
        assert np.max(np.linalg.norm(solution.states - exact, axis=1)) < 1.0e-2


class TestHeteroclinicOrbit(TestCase):
    """Saddle-to-saddle connection against the exact double-well oracle."""

    def _field(self) -> HeteroclinicField:
        return HeteroclinicField(
            variables=[State(), State()],
            parameters=[],
            results=[State(), State()],
            time=None,
        )

    def _jacobian(self, x: float) -> np.ndarray:
        return np.array([[0.0, 1.0], [-1.0 + 3.0 * x * x, 0.0]])

    def test_converges_to_exact_tanh_connection(self):
        half_length, intervals = 8.0, 200
        times = np.linspace(-half_length, half_length, intervals + 1)
        exact = np.column_stack(
            [
                np.tanh(times / np.sqrt(2.0)),
                (1.0 / np.sqrt(2.0)) / np.cosh(times / np.sqrt(2.0)) ** 2,
            ],
        )
        seed = np.column_stack([np.tanh(times / 2.0), np.zeros_like(times)])
        source = Terminus(np.array([-1.0, 0.0]), self._jacobian(-1.0))
        target = Terminus(np.array([1.0, 0.0]), self._jacobian(1.0))
        orbit = HeteroclinicOrbit(
            self._field(),
            source,
            target,
            MeshSpec(half_length=half_length, intervals=intervals),
        )
        solution = orbit.solve(seed)
        error = np.max(np.linalg.norm(solution.states - exact, axis=1))
        assert error < 1.0e-3

    def test_resolution_estimate_serves_a_heteroclinic_too(self):
        """The estimate lives on the base class and needs no equilibrium knowledge.

        The truncation comparison seeds the longer interval by holding the end
        states, which sit on the two saddles here and on one saddle for a
        homoclinic, so the same code serves both. Calibration is identical - 0.75x
        the true error - and a too-short interval is still caught by truncation:

            T    N      true       discr      trunc      limited_by
            8.0  200    3.77e-4    2.83e-4    3.4e-11    spacing
            3.0  200    2.68e-4    3.98e-5    2.59e-4    length
        """
        source = Terminus(np.array([-1.0, 0.0]), self._jacobian(-1.0))
        target = Terminus(np.array([1.0, 0.0]), self._jacobian(1.0))
        root = np.sqrt(2.0)
        for half_length, limited_by in ((8.0, "spacing"), (3.0, "length")):
            times = np.linspace(-half_length, half_length, 201)
            exact = np.column_stack(
                [
                    np.tanh(times / root),
                    (1.0 / root) / np.cosh(times / root) ** 2,
                ],
            )
            orbit = HeteroclinicOrbit(
                self._field(),
                source,
                target,
                MeshSpec(half_length=half_length, intervals=200),
            )
            solution = orbit.solve(
                np.column_stack([np.tanh(times / 2.0), np.zeros_like(times)]),
            )
            true_error = float(np.max(np.abs(solution.states - exact)))
            resolution = orbit.estimate_orbit_error(solution)
            assert resolution.limited_by == limited_by
            assert 0.6 < resolution.estimate / true_error < 1.2

    def test_lands_on_both_saddles(self):
        half_length, intervals = 8.0, 200
        times = np.linspace(-half_length, half_length, intervals + 1)
        seed = np.column_stack([np.tanh(times / 2.0), np.zeros_like(times)])
        source = Terminus(np.array([-1.0, 0.0]), self._jacobian(-1.0))
        target = Terminus(np.array([1.0, 0.0]), self._jacobian(1.0))
        solution = HeteroclinicOrbit(
            self._field(),
            source,
            target,
            MeshSpec(half_length=half_length, intervals=intervals),
        ).solve(seed)
        assert np.linalg.norm(solution.states[0] - np.array([-1.0, 0.0])) < 1.0e-3
        assert np.linalg.norm(solution.states[-1] - np.array([1.0, 0.0])) < 1.0e-3


class TestSymmetryReduction(TestCase):
    """Cyclic-symmetry (Fourier) reduction of the ring linearisation."""

    def _fluxgate(self, size: int, coupling: float, gain: float) -> FluxgateRing:
        parameters = [Alpha(value=coupling), BetaOne(value=gain), BetaTwo(value=0.0)]
        return FluxgateRing(
            variables=[State() for _ in range(size)],
            parameters=parameters,
            results=[State() for _ in range(size)],
            time=None,
        )

    def _ferroelectric(
        self,
        size: int,
        coupling: float,
        gain: float,
    ) -> FerroelectricRing:
        parameters = [Alpha(value=coupling), BetaOne(value=gain), BetaTwo(value=0.0)]
        return FerroelectricRing(
            variables=[State() for _ in range(size)],
            parameters=parameters,
            results=[State() for _ in range(size)],
            time=None,
        )

    def test_ring_is_cyclically_equivariant(self):
        field = self._fluxgate(5, 0.4, 0.3)
        point = np.array([0.1, -0.2, 0.3, 0.05, -0.15])
        assert equivariance_defect(field, cyclic_action(5), point) < 1.0e-9

    def test_fluxgate_breaks_symmetry_steadily(self):
        coupling, gain, size = 0.4, 0.3, 5
        reduction = fourier_reduce(
            _jacobian(self._fluxgate(size, coupling, gain), np.zeros(size)),
        )
        assert reduction.is_circulant
        modes = np.exp(2j * np.pi * np.arange(size) / size)
        oracle = (gain - 1.0) + coupling * modes
        assert np.allclose(
            np.sort(reduction.eigenvalues.real),
            np.sort(oracle.real),
            atol=1.0e-5,
        )
        assert reduction.critical_mode == 0
        assert not reduction.is_oscillatory

    def test_ferroelectric_hopf_is_a_complex_mode(self):
        gain, size = -0.3, 3
        coupling = -2.0 * gain  # Hopf at lambda = -2a
        reduction = fourier_reduce(
            _jacobian(self._ferroelectric(size, coupling, gain), np.zeros(size)),
        )
        assert reduction.is_circulant
        assert reduction.is_oscillatory
        assert abs(reduction.critical_eigenvalue.real) < 1.0e-3
