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

from discrecontinual_equations.continuation.connecting_orbit import (
    HeteroclinicOrbit,
    HomoclinicOrbit,
    MeshSpec,
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

    There is no closed-form oracle for a saddle-focus homoclinic locus, so this is
    validated by convergent structural evidence: the located curve is smooth and
    monotone, the equilibrium is a saddle-focus along it, and the saddle index
    crosses one (the Belyakov transition from Shilnikov chaos to a tame homoclinic).
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

    def test_traces_saddle_focus_homoclinic_through_belyakov(self):
        shooter = HomoclinicShooting(
            self._jerk(0.5, 0.9),
            0,
            Departure(np.zeros(3), np.array([1.0, 0.0, 0.0])),
            ReturnSettings(dt=0.002, horizon=100.0, departure_radius=0.6),
        )
        seed = shooter.locate(0.45, 0.58)
        curve = shooter.trace(1, [0.90, 0.85, 0.80, 0.75, 0.70], seed, 0.07)
        controls = [b for b, _ in curve]
        located = [a for _, a in curve]
        assert controls == [0.90, 0.85, 0.80, 0.75, 0.70]
        # smooth, monotone locus: a increases as b decreases
        for earlier, later in itertools.pairwise(located):
            assert later > earlier
        # every point is a genuine saddle-focus
        for b, a in curve:
            assert classify_saddle_focus(self._jacobian(a, b)) is not None
        # the saddle index crosses one along the curve: Shilnikov -> tame (Belyakov)
        top = classify_saddle_focus(self._jacobian(located[0], controls[0]))
        bottom = classify_saddle_focus(self._jacobian(located[-1], controls[-1]))
        assert top.saddle_index < 1.0
        assert bottom.saddle_index > 1.0


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

    def test_locates_saddle_focus_homoclinic(self):
        departure = Departure(np.zeros(3), np.array([1.0, 0.0, 0.0]))
        shooter = HomoclinicShooting(
            self._jerk(0.48),
            0,
            departure,
            ReturnSettings(dt=0.002, horizon=100.0, departure_radius=0.6),
        )
        located = shooter.locate(0.46, 0.50)
        assert 0.46 < located < 0.50
        focus = classify_saddle_focus(self._jacobian(located))
        assert focus is not None
        assert focus.satisfies_shilnikov_criterion

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
        departure = Departure(np.array([0.05, 0.05]), np.array([1.0, 1.0]))
        return HomoclinicShooting(self._field(0.0), 0, departure)

    def test_return_gap_changes_sign_at_homoclinic(self):
        shooter = self._shooter()
        assert shooter.gap(-0.03) > 0.0
        assert shooter.gap(0.03) < 0.0

    def test_locates_homoclinic_parameter(self):
        assert abs(self._shooter().locate(-0.05, 0.05)) < 1.0e-3

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
            ReturnSettings(dt=0.002, horizon=120.0, departure_radius=0.6),
        )
        curve = shooter.trace(1, [0.0, 0.05, 0.10, 0.15, 0.20], 0.0, 0.06)
        for control, located in curve:
            assert abs(located - slope * control) < 5.0e-3

    def test_shooting_seed_refines_to_exact_loop(self):
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
                phase_index=0,
                phase_value=1.5,
            ),
        ).solve(seed)
        assert np.max(np.linalg.norm(solution.states - exact, axis=1)) < 2.0e-2


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
