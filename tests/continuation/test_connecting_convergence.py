"""A connecting-orbit solve that did not converge is refused, never returned.

The oracle is the DEQ-81 verifier's finding (DEQ-120): Kirk-Silber in sheared
coordinates, solved in the full space on the two edges leaving xi2. There the
source's unstable manifold is 2-D and the target is a sink inside a 3-D
invariant subspace, so the connections come in a one-parameter family and the
Newton matrix is singular along it. Gauss-Newton's step along the family was
rounding divided by rounding, and the old "best iterate" rule handed back the
seed, unchanged and unannounced, with a resolution estimate of 0.0. DEQ-120
made that a refusal; DEQ-175 takes the minimum-norm step there, and both edges
now converge to genuine trajectories. The refusal is pinned on xi3 -> xi2, a
connection that does not exist.

The convergence test is AUTO's and COLNEW's: an iterate is converged when its
Newton residual is below tolerance or its Newton correction is, component by
component, below ``tol * (1 + |x|)`` (AUTO-07p manual, EPSU; COLNEW CONTRL).
Neither returns an iterate that fails it: AUTO stops with ``MX`` (no
convergence) and COLNEW with ``IFLAG = -2``.

The trapezoidal residuals checked here are computed from the field directly,
not through the solver.
"""

import math
import warnings
from unittest import TestCase, mock

import numpy as np
import pytest

from discrecontinual_equations.continuation import connecting_orbit
from discrecontinual_equations.continuation.connecting_orbit import (
    CollocationResidual,
    HeteroclinicOrbit,
    HomoclinicOrbit,
    MeshSpec,
    NotConvergedError,
    OrbitSolution,
    Terminus,
)
from discrecontinual_equations.continuation.heteroclinic_network import (
    HeteroclinicNetwork,
    coordinate_plane,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import HeteroclinicField, State, _saddle
from tests.continuation.test_connecting_newton import (
    NonAnalyticSaddle,
    _bogdanov_takens_problem,
    _saddle_problem,
)
from tests.continuation.test_heteroclinic_network_subspace import (
    _EDGES,
    _SHEAR4,
    _leak_network,
    _sheared_kirk_silber,
    _trapezoid,
)

_DIVERGING = ((2, 3), (2, 4))


def _full_space(network: HeteroclinicNetwork, edge, intervals: int = 400):
    """The connection's solver in the full space, and the network's seed for it."""
    connection = network.connections[_EDGES.index(edge)]
    mesh = network.mesh(connection, half_length=20.0, intervals=intervals)
    source = network.saddle(connection.source)
    target = network.saddle(connection.target)
    solver = HeteroclinicOrbit(
        network.function,
        Terminus(source.equilibrium, source.jacobian),
        Terminus(target.equilibrium, target.jacobian),
        mesh,
    )
    return solver, network.seed(connection, mesh), mesh


def _tanh_problem(half_length: float = 8.0, intervals: int = 200):
    field = HeteroclinicField(
        variables=[State(), State()],
        parameters=[],
        results=[State(), State()],
        time=None,
    )

    def jacobian(x: float) -> np.ndarray:
        return np.array([[0.0, 1.0], [-1.0 + 3.0 * x * x, 0.0]])

    solver = HeteroclinicOrbit(
        field,
        Terminus(np.array([-1.0, 0.0]), jacobian(-1.0)),
        Terminus(np.array([1.0, 0.0]), jacobian(1.0)),
        MeshSpec(half_length=half_length, intervals=intervals),
    )
    times = np.linspace(-half_length, half_length, intervals + 1)
    seed = np.column_stack([np.tanh(times / 2.0), np.zeros_like(times)])
    exact = np.column_stack(
        [
            np.tanh(times / math.sqrt(2.0)),
            (1.0 / math.sqrt(2.0)) / np.cosh(times / math.sqrt(2.0)) ** 2,
        ],
    )
    return solver, seed, exact, field


def _homoclinic_problem():
    half, intervals = 10.0, 80
    times = np.linspace(-half, half, intervals + 1)
    seed = np.column_stack(
        [
            1.2 / np.cosh(0.45 * times) ** 2,
            -0.9 * np.tanh(0.45 * times) / np.cosh(0.45 * times) ** 2,
        ],
    )
    solver = HomoclinicOrbit(
        _saddle(),
        np.zeros(2),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        MeshSpec(half, intervals, phase_index=1, phase_value=0.0),
    )
    return solver, seed, times


def _pinv_gauss_newton(solver, seed: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Gauss-Newton with the dense SVD pseudo-inverse, not the solver's step.

    Directions below 1e-10 sigma_max are dropped; the Newton matrices of the
    xi2 edges have one singular value at 1e-18 to 7e-17 of sigma_max and the
    next at 8e-3, so any cutoff between gives the same step.
    """
    unknowns = seed.flatten()
    for _ in range(20):
        residual = solver._residual(unknowns, times, seed.shape)  # noqa: SLF001
        if np.linalg.norm(residual) < 1.0e-12:
            break
        jacobian = solver._jacobian(  # noqa: SLF001
            unknowns,
            times,
            seed.shape,
            residual,
            True,  # noqa: FBT003
        ).toarray()
        left, values, right = np.linalg.svd(jacobian, full_matrices=False)
        kept = values > 1.0e-10 * values[0]
        coefficients = (left[:, kept].T @ residual) / values[kept]
        unknowns = unknowns - right[kept].T @ coefficients
    return unknowns.reshape(seed.shape)


def _no_connection(network: HeteroclinicNetwork, intervals: int = 100):
    """xi3 -> xi2, which does not exist: Kirk-Silber's edges run xi2 -> xi3."""
    connection = network.add_connection(
        "xi3",
        "xi2",
        _SHEAR4 @ coordinate_plane(4, (2, 1)),
    )
    mesh = network.mesh(connection, half_length=20.0, intervals=intervals)
    source = network.saddle("xi3")
    target = network.saddle("xi2")
    solver = HeteroclinicOrbit(
        network.function,
        Terminus(source.equilibrium, source.jacobian),
        Terminus(target.equilibrium, target.jacobian),
        mesh,
    )
    return solver, network.seed(connection, mesh), mesh


class TestFamilyOfConnectionsConverges(TestCase):
    """The sheared Kirk-Silber edges leaving xi2, solved in the full space.

    The connections leaving xi2 come in a one-parameter family, so the Newton
    matrix is singular along it: measured sigma_min / sigma_max 1e-18 to 7e-17
    at every iterate, the next singular value 8e-3. The Gauss-Newton step's
    component along the family was rounding divided by rounding, 10 to 95 long
    and different on every BLAS kernel, and the solve refused (DEQ-120). The
    minimum-norm step converges (DEQ-175).
    """

    def test_both_edges_leaving_xi2_converge_to_genuine_trajectories(self):
        """Measured, n = 400: trapezoidal residual 4e-16 and 3e-14, Newton
        residual 3e-15 and 1e-13, the path moving 0.60 and 0.27 from the seed
        to a member 0.025 and 0.16 off the plane. The same on the Prescott,
        Nehalem, Sandybridge, Haswell and Zen OpenBLAS kernels
        (outputs/verification/deq-175/coretype_after.out).
        """
        _, network = _sheared_kirk_silber(_SHEAR4)
        for edge in _DIVERGING:
            solver, seed, _ = _full_space(network, edge)
            solution = solver.solve(seed)
            # A trajectory to rounding, measured from the field directly.
            assert float(np.max(np.abs(_trapezoid(network.function, solution)))) < 1e-12
            assert solution.residual.is_trajectory
            unknowns = solution.states.flatten()
            newton = solver._residual(  # noqa: SLF001
                unknowns,
                solution.times,
                seed.shape,
            )
            assert float(np.linalg.norm(newton)) < 1.0e-9
            # It went somewhere: not the seed, whose residual is 0.09.
            assert float(np.max(np.abs(solution.states - seed))) > 0.1

    def test_the_member_found_is_the_pseudo_inverse_iterations(self):
        """Against Gauss-Newton with numpy's SVD pseudo-inverse, measured 7e-13
        and 9e-13 apart: the solver takes the minimum-norm step."""
        _, network = _sheared_kirk_silber(_SHEAR4)
        for edge in _DIVERGING:
            solver, seed, mesh = _full_space(network, edge, intervals=100)
            times = np.linspace(-mesh.half_length, mesh.half_length, seed.shape[0])
            reference = _pinv_gauss_newton(solver, seed, times)
            solution = solver.solve(seed)
            np.testing.assert_allclose(solution.states, reference, rtol=0.0, atol=1e-8)
            # Off the plane: the family's other members are genuine connections.
            basis = network.connections[_EDGES.index(edge)].basis
            outside = solution.states - (solution.states @ basis) @ basis.T
            assert float(np.max(np.linalg.norm(outside, axis=1))) > 1.0e-2

    def test_last_bit_changes_to_the_seed_change_nothing(self):
        """Before, a one-ulp change in 1% of the seed's entries decided whether
        xi2->xi4 moved at all (5 of 40 did; fix-120/sweep.out). Now every
        perturbed solve converges, to within 5e-12 of the unperturbed member
        (measured 20 per edge and mesh, deq-175/sweep_after.out)."""
        _, network = _sheared_kirk_silber(_SHEAR4)
        generator = np.random.default_rng(1)
        for edge in _DIVERGING:
            solver, seed, _ = _full_space(network, edge, intervals=100)
            unperturbed = solver.solve(seed).states
            for _ in range(3):
                flips = generator.choice([-1.0, 1.0], seed.shape)
                chosen = generator.random(seed.shape) < 0.01
                perturbed = seed * (1.0 + 2.2e-16 * flips * chosen)
                found = solver.solve(perturbed).states
                np.testing.assert_allclose(found, unperturbed, rtol=0.0, atol=1e-8)

    def test_an_estimate_from_the_seed_reports_how_far_it_is(self):
        """On main the seed's estimate read 0.0: both refinements returned seeds.
        Now the refinements converge, and the estimate is the seed's distance
        from the connection they reach (measured 0.60 and 0.27)."""
        _, network = _sheared_kirk_silber(_SHEAR4)
        for edge in _DIVERGING:
            solver, seed, mesh = _full_space(network, edge, intervals=100)
            times = np.linspace(-mesh.half_length, mesh.half_length, seed.shape[0])
            resolution = solver.estimate_orbit_error(OrbitSolution(times, seed))
            assert resolution.valid
            assert resolution.estimate > 0.1

    def test_the_converged_member_resolves_at_second_order(self):
        """Its refinements reach the same member: the discretisation estimate
        falls 4x per halving of h (measured 3.7e-3, 9.1e-4 on xi2->xi3)."""
        _, network = _sheared_kirk_silber(_SHEAR4)
        estimates = []
        for intervals in (100, 200):
            solver, seed, _ = _full_space(network, (2, 3), intervals=intervals)
            resolution = solver.estimate_orbit_error(solver.solve(seed))
            assert resolution.valid
            estimates.append(resolution.discretisation)
        assert estimates[0] < 1.0e-2
        assert 3.0 < estimates[0] / estimates[1] < 5.0

    def test_edges_whose_full_space_solve_is_isolated_still_converge(self):
        """xi1->xi2, xi3->xi1 and xi4->xi1 leave a 1-D unstable manifold."""
        _, network = _sheared_kirk_silber(_SHEAR4)
        for edge in ((1, 2), (3, 1), (4, 1)):
            solver, seed, _ = _full_space(network, edge, intervals=200)
            solution = solver.solve(seed)
            assert solution.residual.is_trajectory
            assert float(np.max(np.abs(_trapezoid(network.function, solution)))) < 1e-8

    def test_the_network_records_the_full_space_member(self):
        """The network's own solve is in the plane; the full-space check now
        converges too, to a member off it (measured 0.025)."""
        _, network = _sheared_kirk_silber(_SHEAR4)
        connection = network.connections[_EDGES.index((2, 3))]
        mesh = network.mesh(connection, half_length=20.0, intervals=200)
        orbit = network.solve(connection, mesh)
        assert orbit.residual.is_trajectory
        assert orbit.resolution.valid
        assert orbit.full_space.converged
        assert orbit.full_space.residual.is_trajectory
        assert orbit.full_space.subspace_defect > 1.0e-2

    def test_the_network_full_space_check_converges_where_isolated(self):
        _, network = _sheared_kirk_silber(_SHEAR4)
        connection = network.connections[_EDGES.index((1, 2))]
        mesh = network.mesh(connection, half_length=20.0, intervals=200)
        orbit = network.solve(connection, mesh)
        assert orbit.full_space.converged
        assert orbit.full_space.residual.is_trajectory


class TestAbsentConnectionIsRefused(TestCase):
    """xi3 -> xi2 does not exist, and its solve does not converge.

    Measured: Newton residual 1.15 at n = 100 and 0.82 at 200, correction
    0.99 and 0.98 (inf on one of five OpenBLAS kernels, whose best step was
    rank-deficient), trapezoidal residual 0.35 and 0.18, 357 and 2.3e3 times
    what the mesh allows (outputs/verification/deq-175/no_connection.out,
    coretype_no_connection.out).
    """

    def test_it_refuses_with_its_best_iterate(self):
        _, network = _sheared_kirk_silber(_SHEAR4)
        solver, seed, _ = _no_connection(network)
        with pytest.raises(NotConvergedError, match="did not converge") as caught:
            solver.solve(seed)
        error = caught.value
        # 1e7 times the Newton tolerance, 100 times below the measured 1.15.
        assert error.newton_residual > 1.0e-2
        # 1e5 times the correction tolerance, 100 times below the measured 0.99.
        assert error.correction > 1.0e-2
        best = error.solution
        assert isinstance(best.residual, CollocationResidual)
        assert not best.residual.is_trajectory
        defect = float(np.max(np.abs(_trapezoid(network.function, best))))
        assert defect == pytest.approx(best.residual.largest, rel=1e-9)
        assert defect > 1.0e-2

    def test_the_refusal_is_a_runtime_error(self):
        _, network = _sheared_kirk_silber(_SHEAR4)
        solver, seed, _ = _no_connection(network)
        with pytest.raises(RuntimeError):
            solver.solve(seed)

    def test_an_estimate_from_diverged_refinements_is_invalid_not_zero(self):
        _, network = _sheared_kirk_silber(_SHEAR4)
        solver, seed, mesh = _no_connection(network)
        times = np.linspace(-mesh.half_length, mesh.half_length, seed.shape[0])
        resolution = solver.estimate_orbit_error(OrbitSolution(times, seed))
        assert not resolution.valid
        assert resolution.discretisation == math.inf
        assert resolution.estimate == math.inf


class TestNonFiniteIteratesAreRefused(TestCase):
    """An iterate that overflows ends the solve as a refusal, not a crash."""

    def test_an_overflowing_step_refuses(self):
        solver, seed, _, _ = _tanh_problem()

        def overflow(_self, _jacobian, residual):
            return np.full(seed.size, 1.0e300) * (1.0 + 0.0 * residual[0]), True

        with (
            mock.patch.object(HeteroclinicOrbit, "_newton_step", overflow),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(NotConvergedError):
                solver.solve(seed)

    def test_a_field_that_raises_on_overflow_refuses(self):
        """``math.pow`` raises ``OverflowError`` where numpy would return inf."""
        solver, _, seed = _saddle_problem(
            60,
            NonAnalyticSaddle(
                variables=[State(), State()],
                parameters=[],
                results=[State(), State()],
                time=None,
            ),
        )

        def overflow(_self, _jacobian, residual):
            return np.full(seed.size, 1.0e200) * (1.0 + 0.0 * residual[0]), True

        with (
            mock.patch.object(HomoclinicOrbit, "_newton_step", overflow),
            pytest.raises(NotConvergedError),
        ):
            solver.solve(seed)

    def test_a_non_finite_seed_refuses(self):
        solver, seed, _, _ = _tanh_problem()
        seed = seed.copy()
        seed[3, 0] = np.nan
        with pytest.raises(NotConvergedError):
            solver.solve(seed)


class TestConvergedSolveCarriesItsResidual(TestCase):
    """Every returned orbit carries its collocation residual."""

    def test_heteroclinic(self):
        solver, seed, exact, field = _tanh_problem()
        solution = solver.solve(seed)
        assert isinstance(solution.residual, CollocationResidual)
        assert solution.residual.is_trajectory
        direct = float(np.max(np.abs(_trapezoid(field, solution))))
        assert solution.residual.largest == pytest.approx(direct, rel=1e-9, abs=1e-15)
        assert np.max(np.abs(solution.states - exact)) < 1.0e-3

    def test_homoclinic(self):
        solver, seed, times = _homoclinic_problem()
        solution = solver.solve(seed)
        assert solution.residual.is_trajectory
        exact = 1.5 / np.cosh(times / 2.0) ** 2
        assert np.max(np.abs(solution.component(0) - exact)) < 5.0e-2

    def test_a_converged_least_squares_minimum_is_returned_with_its_residual(self):
        """The leaking plane: Gauss-Newton converges - its correction falls to
        1e-14 - to a least-squares minimum that is not a trajectory. That is a
        converged solve of an inconsistent system, not a diverged one; the
        residual it carries is what says so (measured 1.9e-3, 119x the allowance).
        """
        network, connection = _leak_network(0.05)
        source = network.saddle("xi1")
        target = network.saddle("xi2")
        mesh = network.mesh(connection, half_length=20.0, intervals=200)
        solver = HeteroclinicOrbit(
            network.function,
            Terminus(source.equilibrium, source.jacobian),
            Terminus(target.equilibrium, target.jacobian),
            mesh,
        )
        solution = solver.solve(network.seed(connection, mesh))
        assert not solution.residual.is_trajectory
        assert solution.residual.worst > 10.0

    def test_an_orbit_built_by_hand_has_no_residual(self):
        solution = OrbitSolution(np.zeros(3), np.zeros((3, 2)))
        assert solution.residual is None

    def test_a_homoclinic_curve_point_carries_its_residual(self):
        curve, _, unknowns, sweep = _bogdanov_takens_problem(100)
        orbit = unknowns[:-3].reshape(-1, 2)
        point = curve.solve_point(sweep, orbit, unknowns[-3:-1], unknowns[-1])
        assert point is not None
        assert isinstance(point.residual, CollocationResidual)
        assert point.residual.is_trajectory
        derivative = curve._equation.derivative  # noqa: SLF001
        derivative.parameters[0].value = point.continuation
        derivative.parameters[1].value = sweep
        direct = float(np.max(np.abs(_trapezoid(derivative, point.orbit))))
        assert point.residual.largest == pytest.approx(direct, rel=1e-6, abs=1e-15)
        assert point.orbit.residual is point.residual


class _ScaledDoubleWell(DeterministicFunction):
    """The double well in coordinates ``1e8`` times larger.

    Rounding then leaves a Newton residual near 1e-7 at the solution, above
    the 1e-9 tolerance, so only the correction test can say it converged.
    """

    scale = 1.0e8

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x, y = point[0] / self.scale, point[1] / self.scale
        return [self.scale * y, self.scale * (-x + x * x * x)]


class TestConvergedAtTheRoundingFloor(TestCase):
    """A residual that rounding keeps above tolerance is still converged."""

    def test_large_coordinates_converge_by_the_correction_test(self):
        scale = _ScaledDoubleWell.scale
        field = _ScaledDoubleWell(
            variables=[State(), State()],
            parameters=[],
            results=[State(), State()],
            time=None,
        )

        def jacobian(x: float) -> np.ndarray:
            return np.array([[0.0, 1.0], [-1.0 + 3.0 * x * x, 0.0]])

        solver = HeteroclinicOrbit(
            field,
            Terminus(np.array([-scale, 0.0]), jacobian(-1.0)),
            Terminus(np.array([scale, 0.0]), jacobian(1.0)),
            MeshSpec(half_length=8.0, intervals=200),
        )
        times = np.linspace(-8.0, 8.0, 201)
        seed = scale * np.column_stack([np.tanh(times / 2.0), np.zeros_like(times)])
        root = math.sqrt(2.0)
        exact = scale * np.column_stack(
            [np.tanh(times / root), (1.0 / root) / np.cosh(times / root) ** 2],
        )
        solution = solver.solve(seed)
        # Measured 3.8e-4, the trapezoidal error of the unscaled problem.
        assert np.max(np.abs(solution.states - exact)) / scale < 1.0e-3
        assert solution.residual.is_trajectory

    def test_a_square_system_converges_by_the_correction_test(self):
        """The normal-equation defect is scaled by ``|r|``: measured against
        ``|J dx + r|``, which is rounding here, it would refuse this orbit.
        Measured: Newton residual 7.2e-8, trapezoidal worst 2.1e-11.
        """
        scale = _ScaledDampedWell.scale
        field = _ScaledDampedWell(
            variables=[State(), State()],
            parameters=[],
            results=[State(), State()],
            time=None,
        )
        solver = HeteroclinicOrbit(
            field,
            Terminus(np.zeros(2), np.array([[0.0, 1.0], [1.0, -0.5]])),
            Terminus(np.array([scale, 0.0]), np.array([[0.0, 1.0], [-2.0, -0.5]])),
            MeshSpec(half_length=15.0, intervals=300, phase_value=0.5 * scale),
        )
        times = np.linspace(-15.0, 15.0, 301)
        seed = scale * np.column_stack(
            [(1.0 + np.tanh(times / 2.0)) / 2.0, np.zeros_like(times)],
        )
        solution = solver.solve(seed)
        assert solution.residual.is_trajectory


class _ScaledDampedWell(DeterministicFunction):
    """``x'' = x - x^3 - x'/2`` in coordinates ``1e8`` times larger.

    The saddle at the origin connects to the spiral sink at ``x = 1``: one stable
    direction at the source, none unstable at the target, and the phase
    condition make two boundary rows for two dimensions, so the Newton system is
    square and its correction solves it exactly - the remainder ``J dx + r`` is
    rounding, not orthogonal to the range of ``J``.
    """

    scale = 1.0e8

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x, y = point[0] / self.scale, point[1] / self.scale
        return [self.scale * y, self.scale * (x - x * x * x - 0.5 * y)]


class TestTheCertificateBelongsToTheReturnedIterate(TestCase):
    """A small correction certifies the iterate it was computed at, no other."""

    def test_later_small_corrections_do_not_certify_the_best_iterate(self):
        """The seed sits 1e-3 off the orbit and every later iterate's residual
        is reported one larger, so the seed stays the best iterate while
        Gauss-Newton converges behind it. Its own correction is about 1e-3: it
        is refused, whatever the later iterates' corrections are.
        """
        solver, seed, _, _ = _tanh_problem()
        orbit = solver.solve(seed).states
        start = orbit + 1.0e-3 * np.cos(np.arange(orbit.size)).reshape(orbit.shape)
        real = HeteroclinicOrbit._residual_norm  # noqa: SLF001
        calls = []

        def inflated(self, unknowns, times, shape):
            norm, residual = real(self, unknowns, times, shape)
            calls.append(norm)
            return (norm if len(calls) == 1 else norm + 1.0), residual

        with (
            mock.patch.object(HeteroclinicOrbit, "_residual_norm", inflated),
            pytest.raises(NotConvergedError) as caught,
        ):
            solver.solve(start)
        assert np.array_equal(caught.value.solution.states, start)
        assert caught.value.correction > 1.0e-4
        # Behind it, Gauss-Newton did converge.
        assert min(calls[1:]) < 1.0e-9


class TestFallbacksConvergeOrRefuse(TestCase):
    """Each DEQ-50 fallback either converges below tolerance or refuses."""

    def test_lsmr_after_a_singular_factorisation_converges(self):
        solver, seed, exact, _ = _tanh_problem()

        def singular(_matrix):
            message = "Factor is exactly singular"
            raise RuntimeError(message)

        with (
            mock.patch.object(connecting_orbit, "_factor", singular),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            solution = solver.solve(seed)
        assert any("lsmr" in str(warning.message) for warning in caught)
        assert solution.residual.is_trajectory
        assert np.max(np.abs(solution.states - exact)) < 1.0e-3

    def test_dense_least_squares_after_lsmr_fails_converges(self):
        solver, seed, exact, _ = _tanh_problem(intervals=100)

        def singular(_matrix):
            message = "Factor is exactly singular"
            raise RuntimeError(message)

        def stuck(jacobian, *_args, **_kwargs):
            # lsmr's 8-tuple, stopped at its iteration limit (istop = 7).
            return (np.zeros(jacobian.shape[1]), 7, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        with (
            mock.patch.object(connecting_orbit, "_factor", singular),
            mock.patch.object(connecting_orbit, "lsmr", stuck),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            solution = solver.solve(seed)
        assert any("dense" in str(warning.message) for warning in caught)
        assert solution.residual.is_trajectory
        assert np.max(np.abs(solution.states - exact)) < 3.0e-3

    def test_lsmr_on_the_diverging_edges_converges_to_a_genuine_connection(self):
        """With the sparse factorisation refused, lsmr's steps do converge here.

        Not to the connection in the plane: to another member of the family
        leaving xi2, off the plane (measured 0.025 and 0.16) and a trajectory to
        rounding (trapezoidal residual 5e-16 and 5e-14). The connection is not
        isolated in the full space, so this is a correct answer to the problem
        posed - converged below tolerance - and the plane is the network's
        business, which solves in it.
        """
        _, network = _sheared_kirk_silber(_SHEAR4)

        def singular(_matrix):
            message = "Factor is exactly singular"
            raise RuntimeError(message)

        for edge in _DIVERGING:
            solver, seed, _ = _full_space(network, edge, intervals=100)
            with (
                mock.patch.object(connecting_orbit, "_factor", singular),
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", RuntimeWarning)
                solution = solver.solve(seed)
            direct = float(np.max(np.abs(_trapezoid(network.function, solution))))
            assert direct < 1.0e-10
            assert solution.residual.is_trajectory
            basis = network.connections[_EDGES.index(edge)].basis
            outside = solution.states - (solution.states @ basis) @ basis.T
            assert float(np.max(np.linalg.norm(outside, axis=1))) > 1.0e-2

    def test_a_fallback_step_that_goes_nowhere_refuses(self):
        """A zero step stagnates on the seed; the seed is not converged."""
        solver, seed, _, _ = _tanh_problem()
        with (
            mock.patch.object(
                HeteroclinicOrbit,
                "_newton_step",
                lambda _self, _jacobian, residual: (
                    np.zeros(seed.size) * residual[0],
                    True,
                ),
            ),
            pytest.raises(NotConvergedError),
        ):
            solver.solve(seed)

    def test_a_slowly_converging_solve_short_of_tolerance_refuses(self):
        """Steps cut to a quarter converge linearly, at 3/4 a step: after the
        iteration cap the residual is 1e-8 (measured), above the tolerance, and
        a cut step is not the least-squares step, so its size certifies nothing.
        The iterate is within the mesh's accuracy of the orbit, yet unconverged.
        """
        solver, seed, _, _ = _tanh_problem()
        full = HeteroclinicOrbit._newton_step  # noqa: SLF001 (wrapping the step)

        def quarter(self, jacobian, residual):
            step, resolved = full(self, jacobian, residual)
            return 0.25 * step, resolved

        with (
            mock.patch.object(HeteroclinicOrbit, "_newton_step", quarter),
            pytest.raises(NotConvergedError) as caught,
        ):
            solver.solve(seed)
        assert 1.0e-9 < caught.value.newton_residual < 1.0e-6
        assert caught.value.correction == math.inf

    def test_a_curve_step_that_overflows_refuses_rather_than_crashes(self):
        curve, _, unknowns, sweep = _bogdanov_takens_problem(40)
        orbit = unknowns[:-3].reshape(-1, 2)
        with (
            mock.patch.object(
                type(curve),
                "_step",
                staticmethod(lambda _jacobian, residual: np.full(residual.size, 1e300)),
            ),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", RuntimeWarning)
            point = curve.solve_point(sweep, orbit, unknowns[-3:-1], unknowns[-1])
        assert point is None


class TestResolutionValidity(TestCase):
    """The estimate is computed only from refinement solves that converged."""

    def test_a_converged_orbit_has_a_valid_finite_estimate(self):
        solver, seed, _, _ = _tanh_problem()
        resolution = solver.estimate_orbit_error(solver.solve(seed))
        assert resolution.valid
        assert 0.0 < resolution.estimate < 1.0e-3

    def test_each_failed_refinement_makes_its_component_infinite(self):
        solver, seed, _, _ = _tanh_problem()
        solution = solver.solve(seed)
        original = solver._solve_on  # noqa: SLF001

        for failing in ("doubled", "longer"):

            def refine(_self, mesh, start, failing=failing):
                doubled = mesh.intervals == 2 * solver._mesh.intervals  # noqa: SLF001
                if (failing == "doubled") == doubled:
                    message = "refinement did not converge"
                    raise NotConvergedError(
                        message,
                        OrbitSolution(np.zeros(1), start[:1]),
                        newton_residual=1.0,
                        correction=1.0,
                    )
                return original(mesh, start)

            with mock.patch.object(HeteroclinicOrbit, "_solve_on", refine):
                resolution = solver.estimate_orbit_error(solution)
            with self.subTest(failing=failing):
                assert not resolution.valid
                assert resolution.estimate == math.inf
                if failing == "doubled":
                    assert resolution.discretisation == math.inf
                    assert math.isfinite(resolution.truncation)
                else:
                    assert resolution.truncation == math.inf
                    assert math.isfinite(resolution.discretisation)
