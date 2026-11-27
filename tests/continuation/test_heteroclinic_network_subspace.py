"""Connections solved inside their invariant subspace, and refused off a trajectory.

Two failures of the full-space solve, found by the DEQ-40 verification, are the
oracles here:

* with ``eps x1^2 x2^2`` added to ``x3'`` the x1x2 plane is not invariant and
  ``xi1 -> xi2`` does not exist, yet the full-space solve returned a path whose
  resolution estimate (1.9e-4) passed: only its trapezoidal residual (1.9e-3)
  says it is not a trajectory;
* at Kirk-Silber's xi2 the unstable manifold is 2-D, so in the full space the
  connections leaving it come in a family and are not isolated. In sheared
  coordinates the full-space solve does not find the one in the plane: its
  Newton matrix is near-singular (not singular - no factorisation fails),
  Gauss-Newton diverges, and the best iterate it returns is the seed.

The reference connection is computed by shooting: ``solve_ivp`` on the field in
its original coordinates, restricted to the plane by hand and written here in
numpy, started a distance ``1e-14`` along the source's unstable eigenvector, and
mapped into sheared coordinates afterwards. Nothing in it comes from the library.
"""

from unittest import TestCase

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.connecting_orbit import (
    HeteroclinicOrbit,
    MeshSpec,
    OrbitSolution,
    Terminus,
)
from discrecontinual_equations.continuation.heteroclinic_network import (
    HeteroclinicNetwork,
    NotATrajectoryError,
    coordinate_plane,
)
from discrecontinual_equations.systems.reflection_cubic import (
    Amplitude,
    guckenheimer_holmes,
    reflection_cubic,
)
from tests.continuation.test_heteroclinic_network_independent import (
    _kirk_silber_interaction,
    _Leak,
    _sheared,
    _sheared_gh_network,
)

_NAMES = ("xi1", "xi2", "xi3")
# The DEQ-40 verifier's sheared Kirk-Silber case: ``I + 0.6 N(0, 1)`` from
# ``default_rng(3)``, rounded; condition number 6.2.
_SHEAR4 = np.array(
    [
        [2.22, -1.53, 0.25, -0.34],
        [-0.27, 0.87, -1.21, -0.14],
        [-0.52, 1.99, 1.14, -0.21],
        [-0.17, -0.40, -0.63, 0.77],
    ],
)
# e23 = e24: xi2's unstable eigenvalue is repeated, and its unstable manifold 2-D.
_KIRK_SILBER = {
    "e12": 1.0, "c13": 1.0, "c14": 1.0, "c21": 1.0, "e23": 0.7, "e24": 0.7,
    "e31": 1.0, "c32": 1.0, "c34": 1.0, "e41": 1.0, "c42": 1.0, "c43": 1.0,
}  # fmt: skip
_EDGES = ((1, 2), (2, 3), (3, 1), (2, 4), (4, 1))


def _leak_network(strength: float):
    inner = guckenheimer_holmes(1.0, -1.0, -1.5, -0.6)
    field = _Leak(
        variables=[Amplitude() for _ in range(3)],
        parameters=inner.parameters,
        results=[Amplitude() for _ in range(3)],
        time=None,
    )
    field.inner, field.strength = inner, strength
    network = HeteroclinicNetwork(field)
    for index, name in enumerate(_NAMES):
        network.add_saddle(name, np.eye(3)[index])
    connection = network.add_connection("xi1", "xi2", coordinate_plane(3, (0, 1)))
    return network, connection


def _sheared_kirk_silber(shear: np.ndarray):
    interaction = _kirk_silber_interaction(_KIRK_SILBER)
    network = HeteroclinicNetwork(
        _sheared(reflection_cubic(np.ones(4), interaction), shear),
    )
    for index in range(4):
        network.add_saddle(f"xi{index + 1}", shear @ np.eye(4)[index])
    for source, target in _EDGES:
        network.add_connection(
            f"xi{source}",
            f"xi{target}",
            shear @ coordinate_plane(4, (source - 1, target - 1)),
        )
    return interaction, network


def _trapezoid(field, solution: OrbitSolution) -> np.ndarray:
    """Per-interval trapezoidal residual of a path, from the field directly."""
    states, times = solution.states, solution.times
    rates = np.array([field.eval(list(state)) for state in states], dtype=float)
    return (
        states[1:]
        - states[:-1]
        - 0.5 * np.diff(times)[:, None] * (rates[1:] + rates[:-1])
    )


def _shoot(interaction, plane, shear, mesh: MeshSpec, times):
    """The connection by shooting in the plane, sampled at the BVP's mesh times.

    In the original coordinates the plane is ``x_s x_t`` with the source at
    ``x_s = 1``. There ``x_k' = x_k (1 + sum_j A_kj x_j^2)``, and the source's
    unstable eigenvector in the plane is the ``x_t`` axis. Time is aligned by the
    same phase condition the BVP uses, located on the dense output.
    """
    source, target = plane
    block = interaction[np.ix_(plane, plane)]

    def rate(_time, x):
        return x * (1.0 + block @ (x * x))

    start = np.zeros(2)
    start[0], start[1] = 1.0, 1.0e-14
    solution = solve_ivp(
        rate,
        (0.0, 100.0),
        start,
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-14,
        dense_output=True,
    )
    embed = np.zeros((4, 2))
    embed[source, 0], embed[target, 1] = 1.0, 1.0
    lift = shear @ embed

    def phase(time: float) -> float:
        return float((lift @ solution.sol(time))[mesh.phase_index] - mesh.phase_value)

    grid = np.linspace(0.0, 100.0, 10001)
    values = np.array([phase(time) for time in grid])
    crossing = int(np.flatnonzero(np.sign(values[1:]) != np.sign(values[:-1]))[0])
    low, high = grid[crossing], grid[crossing + 1]
    for _ in range(80):
        middle = 0.5 * (low + high)
        if np.sign(phase(middle)) == np.sign(phase(low)):
            low = middle
        else:
            high = middle
    centre = 0.5 * (low + high)
    assert centre + times[0] > 0.0
    return (lift @ solution.sol(centre + times)).T


class TestResidualIsReported(TestCase):
    """The solver and the network report the collocation residual of a path."""

    def test_solver_residual_is_the_largest_trapezoidal_defect(self):
        network, connection = _leak_network(0.05)
        source = network.saddle("xi1")
        target = network.saddle("xi2")
        mesh = network.mesh(connection, half_length=20.0, intervals=200)
        orbit = HeteroclinicOrbit(
            network.function,
            Terminus(source.equilibrium, source.jacobian),
            Terminus(target.equilibrium, target.jacobian),
            mesh,
        )
        solution = orbit.solve(network.seed(connection, mesh))
        residual = orbit.collocation_residual(solution)
        defects = _trapezoid(network.function, solution)
        assert residual.largest == pytest.approx(np.max(np.abs(defects)), rel=1e-12)
        # The trapezoidal rule's local truncation error, h^3 |x'''| / 12, read
        # from third differences of the path.
        # With the safety factor 2 and the Newton tolerance 1e-9 as a floor.
        third = np.max(np.abs(np.diff(solution.states, 3, axis=0))) / 12.0
        assert residual.tolerance == pytest.approx(2.0 * third + 1.0e-9, rel=1e-12)
        # Measured: 1.9e-3 against a tolerance of 6.0e-5.
        assert residual.largest > 10.0 * residual.tolerance
        assert not residual.is_trajectory

    def test_a_genuine_connection_has_a_residual_far_below_its_tolerance(self):
        network, connection = _leak_network(0.0)
        mesh = network.mesh(connection, half_length=20.0, intervals=200)
        orbit = network.solve(connection, mesh)
        assert orbit.residual.largest < 1.0e-9
        assert orbit.residual.largest < 1.0e-4 * orbit.residual.tolerance
        assert orbit.residual.is_trajectory
        defects = _trapezoid(network.function, orbit.solution)
        assert orbit.residual.largest == pytest.approx(
            np.max(np.abs(defects)),
            rel=1e-6,
            abs=1e-15,
        )


def _gh_plane_orbit(mesh: MeshSpec, times: np.ndarray) -> np.ndarray:
    """GH ``xi1 -> xi2`` shot with Radau in the x1x2 plane, at the mesh's phase.

    ``x_k' = x_k (1 + A x^2)`` restricted to the plane, with ``a1 = -1``,
    ``a2 = -1.5`` and ``a3 = -0.6``; it leaves xi1 along x2 from ``1e-12``.
    """
    block = np.array([[-1.0, -1.5], [-0.6, -1.0]])
    solution = solve_ivp(
        lambda _time, x: x * (1.0 + block @ (x * x)),
        (0.0, 250.0),
        np.array([1.0, 1.0e-12]),
        method="Radau",
        rtol=1.0e-13,
        atol=1.0e-16,
        dense_output=True,
    )

    def phase(time: float) -> float:
        return float(solution.sol(time)[mesh.phase_index] - mesh.phase_value)

    low, high = 0.0, 250.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if phase(middle) > 0.0:  # x1 still above its midpoint: not yet there
            low = middle
        else:
            high = middle
    centre = 0.5 * (low + high)
    assert centre + times[0] > 0.0
    assert centre + times[-1] < 250.0
    return np.column_stack([solution.sol(centre + times).T, np.zeros(len(times))])


class TestResidualIsCheckedPerInterval(TestCase):
    """Each interval's residual is held to its own truncation error, with margin.

    Read against the largest third difference anywhere, a defect near a saddle -
    where the third derivative is 50 to 250 times smaller than mid-transition -
    passes when it is as large as the whole orbit's error estimate. And the exact
    orbit, sampled, leaves a residual 1.005 to 1.24 times ``max |D^3 x| / 12``, so
    without a safety factor the test refuses the orbit it is meant to accept.
    """

    def _solver(self, intervals: int):
        network, connection = _leak_network(0.0)
        source, target = network.saddle("xi1"), network.saddle("xi2")
        mesh = network.mesh(connection, half_length=20.0, intervals=intervals)
        solver = HeteroclinicOrbit(
            network.function,
            Terminus(source.equilibrium, source.jacobian),
            Terminus(target.equilibrium, target.jacobian),
            mesh,
        )
        return network, connection, mesh, solver

    def test_the_exact_orbit_sampled_on_the_mesh_is_a_trajectory(self):
        for intervals in (25, 200):
            _, _, mesh, solver = self._solver(intervals)
            times = np.linspace(-20.0, 20.0, intervals + 1)
            exact = OrbitSolution(times, _gh_plane_orbit(mesh, times))
            residual = solver.collocation_residual(exact)
            # Measured: residual / (max |D^3 x| / 12) = 1.24 at h = 1.6 and
            # 1.005 at h = 0.2; against the per-interval allowance, 0.62 and 0.50.
            assert residual.is_trajectory

    def test_a_defect_near_a_saddle_is_refused_where_it_is(self):
        network, connection, mesh, solver = self._solver(200)
        orbit = network.solve(connection, mesh)
        times = orbit.solution.times
        for centre in (-12.0, 12.0):
            bumped = orbit.solution.states.copy()
            bumped[:, 2] += 1.0e-5 * np.exp(-((times - centre) ** 2))
            residual = solver.collocation_residual(OrbitSolution(times, bumped))
            # The largest defect is far under the whole path's tolerance, but
            # not under what its own interval's truncation error allows.
            assert residual.largest < 0.1 * residual.tolerance
            assert not residual.is_trajectory


class TestLeakingPlaneIsRefused(TestCase):
    """A path that is not a trajectory is refused on its residual."""

    def test_refused_on_the_residual_although_the_estimate_passes(self):
        network, connection = _leak_network(0.05)
        for intervals in (200, 400):
            mesh = network.mesh(connection, half_length=20.0, intervals=intervals)
            with pytest.raises(NotATrajectoryError, match="residual") as caught:
                network.solve(connection, mesh)
            orbit = caught.value.orbit
            # The estimate alone would certify it, and the returned path is in
            # the plane: neither the estimate nor the defect is what refuses it.
            assert orbit.resolution.estimate < 1.0e-3
            assert orbit.subspace_defect < 1.0e-14
            assert orbit.residual.largest > 10.0 * orbit.residual.tolerance
            # The full-space check leaves the plane, as the flow does.
            assert orbit.full_space.subspace_defect > 1.0e-3
            assert not orbit.full_space.residual.is_trajectory

    def test_the_refusal_is_a_value_error(self):
        network, connection = _leak_network(0.05)
        mesh = network.mesh(connection, half_length=20.0, intervals=200)
        with pytest.raises(ValueError, match="not a trajectory"):
            network.solve(connection, mesh)


class TestSolvedInTheSubspace(TestCase):
    """Kirk-Silber in sheared coordinates: the connections leaving xi2."""

    def test_edges_leaving_xi2_lie_in_their_plane_to_rounding(self):
        """Measured defects 2.7e-15 and 3.7e-15; residuals 3e-12 and 2e-13."""
        interaction, network = _sheared_kirk_silber(_SHEAR4)
        for source, target in ((2, 3), (2, 4)):
            connection = network.connections[_EDGES.index((source, target))]
            mesh = network.mesh(connection, half_length=20.0, intervals=400)
            orbit = network.solve(connection, mesh)
            assert orbit.subspace_defect < 1.0e-13
            assert orbit.residual.largest < 1.0e-9
            assert orbit.resolution.estimate < 1.0e-3
            ends = (
                network.saddle(f"xi{source}").equilibrium,
                network.saddle(f"xi{target}").equilibrium,
            )
            assert np.linalg.norm(orbit.solution.states[0] - ends[0]) < 1.0e-3
            assert np.linalg.norm(orbit.solution.states[-1] - ends[1]) < 1.0e-3
            reference = _shoot(
                interaction,
                (source - 1, target - 1),
                _SHEAR4,
                mesh,
                orbit.solution.times,
            )
            error = float(np.max(np.abs(orbit.solution.states - reference)))
            # Measured 3.0e-4 and 3.4e-4: 4/3 of the estimate, as E - E/4 predicts.
            assert error < 2.0 * orbit.resolution.estimate
            # The full-space solve, from the same seed, finds another member
            # of the family: a trajectory off the plane (measured 0.025 and
            # 0.16; DEQ-175). Before, it diverged and kept the seed.
            assert orbit.full_space.converged
            assert orbit.full_space.residual.is_trajectory
            assert orbit.full_space.subspace_defect > 1.0e-2

    def test_every_edge_agrees_with_shooting_and_converges_at_second_order(self):
        interaction, network = _sheared_kirk_silber(_SHEAR4)
        for (source, target), connection in zip(
            _EDGES,
            network.connections,
            strict=True,
        ):
            errors = []
            for intervals in (200, 400):
                mesh = network.mesh(connection, half_length=20.0, intervals=intervals)
                orbit = network.solve(connection, mesh)
                assert orbit.subspace_defect < 1.0e-13
                reference = _shoot(
                    interaction,
                    (source - 1, target - 1),
                    _SHEAR4,
                    mesh,
                    orbit.solution.times,
                )
                errors.append(float(np.max(np.abs(orbit.solution.states - reference))))
            assert errors[1] < 1.0e-3
            # Measured: 8.5e-4 at most, and ratios 3.95 to 4.00.
            assert 3.9 < errors[0] / errors[1] < 4.1

    def test_a_phase_coordinate_orthogonal_to_the_plane_is_rejected(self):
        network = HeteroclinicNetwork(guckenheimer_holmes(1.0, -1.0, -1.5, -0.6))
        for index, name in enumerate(_NAMES):
            network.add_saddle(name, np.eye(3)[index])
        connection = network.add_connection(
            "xi1",
            "xi2",
            coordinate_plane(3, (0, 1)),
        )
        mesh = MeshSpec(20.0, 200, phase_index=2, phase_value=0.0)
        with pytest.raises(ValueError, match="phase"):
            network.solve(connection, mesh)


class TestEstimateIsInFullCoordinates(TestCase):
    """The resolution estimate measures shifts in the coordinates the orbit is read in.

    Sheared GH ``xi1 -> xi2`` is isolated in the full space too, so the full-space
    solver can be refined independently: interpolate the orbit onto the doubled
    mesh, solve there, and take the largest component of the shift. The network's
    estimate, from the subspace solve, must be that number.
    """

    def test_discretisation_shift_matches_an_independent_full_space_refinement(self):
        network = _sheared_gh_network(-1.4, -0.6)
        connection = network.connections[0]
        mesh = network.mesh(connection, half_length=16.0, intervals=100)
        orbit = network.solve(connection, mesh)
        source, target = network.saddle("xi1"), network.saddle("xi2")
        fine = MeshSpec(16.0, 200, mesh.phase_index, mesh.phase_value)
        solver = HeteroclinicOrbit(
            network.function,
            Terminus(source.equilibrium, source.jacobian),
            Terminus(target.equilibrium, target.jacobian),
            fine,
        )
        times = np.linspace(-16.0, 16.0, 201)
        seed = np.column_stack(
            [
                np.interp(times, orbit.solution.times, column)
                for column in orbit.solution.states.T
            ],
        )
        refined = solver.solve(seed)
        shift = float(np.max(np.abs(refined.states[::2] - orbit.solution.states)))
        assert orbit.resolution.discretisation == pytest.approx(shift, rel=1e-6)
        # The full-space solver's own estimate measures the same shift.
        coarse = HeteroclinicOrbit(
            network.function,
            Terminus(source.equilibrium, source.jacobian),
            Terminus(target.equilibrium, target.jacobian),
            mesh,
        )
        own = coarse.estimate_orbit_error(orbit.solution).discretisation
        assert own == pytest.approx(shift, rel=1e-6)


class TestSaddlesMustLieInTheSubspace(TestCase):
    """The subspace solve projects each saddle into the plane; it must already be there.

    Kirk-Silber's x2x3 plane is invariant for the flow and for the Jacobian at
    both xi2 and xi4, and the margin of ``xi2 -> xi4`` in it is positive (0.7),
    so neither existing check rejects it. But xi4 = e4 is not in the plane: the
    reduced problem's target would be ``B^T xi4 = 0``. Before this check the solve
    was refused only by accident, on a residual of 0.15, with the wrong reason
    ("the flow leaves the subspace").
    """

    def test_a_plane_that_misses_a_saddle_is_rejected_when_added(self):
        interaction = _kirk_silber_interaction(_KIRK_SILBER)
        network = HeteroclinicNetwork(reflection_cubic(np.ones(4), interaction))
        for index in range(4):
            network.add_saddle(f"xi{index + 1}", np.eye(4)[index])
        with pytest.raises(ValueError, match="does not contain the saddle 'xi4'"):
            network.add_connection("xi2", "xi4", coordinate_plane(4, (1, 2)))
        # The plane that does contain both is still accepted.
        network.add_connection("xi2", "xi4", coordinate_plane(4, (1, 3)))
