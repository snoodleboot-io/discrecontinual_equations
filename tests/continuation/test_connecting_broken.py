"""A connection broken by less than the mesh's truncation error is not a trajectory.

The oracle is the DEQ-120 verifier's finding (DEQ-155). The homoclinic loop of
``x'' = x - x^2`` is broken by damping ``delta y``. The solve converges to the
least-squares minimum, as it should, but the per-interval allowance passed it
up to ``delta = 9e-3`` at ``h = 0.25``. Its residual, about ``0.4 delta
sqrt(h)``, is spread along the orbit, under allowances of order ``h^3``.

The consistency test asks whether the discrete system, corrected by its own
truncation error, can be satisfied (deferred correction). The genuine controls
need no structure to protect them. ``y' = x - x^2 + k y H``, with ``H`` the
saddle's energy, keeps the loop ``x = 1.5 sech^2(t/2)`` exactly, but the flow is
neither Hamiltonian nor reversible. The trapezoidal rule therefore breaks the
connection at ``O(h^2)``, and only the corrected scheme sees that it is genuine.
Its exact orbit comes from the formula, not from the solver.

Kirk-Silber and Bogdanov-Takens keep reporting trajectories in
``test_heteroclinic_network_subspace`` and ``test_connecting_convergence``,
whose ``is_trajectory`` assertions now include this test.
"""

import math
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.connecting_orbit import (
    CollocationResidual,
    HomoclinicOrbit,
    MeshSpec,
    OrbitSolution,
    Terminus,
    _difference,
    _manifold_miss,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import State


class _GuardedWell(DeterministicFunction):
    """``x' = y, y' = x - x^2 + k y H - delta y``, ``H = y^2 - x^2 + 2 x^3 / 3``.

    ``H`` vanishes on the saddle's homoclinic loop, so at ``delta = 0`` the loop
    is an exact orbit for every ``k``. ``k = 0`` is the reversible well of the
    ticket.
    """

    guard = 0.0
    damping = 0.0

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x, y = point[0], point[1]
        energy = y * y - x * x + 2.0 * x * x * x / 3.0
        return [y, x - x * x + self.guard * y * energy - self.damping * y]


def _loop(times: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            1.5 / np.cosh(times / 2.0) ** 2,
            -1.5 * np.tanh(times / 2.0) / np.cosh(times / 2.0) ** 2,
        ],
    )


def _problem(damping: float, guard: float = 0.0, half: float = 10.0, n: int = 80):
    field = _GuardedWell(
        variables=[State(), State()],
        parameters=[],
        results=[State(), State()],
        time=None,
    )
    field.guard, field.damping = guard, damping
    jacobian = np.array([[0.0, 1.0], [1.0, -damping]])
    solver = HomoclinicOrbit(field, np.zeros(2), jacobian, MeshSpec(half, n, 1, 0.0))
    times = np.linspace(-half, half, n + 1)
    return solver, times


def _ratio(residual) -> float:
    return residual.inconsistency / residual.inconsistency_tolerance


class TestBrokenConnectionIsNotATrajectory(TestCase):
    """The damped family, on the verifier's mesh (h = 0.25) and on a finer one."""

    def test_the_tickets_case_is_refused_on_consistency(self):
        solver, times = _problem(1.0e-4)
        residual = solver.solve(_loop(times)).residual
        # The per-interval test alone passed it: measured worst 1.12e-2.
        assert residual.worst < 0.1
        assert not residual.consistent
        assert not residual.is_trajectory
        # Measured 4.1: the miss is 0.415 delta against a bound of 1.0e-5.
        assert _ratio(residual) > 2.0

    def test_the_smallest_resolved_damping_falls_with_the_mesh(self):
        # Measured at T = 10: refused from delta = 3.2e-5 at h = 0.25 and from
        # 5.6e-7 at h = 0.125 (quarter-decade grid). Below that the miss is
        # inside the corrected scheme's own error, and the path is accepted.
        for intervals, refused, accepted in (
            (80, 5.0e-5, 1.0e-5),
            (160, 1.0e-6, 2.0e-7),
        ):
            with self.subTest(intervals=intervals):
                solver, times = _problem(refused, n=intervals)
                assert not solver.solve(_loop(times)).residual.is_trajectory
                solver, times = _problem(accepted, n=intervals)
                assert solver.solve(_loop(times)).residual.is_trajectory

    def test_a_fine_mesh_resolves_what_its_ends_would_hide(self):
        """At h = 0.0625 a loop broken by 1e-7 is refused (measured 7x its bound).
        The ends' truncation term is weighted by each boundary row's share of
        the inconsistent direction. Unweighted, it would be 2.4e-8 and hide
        this miss of 2.1e-8."""
        solver, times = _problem(1.0e-7, n=320)
        residual = solver.solve(_loop(times)).residual
        assert not residual.is_trajectory
        assert _ratio(residual) > 3.0

    def test_the_miss_grows_linearly_with_the_damping(self):
        misses = []
        for damping in (1.0e-6, 1.0e-4):
            solver, times = _problem(damping)
            misses.append(solver.solve(_loop(times)).residual.inconsistency)
        # The least-squares miss of a loop broken by delta is linear in delta.
        assert misses[1] / misses[0] == pytest.approx(100.0, rel=0.02)


class TestGenuineConnectionsStayTrajectories(TestCase):
    """Genuine loops whose discrete systems are inconsistent at O(h^2)."""

    def test_the_reversible_loop_is_consistent_to_rounding(self):
        solver, times = _problem(0.0)
        residual = solver.solve(_loop(times)).residual
        assert residual.is_trajectory
        assert residual.inconsistency < 1.0e-15

    def test_a_dissipative_loop_solved_and_sampled_is_a_trajectory(self):
        for intervals in (40, 80, 160):
            solver, times = _problem(0.0, guard=1.0, n=intervals)
            for label, residual in (
                ("solved", solver.solve(_loop(times)).residual),
                (
                    "exact",
                    solver.collocation_residual(OrbitSolution(times, _loop(times))),
                ),
            ):
                with self.subTest(intervals=intervals, path=label):
                    # Measured 0.16, 0.088 and 0.072 solved; 0.022, 0.014 and
                    # 0.012 exact.
                    assert residual.is_trajectory
                    assert _ratio(residual) < 0.3

    def test_the_uncorrected_inconsistency_would_refuse_it(self):
        """The trapezoidal system itself is inconsistent here: without the
        correction the genuine loop's miss is 30x the bound (measured 8.7e-4
        against 1.0e-5 at h = 0.25)."""
        solver, times = _problem(0.0, guard=1.0)
        solution = solver.solve(_loop(times))
        shape = solution.states.shape
        unknowns = solution.states.flatten()
        plain = solver._residual(unknowns, times, shape)  # noqa: SLF001
        differentiate = solver._jacobian  # noqa: SLF001
        jacobian = differentiate(unknowns, times, shape, plain, analytic=True)
        step = solver._least_squares(jacobian, plain)  # noqa: SLF001
        uncorrected = float(np.linalg.norm(plain + jacobian @ step))
        assert uncorrected > 10.0 * solution.residual.inconsistency_tolerance
        assert solution.residual.is_trajectory

    def test_a_short_interval_is_allowed_its_truncation(self):
        """At T = 5 the projection conditions are missed by the manifold's
        curvature. The miss term of the bound carries it (measured 0.42 of the
        bound at h = 0.125, solved and exact), and without that term the genuine
        loop would be refused."""
        solver, times = _problem(0.0, guard=1.0, half=5.0, n=80)
        for residual in (
            solver.solve(_loop(times)).residual,
            solver.collocation_residual(OrbitSolution(times, _loop(times))),
        ):
            assert residual.is_trajectory
            assert 0.33 < _ratio(residual) < 0.6


class TestTheBoundsIngredients(TestCase):
    """The difference stencil and the manifold miss, against closed forms."""

    def test_differences_of_a_polynomial_are_exact(self):
        times = np.linspace(0.0, 1.0, 11)
        spacing = times[1] - times[0]
        cubic = (times**3)[:, None]
        # D3 of t^3 is 6 h^3 everywhere, on interior and end stencils alike.
        for interval in (0, 4, 9):
            assert _difference(cubic, interval, 3)[0] == pytest.approx(
                6.0 * spacing**3, rel=1e-9
            )
        seventh = (times**7)[:, None]
        assert _difference(seventh, 2, 7)[0] == pytest.approx(
            5040.0 * spacing**7, rel=1e-6
        )

    def test_the_stencil_is_centred_on_its_interval(self):
        states = np.arange(20.0)[:, None] ** 4
        # A centred D3 of t^4 on nodes i-1..i+2 is 24 h^3 (t_i + h/2) with h = 1.
        assert _difference(states, 7, 3)[0] == pytest.approx(24.0 * 7.5, rel=1e-12)

    def test_the_miss_matches_the_manifolds_quadratic_term(self):
        """``x' = x, y' = -y + x^2``: the unstable manifold is ``y = x^2 / 3``
        exactly, so a point on it misses the linear condition ``y = 0`` by
        ``x^2 / 3`` - what ``l N / (2 mu - lambda) = x^2 / (2 + 1)`` gives."""

        class Curved(DeterministicFunction):
            def eval(self, point, time=None):  # noqa: ARG002
                return [point[0], -point[1] + point[0] ** 2]

        field = Curved(
            variables=[State(), State()],
            parameters=[],
            results=[State(), State()],
            time=None,
        )
        end = Terminus(np.zeros(2), np.array([[1.0, 0.0], [0.0, -1.0]]))
        x = 1.0e-2
        miss = _manifold_miss(field, end, np.array([x, x * x / 3.0]), departing=True)
        assert miss.shape == (1,)
        assert miss[0] == pytest.approx(x * x / 3.0, rel=1e-9)
        # Arriving along the stable manifold x = 0 there is nothing to miss.
        arrival = _manifold_miss(field, end, np.array([0.0, x]), departing=False)
        assert arrival[0] == 0.0

    def test_a_path_too_short_to_difference_is_not_checked(self):
        solver, _ = _problem(1.0e-2, n=6)
        times = np.linspace(-10.0, 10.0, 7)
        residual = solver.collocation_residual(OrbitSolution(times, _loop(times)))
        assert residual.inconsistency == 0.0
        assert residual.inconsistency_tolerance == 0.0
        assert residual.consistent

    def test_a_path_that_overflows_the_measurement_is_inconsistent(self):
        solver, times = _problem(0.0)
        wild = _loop(times) * 1.0e200
        residual = solver.collocation_residual(OrbitSolution(times, wild))
        assert math.isinf(residual.inconsistency)
        assert not residual.consistent

    def test_a_bound_that_is_not_finite_proves_nothing(self):
        for bound in (math.inf, math.nan):
            with self.subTest(bound=bound):
                residual = CollocationResidual(1e-3, 1.0, 0.5, 1.0, bound)
                assert not residual.consistent
                assert not residual.is_trajectory
        nan_miss = CollocationResidual(1e-3, 1.0, 0.5, math.nan, 1.0)
        assert not nan_miss.consistent
