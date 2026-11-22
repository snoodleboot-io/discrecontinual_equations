"""Cycle-to-cycle connections from parameterized manifolds (DEQ-38).

``CylinderLadder`` has two saddle cycles joined, by its rotational symmetry,
along the invariant cylinder ``r = 1`` where ``z' = z (1 - z)``: every orbit there
is a connection, and its height is the logistic curve. Tipping the radial
flow breaks the symmetry; the connections then found are checked against the
flow alone (DOP853), which must carry them back onto one cycle and on to the
other.
"""

import math
from functools import cache
from unittest import TestCase

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.cycle_connection import (
    CycleConnectingOrbit,
)
from discrecontinual_equations.continuation.cycle_manifold import CycleManifold
from discrecontinual_equations.continuation.manifold import (
    StableManifold,
    UnstableManifold,
)
from discrecontinual_equations.continuation.periodic_orbit import (
    PeriodicOrbitSolution,
)
from tests.continuation.cycle_fields import OMEGA, CylinderLadder

PERIOD = 2.0 * math.pi / OMEGA
SIGMA = 0.02


def _ring(height: float) -> PeriodicOrbitSolution:
    s = np.linspace(0.0, 1.0, 41)
    states = np.column_stack(
        [np.cos(2 * math.pi * s), np.sin(2 * math.pi * s), np.full(s.size, height)],
    )
    return PeriodicOrbitSolution(states, PERIOD)


@cache
def _solver(epsilon: float) -> tuple[CylinderLadder, CycleConnectingOrbit]:
    field = CylinderLadder(epsilon)
    builder = CycleManifold(order=10, harmonics=32)
    lower = builder.compute(field, _ring(0.0), UnstableManifold())
    upper = builder.compute(field, _ring(1.0), StableManifold())
    # Point each chart's sigma between the cycles: up from the lower, down
    # from the upper.
    out = SIGMA * math.copysign(1.0, lower.grid[1][0][2])
    into = -SIGMA * math.copysign(1.0, upper.grid[1][0][2])
    return field, CycleConnectingOrbit(field, lower, upper, out, into)


def _flow(field, start: np.ndarray, time: float) -> np.ndarray:
    return solve_ivp(
        lambda _t, x: field.eval(point=list(x), time=None),
        (0.0, time),
        start,
        method="DOP853",
        rtol=1.0e-13,
        atol=1.0e-13,
    ).y[:, -1]


class TestSymmetricConnection(TestCase):
    """The connection known by symmetry: on the cylinder, height logistic."""

    @classmethod
    def setUpClass(cls):
        cls.field, solver = _solver(0.0)
        seeds = solver.seeds(horizon=10.0, phases=12)
        cls.connection = solver.solve(seeds[0])

    def test_residual_is_converged(self):
        assert self.connection.residual < 1.0e-10

    def test_orbit_lies_on_the_cylinder(self):
        times, states = self.connection.orbit(tail=8.0)
        radii = np.hypot(states[:, 0], states[:, 1])
        assert np.max(np.abs(radii - 1.0)) < 1.0e-10
        assert times[0] < 0.0 < times[-1]

    def test_height_is_the_logistic_curve(self):
        times, states = self.connection.orbit(tail=8.0)
        start = states[np.argmin(np.abs(times))][2]
        logistic = 1.0 / (1.0 + (1.0 / start - 1.0) * np.exp(-times))
        assert np.max(np.abs(states[:, 2] - logistic)) < 1.0e-10

    def test_time_of_flight_is_the_logistic_transit(self):
        # On the cylinder z / (1 - z) = C e^t, so the unstable chart of the
        # lower cycle is z = sigma / (1 + sigma) and the stable one of the
        # upper is 1 - z = sigma / (1 + sigma): from z / (1 - z) = SIGMA to
        # 1 / SIGMA takes 2 ln(1 / SIGMA).
        transit = 2.0 * math.log(1.0 / SIGMA)
        # The residual is converged to 1e-10 where the flow moves at about
        # SIGMA, so the time of flight is good to about 1e-10 / SIGMA.
        assert abs(self.connection.time - transit) < 1.0e-8


class TestTransverseConnection(TestCase):
    """With the symmetry broken, the connection is checked by the flow alone."""

    @classmethod
    def setUpClass(cls):
        cls.field, cls.solver = _solver(0.6)
        cls.seeds = cls.solver.seeds(horizon=10.0, phases=12)
        cls.connection = cls.solver.solve(cls.seeds[0])

    def test_seed_comes_from_the_manifold_intersection(self):
        # The nearest approach of the flowed source ring to the target ring is
        # only a seed; the damped Newton step takes it the rest of the way.
        gap = np.linalg.norm(self.solver.residual(self.seeds[0]))
        assert self.connection.residual < 1.0e-10 < gap

    def test_symmetry_is_broken(self):
        _, states = self.connection.orbit(tail=2.0)
        radii = np.hypot(states[:, 0], states[:, 1])
        assert np.max(np.abs(radii - 1.0)) > 1.0e-4

    def test_backward_flow_returns_to_the_lower_cycle(self):
        start = self.connection.states[0]
        distances = []
        for time in (-3.0, -6.0, -9.0):
            state = _flow(self.field, start, time)
            distances.append(abs(math.hypot(state[0], state[1]) - 1.0) + abs(state[2]))
        # The unstable exponent is 1: each 3 time units shrinks it ~e^-3.
        assert distances[2] < distances[1] < distances[0] < 2.0 * SIGMA * math.exp(-3.0)
        assert distances[2] / distances[1] < 0.1

    def test_forward_flow_reaches_the_upper_cycle(self):
        start = self.connection.states[0]
        end = _flow(self.field, start, self.connection.time + 3.0)
        target = self.connection.orbit(tail=3.0)[1][-1]
        assert np.linalg.norm(end - target) < 1.0e-8
        assert abs(end[2] - 1.0) < 2.0 * SIGMA * math.exp(-3.0)

    def test_two_distinct_connections_exist(self):
        # A Melnikov function cos-like in the phase has two zeros per period.
        phases = []
        for seed in self.seeds[:4]:
            found = self.solver.solve(seed)
            phase = found.source_phase % PERIOD
            if all(abs(phase - p) > 1.0e-6 for p in phases):
                phases.append(phase)
        assert len(phases) == 2


class TestRefusals(TestCase):
    def test_source_must_be_unstable_and_target_stable(self):
        field = CylinderLadder(0.0)
        builder = CycleManifold(order=4, harmonics=16)
        upper = builder.compute(field, _ring(1.0), StableManifold())
        with pytest.raises(ValueError, match="source chart"):
            CycleConnectingOrbit(field, upper, upper, SIGMA, SIGMA)
        lower = builder.compute(field, _ring(0.0), UnstableManifold())
        with pytest.raises(ValueError, match="target chart"):
            CycleConnectingOrbit(field, lower, lower, SIGMA, SIGMA)
