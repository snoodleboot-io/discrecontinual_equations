"""SRK4 is Kloeden-Platen (15.1.4): weak order 2.0, strong order 1.0, scalar noise only.

Both orders are checked because both are claimed, and because they differ: a
scheme that reached weak order 2.0 by carrying the exact ``I_(1,0)`` would also
be strong 1.5, and this one is deliberately not that scheme. The strong slope is
bounded above as well as below, so the test says what the scheme is, not only
what it is at least.
"""

import pytest

from discrecontinual_equations.solver.stochastic.srk4.srk4_config import SRK4Config
from discrecontinual_equations.solver.stochastic.srk4.srk4_solver import SRK4Solver
from tests.solver.geometric_brownian_motion import GeometricBrownianMotionCase, slope

_STEPS = [2.0**-k for k in range(2, 7)]


def _final(seed: int | None) -> list[float]:
    solver = SRK4Solver(SRK4Config(end_time=1.0, step_size=0.05, random_seed=seed))
    solver.solve(GeometricBrownianMotionCase().equation(), [1.0])
    return list(solver.solution.results[0].discretization)


class TestSRK4Randomness:
    def test_unseeded_runs_differ(self):
        assert _final(None) != _final(None)

    def test_random_seed_fixes_the_path_and_changes_it(self):
        assert _final(7) == _final(7)
        assert _final(7) != _final(8)


class TestSRK4Convergence:
    def test_weak_order_two_on_geometric_brownian_motion(self):
        errors = GeometricBrownianMotionCase(seed=400).errors(
            SRK4Solver,
            SRK4Config,
            _STEPS[:4],
            paths=1500,
            resolution=2.0**-8,
        )
        assert slope({h: e[1] for h, e in errors.items()}) > 1.6, errors

    def test_strong_order_is_one_not_more(self):
        errors = GeometricBrownianMotionCase().errors(
            SRK4Solver,
            SRK4Config,
            _STEPS,
            paths=60,
            resolution=2.0**-10,
        )
        strong = slope({h: e[0] for h, e in errors.items()})
        assert strong > 0.85, errors
        assert strong < 1.3, errors


class TestSRK4Refusals:
    def test_systems_are_refused_with_a_pointer_to_srk2(self):
        solver = SRK4Solver(SRK4Config(end_time=1.0, step_size=0.1, random_seed=1))
        with pytest.raises(ValueError, match="SRK2Solver"):
            solver.solve(
                GeometricBrownianMotionCase(dimension=2).equation(),
                [1.0, 1.0],
            )
