"""SRK3 is Kloeden-Platen (11.2.1): strong order 1.5 for scalar noise, systems refused.

The convergence thresholds sit between the measured order (1.48 at 100000 paths)
and the order the scheme would have if any of its terms were wrong (1.0 at best),
so they discriminate without being brittle at the small budget a unit test can
afford. The weak slope is checked too, because the harness measured it at 1.98
and the docstring reports that number.
"""

import pytest

from discrecontinual_equations.solver.stochastic.srk3.srk3_config import SRK3Config
from discrecontinual_equations.solver.stochastic.srk3.srk3_solver import SRK3Solver
from tests.solver.geometric_brownian_motion import GeometricBrownianMotionCase, slope

_STEPS = [2.0**-k for k in range(2, 7)]
_RESOLUTION = 2.0**-10


def _final(seed: int | None) -> list[float]:
    solver = SRK3Solver(SRK3Config(end_time=1.0, step_size=0.05, random_seed=seed))
    solver.solve(GeometricBrownianMotionCase().equation(), [1.0])
    return list(solver.solution.results[0].discretization)


class TestSRK3Randomness:
    def test_unseeded_runs_differ(self):
        assert _final(None) != _final(None)

    def test_random_seed_fixes_the_path_and_changes_it(self):
        assert _final(7) == _final(7)
        assert _final(7) != _final(8)


class TestSRK3Convergence:
    def test_strong_order_one_and_a_half_on_geometric_brownian_motion(self):
        errors = GeometricBrownianMotionCase().errors(
            SRK3Solver,
            SRK3Config,
            _STEPS,
            paths=60,
            resolution=_RESOLUTION,
        )
        strong = slope({h: e[0] for h, e in errors.items()})
        assert strong > 1.3, errors
        assert strong < 1.8, errors

    def test_weak_order_two_on_geometric_brownian_motion(self):
        errors = GeometricBrownianMotionCase(seed=300).errors(
            SRK3Solver,
            SRK3Config,
            _STEPS[:4],
            paths=1500,
            resolution=2.0**-8,
        )
        assert slope({h: e[1] for h, e in errors.items()}) > 1.6, errors

    def test_stratonovich_converges_to_the_stratonovich_solution(self):
        errors = GeometricBrownianMotionCase(seed=200, calculus="stratonovich").errors(
            SRK3Solver,
            SRK3Config,
            _STEPS,
            paths=40,
            resolution=_RESOLUTION,
        )
        assert slope({h: e[0] for h, e in errors.items()}) > 1.3, errors


class TestSRK3Refusals:
    def test_systems_are_refused_with_a_pointer_to_srk2(self):
        solver = SRK3Solver(SRK3Config(end_time=1.0, step_size=0.1, random_seed=1))
        with pytest.raises(ValueError, match="SRK2Solver"):
            solver.solve(
                GeometricBrownianMotionCase(dimension=2).equation(),
                [1.0, 1.0],
            )
