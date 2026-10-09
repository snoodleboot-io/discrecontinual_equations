"""SRK2 is Kloeden-Platen (11.1.7): strong order 1.0, and actually random.

The convergence tests here use small budgets so the suite stays quick; the
orders quoted in the solver's docstring come from the same harness at 100000
paths. The thresholds are set well below the measured slopes so that sampling
noise does not fail them, but well above the next order down, so a regression
to Euler-Maruyama's 0.5 - or to a deterministic scheme - cannot pass.
"""

import numpy as np

from discrecontinual_equations.solver.stochastic.srk2.srk2_config import SRK2Config
from discrecontinual_equations.solver.stochastic.srk2.srk2_solver import SRK2Solver
from tests.solver.geometric_brownian_motion import GeometricBrownianMotionCase, slope

_STEPS = [2.0**-k for k in range(3, 8)]
_RESOLUTION = 2.0**-10


def _final(seed: int | None, **config) -> list[float]:
    solver = SRK2Solver(
        SRK2Config(end_time=1.0, step_size=0.05, random_seed=seed, **config),
    )
    solver.solve(GeometricBrownianMotionCase().equation(), [1.0])
    return list(solver.solution.results[0].discretization)


class TestSRK2Randomness:
    def test_unseeded_runs_differ(self):
        # The defect of DEQ-25: five runs agreed to the last digit because nothing
        # was drawn. Two unseeded runs must now disagree.
        assert _final(None) != _final(None)

    def test_random_seed_fixes_the_path_and_changes_it(self):
        assert _final(7) == _final(7)
        assert _final(7) != _final(8)

    def test_the_global_stream_is_not_touched(self):
        np.random.seed(1)  # noqa: NPY002 (the stream under test)
        expected = np.random.normal(size=4)  # noqa: NPY002
        np.random.seed(1)  # noqa: NPY002
        _final(3)
        assert np.array_equal(expected, np.random.normal(size=4))  # noqa: NPY002


class TestSRK2Convergence:
    def test_strong_order_one_on_geometric_brownian_motion(self):
        errors = GeometricBrownianMotionCase().errors(
            SRK2Solver,
            SRK2Config,
            _STEPS,
            paths=60,
            resolution=_RESOLUTION,
        )
        strong = slope({h: e[0] for h, e in errors.items()})
        assert strong > 0.85, errors
        assert strong < 1.3, errors

    def test_strong_order_one_is_kept_on_a_decoupled_system(self):
        # Two independent copies share nothing, so the omitted Levy areas are zero
        # and the diagonal scheme keeps its order; this is the case the docstring
        # says is safe.
        errors = GeometricBrownianMotionCase(dimension=2, seed=100).errors(
            SRK2Solver,
            SRK2Config,
            _STEPS,
            paths=40,
            resolution=_RESOLUTION,
        )
        assert slope({h: e[0] for h, e in errors.items()}) > 0.85, errors

    def test_stratonovich_converges_to_the_stratonovich_solution(self):
        # Pins the sign of the drift correction: the Stratonovich solution
        # X0 exp(mu t + sigma W) is only reached if (1/2) b b' is added, not taken off.
        errors = GeometricBrownianMotionCase(seed=200, calculus="stratonovich").errors(
            SRK2Solver,
            SRK2Config,
            _STEPS,
            paths=40,
            resolution=_RESOLUTION,
        )
        assert slope({h: e[0] for h, e in errors.items()}) > 0.85, errors
        assert errors[_STEPS[-1]][0] < 0.05, errors
