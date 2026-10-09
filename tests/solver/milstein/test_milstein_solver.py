"""Milstein is strong order 1.0 in either calculus, against that calculus's solution.

The convergence tests use the budget of the srk tests so the suite stays quick.
The thresholds sit well below the order 1.0 the scheme has, so sampling noise
does not fail them, but well above 0.5, so a regression to Euler-Maruyama cannot
pass; the bound on the finest-step error is what catches a wrong drift, whose
error does not shrink with the step at all.
"""

from discrecontinual_equations.solver.stochastic.milstein.milstein_config import (
    MilsteinConfig,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_solver import (
    MilsteinSolver,
)
from tests.solver.geometric_brownian_motion import GeometricBrownianMotionCase, slope

_STEPS = [2.0**-k for k in range(3, 8)]
_RESOLUTION = 2.0**-10


class TestMilsteinConvergence:
    def test_strong_order_one_on_geometric_brownian_motion(self):
        errors = GeometricBrownianMotionCase().errors(
            MilsteinSolver,
            MilsteinConfig,
            _STEPS,
            paths=40,
            resolution=_RESOLUTION,
        )
        strong = slope({h: e[0] for h, e in errors.items()})
        assert strong > 0.85, errors
        assert strong < 1.3, errors

    def test_stratonovich_converges_to_the_stratonovich_solution(self):
        # Pins the sign of the drift correction (DEQ-27): the Stratonovich solution
        # X0 exp(mu t + sigma W) is only reached if (1/2) b b' is added to the drift,
        # not taken off. With the sign inverted the error at the finest step stays
        # near 0.5 instead of falling below 0.05.
        errors = GeometricBrownianMotionCase(seed=200, calculus="stratonovich").errors(
            MilsteinSolver,
            MilsteinConfig,
            _STEPS,
            paths=40,
            resolution=_RESOLUTION,
        )
        assert slope({h: e[0] for h, e in errors.items()}) > 0.85, errors
        assert errors[_STEPS[-1]][0] < 0.05, errors
