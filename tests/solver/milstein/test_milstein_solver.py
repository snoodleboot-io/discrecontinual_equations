"""Milstein is strong order 1.0 in either calculus, against that calculus's solution.

The convergence tests use the budget of the srk tests so the suite stays quick.
The thresholds sit well below the order 1.0 the scheme has, so sampling noise
does not fail them, but well above 0.5, so a regression to Euler-Maruyama cannot
pass; the bound on the finest-step error is what catches a wrong drift, whose
error does not shrink with the step at all.
"""

import pytest

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_solver import (
    EulerMaruyamaSolver,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_config import (
    MilsteinConfig,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_solver import (
    MilsteinSolver,
)
from discrecontinual_equations.variable import Variable
from tests.solver.conformal_geometric_brownian_motion import ConformalCase
from tests.solver.geometric_brownian_motion import GeometricBrownianMotionCase, slope
from tests.solver.rotating_ornstein_uhlenbeck import (
    RotatingOrnsteinUhlenbeck,
    equation,
)

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


class ShearNoise(StochasticFunction):
    """``G = [[1, x], [0, 1]]``: ``(dG_2/dx) G_1 = (1, 0)``, ``(dG_1/dx) G_2 = 0``."""

    def eval(self, point: list[float], time: float | None = None) -> list[float]:  # noqa: ARG002
        return [0.0, 0.0]

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[list[float]]:
        return [[1.0, point[0]], [0.0, 1.0]]


def _shear_equation() -> DifferentialEquation:
    time = Variable(name="t", discretization=[])
    variables = [
        Variable(name="x", discretization=[]),
        Variable(name="y", discretization=[]),
    ]
    results = [
        Variable(name="dx", discretization=[]),
        Variable(name="dy", discretization=[]),
    ]
    return DifferentialEquation(
        variables=variables,
        time=time,
        parameters=[],
        derivative=ShearNoise(variables, [], results, time),
    )


class TestMilsteinMatrixNoise:
    """Commuting matrix noise gets the full Milstein term; non-commuting is refused.

    On complex geometric Brownian motion (two coupled multiplicative drivers whose
    noise commutes) the slope measured 0.97 against Euler-Maruyama's 0.59 on the
    same paths. On a decoupled two-component system the scheme used to fall back
    to Euler-Maruyama with a printed warning; it is now Milstein in each component.
    """

    def test_strong_order_one_on_conformal_noise(self):
        errors = ConformalCase().errors(
            MilsteinSolver,
            MilsteinConfig,
            _STEPS,
            paths=40,
            resolution=_RESOLUTION,
        )
        order = slope(errors)
        assert order > 0.85, errors
        assert order < 1.3, errors
        assert errors[_STEPS[-1]] < 0.02, errors

    def test_beats_euler_maruyama_on_the_same_paths(self):
        milstein = ConformalCase().errors(
            MilsteinSolver,
            MilsteinConfig,
            [_STEPS[-1]],
            paths=20,
            resolution=_RESOLUTION,
        )
        euler = ConformalCase().errors(
            EulerMaruyamaSolver,
            EulerMaruyamaConfig,
            [_STEPS[-1]],
            paths=20,
            resolution=_RESOLUTION,
        )
        assert milstein[_STEPS[-1]] < 0.5 * euler[_STEPS[-1]], (milstein, euler)

    def test_strong_order_one_is_kept_on_a_decoupled_system(self):
        errors = GeometricBrownianMotionCase(dimension=2, seed=300).errors(
            MilsteinSolver,
            MilsteinConfig,
            _STEPS,
            paths=40,
            resolution=_RESOLUTION,
        )
        assert slope({h: e[0] for h, e in errors.items()}) > 0.85, errors

    def test_accepts_additive_matrix_noise(self):
        # Constant G: every column Jacobian is zero, so the noise commutes trivially
        # and the step is Euler-Maruyama's, which is exact for the Milstein term.
        config = MilsteinConfig(
            start_time=0.0,
            end_time=1.0,
            step_size=0.01,
            random_seed=3,
        )
        solver = MilsteinSolver(config)
        solver.solve(equation(RotatingOrnsteinUhlenbeck), [0.0, 0.0])
        assert len(solver.solution.results[0].discretization) == 101

    def test_refuses_non_commuting_noise(self):
        config = MilsteinConfig(start_time=0.0, end_time=1.0, step_size=0.01)
        with pytest.raises(ValueError, match="commuting noise"):
            MilsteinSolver(config).solve(_shear_equation(), [1.0, 0.0])
