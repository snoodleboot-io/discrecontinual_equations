"""The adaptive stepping of the embedded pairs: termination, error norm, PI control.

The controller is checked against ``DOPRI5``'s ``DOPCOR`` written out by hand:
``FAC11 = ERR**EXPO1``, ``EXPO1 = 1/q - 0.75 BETA``, ``FAC = FAC11 /
FACOLD**BETA`` held to ``[1/FAC2, 1/FAC1] = [0.1, 5]`` after dividing by
``SAFE``, ``HNEW = H / FAC``; on rejection ``HNEW = H / MIN(5, FAC11 / SAFE)``.
"""

import math
import time

import numpy as np
import pytest

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.solver.deterministic.cash_karp.cash_karp_config import (
    CashKarpConfig,
)
from discrecontinual_equations.solver.deterministic.cash_karp.cash_karp_solver import (
    CashKarpSolver,
)
from discrecontinual_equations.solver.deterministic.dormand_prince.dormand_prince_config import (  # noqa: E501
    DormandPrinceConfig,
)
from discrecontinual_equations.solver.deterministic.dormand_prince.dormand_prince_solver import (  # noqa: E501
    DormandPrinceSolver,
)
from discrecontinual_equations.solver.deterministic.embedded_runge_kutta import (
    StepSizeTooSmallError,
)
from discrecontinual_equations.solver.deterministic.fehlberg.fehlberg_config import (
    FehlbergConfig,
)
from discrecontinual_equations.solver.deterministic.fehlberg.fehlberg_solver import (
    FehlbergSolver,
)
from discrecontinual_equations.solver.deterministic.prince_dormand87.prince_dormand87_config import (  # noqa: E501
    PrinceDormand87Config,
)
from discrecontinual_equations.solver.deterministic.prince_dormand87.prince_dormand87_solver import (  # noqa: E501
    PrinceDormand87Solver,
)
from discrecontinual_equations.variable import Variable

PAIRS = [
    pytest.param(DormandPrinceSolver, DormandPrinceConfig, id="dormand_prince"),
    pytest.param(CashKarpSolver, CashKarpConfig, id="cash_karp"),
    pytest.param(FehlbergSolver, FehlbergConfig, id="fehlberg"),
    pytest.param(PrinceDormand87Solver, PrinceDormand87Config, id="prince_dormand87"),
]
# q: one more than the order of each pair's error estimate.
Q = {
    DormandPrinceSolver: 5,
    CashKarpSolver: 5,
    FehlbergSolver: 5,
    PrinceDormand87Solver: 8,
}

# More field evaluations than any of these solves needs (about 900): a solver
# that loops fails the test here instead of hanging it.
BUDGET = 100_000


class Singular(DeterministicFunction):
    """``y' = 1 / (1 - t)``: ``y = -log(1 - t)``, unbounded at ``t = 1``."""

    calls = 0

    def eval(self, point, time=None):  # noqa: ARG002
        Singular.calls += 1
        if Singular.calls > BUDGET:
            message = "the solver did not stop: evaluation budget spent"
            raise AssertionError(message)
        return [1.0 / (1.0 - time)]


class Decay(DeterministicFunction):
    """``y' = -y``."""

    def eval(self, point, time=None):  # noqa: ARG002
        return [-point[0]]


def equation(field: type[DeterministicFunction]) -> DifferentialEquation:
    t, y, r = Variable(name="t"), Variable(name="y"), Variable(name="r")
    return DifferentialEquation(
        variables=[y],
        time=t,
        parameters=[],
        derivative=field([y], [], [r], t),
    )


class TestTermination:
    """A step rejected at ``min_step_size`` raises; it retried forever (DEQ-140)."""

    @pytest.mark.parametrize(("solver", "config"), PAIRS)
    def test_singular_field_stops_with_a_clear_error(self, solver, config):
        Singular.calls = 0
        started = time.perf_counter()
        with pytest.raises(StepSizeTooSmallError) as raised:
            solver(config(end_time=1.0)).solve(equation(Singular), [0.0])
        assert time.perf_counter() - started < 5.0
        error = raised.value
        assert isinstance(error, RuntimeError)
        assert 0.999 < error.time < 1.0
        assert error.step <= 1e-8
        assert error.error_ratio is not None
        assert error.error_ratio > 1.0
        assert f"t = {error.time!r}" in str(error)
        assert "h = 1e-08" in str(error)

    @pytest.mark.parametrize(("solver", "config"), PAIRS)
    def test_unreachable_tolerance_at_the_minimum_step_raises_at_once(
        self, solver, config
    ):
        settings = config(
            end_time=1.0,
            initial_step_size=0.5,
            min_step_size=0.5,
            max_step_size=0.5,
            absolute_tolerance=1e-30,
            relative_tolerance=1e-30,
        )
        with pytest.raises(StepSizeTooSmallError) as raised:
            solver(settings).solve(equation(Decay), [1.0])
        assert raised.value.time == 0.0
        assert raised.value.step == 0.5

    @pytest.mark.parametrize(("solver", "config"), PAIRS)
    def test_last_point_is_end_time(self, solver, config):
        curve = solver(config(end_time=1.7)).solve(equation(Decay), [1.0])
        assert curve.time.discretization[-1] == 1.7
        assert abs(curve.results[0].discretization[-1] - math.exp(-1.7)) < 1e-5

    @pytest.mark.parametrize(("solver", "config"), PAIRS)
    def test_beta_too_large_for_the_exponent_is_refused(self, solver, config):
        q = Q[solver]
        with pytest.raises(ValueError, match="beta"):
            solver(config(beta=4 / (3 * q)))


class TestController:
    """The PI controller, step for step against DOPRI5's formulas."""

    @pytest.mark.parametrize(("solver", "config"), PAIRS)
    def test_accepted_step_is_proportional_integral(self, solver, config):
        stepper = solver(config(max_step_size=100.0))
        h, ratio, previous = 0.1, 0.3, 0.7
        alpha = 1 / Q[solver] - 0.75 * 0.04
        expected = h / ((ratio**alpha / previous**0.04) / 0.9)
        assert stepper.accepted_step_size(h, ratio, previous) == pytest.approx(
            expected,
            rel=1e-14,
        )

    @pytest.mark.parametrize(("solver", "config"), PAIRS)
    def test_previous_error_enters(self, solver, config):
        stepper = solver(config(max_step_size=100.0))
        calm = stepper.accepted_step_size(0.1, 0.5, 0.01)
        rough = stepper.accepted_step_size(0.1, 0.5, 1.0)
        assert calm / rough == pytest.approx(0.01**0.04, rel=1e-14)

    @pytest.mark.parametrize(("solver", "config"), PAIRS)
    def test_growth_is_at_most_tenfold(self, solver, config):
        stepper = solver(config(max_step_size=100.0))
        assert stepper.accepted_step_size(0.1, 0.0, 1e-4) == pytest.approx(1.0)
        assert stepper.accepted_step_size(0.1, 1e-30, 1e-4) == pytest.approx(1.0)

    @pytest.mark.parametrize(("solver", "config"), PAIRS)
    def test_accepted_step_stays_within_the_bounds(self, solver, config):
        stepper = solver(config(max_step_size=0.2, min_step_size=0.05))
        assert stepper.accepted_step_size(0.1, 1e-12, 1e-4) == 0.2
        assert stepper.accepted_step_size(0.06, 1.0, 1e-4) == 0.05

    @pytest.mark.parametrize(("solver", "config"), PAIRS)
    def test_rejected_step_shrinks(self, solver, config):
        stepper = solver(config())
        alpha = 1 / Q[solver] - 0.75 * 0.04
        assert stepper.rejected_step_size(0.1, 2.0) == pytest.approx(
            0.1 / (2.0**alpha / 0.9),
            rel=1e-14,
        )
        assert stepper.rejected_step_size(0.1, 1e9) == pytest.approx(0.02)
        assert stepper.rejected_step_size(0.1, math.inf) == pytest.approx(0.02)

    @pytest.mark.parametrize(("solver", "config"), PAIRS)
    def test_error_is_the_rms_of_the_scaled_components(self, solver, config):
        stepper = solver(config(absolute_tolerance=1e-3, relative_tolerance=1e-2))
        y = np.array([1.0, -2.0])
        y_new = np.array([3.0, -1.0])
        error = np.array([3e-3, 8e-3])
        scale = 1e-3 + 1e-2 * np.array([3.0, 2.0])
        expected = math.sqrt(((3e-3 / scale[0]) ** 2 + (8e-3 / scale[1]) ** 2) / 2)
        assert stepper.error_ratio(error, y, y_new) == pytest.approx(
            expected,
            rel=1e-14,
        )

    @pytest.mark.parametrize(("solver", "config"), PAIRS)
    def test_nan_error_is_rejected(self, solver, config):
        stepper = solver(config())
        ratio = stepper.error_ratio(
            np.array([math.nan]),
            np.array([1.0]),
            np.array([1.0]),
        )
        assert ratio == math.inf
