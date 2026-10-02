"""Each stochastic solver draws from its own stream, not the process-wide one.

These are the properties the global ``np.random`` could not provide, and losing them
is a correctness problem rather than an inconvenience. A solver that seeded the global
stream in its constructor and drew from it later had two failure modes. Its path
depended on how many draws anything else in the process took between construction and
solving, so the same configuration gave two different answers in one run. And
constructing a solver reseeded the stream for everything else, so a solver could change
an unrelated result just by existing. Both are pinned below against the behaviour they
replace; neither shows up as a failure anywhere else, because a wrong answer here is a
perfectly plausible sample path.
"""

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (  # noqa: E501
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_solver import (  # noqa: E501
    EulerMaruyamaSolver,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_config import (
    MilsteinConfig,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_solver import (
    MilsteinSolver,
)
from discrecontinual_equations.variable import Variable

_SEED = 42
_OTHER_DRAWS = 17


class OrnsteinUhlenbeck(StochasticFunction):
    """dX = -X dt + sigma dW, whose path is pure noise history."""

    def __init__(self, variables, parameters, time=None):
        super().__init__(
            variables,
            parameters,
            [Variable(name="dX/dt", discretization=[])],
            time,
        )

    def eval(self, point: list[float], time: float | None = None) -> list[float]:  # noqa: ARG002
        return [-point[0]]

    def diffusion(self, point: list[float], time: float | None = None) -> list[float]:  # noqa: ARG002
        return [self.parameters[0].value]


def _equation():
    time = Variable(name="t", discretization=[])
    variables = [Variable(name="X", discretization=[])]
    parameters = [Parameter(name="sigma", value=0.5)]
    return DifferentialEquation(
        variables=variables,
        time=time,
        parameters=parameters,
        derivative=OrnsteinUhlenbeck(variables, parameters, time),
    )


def _config(config_type=EulerMaruyamaConfig):
    return config_type(
        start_time=0.0,
        end_time=1.0,
        step_size=0.05,
        random_seed=_SEED,
    )


def _path(solver_type, config_type):
    solver = solver_type(_config(config_type))
    solver.solve(_equation(), [1.0])
    return list(solver.solution.results[0].discretization)


class TestStochasticSolverStreams:
    def test_the_same_seed_gives_the_same_path(self):
        for solver_type, config_type in (
            (EulerMaruyamaSolver, EulerMaruyamaConfig),
            (MilsteinSolver, MilsteinConfig),
        ):
            first = _path(solver_type, config_type)
            second = _path(solver_type, config_type)
            assert first == second

    def test_draws_taken_between_construction_and_solving_change_nothing(self):
        # The discriminating case. On a shared global stream the first solver would
        # draw from wherever those intervening draws had left the state, while the
        # second drew from the freshly reseeded start, and the two paths parted company
        # with nothing to say which one the configuration had asked for.
        first = EulerMaruyamaSolver(_config())
        np.random.normal(size=_OTHER_DRAWS)  # noqa: NPY002 (the stream under test)
        first.solve(_equation(), [1.0])
        expected = list(first.solution.results[0].discretization)
        second = EulerMaruyamaSolver(_config())
        second.solve(_equation(), [1.0])
        assert list(second.solution.results[0].discretization) == expected

    def test_constructing_a_solver_leaves_the_global_stream_alone(self):
        np.random.seed(1)  # noqa: NPY002 (the stream under test)
        expected = np.random.normal(size=5)  # noqa: NPY002 (the stream under test)
        np.random.seed(1)  # noqa: NPY002 (the stream under test)
        EulerMaruyamaSolver(_config())
        assert np.array_equal(
            expected,
            np.random.normal(size=5),  # noqa: NPY002 (the stream under test)
        )

    def test_two_solvers_do_not_share_one_stream(self):
        # Constructing both up front, then solving, must give what solving each in turn
        # gives. A shared stream makes the order of construction visible in the answer.
        first = EulerMaruyamaSolver(_config())
        second = EulerMaruyamaSolver(_config())
        first.solve(_equation(), [1.0])
        second.solve(_equation(), [1.0])
        first_path = list(first.solution.results[0].discretization)
        second_path = list(second.solution.results[0].discretization)
        assert first_path == second_path
