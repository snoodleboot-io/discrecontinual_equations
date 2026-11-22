"""A fixed-step solve reads its time grid once, so a path costs O(K) (DEQ-47).

``FixedStepConfig.times`` builds its ``np.arange`` afresh on every read. The
stochastic solvers read it twice per step, so a path of K steps cost O(K^2):
64k Euler-Maruyama steps took 4.0 s, growing as K^1.85. Two things are pinned
here. Structurally, every fixed-step solver reads the grid a bounded number of
times however many steps it takes, which a config that counts its reads shows
without a clock. And by the clock, Euler-Maruyama's time grows linearly in K:
a local slope of at most 1.1 between K = 16k and 128k. The grid each solve
records is the config's grid, value for value.
"""

import time
from typing import ClassVar

import numpy as np
import pytest

from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.simple_ode_function import SimpleODEFunction
from discrecontinual_equations.solver.deterministic.adams_bashforth2.adams_bashforth2_config import (  # noqa: E501
    AdamsBashforth2Config,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth2.adams_bashforth2_solver import (  # noqa: E501
    AdamsBashforth2Solver,
)
from discrecontinual_equations.solver.deterministic.euler.euler_config import (
    EulerConfig,
)
from discrecontinual_equations.solver.deterministic.euler.euler_solver import (
    EulerSolver,
)
from discrecontinual_equations.solver.stochastic.ensemble import StochasticEnsemble
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_config import (  # noqa: E501
    EulerMaruyamaConfig,
)
from discrecontinual_equations.solver.stochastic.euler_maruyama.euler_maruyama_solver import (  # noqa: E501
    EulerMaruyamaSolver,
)
from discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_config import (  # noqa: E501
    GottwaldMelbourneConfig,
)
from discrecontinual_equations.solver.stochastic.gottwald_melbourne.gottwald_melbourne_solver import (  # noqa: E501
    GottwaldMelbourneSolver,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_config import (
    MilsteinConfig,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_solver import (
    MilsteinSolver,
)
from discrecontinual_equations.solver.stochastic.srk2.srk2_config import SRK2Config
from discrecontinual_equations.solver.stochastic.srk2.srk2_solver import SRK2Solver
from discrecontinual_equations.solver.stochastic.srk3.srk3_config import SRK3Config
from discrecontinual_equations.solver.stochastic.srk3.srk3_solver import SRK3Solver
from discrecontinual_equations.solver.stochastic.srk4.srk4_config import SRK4Config
from discrecontinual_equations.solver.stochastic.srk4.srk4_solver import SRK4Solver
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_config import (  # noqa: E501
    ZhanDuanLiLiConfig,
)
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_solver import (  # noqa: E501
    ZhanDuanLiLiSolver,
)
from discrecontinual_equations.variable import Variable

# A solve may read the grid a few times (``n_steps`` reads it too); what it may
# not do is read it once per step.
_MOST_READS = 3
_STEPS = 200
_SMALL, _LARGE = 16_000, 128_000
_REPEATS = 5
_MOST_SLOPE = 1.1


class _Linear(StochasticFunction):
    """``dX = -X dt + 0.3 X dW``, scalar, so every solver takes it."""

    def __init__(self, variables, parameters, time=None):
        super().__init__(variables, parameters, [Variable(name="dX/dt")], time)

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[float]:
        return [-point[0]]

    def diffusion(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002
    ) -> list[float]:
        return [0.3 * point[0]]

    def eval_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        return -points

    def diffusion_batch(
        self,
        points: np.ndarray,
        time: float | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        return 0.3 * points


def _stochastic() -> DifferentialEquation:
    time_variable = Variable(name="t")
    variables = [Variable(name="X")]
    parameters = [Parameter(name="sigma", value=0.3)]
    return DifferentialEquation(
        variables=variables,
        time=time_variable,
        parameters=parameters,
        derivative=_Linear(variables, parameters, time_variable),
    )


def _ordinary() -> DifferentialEquation:
    time_variable = Variable(name="t")
    variables = [Variable(name="y")]
    parameters = [Parameter(name="k", value=1.0)]
    function = SimpleODEFunction(
        variables,
        parameters,
        [Variable(name="dy/dt")],
        time_variable,
    )
    return DifferentialEquation(
        variables=variables,
        time=time_variable,
        parameters=parameters,
        derivative=function,
    )


def _counting(config_class: type) -> type:
    """``config_class`` with a ``times`` that counts how often it is read."""

    class Counting(config_class):
        reads: ClassVar[int] = 0

        @property
        def times(self) -> np.ndarray:
            type(self).reads += 1
            return super().times

    return Counting


_GRID = {"start_time": 0.0, "end_time": 1.0, "step_size": 1.0 / _STEPS}
_SEED = {"random_seed": 1}
_FIXED_STEP = [
    ("euler", EulerSolver, EulerConfig, _ordinary, {}),
    ("adams_bashforth2", AdamsBashforth2Solver, AdamsBashforth2Config, _ordinary, {}),
    ("euler_maruyama", EulerMaruyamaSolver, EulerMaruyamaConfig, _stochastic, _SEED),
    ("milstein", MilsteinSolver, MilsteinConfig, _stochastic, _SEED),
    (
        "gottwald_melbourne",
        GottwaldMelbourneSolver,
        GottwaldMelbourneConfig,
        _stochastic,
        {"epsilon": 1.0, **_SEED},
    ),
    ("zhan_duan_li_li", ZhanDuanLiLiSolver, ZhanDuanLiLiConfig, _stochastic, _SEED),
    ("srk2", SRK2Solver, SRK2Config, _stochastic, _SEED),
    ("srk3", SRK3Solver, SRK3Config, _stochastic, _SEED),
    ("srk4", SRK4Solver, SRK4Config, _stochastic, _SEED),
]


@pytest.mark.parametrize(
    ("solver_class", "config_class", "make_equation", "extra"),
    [entry[1:] for entry in _FIXED_STEP],
    ids=[entry[0] for entry in _FIXED_STEP],
)
def test_a_solve_reads_the_grid_a_bounded_number_of_times(
    solver_class,
    config_class,
    make_equation,
    extra,
):
    # Arrange
    counting = _counting(config_class)
    config = counting(**_GRID, **extra)

    # Act
    curve = solver_class(config).solve(make_equation(), [1.0])

    # Assert: a few reads for 200 steps, and the recorded grid is the config's.
    assert counting.reads <= _MOST_READS
    assert curve.time.discretization == config_class(**_GRID, **extra).times.tolist()


def test_the_ensemble_reads_the_grid_a_bounded_number_of_times():
    # Arrange
    counting = _counting(EulerMaruyamaConfig)

    # Act
    StochasticEnsemble(counting(**_GRID, random_seed=1)).solve(
        _stochastic(),
        [1.0],
        paths=4,
    )

    # Assert
    assert counting.reads <= _MOST_READS


def _cpu_seconds(steps: int) -> float:
    """CPU time of one seeded solve: less disturbed than wall time by other work."""
    config = EulerMaruyamaConfig(
        start_time=0.0,
        end_time=1.0,
        step_size=1.0 / steps,
        random_seed=1,
    )
    solver = EulerMaruyamaSolver(config)
    equation = _stochastic()
    start = time.process_time()
    solver.solve(equation, [1.0])
    return time.process_time() - start


def test_euler_maruyama_time_grows_linearly_in_the_steps():
    # Arrange / Act: the best of five of each size, the sizes interleaved so a
    # spell of contention from other work on the machine hits both alike.
    small, large = float("inf"), float("inf")
    for _ in range(_REPEATS):
        small = min(small, _cpu_seconds(_SMALL))
        large = min(large, _cpu_seconds(_LARGE))

    # Assert: the local slope between 16k and 128k steps (it was 1.85 before).
    slope = np.log(large / small) / np.log(_LARGE / _SMALL)
    assert slope <= _MOST_SLOPE, f"slope {slope:.2f}: {small:.3f} s -> {large:.3f} s"
