"""Solving an equation again gives the same answer, not both answers (DEQ-45).

Every per-path solver used to build its ``Curve`` on the equation's own time and
state ``Variable`` objects and append to them, so a second solve of the same
equation appended a second run to the first: lengths 1001, 2002, 3003 for three
solves, and every later reader of ``solution.time`` - a parameter sweep, a loop over
solvers, a test that reuses a fixture - got the runs concatenated. The ensemble
solver never had the defect, because it builds an ``EnsembleResult`` of its own
and leaves the equation alone; the per-path solvers now do the same with their
``Curve``, and return it.

Each solver family is solved three times on one equation. The stochastic ones are
rebuilt from the same seed each time, so the three runs must agree exactly; the
equation's own variables must still be empty afterwards.
"""

from collections.abc import Callable

import pytest

from discrecontinual_equations.curve import Curve
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
from discrecontinual_equations.solver.deterministic.adams_bashforth3.adams_bashforth3_config import (  # noqa: E501
    AdamsBashforth3Config,
)
from discrecontinual_equations.solver.deterministic.adams_bashforth3.adams_bashforth3_solver import (  # noqa: E501
    AdamsBashforth3Solver,
)
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
from discrecontinual_equations.solver.deterministic.euler.euler_config import (
    EulerConfig,
)
from discrecontinual_equations.solver.deterministic.euler.euler_solver import (
    EulerSolver,
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
from discrecontinual_equations.solver.solver import Solver
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
from discrecontinual_equations.solver.stochastic.srk5.srk5_config import SRK5Config
from discrecontinual_equations.solver.stochastic.srk5.srk5_solver import SRK5Solver
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_config import (  # noqa: E501
    ZhanDuanLiLiConfig,
)
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li.zhan_duan_li_li_solver import (  # noqa: E501
    ZhanDuanLiLiSolver,
)
from discrecontinual_equations.variable import Variable

_SOLVES = 3
_SEED = 7
_INITIAL = [1.0]


class _MultiplicativeNoise(StochasticFunction):
    """``dX = -X dt + 0.3 X dW``: scalar, so every stochastic solver accepts it."""

    def __init__(self, variables, parameters, time=None):
        super().__init__(
            variables,
            parameters,
            [Variable(name="dX/dt", discretization=[])],
            time,
        )

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
        return [self.parameters[0].value * point[0]]


def _ordinary() -> DifferentialEquation:
    time = Variable(name="t", discretization=[])
    variables = [Variable(name="y", discretization=[])]
    parameters = [Parameter(name="k", value=1.0)]
    results = [Variable(name="dy/dt", discretization=[])]
    function = SimpleODEFunction(variables, parameters, results, time)
    return DifferentialEquation(
        variables=variables,
        time=time,
        parameters=parameters,
        derivative=function,
    )


def _stochastic() -> DifferentialEquation:
    time = Variable(name="t", discretization=[])
    variables = [Variable(name="X", discretization=[])]
    parameters = [Parameter(name="sigma", value=0.3)]
    return DifferentialEquation(
        variables=variables,
        time=time,
        parameters=parameters,
        derivative=_MultiplicativeNoise(variables, parameters, time),
    )


_FIXED = {"start_time": 0.0, "end_time": 1.0, "step_size": 0.1}
_ADAPTIVE = {"start_time": 0.0, "end_time": 1.0, "initial_step_size": 0.1}
_SEEDED = {**_FIXED, "random_seed": _SEED}

# (name, equation factory, solver factory): every per-path solver family
_Family = tuple[str, Callable[[], DifferentialEquation], Callable[[], Solver]]
_FAMILIES: list[_Family] = [
    ("euler", _ordinary, lambda: EulerSolver(EulerConfig(**_FIXED))),
    (
        "adams_bashforth2",
        _ordinary,
        lambda: AdamsBashforth2Solver(AdamsBashforth2Config(**_FIXED)),
    ),
    (
        "adams_bashforth3",
        _ordinary,
        lambda: AdamsBashforth3Solver(AdamsBashforth3Config(**_FIXED)),
    ),
    ("cash_karp", _ordinary, lambda: CashKarpSolver(CashKarpConfig(**_ADAPTIVE))),
    ("fehlberg", _ordinary, lambda: FehlbergSolver(FehlbergConfig(**_ADAPTIVE))),
    (
        "dormand_prince",
        _ordinary,
        lambda: DormandPrinceSolver(DormandPrinceConfig(**_ADAPTIVE)),
    ),
    (
        "prince_dormand87",
        _ordinary,
        lambda: PrinceDormand87Solver(PrinceDormand87Config(**_ADAPTIVE)),
    ),
    (
        "euler_maruyama",
        _stochastic,
        lambda: EulerMaruyamaSolver(EulerMaruyamaConfig(**_SEEDED)),
    ),
    ("milstein", _stochastic, lambda: MilsteinSolver(MilsteinConfig(**_SEEDED))),
    (
        "gottwald_melbourne",
        _stochastic,
        lambda: GottwaldMelbourneSolver(GottwaldMelbourneConfig(**_SEEDED)),
    ),
    (
        "zhan_duan_li_li",
        _stochastic,
        lambda: ZhanDuanLiLiSolver(ZhanDuanLiLiConfig(**_SEEDED)),
    ),
    ("srk2", _stochastic, lambda: SRK2Solver(SRK2Config(**_SEEDED))),
    ("srk3", _stochastic, lambda: SRK3Solver(SRK3Config(**_SEEDED))),
    ("srk4", _stochastic, lambda: SRK4Solver(SRK4Config(**_SEEDED))),
]


def _snapshot(curve: Curve) -> tuple[list, list, list]:
    return (
        list(curve.time.discretization),
        [list(variable.discretization) for variable in curve.variables],
        [list(result.discretization) for result in curve.results],
    )


@pytest.mark.parametrize(
    ("make_equation", "make_solver"),
    [(make_equation, make_solver) for _, make_equation, make_solver in _FAMILIES],
    ids=[name for name, _, _ in _FAMILIES],
)
class TestSolveRepeatedly:
    def test_three_solves_of_one_equation_give_three_identical_single_runs(
        self,
        make_equation,
        make_solver,
    ):
        # Arrange: the reference is one run on an equation nobody solved before.
        reference_solver = make_solver()
        reference_solver.solve(make_equation(), _INITIAL)
        reference = _snapshot(reference_solver.solution)
        equation = make_equation()

        # Act: three solves of one equation, each by a solver from the same seed.
        runs = []
        for _ in range(_SOLVES):
            solver = make_solver()
            solver.solve(equation, _INITIAL)
            runs.append(_snapshot(solver.solution))

        # Assert: each run is the single run, point for point.
        assert len(reference[0]) > 1
        for run in runs:
            assert run == reference

    def test_a_solve_leaves_the_equation_unchanged(self, make_equation, make_solver):
        # Arrange
        equation = make_equation()

        # Act
        make_solver().solve(equation, _INITIAL)

        # Assert: the equation's own variables (and the function's) hold nothing.
        assert equation.time.discretization == []
        assert all(variable.discretization == [] for variable in equation.variables)
        assert equation.derivative.time.discretization == []
        assert all(
            variable.discretization == [] for variable in equation.derivative.variables
        )

    def test_one_solver_solving_again_returns_one_run(self, make_equation, make_solver):
        # Arrange: the same solver instance, reused; its stream moves on, so only
        # the shape of the result is pinned here, not its values.
        equation = make_equation()
        solver = make_solver()
        first = solver.solve(equation, _INITIAL)

        # Act
        second = solver.solve(equation, _INITIAL)

        # Assert: solve returns the solution it built, a new one each time.
        assert first is not second
        assert second is solver.solution
        assert len(second) == len(first)
        assert len(first.time.discretization) == len(first.results[0].discretization)

    def test_a_second_solve_leaves_the_first_curve_as_it_was(
        self,
        make_equation,
        make_solver,
    ):
        # Arrange: a curve kept by the caller, then the same solver run again on
        # the same equation - the curve must not share storage with the next run.
        equation = make_equation()
        solver = make_solver()
        first = solver.solve(equation, _INITIAL)
        before = _snapshot(first)

        # Act
        solver.solve(equation, _INITIAL)
        make_solver().solve(equation, _INITIAL)

        # Assert
        assert _snapshot(first) == before

    def test_the_solution_keeps_the_equations_names(self, make_equation, make_solver):
        # Arrange
        equation = make_equation()

        # Act
        curve = make_solver().solve(equation, _INITIAL)

        # Assert: fresh variables, but named as the equation names them.
        assert curve.time is not equation.derivative.time
        assert curve.time.name == equation.derivative.time.name
        assert [variable.name for variable in curve.variables] == [
            variable.name for variable in equation.derivative.variables
        ]
        assert all(
            mine is not theirs
            for mine, theirs in zip(
                curve.variables,
                equation.derivative.variables,
                strict=True,
            )
        )


def test_srk5_refuses_without_touching_the_equation():
    # Arrange
    equation = _stochastic()

    # Act / Assert
    with pytest.raises(NotImplementedError):
        SRK5Solver(SRK5Config(**_SEEDED)).solve(equation, _INITIAL)
    assert equation.time.discretization == []
