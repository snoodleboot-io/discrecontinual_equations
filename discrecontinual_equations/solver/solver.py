"""The solver interface: an equation and initial values in, a curve out.

Every solver, deterministic or stochastic, is built from a configuration
(``SolverConfig``: interval, step or tolerances) and has one entry point,
``solve(equation, initial_values)``, which integrates ``y' = f(t, y)`` (or an
SDE) from the configuration's start time and returns the solution as a
:class:`~discrecontinual_equations.curve.Curve`. Each solve writes into a fresh
curve, so an equation can be solved repeatedly and by several solvers.

The deterministic solvers are explicit: Euler and the Adams-Bashforth methods
on a fixed grid, and the embedded Runge-Kutta pairs (Cash-Karp, Fehlberg,
Dormand-Prince 5(4) and 8(7)) with adaptive steps. Their sources are cited in
their own modules and collected in ``docs/references.md``.
"""

from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver_config import SolverConfig
from discrecontinual_equations.variable import Variable


class Solver:
    """
    Abstract base class for numerical differential equation solvers.

    This class provides the common interface and structure for all numerical
    methods implemented in the discrecontinual_equations library. Each solver
    implements a specific numerical algorithm for solving ordinary differential
    equations (ODEs) or stochastic differential equations (SDEs).

    Attributes:
        solution: The computed solution curve containing time series data.
                 Set to None until solve() is called.

    The solver workflow:
    1. Initialize with solver configuration
    2. Call solve() with equation and initial conditions
    3. Use the curve solve() returns, which is also kept as the solution attribute

    Each solve builds a curve of its own and leaves the equation unchanged, so an
    equation can be solved any number of times, by one solver or several (DEQ-45).
    """

    solution: Curve | None = None

    __slots__ = ["_solver_config"]

    def __init__(self, solver_config: SolverConfig):
        """
        Initialize the solver with configuration.

        Args:
            solver_config: Configuration object containing solver parameters
                          (time range, tolerances, step sizes, etc.)
        """
        self._solver_config = solver_config

    @property
    def solver_config(self) -> SolverConfig:
        """Get the solver configuration."""
        return self._solver_config

    def solve(self, equation: DifferentialEquation, initial_values: list[float]):
        """
        Solve the differential equation with given initial conditions.

        This method must be implemented by all concrete solver subclasses.
        The solution is returned and also stored in the 'solution' attribute:
        a curve whose ``time`` holds the nodes and whose ``results`` hold the
        state at each node.

        Args:
            equation: The differential equation to solve, containing
                     variables, parameters, and derivative function.
            initial_values: Initial values for all variables at start time.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError

    @staticmethod
    def _new_solution(equation: DifferentialEquation) -> Curve:
        """An empty curve for one solve, on variables of its own.

        The time and state variables are copies of the equation's, named alike
        but with an empty discretization, so appending a run never writes into
        the equation: a solver that appended to the equation's own ``Variable``
        objects handed back every run so far, concatenated, from the second
        solve on (DEQ-45). This is what the ensemble solver has always done with
        its ``EnsembleResult``. The results are ``Integral of <name>``, as before.
        """
        variables = equation.derivative.variables
        time = equation.derivative.time
        return Curve(
            time=None if time is None else _empty_copy(time),
            variables=[_empty_copy(variable) for variable in variables],
            results=[
                Variable(name=f"Integral of {variable.name}") for variable in variables
            ],
        )


def _empty_copy(variable: Variable) -> Variable:
    """``variable`` with its type, name and abbreviation, and nothing recorded."""
    return variable.model_copy(
        update={"discretization": [], "range": list(variable.range)},
    )
