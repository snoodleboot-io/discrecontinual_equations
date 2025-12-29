from discrecontinual_equations.curve import Curve
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.solver.solver_config import SolverConfig


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
    3. Access results through the solution attribute
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
        The solution will be stored in the 'solution' attribute.

        Args:
            equation: The differential equation to solve, containing
                     variables, parameters, and derivative function.
            initial_values: Initial values for all variables at start time.

        Raises:
            NotImplementedError: This is an abstract method.
        """
        raise NotImplementedError
