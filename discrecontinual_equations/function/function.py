from discrecontinual_equations.curve import Curve
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class Function:
    """Base class for differential equation functions."""

    @property
    def variables(self) -> list[Variable]:
        return self._variables

    @property
    def parameters(self) -> list[Parameter]:
        return self._parameters

    @property
    def time(self) -> Variable | None:
        return self._time

    __slots__ = ["_curve", "_parameters", "_time", "_variables"]

    def __init__(
        self,
        variables: list[Variable],
        parameters: list[Parameter],
        results: list[Variable],
        time: Variable | None = None,
    ):
        self._variables: list[Variable] = variables
        self._parameters: list[Parameter] = parameters
        self._time: Variable | None = time
        self._curve: Curve = Curve(
            variables=self._variables,
            time=self._time,
            results=results,
        )

    @property
    def curve(self) -> Curve:
        return self._curve

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        """Evaluate the function at a given point and time.

        For ODEs: returns the derivative f(x, t)
        For SDEs: returns the drift term μ(x, t)

        Args:
            point: Current state vector
            time: Current time

        Returns:
            Function evaluation result
        """
        raise NotImplementedError
