from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class SimpleODEFunction(DeterministicFunction):
    """A simple ODE function for dy/dt = -k * y"""

    def __init__(
        self,
        variables: list[Variable],
        parameters: list[Parameter],
        results: list[Variable],
        time: Variable | None = None,
    ):
        super().__init__(variables, parameters, results, time)

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        # For dy/dt = -k * y, the derivative is -k * y
        # Assuming parameters[0] is k
        k = self.parameters[0].value
        y = point[0]  # assuming single variable
        return [-k * y]
