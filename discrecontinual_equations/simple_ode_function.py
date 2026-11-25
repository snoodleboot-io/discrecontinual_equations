"""The linear decay ``y' = -k y``: a minimal right-hand side for examples and tests.

Its solution is ``y(t) = y(0) exp(-k t)``, so it is a quick check of a solver's
accuracy. The rate ``k`` is the first parameter and only the first component of
the state is used.
"""

from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable


class SimpleODEFunction(DeterministicFunction):
    """A simple ODE function for dy/dt = -k * y

    The rate ``k`` is ``parameters[0].value``; the field ignores ``time`` and
    every state component after the first.
    """

    def __init__(
        self,
        variables: list[Variable],
        parameters: list[Parameter],
        results: list[Variable],
        time: Variable | None = None,
    ):
        """Wrap the variables, the parameter ``k`` and the results.

        Args:
            variables: The state variable ``y``.
            parameters: ``[k]``.
            results: The result variable.
            time: The time variable, if any.
        """
        super().__init__(variables, parameters, results, time)

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        """Return ``[-k y]`` for ``y = point[0]``.

        Args:
            point: The state; only ``point[0]`` is used.
            time: Ignored.

        Returns:
            ``[-k * point[0]]``.
        """
        # For dy/dt = -k * y, the derivative is -k * y
        # Assuming parameters[0] is k
        k = self.parameters[0].value
        y = point[0]  # assuming single variable
        return [-k * y]
