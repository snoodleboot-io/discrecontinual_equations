from discrecontinual_equations.function.function import Function


class DeterministicFunction(Function):
    """Function class for ordinary differential equations (deterministic)."""

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        """Evaluate the derivative for ODEs."""
        raise NotImplementedError
