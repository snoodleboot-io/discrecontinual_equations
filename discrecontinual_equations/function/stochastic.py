from discrecontinual_equations.function.function import Function


class StochasticFunction(Function):
    """Function class for stochastic differential equations."""

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        """Evaluate the drift term μ(x, t) for SDEs."""
        raise NotImplementedError

    def diffusion(self, point: list[float], time: float | None = None) -> list[float]:
        """Evaluate the diffusion term σ(x, t) for SDEs."""
        raise NotImplementedError
