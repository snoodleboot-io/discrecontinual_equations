"""The base class for the right-hand side of an ODE, ``y' = f(t, y)``.

Subclass ``DeterministicFunction`` and implement ``eval(point, time)`` to return
``f(t, y)`` as a list, one entry per state variable; the deterministic solvers
call nothing else.
"""

from discrecontinual_equations.function.function import Function


class DeterministicFunction(Function):
    """Function class for ordinary differential equations (deterministic)."""

    def eval(self, point: list[float], time: float | None = None) -> list[float]:
        """Evaluate the derivative for ODEs.

        Args:
            point: The state ``y``.
            time: The time ``t``.

        Returns:
            ``f(t, y)``, one entry per state component.

        Raises:
            NotImplementedError: Always; subclasses implement it.
        """
        raise NotImplementedError
