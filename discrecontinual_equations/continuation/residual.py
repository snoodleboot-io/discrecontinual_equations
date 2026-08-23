"""The equilibrium residual ``f(u, lambda)`` as an injectable abstraction.

This is the single seam that couples continuation to the domain model. Everything
downstream depends on :class:`ResidualFunction`, not on
:class:`~discrecontinual_equations.differential_equation.DifferentialEquation`.
"""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation


class ResidualFunction(ABC):
    """Evaluate the equilibrium residual of a system at ``(state, parameter)``."""

    @abstractmethod
    def evaluate(self, state: np.ndarray, parameter: float) -> np.ndarray:
        """Return ``f(state, parameter)`` as a numpy array."""
        raise NotImplementedError


class EquationResidual(ResidualFunction):
    """Adapt a ``DifferentialEquation`` to a residual over one of its parameters.

    The continuation parameter is the equation parameter at
    ``parameter_index``; its value is set live before each evaluation, so the
    continued function must read it from ``self.parameters[index].value`` rather
    than caching it.
    """

    __slots__ = ["_equation", "_parameter", "_time"]

    def __init__(
        self,
        equation: DifferentialEquation,
        parameter_index: int,
        evaluation_time: float,
    ) -> None:
        self._equation = equation
        self._parameter = equation.derivative.parameters[parameter_index]
        self._time = evaluation_time

    def evaluate(self, state: np.ndarray, parameter: float) -> np.ndarray:
        self._parameter.value = float(parameter)
        evaluated = self._equation.derivative.eval(
            point=[float(component) for component in state],
            time=self._time,
        )
        return np.asarray(evaluated, dtype=float)
