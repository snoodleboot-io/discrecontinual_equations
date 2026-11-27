"""The equilibrium residual ``f(u, lambda)`` as an injectable abstraction.

This is the single seam that couples continuation to the domain model. Everything
downstream depends on :class:`ResidualFunction`, not on
:class:`~discrecontinual_equations.differential_equation.DifferentialEquation`.

Equilibria of ``du/dt = f(u, lambda)`` satisfy ``f(u, lambda) = 0``: ``n``
equations in ``n + 1`` unknowns, which generically define a curve in
``(u, lambda)``-space (Kuznetsov (1998), eqs. (10.27)-(10.28)). Continuation
traces that curve; the residual is the ``F`` it traces. Two-parameter continuation
substitutes an augmented residual (the equilibrium equations plus a scalar defining
a fold or Hopf) with the same interface.

References:
- Kuznetsov, Yu. A. "Elements of Applied Bifurcation Theory", 2nd ed., Applied
  Mathematical Sciences 112, Springer, 1998, doi:10.1007/b98848, Section 10.2.1,
  eqs. (10.27)-(10.28).
"""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.differential_equation import DifferentialEquation


class ResidualFunction(ABC):
    """Evaluate the equilibrium residual of a system at ``(state, parameter)``."""

    @abstractmethod
    def evaluate(self, state: np.ndarray, parameter: float) -> np.ndarray:
        """Return ``f(state, parameter)`` as a numpy array.

        Args:
            state: The state ``u``.
            parameter: The continuation parameter ``lambda``.

        Returns:
            The residual vector.
        """
        raise NotImplementedError

    def adapt(self, state: np.ndarray, parameter: float) -> None:
        """Refresh auxiliary data at an accepted point; most residuals have none.

        Minimally augmented systems override this to move their bordering vectors
        to the null vectors at ``(state, parameter)``. The zero set does not depend
        on the borders, so the continued curve is unchanged.

        Args:
            state: The accepted state.
            parameter: The accepted parameter value.
        """
        del state, parameter


class DifferentiableResidual(ResidualFunction):
    """A residual that supplies its own ``[df/du | df/dlambda]``.

    Bordered augmented systems - the cusp and generalized-Hopf curves - know
    most of their columns in closed form and difference only the rest. Those
    columns take ``epsilon`` as a *relative* step (DEQ-159), the rule every
    other finite difference in the package follows (DEQ-118): the unknown ``x``
    a column moves is stepped by ``epsilon max(|x|, typical_x)`` - the typical
    size one unless the caller gives one per unknown (DEQ-158) - and the
    quotient is divided by the width actually stepped, which
    :func:`~discrecontinual_equations.finite_difference.central_column` does.
    On unknowns of size one or less the step is ``epsilon`` itself, as it was
    when it was absolute; above one it grows with the unknown, where an
    absolute step rounds away (an ulp of ``3e11`` is ``6e-5``) and returns a
    zero column.

    References:
        Dennis and Schnabel (1983), Sec. 5.4 and Algorithm A5.4.1: the step
        scaled by ``max(|x_j|, typx_j)``. SciPy 1.14.1,
        ``scipy/optimize/_numdiff.py``: the division by the width actually
        stepped. See ``docs/references.md``.
    """

    @abstractmethod
    def extended_jacobian(
        self,
        state: np.ndarray,
        parameter: float,
        epsilon: float,
        typical: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the ``(m, m + 1)`` extended Jacobian at ``(state, parameter)``.

        ``epsilon`` is the relative step for any column the residual
        differences: the moved unknown ``x`` steps ``epsilon max(|x|,
        typical_x)``.

        Args:
            state: The state ``u``.
            parameter: The parameter ``lambda``.
            epsilon: Finite-difference step for any column not known analytically.
            typical: The typical size of each unknown, the ``m`` state
                components then the parameter, or ``None`` for one each.

        Returns:
            The ``(m, m + 1)`` matrix ``[df/du | df/dlambda]``.
        """
        raise NotImplementedError


class EquationResidual(ResidualFunction):
    """Adapt a ``DifferentialEquation`` to a residual over one of its parameters.

    The continuation parameter is the equation parameter at
    ``parameter_index``; its value is set live before each evaluation, so the
    continued function must read it from ``self.parameters[index].value`` rather
    than caching it.

    Args:
        equation: The equation whose right-hand side is continued.
        parameter_index: Which of its parameters is ``lambda``.
        evaluation_time: Time passed to the right-hand side (ignored by autonomous
            systems; a non-autonomous one is continued at this frozen time).
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
