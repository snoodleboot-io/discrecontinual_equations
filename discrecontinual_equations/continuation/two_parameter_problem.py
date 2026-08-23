"""The underlying system as a function of two active parameters.

Two-parameter continuation traces a curve of codim-1 bifurcations. The augmented
curve residuals and the codim-2 detectors need the original vector field
``f(u; p_a, p_b)``, its state Jacobian, and higher directional derivatives; this
class provides them, differentiating through an injected
:class:`~.derivative_provider.DerivativeProvider` (automatic differentiation by
default, so the curves and normal-form coefficients are exact).
"""

import numpy as np

from discrecontinual_equations.continuation.center_manifold import (
    TaylorCenterManifold,
)
from discrecontinual_equations.continuation.derivative_provider import (
    DerivativeProvider,
)
from discrecontinual_equations.continuation.focal import PlanarLyapunov
from discrecontinual_equations.differential_equation import DifferentialEquation

_PLANAR_DIMENSION = 2


class CurveDecoded:
    """The original-system quantities decoded from an augmented curve point."""

    __slots__ = ["omega", "parameter_a", "parameter_b", "state"]

    def __init__(
        self,
        state: np.ndarray,
        parameter_a: float,
        parameter_b: float,
        omega: float | None,
    ) -> None:
        self.state = state
        self.parameter_a = parameter_a
        self.parameter_b = parameter_b
        self.omega = omega


class TwoParameterProblem:
    """Evaluate the original system and its derivatives over two active parameters."""

    __slots__ = ["_equation", "_first", "_provider", "_second", "_time"]

    def __init__(
        self,
        equation: DifferentialEquation,
        first_index: int,
        second_index: int,
        time: float,
        provider: DerivativeProvider,
    ) -> None:
        self._equation = equation
        self._first = equation.derivative.parameters[first_index]
        self._second = equation.derivative.parameters[second_index]
        self._time = time
        self._provider = provider

    def _activate(self, parameter_a: float, parameter_b: float) -> None:
        self._first.value = float(parameter_a)
        self._second.value = float(parameter_b)

    def evaluate(
        self,
        state: np.ndarray,
        parameter_a: float,
        parameter_b: float,
    ) -> np.ndarray:
        """Return ``f(state; p_a, p_b)`` with both active parameters set."""
        self._activate(parameter_a, parameter_b)
        result = self._equation.derivative.eval(
            point=[float(component) for component in state],
            time=self._time,
        )
        return np.asarray(result, dtype=float)

    def directional(
        self,
        state: np.ndarray,
        direction: np.ndarray,
        parameter_a: float,
        parameter_b: float,
    ) -> np.ndarray:
        """Return the real directional derivative ``f_u(state) @ direction``."""
        self._activate(parameter_a, parameter_b)
        coefficients = self._provider.taylor(
            self._equation.derivative,
            np.asarray(state, dtype=float),
            np.asarray(direction, dtype=float),
            1,
            self._time,
        )
        return coefficients[1].real

    def taylor(
        self,
        state: np.ndarray,
        direction: np.ndarray,
        order: int,
        parameter_a: float,
        parameter_b: float,
    ) -> list[np.ndarray]:
        """Return complex directional Taylor coefficients up to ``order``."""
        self._activate(parameter_a, parameter_b)
        return self._provider.taylor(
            self._equation.derivative,
            np.asarray(state, dtype=complex),
            np.asarray(direction, dtype=complex),
            order,
            self._time,
        )

    def state_jacobian(
        self,
        state: np.ndarray,
        parameter_a: float,
        parameter_b: float,
    ) -> np.ndarray:
        """Return the state Jacobian ``f_u`` of the original system."""
        self._activate(parameter_a, parameter_b)
        return self._provider.jacobian(
            self._equation.derivative,
            np.asarray(state, dtype=float),
            self._time,
        )

    def eigenvalues(
        self,
        state: np.ndarray,
        parameter_a: float,
        parameter_b: float,
    ) -> np.ndarray:
        """Return the eigenvalues of the original state Jacobian at a point."""
        return np.linalg.eigvals(
            self.state_jacobian(state, parameter_a, parameter_b),
        )

    def lyapunov_quantities(
        self,
        state: np.ndarray,
        parameter_a: float,
        parameter_b: float,
    ) -> tuple[float, float]:
        """Return the first and second Lyapunov quantities at a Hopf point.

        Planar systems use the focal-value recursion directly; higher-dimensional
        systems are first reduced to their two-dimensional center manifold.
        """
        self._activate(parameter_a, parameter_b)
        equilibrium = np.asarray(state, dtype=float)
        jacobian = self.state_jacobian(state, parameter_a, parameter_b)
        function = self._equation.derivative
        if jacobian.shape[0] == _PLANAR_DIMENSION:
            return PlanarLyapunov(
                function,
                equilibrium,
                jacobian,
                self._time,
            ).quantities()
        return TaylorCenterManifold().lyapunov_quantities(
            function,
            equilibrium,
            jacobian,
            self._time,
        )
