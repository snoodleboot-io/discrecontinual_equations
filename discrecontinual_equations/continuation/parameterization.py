"""The continuation constraint (parameterization) as an injectable strategy.

The corrector solves ``f(u, lambda) = 0`` together with one extra scalar equation
that pins down where along the branch the corrected point sits. *Which* extra
equation that is - natural parameter, pseudo-arclength, Moore-Penrose - is a
family of interchangeable solutions, so it is injected rather than baked into the
corrector.
"""

from abc import ABC, abstractmethod

import numpy as np

_PARAMETER_BORDER = 1.0


class CorrectionStep:
    """The data a parameterization needs at one corrector iteration."""

    __slots__ = [
        "_parameter",
        "_predicted_parameter",
        "_predicted_state",
        "_state",
        "_tangent",
    ]

    def __init__(
        self,
        state: np.ndarray,
        parameter: float,
        predicted_state: np.ndarray,
        predicted_parameter: float,
        tangent: np.ndarray,
    ) -> None:
        self._state = state
        self._parameter = parameter
        self._predicted_state = predicted_state
        self._predicted_parameter = predicted_parameter
        self._tangent = tangent

    @property
    def state(self) -> np.ndarray:
        """Current corrector iterate for the state."""
        return self._state

    @property
    def parameter(self) -> float:
        """Current corrector iterate for the parameter."""
        return self._parameter

    @property
    def predicted_state(self) -> np.ndarray:
        """Predictor's state, the reference for the constraint."""
        return self._predicted_state

    @property
    def predicted_parameter(self) -> float:
        """Predictor's parameter, the reference for the constraint."""
        return self._predicted_parameter

    @property
    def tangent(self) -> np.ndarray:
        """Branch tangent in augmented space."""
        return self._tangent


class Parameterization(ABC):
    """The extra scalar equation that closes the continuation system."""

    @abstractmethod
    def constraint(self, step: CorrectionStep) -> float:
        """Return the scalar constraint value ``g`` for the current iterate."""
        raise NotImplementedError

    @abstractmethod
    def border(self, step: CorrectionStep) -> np.ndarray:
        """Return the row appended to the extended Jacobian (``dg/d(u, lambda)``)."""
        raise NotImplementedError


class PseudoArclength(Parameterization):
    """Keller's constraint: stay on the hyperplane normal to the tangent.

    Well posed through folds, where the parameter is not a valid coordinate.
    """

    def constraint(self, step: CorrectionStep) -> float:
        tangent = step.tangent
        state_part = tangent[:-1] @ (step.state - step.predicted_state)
        parameter_part = tangent[-1] * (step.parameter - step.predicted_parameter)
        return float(state_part + parameter_part)

    def border(self, step: CorrectionStep) -> np.ndarray:
        return step.tangent


class Natural(Parameterization):
    """Natural-parameter continuation: fix ``lambda`` and solve for the state.

    Simpler and cheaper than pseudo-arclength, but singular at folds - a genuine
    alternative with a known trade-off, which is exactly why it is a separate,
    swappable strategy.
    """

    def constraint(self, step: CorrectionStep) -> float:
        return float(step.parameter - step.predicted_parameter)

    def border(self, step: CorrectionStep) -> np.ndarray:
        row = np.zeros(step.tangent.size)
        row[-1] = _PARAMETER_BORDER
        return row
