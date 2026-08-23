"""Predictor step of the predictor-corrector scheme."""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.continuation.continuation_state import ContinuationState


class Predictor(ABC):
    """Produce a first guess for the next point on the branch."""

    @abstractmethod
    def predict(
        self,
        state: ContinuationState,
        step: float,
    ) -> tuple[np.ndarray, float]:
        """Return a predicted ``(state, parameter)`` a distance ``step`` ahead."""
        raise NotImplementedError


class TangentPredictor(Predictor):
    """Linear predictor along the branch tangent."""

    def predict(
        self,
        state: ContinuationState,
        step: float,
    ) -> tuple[np.ndarray, float]:
        tangent = state.tangent
        predicted_state = state.state + step * tangent[:-1]
        predicted_parameter = state.parameter + step * float(tangent[-1])
        return predicted_state, predicted_parameter
