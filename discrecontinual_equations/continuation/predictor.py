"""Predictor step of the predictor-corrector scheme.

Continuation traces the curve of solutions of ``F(y) = 0``, ``F: R^(n+1) -> R^n``,
with ``y = (u, lambda)`` the state and the parameter. Every step starts from an
accepted point ``y_j`` and its unit tangent ``v_j`` and *guesses* the next point;
the corrector then pulls the guess back onto the curve. The guess here is the
Euler (tangent) predictor

    y~_(j+1) = y_j + h_j v_j,

a step ``h_j`` of arclength along the tangent (Allgower and Georg (1990), eq.
(2.2.2); Kuznetsov (1998), eq. (10.29)). It is first-order accurate: the distance
from the guess to the curve grows like ``h^2`` times the curvature, which the step
controller keeps small by watching how hard the corrector has to work.

References:
- Allgower, E. L. and Georg, K. "Numerical Continuation Methods: An Introduction",
  Springer Series in Computational Mathematics 13, Springer, 1990,
  doi:10.1007/978-3-642-61257-2, Section 2.2, eq. (2.2.2).
- Kuznetsov, Yu. A. "Elements of Applied Bifurcation Theory", 2nd ed., Applied
  Mathematical Sciences 112, Springer, 1998, doi:10.1007/b98848, Section 10.2.1,
  eq. (10.29).
"""

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
        """Return a predicted ``(state, parameter)`` a distance ``step`` ahead.

        Args:
            state: The last accepted point, with its unit tangent.
            step: The arclength step ``h`` (positive; the tangent sets the
                direction).

        Returns:
            The predicted state vector and parameter value.
        """
        raise NotImplementedError


class TangentPredictor(Predictor):
    """Linear (Euler) predictor along the branch tangent.

    Returns ``(u, lambda) + step * v`` for the stored unit tangent ``v``. A secant
    predictor (Kuznetsov (1998), eq. (10.31)) is not provided.

    References:
    - Allgower and Georg (1990), eq. (2.2.2).
    - Kuznetsov (1998), Section 10.2.1, eq. (10.29).
    """

    def predict(
        self,
        state: ContinuationState,
        step: float,
    ) -> tuple[np.ndarray, float]:
        tangent = state.tangent
        predicted_state = state.state + step * tangent[:-1]
        predicted_parameter = state.parameter + step * float(tangent[-1])
        return predicted_state, predicted_parameter
