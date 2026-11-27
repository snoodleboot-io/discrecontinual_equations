"""The continuation constraint (parameterization) as an injectable strategy.

The corrector solves ``f(u, lambda) = 0`` together with one extra scalar equation
that pins down where along the branch the corrected point sits. *Which* extra
equation that is - natural parameter, pseudo-arclength, Moore-Penrose - is a
family of interchangeable solutions, so it is injected rather than baked into the
corrector.

This is the *branch's* parameterization: how a point of a solution curve of
``f(u, lambda) = 0`` is named. It is unrelated to the parameterization method of
:mod:`~.manifold`, which names the points of an invariant manifold.

``f(u, lambda) = 0`` is ``n`` equations in ``n + 1`` unknowns ``x = (u, lambda)``,
and near a regular point its solutions form a curve. Newton's method needs a
square system, so the corrector appends one scalar equation ``g(x) = 0`` and the
row ``dg/dx`` to the ``n x (n + 1)`` extended Jacobian ``[f_u | f_lambda]``:

* :class:`Natural` fixes ``lambda`` at its predicted value. The bordered matrix
  is then non-singular exactly where ``f_u`` is, so it fails at a fold, where
  the branch turns back in ``lambda``.
* :class:`PseudoArclength` is Keller's constraint: the corrected point lies on
  the hyperplane through the predicted point ``x_p`` normal to the unit tangent
  ``t``, ``t . (x - x_p) = 0``. With ``x_p = x_0 + ds t`` and ``|t| = 1`` this
  is the printed form ``(x - x_0) . t - ds = 0``. Since the extended Jacobian's
  null space at a regular point is spanned by ``t``, the bordered matrix
  ``[f_x; t]`` is non-singular there - folds included.

Neither handles a branch point, where the extended Jacobian loses rank and the
tangent itself is not unique; :mod:`~.switching` deals with those.

References:
- Doedel, E. J. "Numerical Analysis of Nonlinear Equations", lecture notes,
  Kyoto, July 2006 (https://www.math.kyoto-u.ac.jp/~kokubu/RIMS2006/doedel.pdf),
  slides 18-27: regular solutions and folds (slides 18-20), "Keller's
  Pseudo-Arclength Continuation" and its Newton system (slides 22-25), and the
  fact that the pseudo-arclength Jacobian is non-singular at a regular solution
  point (slide 27).
"""

from abc import ABC, abstractmethod

import numpy as np

_PARAMETER_BORDER = 1.0


class CorrectionStep:
    """The data a parameterization needs at one corrector iteration.

    Args:
        state: The corrector's current state iterate ``u``.
        parameter: The corrector's current parameter iterate ``lambda``.
        predicted_state: The predictor's state, the constraint's reference.
        predicted_parameter: The predictor's parameter, the constraint's
            reference.
        tangent: The unit branch tangent in augmented space, ``n + 1``
            entries with the parameter's last.
    """

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
    """The extra scalar equation that closes the continuation system.

    The corrector appends :meth:`constraint` to the residual ``f(u, lambda)`` and
    :meth:`border` to the extended Jacobian, and Newton's method solves the
    resulting square system.

    References:
    - Doedel (2006), Kyoto lecture notes, slides 22-25 (see the module docstring).
    """

    @abstractmethod
    def constraint(self, step: CorrectionStep) -> float:
        """Return the scalar constraint value ``g`` for the current iterate.

        Args:
            step: The current iterate, the prediction and the tangent.

        Returns:
            ``g(u, lambda)``, zero on the constraint.
        """
        raise NotImplementedError

    @abstractmethod
    def border(self, step: CorrectionStep) -> np.ndarray:
        """Return the row appended to the extended Jacobian (``dg/d(u, lambda)``).

        Args:
            step: The current iterate, the prediction and the tangent.

        Returns:
            The ``n + 1`` derivatives of ``g``, the parameter's last.
        """
        raise NotImplementedError


class PseudoArclength(Parameterization):
    """Keller's constraint: stay on the hyperplane normal to the tangent.

    Well posed through folds, where the parameter is not a valid coordinate.
    ``g = t . (x - x_p)`` with ``x = (u, lambda)``, ``x_p`` the prediction and
    ``t`` the unit tangent: Doedel's ``(x_1 - x_0)* x_0' - ds = 0`` with the
    predictor's step ``x_p = x_0 + ds t`` folded in. The bordered matrix
    ``[f_x; t]`` is non-singular at every regular point of the branch.

    References:
    - Doedel (2006), Kyoto lecture notes, slides 22 and 25 (the constraint)
      and slide 27 (non-singularity at a regular point); see the module
      docstring.
    """

    def constraint(self, step: CorrectionStep) -> float:
        """Return ``t . (x - x_p)``, the iterate's offset along the tangent.

        Args:
            step: The current iterate, the prediction and the tangent.

        Returns:
            The constraint value; zero on the hyperplane normal to the tangent.
        """
        tangent = step.tangent
        state_part = tangent[:-1] @ (step.state - step.predicted_state)
        parameter_part = tangent[-1] * (step.parameter - step.predicted_parameter)
        return float(state_part + parameter_part)

    def border(self, step: CorrectionStep) -> np.ndarray:
        """Return the tangent itself, the gradient of ``t . (x - x_p)``.

        Args:
            step: The current iterate, the prediction and the tangent.

        Returns:
            The unit tangent ``t``.
        """
        return step.tangent


class Natural(Parameterization):
    """Natural-parameter continuation: fix ``lambda`` and solve for the state.

    Simpler and cheaper than pseudo-arclength, but singular at folds - a genuine
    alternative with a known trade-off, which is exactly why it is a separate,
    swappable strategy. ``g = lambda - lambda_p``, so the bordered matrix is
    ``[[f_u, f_lambda], [0, 1]]``, whose determinant is ``det f_u``: it is
    singular where the branch turns back in ``lambda``.

    References:
    - Doedel (2006), Kyoto lecture notes, slides 18-20 (folds, where ``f_u`` is
      singular); see the module docstring.
    """

    def constraint(self, step: CorrectionStep) -> float:
        """Return ``lambda - lambda_p``.

        Args:
            step: The current iterate and the prediction.

        Returns:
            The constraint value; zero at the predicted parameter.
        """
        return float(step.parameter - step.predicted_parameter)

    def border(self, step: CorrectionStep) -> np.ndarray:
        """Return ``(0, ..., 0, 1)``, the gradient of ``lambda - lambda_p``.

        Args:
            step: The current iterate; only the tangent's size is used.

        Returns:
            A row of zeros with a one in the parameter's place.
        """
        row = np.zeros(step.tangent.size)
        row[-1] = _PARAMETER_BORDER
        return row
