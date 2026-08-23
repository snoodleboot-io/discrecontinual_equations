"""Bifurcation detectors, one focused class per codimension-one type.

Each detector exposes a scalar test function whose sign change along the branch
brackets a bifurcation. New detectors are added by implementing
:class:`BifurcationDetector`; the driver never changes (open/closed).
"""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.continuation.bialternate import bialternate_determinant
from discrecontinual_equations.continuation.continuation_state import ContinuationState
from discrecontinual_equations.continuation.jacobian import JacobianProvider

_BRANCH_STEP = 1.0e-4
_QUADRATIC_FLOOR = 1.0e-3
_CUBIC_FLOOR = 1.0e-6
_NORMALIZATION_FLOOR = 1.0e-9


def _unit_null(matrix: np.ndarray) -> np.ndarray:
    _left, _singular, right = np.linalg.svd(matrix)
    return right[-1].conj()


def _branch_coefficients(
    context: "DetectionContext",
) -> tuple[float, float] | None:
    """Quadratic ``a`` and cubic ``c`` normal-form coefficients at a branch point.

    Both are directional derivatives of ``J(x) q`` along the kernel ``q``: with a
    simple zero eigenvalue, ``B(q, q) = d/de [J(x + e q) q]`` and
    ``C(q, q, q) = d^2/de^2 [J(x + e q) q]``, projected onto the left null vector
    ``p`` normalized so ``<p, q> = 1``. This needs only the Jacobian, no forms.
    """
    jacobian = context.state_jacobian()
    right = _unit_null(jacobian)
    left = _unit_null(jacobian.T)
    denominator = left @ right
    if abs(denominator) < _NORMALIZATION_FLOOR:
        return None
    left = left / denominator
    state = context.continuation_state.state
    step = _BRANCH_STEP
    forward = context.jacobian_at(state + step * right) @ right
    backward = context.jacobian_at(state - step * right) @ right
    centre = jacobian @ right
    quadratic = (forward - backward) / (2.0 * step)
    cubic = (forward - 2.0 * centre + backward) / (step * step)
    return float((left @ quadratic).real), float((left @ cubic).real)


class DetectionContext:
    """Everything a detector may need to evaluate its test function at a point."""

    __slots__ = ["_jacobian", "_state"]

    def __init__(self, state: ContinuationState, jacobian: JacobianProvider) -> None:
        self._state = state
        self._jacobian = jacobian

    @property
    def continuation_state(self) -> ContinuationState:
        """The point at which the test function is evaluated."""
        return self._state

    def state_jacobian(self) -> np.ndarray:
        """State Jacobian at this point."""
        return self._jacobian.state_jacobian(
            self._state.state,
            self._state.parameter,
        )

    def extended_jacobian(self) -> np.ndarray:
        """Extended Jacobian at this point."""
        return self._jacobian.extended_jacobian(
            self._state.state,
            self._state.parameter,
        )

    def jacobian_at(self, state: np.ndarray) -> np.ndarray:
        """State Jacobian at an arbitrary state, holding this point's parameter."""
        return self._jacobian.state_jacobian(state, self._state.parameter)


class BifurcationDetector(ABC):
    """A codim-1 detector: a named, sign-changing scalar test function."""

    @property
    @abstractmethod
    def kind(self) -> str:
        """Label recorded on a detected point (e.g. ``fold``)."""
        raise NotImplementedError

    @abstractmethod
    def test_value(self, context: DetectionContext) -> float:
        """Scalar whose sign change brackets the bifurcation."""
        raise NotImplementedError

    def characteristic_frequency(self, context: DetectionContext) -> float | None:
        """Optional oscillation frequency (only meaningful for Hopf)."""
        _ = context
        return None

    def refine_kind(self, context: DetectionContext) -> str:
        """Refine the label using local structure; defaults to :attr:`kind`."""
        _ = context
        return self.kind


class Fold(BifurcationDetector):
    """Fold (saddle-node): the tangent's parameter component passes through zero."""

    @property
    def kind(self) -> str:
        return "fold"

    def test_value(self, context: DetectionContext) -> float:
        return float(context.continuation_state.tangent[-1])


class BranchPoint(BifurcationDetector):
    """Branch point: the bordered extended-Jacobian determinant changes sign.

    A simple branch point is refined into a *transcritical* bifurcation when the
    quadratic normal-form coefficient ``a = <p, B(q, q)>`` is non-zero, or a
    *pitchfork* when that quadratic coefficient vanishes (as forced by a reflection
    symmetry) while the cubic coefficient ``c = <p, C(q, q, q)>`` does not.
    """

    @property
    def kind(self) -> str:
        return "branch_point"

    def test_value(self, context: DetectionContext) -> float:
        extended = context.extended_jacobian()
        tangent = context.continuation_state.tangent
        return float(np.linalg.det(np.vstack([extended, tangent])))

    def refine_kind(self, context: DetectionContext) -> str:
        coefficients = _branch_coefficients(context)
        if coefficients is None:
            return "branch_point"
        quadratic, cubic = coefficients
        if abs(quadratic) >= _QUADRATIC_FLOOR:
            return "transcritical"
        if abs(cubic) >= _CUBIC_FLOOR:
            return "pitchfork"
        return "branch_point"


class Hopf(BifurcationDetector):
    """Hopf: the bialternate-product determinant of the state Jacobian changes sign."""

    @property
    def kind(self) -> str:
        return "hopf"

    def test_value(self, context: DetectionContext) -> float:
        return bialternate_determinant(context.state_jacobian())

    def characteristic_frequency(self, context: DetectionContext) -> float | None:
        eigenvalues = np.linalg.eigvals(context.state_jacobian())
        nearest = int(np.argmin(np.abs(eigenvalues.real)))
        return float(abs(eigenvalues[nearest].imag))
