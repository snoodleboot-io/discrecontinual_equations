"""Bifurcation detectors, one focused class per codimension-one type.

Each detector exposes a scalar test function whose sign change along the branch
brackets a bifurcation. New detectors are added by implementing
:class:`BifurcationDetector`; the driver never changes (open/closed).
"""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.continuation.bialternate import bialternate_test_value
from discrecontinual_equations.continuation.continuation_state import ContinuationState
from discrecontinual_equations.continuation.critical_forms import bordered_solve
from discrecontinual_equations.continuation.jacobian import JacobianProvider

_BRANCH_STEP = 1.0e-4
_RELATIVE_FLOOR = 1.0e-5
_NOISE_FACTOR = 100.0
_NORMALIZATION_FLOOR = 1.0e-9


def _unit_null(matrix: np.ndarray) -> np.ndarray:
    _left, _singular, right = np.linalg.svd(matrix)
    return right[-1].conj()


def _null_pair(
    context: "DetectionContext",
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Jacobian ``A``, unit ``A q = 0``, and ``A^T p = 0`` with ``<p, q> = 1``."""
    jacobian = context.state_jacobian()
    right = _unit_null(jacobian)
    left = _unit_null(jacobian.T)
    denominator = left @ right
    if abs(denominator) < _NORMALIZATION_FLOOR:
        return None
    return jacobian, right, left / denominator


def _branch_expansion(
    context: "DetectionContext",
    null_pair: tuple[np.ndarray, np.ndarray, np.ndarray],
    step: float,
) -> tuple[float, float, float, float]:
    """``(a, c)`` by differences of step ``step``, and the sizes they are summed from.

    ``a = <p, B(q, q)>`` is summed from ``B(q, q)``, so its size is
    ``|p| |B(q, q)|``; ``c = <p, C(q, q, q) + 3 B(q, h2)>`` is summed from both
    terms, so its size is ``|p| (|C(q, q, q)| + 3 |B(q, h2)|)``. Rescaling time
    (``f -> s f``) or the state multiplies a coefficient and its size alike.
    """
    jacobian, right, left = null_pair
    state = context.continuation_state.state
    forward_jacobian = context.jacobian_at(state + step * right)
    backward_jacobian = context.jacobian_at(state - step * right)
    forward = forward_jacobian @ right
    backward = backward_jacobian @ right
    centre = jacobian @ right
    quadratic = (forward - backward) / (2.0 * step)
    quadratic_coefficient = left @ quadratic
    correction = bordered_solve(
        jacobian,
        right,
        left,
        quadratic - quadratic_coefficient * right,
    )
    curvature = -correction
    along = (forward_jacobian - backward_jacobian) @ curvature / (2.0 * step)
    third = (forward - 2.0 * centre + backward) / (step * step)
    cubic_coefficient = left @ (third + 3.0 * along)
    left_size = float(np.linalg.norm(left))
    return (
        float(quadratic_coefficient.real),
        float(cubic_coefficient.real),
        left_size * float(np.linalg.norm(quadratic)),
        left_size * (float(np.linalg.norm(third)) + 3.0 * float(np.linalg.norm(along))),
    )


def _branch_coefficients(
    context: "DetectionContext",
) -> tuple[float, float] | None:
    """Quadratic ``a`` and cubic ``c`` normal-form coefficients at a branch point.

    On the center manifold at the critical parameter the field reads
    ``w' = (a / 2) w^2 + (c / 6) w^3 + ...`` with (Kuznetsov 1999, section 5)

        ``a = <p, B(q, q)>``,  ``c = <p, C(q, q, q) + 3 B(q, h2)>``,

    ``h2 = -A^INV [B(q, q) - a q]`` with ``<p, h2> = 0`` and ``<p, q> = 1``. The
    ``3 B(q, h2)`` share is the center manifold's curvature: it vanishes in one
    dimension, but once the critical mode drives another one it can make a
    pitchfork out of a vanishing ``<p, C(q, q, q)>``, or cancel one that does not.
    All three forms are directional derivatives of ``J(x) q`` along ``q``:
    ``B(q, y) = d/de [J(x + e q) y]`` and ``C(q, q, q) = d^2/de^2 [J(x + e q) q]``,
    so only the Jacobian is needed. Both are even in the sign of ``q``.
    """
    null_pair = _null_pair(context)
    if null_pair is None:
        return None
    quadratic, cubic, _quadratic_size, _cubic_size = _branch_expansion(
        context,
        null_pair,
        _BRANCH_STEP,
    )
    return quadratic, cubic


def _resolved(value: float, size: float, error: float) -> bool:
    """Whether a coefficient is told apart from zero, relative to scale and noise."""
    return abs(value) > max(_RELATIVE_FLOOR * size, _NOISE_FACTOR * error)


def _branch_label(context: "DetectionContext") -> str:
    """``transcritical``, ``pitchfork`` or ``branch_point`` from ``a`` and ``c``.

    Neither coefficient is compared with an absolute floor, which would make the
    label depend on the units of time and state. A coefficient counts as non-zero
    when it exceeds both a fraction :data:`_RELATIVE_FLOOR` of the terms it is
    summed from (a cancellation that leaves less than that is zero) and
    :data:`_NOISE_FACTOR` times its finite-difference error, estimated by
    repeating the differences with half and twice the step. Both tests are
    unchanged when ``f`` is multiplied by a constant.
    """
    null_pair = _null_pair(context)
    if null_pair is None:
        return "branch_point"
    quadratic, cubic, quadratic_size, cubic_size = _branch_expansion(
        context,
        null_pair,
        _BRANCH_STEP,
    )
    quadratic_error = 0.0
    cubic_error = 0.0
    for step in (0.5 * _BRANCH_STEP, 2.0 * _BRANCH_STEP):
        other_quadratic, other_cubic, _q, _c = _branch_expansion(
            context,
            null_pair,
            step,
        )
        quadratic_error = max(quadratic_error, abs(quadratic - other_quadratic))
        cubic_error = max(cubic_error, abs(cubic - other_cubic))
    if _resolved(quadratic, quadratic_size, quadratic_error):
        return "transcritical"
    if _resolved(cubic, cubic_size, cubic_error):
        return "pitchfork"
    return "branch_point"


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
    symmetry) while the center-manifold cubic ``c = <p, C(q, q, q) + 3 B(q, h2)>``
    does not.

    "Non-zero" is decided without units, so that rescaling time or the field does
    not change the label. A coefficient must exceed ``1e-5`` times the size of the
    terms it is summed from (``|p| |B(q, q)|`` for ``a``, ``|p| (|C(q, q, q)| +
    3 |B(q, h2)|)`` for ``c``), and ``100`` times its finite-difference error,
    estimated by repeating the differences with half and twice the step. A
    coefficient that cannot be resolved is treated as zero, so a point whose ``c``
    is lost in noise stays ``branch_point``.

    References:
        The coefficients are Kuznetsov's center-manifold reduction (see
        :func:`_branch_coefficients`). The thresholds are this library's own
        numerical choice, not taken from a source; they are measured in the
        "Telling bifurcation points apart" section of the continuation API page.
    """

    @property
    def kind(self) -> str:
        return "branch_point"

    def test_value(self, context: DetectionContext) -> float:
        extended = context.extended_jacobian()
        tangent = context.continuation_state.tangent
        return float(np.linalg.det(np.vstack([extended, tangent])))

    def refine_kind(self, context: DetectionContext) -> str:
        return _branch_label(context)


class Hopf(BifurcationDetector):
    """Hopf: the bialternate-product determinant of the state Jacobian changes sign.

    The determinant is read through its eigenvalues (see :mod:`.bialternate`), so
    the test costs one eigenvalue solve and cannot overflow.
    """

    @property
    def kind(self) -> str:
        return "hopf"

    def test_value(self, context: DetectionContext) -> float:
        return bialternate_test_value(context.state_jacobian())

    def characteristic_frequency(self, context: DetectionContext) -> float | None:
        eigenvalues = np.linalg.eigvals(context.state_jacobian())
        nearest = int(np.argmin(np.abs(eigenvalues.real)))
        return float(abs(eigenvalues[nearest].imag))
