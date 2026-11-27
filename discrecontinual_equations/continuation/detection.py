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
_STEP_BAND = 2.0
_STEP_GROWTH = 10.0
_STEP_SEARCHES = 12
_RELATIVE_FLOOR = 1.0e-5
_NOISE_FACTOR = 100.0
_OFFSET_FACTOR = 30.0
_VERTEX_FRACTION = 0.5
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


def _relative_change(
    context: "DetectionContext",
    null_pair: tuple[np.ndarray, np.ndarray, np.ndarray],
    step: float,
) -> float:
    """How far the Jacobian moves over ``step`` along ``q``, as a unit-free length.

    With ``J+-`` the Jacobian at ``x +- step q``, the odd part
    ``|J+ - J-| / (2 |J|)`` is ``step |B(q, .)| / |J|`` and the square root of the
    even part ``|J+ - 2 J + J-| / |J|`` is ``step sqrt(|C(q, q, .)| / |J|)``; both
    are ``step`` divided by a length on which the field is nonlinear. The larger
    is returned. Rescaling the state (``x -> sigma x``) multiplies that length by
    ``sigma``; rescaling time multiplies ``J``, ``B`` and ``C`` alike and leaves it.
    """
    jacobian, right, _left = null_pair
    size = float(np.linalg.norm(jacobian))
    if size == 0.0:
        return 0.0
    state = context.continuation_state.state
    forward = context.jacobian_at(state + step * right)
    backward = context.jacobian_at(state - step * right)
    odd = float(np.linalg.norm(forward - backward)) / (2.0 * size)
    even = float(np.linalg.norm(forward - 2.0 * jacobian + backward)) / size
    return max(odd, float(np.sqrt(even)))


def _branch_step(
    context: "DetectionContext",
    null_pair: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    """The difference step for ``a`` and ``c``, scaled to the field's nonlinearity.

    A fixed step has units: once the state is rescaled by ``sigma``, the third
    difference behind ``c`` moves the Jacobian by ``(step / sigma)^2`` of itself
    and is lost in rounding for ``sigma`` of a few hundred (DEQ-147). Scaling it
    by the state norm, as the finite-difference Jacobians do (DEQ-118), does not
    help at a branch point on the trivial branch ``x = 0``. The step is instead
    chosen so that :func:`_relative_change` is :data:`_BRANCH_STEP`: it starts at
    ``_BRANCH_STEP max(1, |x|)``, moves by ``_BRANCH_STEP / change`` (at least
    ten-fold when it grows, since rounding inflates a small change), and stops
    within a factor of two or after :data:`_STEP_SEARCHES` moves. A Jacobian that
    does not move at all grows the step ten-fold: its change was rounded away,
    or the field is linear along ``q`` and its label is ``branch_point`` at any
    step.
    The search only uses Jacobians, so it is unchanged when ``f`` is multiplied
    by a constant, and it multiplies the step by ``sigma`` when the state is.
    Limits: the change is measured against the whole Jacobian, so a stiff system
    whose other eigenvalues dwarf the critical dynamics gets too large a step.

    References:
        The relative-step idea follows the library's difference Jacobians
        (DEQ-118); the target change and the search are this library's own.
    """
    state = context.continuation_state.state
    step = _BRANCH_STEP * max(1.0, float(np.linalg.norm(state)))
    for _ in range(_STEP_SEARCHES):
        change = _relative_change(context, null_pair, step)
        if not np.isfinite(change):
            return step
        # An unmoved Jacobian is a change lost in rounding, or a linear field.
        factor = _BRANCH_STEP / change if change > 0.0 else _STEP_GROWTH
        if 1.0 / _STEP_BAND <= factor <= _STEP_BAND:
            return step
        if factor > 1.0:
            factor = max(factor, _STEP_GROWTH)
        step *= factor
    return step


def _vertex_offset(
    context: "DetectionContext",
    null_pair: tuple[np.ndarray, np.ndarray, np.ndarray],
    step: float,
) -> float:
    """How far along ``q`` the point sits from the branch the parameter leaves fixed.

    On the center manifold, with ``x = x* + w q``, the reduced field ``g(w, mu)``
    of a transcritical or pitchfork point is ``mu w + (a/2) w^2 + (c/6) w^3`` up
    to scaling, centred on the branch ``w = 0`` that persists for every ``mu``.
    Seen from a point ``w0`` on the other branch, ``g_mu = <p, f_mu> = w0 g_wmu``,
    so ``w0 = g_mu / g_wmu``; it is zero on the persisting branch itself. The
    center manifold moves with ``mu`` too, by ``h_mu`` with
    ``A h_mu = -(f_mu - <p, f_mu> q)``, so ``g_wmu = <p, f_xmu q + B(q, h_mu)>``.
    Without ``B(q, h_mu)`` a parameter that enters through another variable
    (``u' = v u - u^3``, ``v' = -v + mu``) gives ``g_wmu = 0`` and the pitchfork
    is lost. ``f_xmu q`` is the difference of ``f_mu`` over ``x +- step q``, and
    ``B(q, h_mu)`` that of the Jacobian applied to ``h_mu``. Zero is returned
    when ``g_wmu`` vanishes (no transversal crossing to recentre on).

    References:
        A Taylor shift of the normal form of Kuznetsov 1999, section 5, derived
        here; checked against ``<p, x - x*>`` to leading order in
        ``tests/continuation/test_branch_point_units.py``. The ``z beta``
        coefficient ``p (B(q, H_0010) + A_1(q, K))`` with ``H_0010`` from the
        parameter's linear term is Bosschaert, Janssens & Kuznetsov,
        "Switching to nonhyperbolic cycles from codimension two bifurcations of
        equilibria of delay differential equations", preprint 2019,
        arXiv:1903.08276v1, eqs. (88)-(89).
    """
    jacobian, right, left = null_pair
    state = context.continuation_state.state
    derivative = context.parameter_derivative_at(state)
    drift = float((left @ derivative).real)
    shift = -bordered_solve(jacobian, right, left, derivative - drift * right)
    forward = context.parameter_derivative_at(state + step * right)
    backward = context.parameter_derivative_at(state - step * right)
    moved = context.jacobian_at(state + step * right) - context.jacobian_at(
        state - step * right,
    )
    crossing = float((left @ (forward - backward + moved @ shift)).real) / (2.0 * step)
    if crossing == 0.0:
        return 0.0
    return drift / crossing


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
        _branch_step(context, null_pair),
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

    The point is located only to within the localizer's tolerance, and off the
    bifurcation a pitchfork has a quadratic coefficient of its own. In the normal
    form ``w' = mu w + (a/2) w^2 + (c/6) w^3`` with ``a = 0``, the nontrivial
    branch at ``w0`` has critical eigenvalue ``lambda = (c/3) w0^2`` and quadratic
    coefficient ``c w0``, so ``a^2 = 3 |c lambda|``; a transcritical point gives
    ``a^2 / |c lambda| = 2 |a| / |c w0|``, which grows as the point is located
    better. ``a`` therefore names a transcritical point only when also
    ``a^2 > 30 |c lambda|`` (:data:`_OFFSET_FACTOR`), with ``lambda = <p, A q>``.
    This test, too, is unchanged by rescaling time or the state.

    A resolved ``a`` that fails the offset test no longer falls through to the
    cubic (DEQ-147): a weak transcritical located a little off its bifurcation
    was named ``pitchfork`` that way. It is a pitchfork only when the pitchfork's
    own ``a = c w0`` explains it, ``|a - c w0| <= |a| / 2`` with ``w0`` from
    :func:`_vertex_offset`; otherwise the point is ``branch_point``. The
    differences use the step of :func:`_branch_step`, which follows the state's
    scale.

    References:
        Kuznetsov 1999, section 5, for ``a`` and ``c`` (see
        :func:`_branch_coefficients`). The offset test, the recentring and the
        thresholds are this library's own, derived from the normal form above
        and not taken from a source.
    """
    null_pair = _null_pair(context)
    if null_pair is None:
        return "branch_point"
    branch_step = _branch_step(context, null_pair)
    quadratic, cubic, quadratic_size, cubic_size = _branch_expansion(
        context,
        null_pair,
        branch_step,
    )
    quadratic_error = 0.0
    cubic_error = 0.0
    for step in (0.5 * branch_step, 2.0 * branch_step):
        other_quadratic, other_cubic, _q, _c = _branch_expansion(
            context,
            null_pair,
            step,
        )
        quadratic_error = max(quadratic_error, abs(quadratic - other_quadratic))
        cubic_error = max(cubic_error, abs(cubic - other_cubic))
    jacobian, right, left = null_pair
    eigenvalue = abs(float((left @ jacobian @ right).real))
    offset = _OFFSET_FACTOR * abs(cubic) * eigenvalue
    if _resolved(quadratic, quadratic_size, quadratic_error):
        if quadratic * quadratic > offset:
            return "transcritical"
        # A pitchfork seen off its vertex owns its a = c w0; anything else that
        # fails the offset test cannot be told apart and is not named (DEQ-147).
        vertex = quadratic - cubic * _vertex_offset(context, null_pair, branch_step)
        if abs(vertex) > _VERTEX_FRACTION * abs(quadratic):
            return "branch_point"
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

    def parameter_derivative_at(self, state: np.ndarray) -> np.ndarray:
        """``df/dmu`` at an arbitrary state, holding this point's parameter."""
        return self._jacobian.parameter_derivative(state, self._state.parameter)


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
    is lost in noise stays ``branch_point``. Since a located point sits a little
    off the bifurcation, where a pitchfork's own ``a`` is ``c w0``, ``a`` must
    also satisfy ``a^2 > 30 |c lambda|`` with ``lambda = <p, A q>``; a resolved
    ``a`` that fails this is a pitchfork only if it is the pitchfork's own ``c w0``
    seen from its other branch, and is otherwise left ``branch_point`` (see
    :func:`_branch_label`). The difference step follows the scale of the state
    (:func:`_branch_step`), so rescaling the state does not change the label
    either.

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
