"""Branch-point labels do not depend on the units of time.

The four three-dimensional reference systems are those of the DEQ-102 verifier,

    ``u' = mu u + r u^2 + al u v + be u w + k u^3``,
    ``v' = -2 v + 3 d u^2``,  ``w' = -w + v - d u^2``,

mixed by a fixed non-orthogonal ``x = M (u, v, w)``, with a branch point at
``mu = 0``, ``x = 0``. A separate sympy graph center-manifold reduction (not the
library's closed formula) gives, with ``d = 1``:

    ``(r, al, be, k) = (1, 1, -2, 0.1)``: ``a = 1.6903``, ``c = 2.5714``  transcritical
    ``(0, 1, -2, 0.1)``:                  ``a = 0``, ``c = 2.5714``   pitchfork
    ``(0, -1, -1, 0.4)``:                 ``a = 0``, ``c = -6.8571``  pitchfork
    ``(0, 1, -2, -0.5)``:                 ``a = 0``, ``c = 0``        branch point

and, with ``d = 0`` and ``(0, 0, 0, 0.4)``, a reflection-symmetric pitchfork whose
``B(q, q)`` vanishes identically. Rescaling time, ``f -> s f``, multiplies ``a``
and ``c`` by ``s`` and changes no bifurcation type, so every label must hold for
every ``s``, with exact and with finite-difference Jacobians.
"""

from unittest import TestCase, mock

import numpy as np

from discrecontinual_equations.continuation import detection
from discrecontinual_equations.continuation.continuation_state import (
    ContinuationState,
)
from discrecontinual_equations.continuation.detection import (
    _BRANCH_STEP,
    BranchPoint,
    DetectionContext,
    _branch_expansion,
    _branch_label,
    _null_pair,
    _resolved,
)
from discrecontinual_equations.continuation.jacobian import (
    FiniteDifferenceJacobian,
    JacobianProvider,
)
from discrecontinual_equations.continuation.residual import ResidualFunction

M = np.array([[1.0, 0.5, -0.3], [0.2, 1.0, 0.4], [-0.6, 0.1, 1.0]])
MI = np.linalg.inv(M)
SCALES = [10.0**power for power in range(-6, 7)]
FINITE_DIFFERENCE_EPSILON = 1.0e-7
SIZE_TOLERANCE = 1.0e-6
CASES = {
    "transcritical": ((1.0, 1.0, -2.0, 0.1, 1.0), "transcritical"),
    "pitchfork through the curvature": ((0.0, 1.0, -2.0, 0.1, 1.0), "pitchfork"),
    "pitchfork against the cubic": ((0.0, -1.0, -1.0, 0.4, 1.0), "pitchfork"),
    "cancelled": ((0.0, 1.0, -2.0, -0.5, 1.0), "branch_point"),
    "reflection symmetric": ((0.0, 0.0, 0.0, 0.4, 0.0), "pitchfork"),
}


class _Field(ResidualFunction):
    """``s M f(M^-1 x)`` for one reference case."""

    def __init__(self, coefficients: tuple[float, ...], scale: float) -> None:
        self.coefficients = coefficients
        self.scale = scale

    def evaluate(self, state: np.ndarray, parameter: float) -> np.ndarray:
        r, al, be, k, d = self.coefficients
        u, v, w = MI @ state
        rows = np.array(
            [
                parameter * u + r * u * u + al * u * v + be * u * w + k * u**3,
                -2.0 * v + 3.0 * d * u * u,
                -w + v - d * u * u,
            ],
        )
        return self.scale * (M @ rows)


class _Exact(_Field, JacobianProvider):
    """The same field with its Jacobian written out by hand."""

    def state_jacobian(self, state: np.ndarray, parameter: float) -> np.ndarray:
        r, al, be, k, d = self.coefficients
        u, v, w = MI @ state
        rows = np.array(
            [
                [
                    parameter + 2 * r * u + al * v + be * w + 3 * k * u * u,
                    al * u,
                    be * u,
                ],
                [6.0 * d * u, -2.0, 0.0],
                [-2.0 * d * u, 1.0, -1.0],
            ],
        )
        return self.scale * (M @ rows @ MI)

    def parameter_derivative(self, state: np.ndarray, parameter: float) -> np.ndarray:
        _ = parameter
        return self.scale * (M @ np.array([(MI @ state)[0], 0.0, 0.0]))


def _context(coefficients, scale: float, *, exact: bool) -> DetectionContext:
    provider: JacobianProvider = (
        _Exact(coefficients, scale)
        if exact
        else FiniteDifferenceJacobian(
            _Field(coefficients, scale),
            FINITE_DIFFERENCE_EPSILON,
        )
    )
    state = ContinuationState(np.zeros(3), 0.0, np.array([0.0, 0.0, 0.0, 1.0]), 0.0)
    return DetectionContext(state, provider)


class TestBranchLabelUnderTimeRescaling(TestCase):
    def test_labels_hold_from_one_millionth_to_a_million(self):
        for exact in (True, False):
            for name, (coefficients, truth) in CASES.items():
                labels = {
                    scale: BranchPoint().refine_kind(
                        _context(coefficients, scale, exact=exact),
                    )
                    for scale in SCALES
                }
                wrong = {
                    scale: label for scale, label in labels.items() if label != truth
                }
                assert not wrong, (name, exact, wrong)


class TestBranchLabelOffTheBifurcation(TestCase):
    """A located point sits a little off the bifurcation, on a nontrivial branch.

    With ``d = 0`` the critical mode is ``u' = mu u + r u^2 + k u^3``, whose
    nontrivial branch through ``u = u0`` has ``mu = -r u0 - k u0^2``. There a
    pitchfork (``r = 0``) has a quadratic coefficient ``6 k u0`` of its own; the
    label must still be ``pitchfork``, and a transcritical point must stay one.
    """

    @staticmethod
    def _label(coefficients, offset: float, scale: float, *, exact: bool) -> str:
        r, _al, _be, k, _d = coefficients
        parameter = -r * offset - k * offset * offset
        state = M @ np.array([offset, 0.0, 0.0])
        provider: JacobianProvider = (
            _Exact(coefficients, scale)
            if exact
            else FiniteDifferenceJacobian(
                _Field(coefficients, scale),
                FINITE_DIFFERENCE_EPSILON,
            )
        )
        point = ContinuationState(state, parameter, np.array([0.0, 0.0, 0.0, 1.0]), 0.0)
        return BranchPoint().refine_kind(DetectionContext(point, provider))

    def test_a_pitchfork_located_off_its_vertex_is_a_pitchfork(self):
        for exact in (True, False):
            for offset in (1.0e-6, 1.0e-4, 1.0e-2, -1.0e-2):
                for scale in (1.0e-6, 1.0, 1.0e6):
                    label = self._label(
                        (0.0, 0.0, 0.0, 0.4, 0.0),
                        offset,
                        scale,
                        exact=exact,
                    )
                    assert label == "pitchfork", (offset, scale, exact)

    def test_a_transcritical_point_located_off_it_is_transcritical(self):
        for exact in (True, False):
            for offset in (1.0e-6, 1.0e-4, 1.0e-2, -1.0e-2):
                for scale in (1.0e-6, 1.0, 1.0e6):
                    label = self._label(
                        (1.0, 0.0, 0.0, 0.4, 0.0),
                        offset,
                        scale,
                        exact=exact,
                    )
                    assert label == "transcritical", (offset, scale, exact)


class TestBranchExpansionSizes(TestCase):
    """The sizes the floors are relative to, against a hand computation."""

    @staticmethod
    def _reference(coefficients, scale: float) -> tuple[float, float]:
        # In (u, v, w): q_y = e_u, and with x = M y the x-space q is M e_u / |M e_u|.
        r, al, be, k, d = coefficients
        lam = 1.0 / np.linalg.norm(M[:, 0])
        right = lam * M[:, 0]
        jacobian = scale * (M @ np.array([[0, 0, 0], [0, -2, 0], [0, 1, -1.0]]) @ MI)
        left = np.linalg.svd(jacobian.T)[2][-1]
        left = left / (left @ right)

        def second(xi, eta):
            # B_y(xi, eta) of the polynomial rows at y = 0.
            return np.array(
                [
                    2 * r * xi[0] * eta[0]
                    + al * (xi[0] * eta[1] + xi[1] * eta[0])
                    + be * (xi[0] * eta[2] + xi[2] * eta[0]),
                    6.0 * d * xi[0] * eta[0],
                    -2.0 * d * xi[0] * eta[0],
                ],
            )

        def bilinear(x_one, x_two):
            return scale * (M @ second(MI @ x_one, MI @ x_two))

        quadratic = bilinear(right, right)
        a = left @ quadratic
        bordered = np.block(
            [[jacobian, right[:, None]], [left[None, :], np.zeros((1, 1))]],
        )
        curvature = -np.linalg.solve(
            bordered,
            np.concatenate([quadratic - a * right, [0.0]]),
        )[:3]
        cubic = scale * (M @ np.array([6.0 * k * (MI @ right)[0] ** 3, 0.0, 0.0]))
        along = bilinear(right, curvature)
        size = np.linalg.norm(left)
        return (
            size * np.linalg.norm(quadratic),
            size * (np.linalg.norm(cubic) + 3.0 * np.linalg.norm(along)),
        )

    def test_sizes_match_the_terms_they_are_summed_from(self):
        for coefficients, _truth in CASES.values():
            for scale in (1.0e-3, 1.0, 1.0e3):
                context = _context(coefficients, scale, exact=True)
                pair = _null_pair(context)
                assert pair is not None
                _a, _c, quadratic_size, cubic_size = _branch_expansion(
                    context,
                    pair,
                    _BRANCH_STEP,
                )
                expected = self._reference(coefficients, scale)
                assert abs(quadratic_size - expected[0]) <= SIZE_TOLERANCE * max(
                    expected[0],
                    scale,
                )
                assert abs(cubic_size - expected[1]) <= SIZE_TOLERANCE * max(
                    expected[1],
                    scale,
                )


class TestResolvedCoefficient(TestCase):
    def test_a_cancellation_below_the_relative_floor_is_zero(self):
        size = 3.0
        assert not _resolved(0.5e-5 * size, size, 0.0)
        assert _resolved(2.0e-5 * size, size, 0.0)

    def test_a_coefficient_within_its_noise_is_zero(self):
        error = 1.0e-3
        assert not _resolved(50.0 * error, 1.0, error)
        assert _resolved(200.0 * error, 1.0, error)

    def test_an_exact_zero_is_zero(self):
        assert not _resolved(0.0, 0.0, 0.0)

    def test_the_floors_are_unit_free(self):
        for scale in (1.0e-9, 1.0, 1.0e9):
            assert _resolved(1.0e-3 * scale, scale, 1.0e-6 * scale)
            assert not _resolved(1.0e-7 * scale, scale, 1.0e-12 * scale)


class _Expansions:
    """A stand-in for ``_branch_expansion`` that returns given ``(a, c)`` per step."""

    def __init__(
        self,
        values: dict[float, tuple[float, float]],
        sizes: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self._values = values
        self._sizes = sizes

    def __call__(self, context, null_pair, step: float):
        _ = context, null_pair
        quadratic, cubic = self._values[step]
        return quadratic, cubic, *self._sizes


class TestNoiseEstimate(TestCase):
    """The error of ``a`` and ``c`` is read from both neighbouring steps."""

    @staticmethod
    def _label(
        values: dict[float, tuple[float, float]],
        sizes: tuple[float, float] = (1.0, 1.0),
    ) -> str:
        context = _context(CASES["cancelled"][0], 1.0, exact=True)
        expansion = _Expansions(values, sizes)
        # The values are keyed by step, so the step search is held at its start.
        with (
            mock.patch.object(detection, "_branch_expansion", expansion),
            mock.patch.object(detection, "_branch_step", return_value=_BRANCH_STEP),
        ):
            return _branch_label(context)

    def test_each_coefficient_is_measured_against_its_own_terms(self):
        step = _BRANCH_STEP
        steady = {
            0.5 * step: (1.0e-3, 0.0),
            step: (1.0e-3, 0.0),
            2.0 * step: (1.0e-3, 0.0),
        }
        assert self._label(steady, sizes=(1.0, 1.0e3)) == "transcritical"
        assert self._label(steady, sizes=(1.0e3, 1.0)) == "branch_point"
        steady = {
            0.5 * step: (0.0, 1.0e-3),
            step: (0.0, 1.0e-3),
            2.0 * step: (0.0, 1.0e-3),
        }
        assert self._label(steady, sizes=(1.0e3, 1.0)) == "pitchfork"
        assert self._label(steady, sizes=(1.0, 1.0e3)) == "branch_point"

    def test_steady_coefficients_are_resolved(self):
        step = _BRANCH_STEP
        values = {0.5 * step: (0.0, 0.5), step: (0.0, 0.5), 2.0 * step: (0.0, 0.5)}
        assert self._label(values) == "pitchfork"
        values = {0.5 * step: (0.5, 0.5), step: (0.5, 0.5), 2.0 * step: (0.5, 0.5)}
        assert self._label(values) == "transcritical"

    def test_noise_seen_only_at_the_smaller_step_is_noise(self):
        step = _BRANCH_STEP
        values = {0.5 * step: (0.0, -0.5), step: (0.0, 0.5), 2.0 * step: (0.0, 0.5)}
        assert self._label(values) == "branch_point"
        values = {0.5 * step: (-0.5, 0.0), step: (0.5, 0.0), 2.0 * step: (0.5, 0.0)}
        assert self._label(values) == "branch_point"

    def test_noise_seen_only_at_the_larger_step_is_noise(self):
        step = _BRANCH_STEP
        values = {0.5 * step: (0.0, 0.5), step: (0.0, 0.5), 2.0 * step: (0.0, -0.5)}
        assert self._label(values) == "branch_point"
        values = {0.5 * step: (0.5, 0.0), step: (0.5, 0.0), 2.0 * step: (-0.5, 0.0)}
        assert self._label(values) == "branch_point"
