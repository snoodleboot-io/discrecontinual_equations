"""Branch-point labels do not depend on units, end to end (DEQ-146, DEQ-147).

The reference field is the DEQ-102/124/126 verifier's mixed three-dimensional
system (see ``test_branch_label_scale.py``), with time rescaled by ``s``
(``f -> s f``) and the state by ``sigma`` (``x -> sigma x``):

    ``u' = mu u + r u^2 + al u v + be u w + k u^3``,
    ``v' = -2 v + 3 d u^2``,  ``w' = -w + v - d u^2``,  ``x = sigma M (u, v, w)``.

Neither rescaling changes a bifurcation's type, so a label that changes with
``s`` or ``sigma`` is a units bug. DEQ-146: the localizer's absolute value stop
handed the labeller a point ``mu ~ 7e-3`` off the bifurcation for slow systems.
DEQ-147: the absolute difference step ``_BRANCH_STEP`` lost the pitchfork's cubic
in rounding once ``sigma`` reached a few hundred, and a transcritical that failed
the offset test fell through to ``pitchfork``.
"""

from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.continuation.continuation_state import (
    ContinuationState,
)
from discrecontinual_equations.continuation.detection import (
    _BRANCH_STEP,
    DetectionContext,
    _branch_coefficients,
    _branch_label,
    _branch_step,
    _null_pair,
    _relative_change,
    _vertex_offset,
)
from discrecontinual_equations.continuation.jacobian import (
    FiniteDifferenceJacobian,
    JacobianProvider,
)
from discrecontinual_equations.continuation.residual import (
    DifferentiableResidual,
    ResidualFunction,
)
from discrecontinual_equations.continuation.root_finder import (
    Bisection,
    RootFinderSettings,
    Secant,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

M = np.array([[1.0, 0.5, -0.3], [0.2, 1.0, 0.4], [-0.6, 0.1, 1.0]])
MI = np.linalg.inv(M)
EXTREMES = [1.0e-6, 1.0e-3, 1.0, 1.0e3, 1.0e6]
FINITE_DIFFERENCE_EPSILON = 1.0e-7
TRANSCRITICAL = (1.0, 1.0, -2.0, 0.1, 1.0)
WEAK_TRANSCRITICAL = (1.0e-3, 1.0, -2.0, 0.1, 1.0)
PITCHFORKS = {
    "through the curvature": (0.0, 1.0, -2.0, 0.1, 1.0),
    "against the cubic": (0.0, -1.0, -1.0, 0.4, 1.0),
    "reflection symmetric": (0.0, 0.0, 0.0, 0.4, 0.0),
}
LOCATION_TOLERANCE = 1.0e-6
STEP_RATIO_TOLERANCE = 0.5
OFFSET_TOLERANCE = 1.0e-3
COEFFICIENT_TOLERANCE = 1.0e-3
# Sympy center-manifold reduction of the DEQ-102 verifier (test_branch_label_scale).
REFERENCE_CUBIC = 2.5714


class _Field(DifferentiableResidual, JacobianProvider):
    """The reference field, its exact Jacobian and its exact ``df/dmu``."""

    def __init__(self, coefficients, scale=1.0, sigma=1.0) -> None:
        self._coefficients = coefficients
        self._scale = scale
        self._sigma = sigma

    def _mode(self, state):
        return MI @ (np.asarray(state, dtype=float) / self._sigma)

    def evaluate(self, state, parameter):
        r, al, be, k, d = self._coefficients
        u, v, w = self._mode(state)
        rows = np.array(
            [
                parameter * u + r * u * u + al * u * v + be * u * w + k * u**3,
                -2.0 * v + 3.0 * d * u * u,
                -w + v - d * u * u,
            ],
        )
        return self._scale * self._sigma * (M @ rows)

    def state_jacobian(self, state, parameter):
        r, al, be, k, d = self._coefficients
        u, v, w = self._mode(state)
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
        return self._scale * (M @ rows @ MI)

    def parameter_derivative(self, state, parameter):
        _ = parameter
        mode = np.array([self._mode(state)[0], 0.0, 0.0])
        return self._scale * self._sigma * (M @ mode)

    def extended_jacobian(self, state, parameter, epsilon=0.0):
        _ = epsilon
        return np.hstack(
            [
                self.state_jacobian(state, parameter),
                self.parameter_derivative(state, parameter)[:, None],
            ],
        )


class _Mu(Parameter, name="mu", abbreviation="mu"):
    pass


class _X(Variable, name="x", abbreviation="x"):
    pass


class _T(Variable, name="t", abbreviation="t"):
    pass


class _Unused(DeterministicFunction):
    def eval(self, point, time=None):
        _ = point, time
        return [0.0, 0.0, 0.0]


def _equation() -> DifferentialEquation:
    parameters = [_Mu(value=-0.3)]
    return DifferentialEquation(
        variables=[_X() for _ in range(3)],
        time=_T(),
        parameters=parameters,
        derivative=_Unused(
            variables=[_X() for _ in range(3)],
            parameters=parameters,
            results=[_X() for _ in range(3)],
            time=None,
        ),
    )


def _epsilon(sigma: float) -> float:
    """A difference-Jacobian step that resolves a state of size ``sigma``.

    The difference Jacobian's step is ``epsilon max(1, |x_j|)``, absolute near
    ``x = 0``; at ``sigma = 1e-6`` the default ``1e-7`` is a tenth of the state
    and moves the critical eigenvalue to ``4.6e-3``, so the Jacobian, not the
    labeller, is wrong there. The step is therefore scaled with ``sigma`` below 1.
    """
    return FINITE_DIFFERENCE_EPSILON * min(1.0, sigma)


def _continued(coefficients, scale=1.0, sigma=1.0, *, exact=True):
    """Continue ``x = 0`` from ``mu = -0.3`` through ``mu = 0`` with the defaults."""
    config = ContinuationConfig(
        finite_difference_epsilon=_epsilon(sigma),
        continuation_parameter_index=0,
        initial_parameter=-0.3,
        direction=1,
        detectors=["branch_point"],
        maximum_points=40,
        initial_step=0.017,
        maximum_step=0.03,
        parameter_lower_bound=-1.0,
        parameter_upper_bound=0.3,
        jacobian="analytic_jacobian" if exact else "finite_difference_jacobian",
    )
    continuer = ContinuerBuilder.assemble(
        config,
        _Field(coefficients, scale, sigma),
    )
    branch = continuer.solve(_equation(), [0.0, 0.0, 0.0])
    return [(point.kind, point.parameter) for point in branch.special_points]


def _context(coefficients, sigma=1.0, *, exact=True, u0=0.0, mu_shift=0.0):
    """A point on the branch through ``u0`` (``u0 = 0`` is the trivial branch)."""
    r, al, be, k, d = coefficients
    v, w = 1.5 * d * u0 * u0, 0.5 * d * u0 * u0
    mu = -(r * u0 + al * v + be * w + k * u0 * u0) + mu_shift
    field = _Field(coefficients, 1.0, sigma)
    provider = field if exact else FiniteDifferenceJacobian(field, _epsilon(sigma))
    state = sigma * (M @ np.array([u0, v, w]))
    tangent = np.array([0.0, 0.0, 0.0, 1.0])
    return DetectionContext(ContinuationState(state, mu, tangent, 0.0), provider)


class TestRelativeValueStop(TestCase):
    """A relative value stop finds the same root whatever the function's units."""

    @staticmethod
    def _root(finder_class, slope: float, *, relative: bool) -> float:
        settings = RootFinderSettings(1.0e-8, 1.0e-12, 60, relative=relative)
        finder = finder_class(settings)
        located = finder.locate(lambda x: slope * (x - 0.3), 0.0, 1.0)
        assert located is not None
        return located

    def test_bisection_locates_the_root_at_every_scale(self):
        for slope in (1.0e-18, 1.0e-9, 1.0, 1.0e9):
            assert (
                abs(self._root(Bisection, slope, relative=True) - 0.3)
                < LOCATION_TOLERANCE
            )

    def test_an_absolute_stop_accepts_the_first_midpoint_of_a_slow_function(self):
        assert (
            abs(self._root(Bisection, 1.0e-9, relative=False) - 0.5)
            < LOCATION_TOLERANCE
        )

    def test_secant_locates_the_root_at_every_scale(self):
        for slope in (1.0e-18, 1.0e-9, 1.0, 1.0e9):
            assert (
                abs(self._root(Secant, slope, relative=True) - 0.3) < LOCATION_TOLERANCE
            )

    def test_a_root_near_one_end_is_measured_against_the_other(self):
        # The larger end value sets the scale: measured against the small one
        # the stop would wait for the bracket to close (about 40 halvings).
        calls = []

        def function(x: float) -> float:
            calls.append(x)
            return 1.0e-9 * (x - 1.0e-6)

        settings = RootFinderSettings(1.0e-8, 1.0e-12, 60, relative=True)
        located = Bisection(settings).locate(function, 0.0, 1.0)
        assert located is not None
        assert abs(located - 1.0e-6) < 1.0e-7
        assert len(calls) <= 32

    def test_secant_does_not_accept_a_slow_first_chord(self):
        def function(x: float) -> float:
            shift = x - 0.3
            return 1.0e-12 * (shift + 5.0 * shift**3)

        settings = RootFinderSettings(1.0e-8, 1.0e-12, 60, relative=True)
        located = Secant(settings).locate(function, 0.0, 1.0)
        assert located is not None
        assert abs(located - 0.3) < LOCATION_TOLERANCE

    def test_the_default_stays_absolute(self):
        settings = RootFinderSettings(1.0e-8, 1.0e-12, 60)
        assert not settings.relative
        assert settings.value_scale(5.0, -7.0) == 1.0
        relative = RootFinderSettings(1.0e-8, 1.0e-12, 60, relative=True)
        assert relative.value_scale(5.0, -7.0) == 7.0


class TestTranscriticalEndToEnd(TestCase):
    """DEQ-146: a transcritical stays transcritical through continuation."""

    def test_time_rescaling_keeps_the_label_and_the_location(self):
        for r in (1.0, 1.0e-2, 1.0e-4):
            coefficients = (r, *TRANSCRITICAL[1:])
            for exact in (True, False):
                for scale in EXTREMES:
                    found = _continued(coefficients, scale=scale, exact=exact)
                    assert [kind for kind, _mu in found] == ["transcritical"], (
                        r,
                        exact,
                        scale,
                        found,
                    )
                    assert abs(found[0][1]) < LOCATION_TOLERANCE

    def test_state_rescaling_keeps_the_label(self):
        for r in (1.0, 1.0e-4):
            coefficients = (r, *TRANSCRITICAL[1:])
            for sigma in EXTREMES:
                found = _continued(coefficients, sigma=sigma)
                assert [kind for kind, _mu in found] == ["transcritical"], (
                    r,
                    sigma,
                    found,
                )


class TestPitchforkUnderStateRescaling(TestCase):
    """DEQ-147: the pitchfork keeps its label for sigma from 1e-6 to 1e6."""

    def test_at_the_vertex_with_exact_and_difference_jacobians(self):
        for name, coefficients in PITCHFORKS.items():
            for exact in (True, False):
                for power in range(-6, 7):
                    sigma = 10.0**power
                    label = _branch_label(_context(coefficients, sigma, exact=exact))
                    assert label == "pitchfork", (name, exact, sigma, label)

    def test_through_continuation(self):
        for name, coefficients in PITCHFORKS.items():
            for sigma in EXTREMES:
                found = _continued(coefficients, sigma=sigma)
                assert [kind for kind, _mu in found] == ["pitchfork"], (
                    name,
                    sigma,
                    found,
                )
                found = _continued(coefficients, sigma=sigma, exact=False)
                assert [kind for kind, _mu in found] == ["pitchfork"], (
                    name,
                    sigma,
                    found,
                )

    def test_the_step_grows_with_the_state_and_ignores_time(self):
        coefficients = PITCHFORKS["through the curvature"]
        context = _context(coefficients)
        pair = _null_pair(context)
        assert pair is not None
        unit = _branch_step(context, pair)
        for sigma in (1.0e-6, 1.0e3, 1.0e6):
            context = _context(coefficients, sigma)
            pair = _null_pair(context)
            assert pair is not None
            ratio = _branch_step(context, pair) / (sigma * unit)
            assert abs(np.log10(ratio)) < STEP_RATIO_TOLERANCE, (sigma, ratio)

    def test_the_step_of_a_quadratic_field_grows_with_the_state(self):
        # No cubic: the Jacobian moves only through its odd part.
        coefficients = (1.0, 0.0, 0.0, 0.0, 0.0)
        steps = []
        for sigma in (1.0, 1.0e6):
            context = _context(coefficients, sigma)
            pair = _null_pair(context)
            assert pair is not None
            steps.append(_branch_step(context, pair) / sigma)
        assert abs(np.log10(steps[1] / steps[0])) < STEP_RATIO_TOLERANCE

    def test_the_coefficients_follow_the_state_scale(self):
        # x -> sigma x divides c by sigma^2.
        coefficients = PITCHFORKS["through the curvature"]
        for sigma in (1.0e-3, 1.0e3, 1.0e6):
            result = _branch_coefficients(_context(coefficients, sigma))
            assert result is not None
            cubic = result[1] * sigma * sigma
            assert abs(abs(cubic) - REFERENCE_CUBIC) <= COEFFICIENT_TOLERANCE * (
                REFERENCE_CUBIC
            ), (sigma, cubic)

    def test_a_jacobian_that_does_not_move_grows_the_step(self):
        context = _context(PITCHFORKS["reflection symmetric"], 1.0e6)
        pair = _null_pair(context)
        assert pair is not None
        assert _branch_step(context, pair) > 1.0e4 * _BRANCH_STEP


class TestWeakTranscritical(TestCase):
    """DEQ-147: a resolved ``a`` that fails the offset test is not a pitchfork."""

    def test_located_off_the_bifurcation_it_is_not_named(self):
        for shift in (1.0e-7, 3.7e-7, 1.0e-6, 1.0e-5, -1.0e-6):
            label = _branch_label(_context(WEAK_TRANSCRITICAL, mu_shift=shift))
            assert label == "branch_point", (shift, label)

    def test_located_well_it_is_transcritical(self):
        for shift in (0.0, 1.0e-10, 1.0e-8):
            label = _branch_label(_context(WEAK_TRANSCRITICAL, mu_shift=shift))
            assert label == "transcritical", (shift, label)

    def test_a_pitchfork_seen_from_its_other_branch_stays_a_pitchfork(self):
        for name, coefficients in PITCHFORKS.items():
            for u0 in (1.0e-4, 1.0e-2, 5.0e-2):
                label = _branch_label(_context(coefficients, u0=u0))
                assert label == "pitchfork", (name, u0, label)

    def test_the_offset_is_zero_on_the_persisting_branch(self):
        context = _context(WEAK_TRANSCRITICAL, mu_shift=1.0e-6)
        pair = _null_pair(context)
        assert pair is not None
        assert _vertex_offset(context, pair, 1.0e-4) == 0.0

    def test_the_offset_is_the_distance_along_q_on_the_other_branch(self):
        # Independent of the reduction: to leading order the point is
        # x0 = w0 q, so w0 = <p, x0> with <p, q> = 1.
        for sigma in (1.0e-3, 1.0, 1.0e3):
            context = _context(PITCHFORKS["reflection symmetric"], sigma, u0=1.0e-4)
            pair = _null_pair(context)
            assert pair is not None
            _jacobian, _right, left = pair
            expected = float(left @ context.continuation_state.state)
            offset = _vertex_offset(context, pair, 1.0e-4 * sigma)
            assert abs(offset - expected) <= OFFSET_TOLERANCE * abs(expected)


class _SlavedPitchfork(ResidualFunction, JacobianProvider):
    """``u' = (v + b mu) u - u^3``, ``v' = -v + (1 - b) mu``: a pitchfork at ``mu = 0``.

    With ``b < 1`` part of the parameter reaches ``u`` only through ``v``, so
    ``f_mu`` has a component off ``q`` and the center manifold moves with ``mu``.
    The nontrivial branch is ``u^2 = mu``, ``v = (1 - b) mu``.
    """

    def __init__(self, share: float) -> None:
        self._share = share

    def evaluate(self, state, parameter):
        u, v = state
        b = self._share
        return np.array([(v + b * parameter) * u - u**3, -v + (1.0 - b) * parameter])

    def state_jacobian(self, state, parameter):
        u, v = state
        return np.array([[v + self._share * parameter - 3.0 * u * u, u], [0.0, -1.0]])

    def parameter_derivative(self, state, parameter):
        _ = parameter
        return np.array([self._share * state[0], 1.0 - self._share])


def _slaved_context(share, u0, *, exact=True):
    field = _SlavedPitchfork(share)
    provider = field if exact else FiniteDifferenceJacobian(field, 1.0e-7)
    mu = u0 * u0
    state = np.array([u0, (1.0 - share) * mu])
    tangent = np.array([0.0, 0.0, 1.0])
    return DetectionContext(ContinuationState(state, mu, tangent, 0.0), provider)


class TestParameterThroughAnotherVariable(TestCase):
    """The recentring includes the center manifold's own drift, ``B(q, h_mu)``.

    Without it ``g_wmu = <p, f_xmu q>`` is zero for ``b = 0`` and half its value
    for ``b = 1/2``, so the offset ``w0`` comes out 0 or ``2 u0`` and a pitchfork
    seen from its nontrivial branch, even ``1e-3`` from its vertex, was named
    ``branch_point``.
    """

    def test_a_pitchfork_seen_from_its_other_branch_stays_a_pitchfork(self):
        for share in (0.0, 0.5, 1.0):
            for u0 in (1.0e-3, 1.0e-2, 3.0e-2):
                for exact in (True, False):
                    label = _branch_label(_slaved_context(share, u0, exact=exact))
                    assert label == "pitchfork", (share, u0, exact, label)

    def test_the_offset_is_the_distance_along_q_from_the_persisting_branch(self):
        # Independent of the reduction: the persisting branch is (0, (1 - b) mu),
        # so to leading order w0 = <p, x - x*> with <p, q> = 1.
        for share in (0.0, 0.5):
            u0 = 1.0e-3
            context = _slaved_context(share, u0)
            pair = _null_pair(context)
            assert pair is not None
            _jacobian, _right, left = pair
            state = context.continuation_state.state
            expected = float(left @ (state - np.array([0.0, state[1]])))
            offset = _vertex_offset(context, pair, 1.0e-4)
            assert abs(offset - expected) <= 1.0e-2 * abs(expected), (share, offset)


class _CountedField(_Field):
    """The reference field, counting its Jacobian evaluations."""

    def __init__(self, coefficients, scale=1.0, sigma=1.0) -> None:
        super().__init__(coefficients, scale, sigma)
        self.jacobians = 0

    def state_jacobian(self, state, parameter):
        self.jacobians += 1
        return super().state_jacobian(state, parameter)


class TestStopAndStepEdges(TestCase):
    """Edges of the relative stop and the step search the tests above leave open."""

    def test_a_root_near_the_upper_end_is_measured_against_the_lower(self):
        # Mirror of the test above: the small value is now f(high). Measured
        # against it alone, the stop waits for the bracket to close.
        calls = []

        def function(x: float) -> float:
            calls.append(x)
            return 1.0e-9 * (x - (1.0 - 1.0e-6))

        settings = RootFinderSettings(1.0e-8, 1.0e-12, 60, relative=True)
        located = Bisection(settings).locate(function, 0.0, 1.0)
        assert located is not None
        assert abs(located - (1.0 - 1.0e-6)) < 1.0e-7
        assert len(calls) <= 32

    def test_secant_follows_a_steep_function_to_its_root(self):
        # Values of order 1e12: the flat-chord test must scale with the end
        # values once, not with their square, or the secant stops early.
        def function(x: float) -> float:
            shift = x - 0.3
            return 1.0e12 * (shift + 5.0 * shift**3)

        settings = RootFinderSettings(1.0e-8, 1.0e-12, 60, relative=True)
        located = Secant(settings).locate(function, 0.0, 1.0)
        assert located is not None
        assert abs(located - 0.3) < LOCATION_TOLERANCE

    def test_the_change_is_the_step_over_the_nonlinear_length(self):
        # A quadratic field's Jacobian is linear in x, so the odd part is exact:
        # |J(x + h q) - J(x - h q)| / (2 |J|) = h |J(x + q) - J(x)| / |J|.
        coefficients = (1.0, 1.0, -2.0, 0.0, 1.0)
        context = _context(coefficients)
        pair = _null_pair(context)
        assert pair is not None
        jacobian, right, _left = pair
        state = context.continuation_state.state
        slope = np.linalg.norm(context.jacobian_at(state + right) - jacobian)
        expected = 1.0e-3 * float(slope / np.linalg.norm(jacobian))
        change = _relative_change(context, pair, 1.0e-3)
        assert abs(change - expected) <= 1.0e-9 * expected

    def test_the_search_grows_the_step_in_few_moves(self):
        # At sigma = 1e6 the step must grow by about 1e6: at least ten-fold per
        # move it takes a few, not one per decade capped at ten.
        coefficients = PITCHFORKS["through the curvature"]
        field = _CountedField(coefficients, 1.0, 1.0e6)
        state = 1.0e6 * (M @ np.zeros(3))
        tangent = np.array([0.0, 0.0, 0.0, 1.0])
        context = DetectionContext(ContinuationState(state, 0.0, tangent, 0.0), field)
        pair = _null_pair(context)
        assert pair is not None
        field.jacobians = 0
        _branch_step(context, pair)
        assert field.jacobians <= 8, field.jacobians
