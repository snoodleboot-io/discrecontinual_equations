"""Branch points on hard brackets, stiff fields and turning branches (DEQ-179, 178, 168).

* DEQ-179: the relative root-finder stop measured ``|f|`` against the bracket's end
  values. On ``sinh(40 (x - 0.37))`` that accepted the first midpoint, 0.13 from
  the root, and an end value of ``inf`` accepted any midpoint.
* DEQ-178: the difference step for ``a`` and ``c`` was scaled to ``|J|``, so a fast
  mode ``-K`` grew it with ``K`` (0.41 at ``K = 1e8``) and both labels were lost;
  and a transcritical seen far out on its nontrivial branch was named pitchfork.
* DEQ-168: localizing from a pitchfork's nontrivial branch jumped to the trivial
  one (``mu* = 2.9e-3``), and the unweighted arclength detected nothing once the
  state was rescaled by ``1e3``.

The reference field is the one of ``test_branch_point_units.py``; the stiff one is
``u' = mu u + r tanh^2 u + k tanh^3 u``, ``v' = -K v + u^2``, whose branch point
at the origin has ``a = 2 r`` and ``c = 6 k``.
"""

import math
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
    BranchPoint,
    DetectionContext,
    _branch_step,
    _jacobian_step,
    _null_pair,
)
from discrecontinual_equations.continuation.jacobian import (
    FiniteDifferenceJacobian,
    JacobianProvider,
)
from discrecontinual_equations.continuation.localizer import _hermite
from discrecontinual_equations.continuation.parameterization import (
    CorrectionStep,
    PseudoArclength,
)
from discrecontinual_equations.continuation.residual import ResidualFunction
from discrecontinual_equations.continuation.root_finder import (
    Bisection,
    RootFinderSettings,
    Secant,
)
from tests.continuation.test_branch_point_units import (
    MI,
    PITCHFORKS,
    TRANSCRITICAL,
    M,
    _equation,
    _Field,
)

TOLERANCE = 1.0e-8
ROOT = 0.37
STEEP = {
    "sinh(40 (x - 0.37))": lambda x: math.sinh(40.0 * (x - ROOT)),
    "sinh(200 (x - 0.37))": lambda x: math.sinh(200.0 * (x - ROOT)),
    "exp(200 (x - 0.37)) - 1": lambda x: math.expm1(200.0 * (x - ROOT)),
    "1 - exp(200 (0.37 - x))": lambda x: -math.expm1(200.0 * (ROOT - x)),
}
# Bisection from [0, 1] to a bracket of 2e-8: 26 halvings, the midpoint and the ends.
BISECTION_EVALUATIONS = 29
STIFFNESS = (1.0, 1.0e8, 1.0e10)
SHEAR = np.array([[1.0, 0.4], [-0.3, 1.0]])
VERTEX_TOLERANCE = 1.0e-6
SCALE_TOLERANCE = 1.0e-6
NEWTON_TOLERANCE = 1.0e-9


def _settings(*, relative: bool = True) -> RootFinderSettings:
    return RootFinderSettings(TOLERANCE, 1.0e-12, 40, relative=relative)


class TestRelativeStopIsOnTheLocation(TestCase):
    """DEQ-179: a relative stop returns a point within tolerance of the sign change."""

    def test_bisection_finds_steep_roots(self):
        for name, function in STEEP.items():
            located = Bisection(_settings()).locate(function, 0.0, 1.0)
            assert located is not None
            assert abs(located - ROOT) <= TOLERANCE, (name, located)

    def test_secant_finds_steep_roots(self):
        for name, function in STEEP.items():
            located = Secant(_settings()).locate(function, 0.0, 1.0)
            assert located is not None
            assert abs(located - ROOT) <= TOLERANCE, (name, located)

    def test_bisection_stops_once_the_bracket_is_small(self):
        calls = []

        def function(x: float) -> float:
            calls.append(x)
            return math.sinh(40.0 * (x - ROOT))

        Bisection(_settings()).locate(function, 0.0, 1.0)
        assert len(calls) <= BISECTION_EVALUATIONS, len(calls)

    def test_bisection_returns_an_exact_zero_at_once(self):
        calls = []

        def function(x: float) -> float:
            calls.append(x)
            return 1.0e300 * (x - 0.5)

        assert Bisection(_settings()).locate(function, 0.0, 1.0) == 0.5
        assert len(calls) == 3

    def test_the_tolerance_is_a_fraction_of_the_bracket(self):
        located = Bisection(_settings()).locate(
            lambda x: math.sinh(40.0 * (x - 3.7)),
            0.0,
            10.0,
        )
        assert located is not None
        assert abs(located - 3.7) <= 10.0 * TOLERANCE
        assert _settings().location_tolerance(2.0, 12.0) == 10.0 * TOLERANCE

    def test_secant_hands_a_flat_chord_to_bisection(self):
        # A step function: every chord between two points on one side is flat.
        def step(x: float) -> float:
            return -1.0 if x < ROOT else 1.0

        located = Secant(_settings()).locate(step, 0.0, 1.0)
        assert located is not None
        assert abs(located - ROOT) <= TOLERANCE

    def test_secant_confirms_a_short_step_by_a_sign_change(self):
        # Flat then steep: the secant creeps along the plateau in short steps far
        # from the root; a short step alone must not stop it.
        def creeping(x: float) -> float:
            return math.expm1(400.0 * (x - ROOT)) - 1.0e-9 * (1.0 - x)

        located = Secant(_settings()).locate(creeping, 0.0, 1.0)
        assert located is not None
        assert abs(creeping(located)) < 1.0e-4

    def test_absolute_settings_keep_the_value_stop(self):
        located = Bisection(_settings(relative=False)).locate(
            lambda x: 1.0e-9 * (x - 0.3),
            0.0,
            1.0,
        )
        assert located == 0.5


class TestNonFiniteEndsAreRefused(TestCase):
    """DEQ-179: an infinite or undefined end value is refused, not bisected."""

    def test_both_finders_refuse_each_end(self):
        for finder in (Bisection, Secant):
            for bad in (math.inf, -math.inf, math.nan):
                for end in (0.0, 1.0):

                    def function(x: float, bad=bad, end=end) -> float:
                        return bad if x == end else x - 0.3

                    with self.assertRaises(ValueError):
                        finder(_settings()).locate(function, 0.0, 1.0)

    def test_absolute_bisection_refuses_the_end_it_evaluates(self):
        with self.assertRaises(ValueError):
            Bisection(_settings(relative=False)).locate(
                lambda x: math.nan if x == 0.0 else x - 0.3,
                0.0,
                1.0,
            )


class _Stiff(ResidualFunction, JacobianProvider):
    """``u' = mu u + r tanh^2 u + k tanh^3 u``, ``v' = -K v + u^2``, sheared."""

    def __init__(self, quadratic, cubic, stiffness, *, mixed) -> None:
        self._quadratic = quadratic
        self._cubic = cubic
        self._stiffness = stiffness
        self._mix = SHEAR if mixed else np.eye(2)
        self._unmix = np.linalg.inv(self._mix)

    def evaluate(self, state, parameter):
        u, v = self._unmix @ np.asarray(state, dtype=float)
        t = math.tanh(u)
        rows = [
            parameter * u + self._quadratic * t * t + self._cubic * t**3,
            -self._stiffness * v + u * u,
        ]
        return self._mix @ np.array(rows)

    def state_jacobian(self, state, parameter):
        u, _v = self._unmix @ np.asarray(state, dtype=float)
        t = math.tanh(u)
        slope = (2.0 * self._quadratic * t + 3.0 * self._cubic * t * t) * (1 - t * t)
        rows = np.array([[parameter + slope, 0.0], [2.0 * u, -self._stiffness]])
        return self._mix @ rows @ self._unmix

    def parameter_derivative(self, state, parameter):
        _ = parameter
        u, _v = self._unmix @ np.asarray(state, dtype=float)
        return self._mix @ np.array([u, 0.0])


def _origin(provider) -> DetectionContext:
    tangent = np.array([0.0, 0.0, 1.0])
    return DetectionContext(ContinuationState(np.zeros(2), 0.0, tangent, 0.0), provider)


class TestStiffFieldsKeepTheirLabels(TestCase):
    """DEQ-178: a fast mode does not set the difference step."""

    def _labels(self, quadratic, *, mixed, exact):
        labels = []
        for stiffness in STIFFNESS:
            field = _Stiff(quadratic, -1.0, stiffness, mixed=mixed)
            provider = field if exact else FiniteDifferenceJacobian(field, 1.0e-7)
            labels.append(BranchPoint().refine_kind(_origin(provider)))
        return labels

    def test_transcritical_up_to_k_1e10(self):
        for mixed in (False, True):
            for exact in (True, False):
                labels = self._labels(1.0, mixed=mixed, exact=exact)
                assert labels == ["transcritical"] * len(STIFFNESS), (mixed, exact)

    def test_pitchfork_up_to_k_1e10(self):
        for mixed in (False, True):
            for exact in (True, False):
                labels = self._labels(0.0, mixed=mixed, exact=exact)
                assert labels == ["pitchfork"] * len(STIFFNESS), (mixed, exact)

    def test_the_step_stays_inside_the_nonlinear_length(self):
        # tanh is close to its Taylor polynomial only for |u| well below 1.
        context = _origin(_Stiff(1.0, -1.0, 1.0e10, mixed=False))
        pair = _null_pair(context)
        assert pair is not None
        assert _jacobian_step(context, pair) > 1.0
        assert _branch_step(context, pair) < 0.1

    def test_a_step_that_is_right_on_the_jacobian_is_kept(self):
        context = _origin(_Stiff(1.0, -1.0, 1.0, mixed=True))
        pair = _null_pair(context)
        assert pair is not None
        assert _branch_step(context, pair) == _jacobian_step(context, pair)

    def test_a_step_already_agreeing_to_the_threshold_is_kept(self):
        # At K = 1e4 the Jacobian's step gives coefficients that agree with half
        # of it to 1.7e-5 on the block: inside the 1e-3 threshold, so it stays.
        context = _origin(_Stiff(1.0, -1.0, 1.0e4, mixed=False))
        pair = _null_pair(context)
        assert pair is not None
        assert _branch_step(context, pair) == _jacobian_step(context, pair)

    def test_a_purely_quadratic_transcritical_at_k_1e8(self):
        # tanh^2 has no cubic term: the block's change is a h alone.
        field = _Stiff(1.0, 0.0, 1.0e8, mixed=False)
        assert BranchPoint().refine_kind(_origin(field)) == "transcritical"

    def test_a_cancelled_cubic_keeps_the_jacobian_step(self):
        # a = 0 and c cancelled: the block's ratio is rounding over rounding at
        # every rung, so no rung is preferred to the Jacobian's own step.
        cancelled = (0.0, 1.0, -2.0, -0.5, 1.0)
        context = _on_branch(cancelled, 0.0)
        pair = _null_pair(context)
        assert pair is not None
        assert _branch_step(context, pair) == _jacobian_step(context, pair)

    def test_a_block_that_never_settles_keeps_the_jacobian_step(self):
        # Linear along q: a = c = 0 at every step.
        class _Linear(ResidualFunction, JacobianProvider):
            def evaluate(self, state, parameter):
                return np.array([parameter * state[0], -state[1]])

            def state_jacobian(self, state, parameter):
                _ = state
                return np.array([[parameter, 0.0], [0.0, -1.0]])

            def parameter_derivative(self, state, parameter):
                _ = parameter
                return np.array([state[0], 0.0])

        context = _origin(_Linear())
        pair = _null_pair(context)
        assert pair is not None
        assert _branch_step(context, pair) == _jacobian_step(context, pair)
        assert BranchPoint().refine_kind(context) == "branch_point"


def _on_branch(coefficients, u0, *, mu_shift=0.0) -> DetectionContext:
    r, al, be, k, d = coefficients
    v, w = 1.5 * d * u0 * u0, 0.5 * d * u0 * u0
    mu = -(r * u0 + al * v + be * w + k * u0 * u0) + mu_shift
    tangent = np.array([0.0, 0.0, 0.0, 1.0])
    state = M @ np.array([u0, v, w])
    return DetectionContext(
        ContinuationState(state, mu, tangent, 0.0),
        _Field(coefficients),
    )


class TestOffVertexNames(TestCase):
    """DEQ-178: the eigenvalue tells a far transcritical from a pitchfork."""

    def test_a_far_transcritical_is_not_a_pitchfork(self):
        # |a - c w0| / |a| = 0.37 passed the vertex test; 3 lambda / (a w0) = 1.26.
        label = BranchPoint().refine_kind(_on_branch(TRANSCRITICAL, 0.2))
        assert label != "pitchfork"

    def test_pitchforks_off_their_vertex_keep_their_name(self):
        for name, coefficients in PITCHFORKS.items():
            for u0 in (1.0e-2, 5.0e-2, 0.1):
                label = BranchPoint().refine_kind(_on_branch(coefficients, u0))
                assert label == "pitchfork", (name, u0)

    def test_a_point_just_off_the_persisting_branch_is_a_pitchfork(self):
        # A corrector tolerance off u = 0 near the vertex: a = c w0 is tiny and
        # lambda = mu, so 3 lambda = a w0 does not apply there.
        coefficients = PITCHFORKS["reflection symmetric"]
        context = _on_branch(coefficients, 1.0e-9, mu_shift=-1.0e-7)
        assert BranchPoint().refine_kind(context) == "pitchfork"


def _nontrivial(coefficients, sigma=1.0, scale=1.0, **settings):
    """Continue the pitchfork's nontrivial branch from u0 = 0.3 through its vertex."""
    r, al, be, k, d = coefficients
    u0 = 0.3
    v, w = 1.5 * d * u0 * u0, 0.5 * d * u0 * u0
    mu0 = -(r * u0 + al * v + be * w + k * u0 * u0)
    config = ContinuationConfig(
        continuation_parameter_index=0,
        initial_parameter=mu0,
        direction=1 if mu0 < 0 else -1,
        detectors=["branch_point"],
        maximum_points=80,
        initial_step=0.01,
        maximum_step=0.02,
        parameter_lower_bound=-1.0,
        parameter_upper_bound=1.0,
        jacobian="analytic_jacobian",
        **settings,
    )
    seed = sigma * (M @ np.array([u0, v, w]))
    continuer = ContinuerBuilder.assemble(config, _Field(coefficients, scale, sigma))
    return continuer.solve(_equation(), list(seed))


class TestLocalizationStaysOnTheBranch(TestCase):
    """DEQ-168: from the nontrivial branch the localizer lands on the vertex."""

    def test_each_pitchfork_is_located_at_its_vertex(self):
        for name, coefficients in PITCHFORKS.items():
            special = _nontrivial(coefficients).special_points
            assert [point.kind for point in special] == ["pitchfork"], name
            point = special[0]
            assert abs(point.parameter) < VERTEX_TOLERANCE, (name, point.parameter)
            mode = MI @ np.asarray(point.state, dtype=float)
            assert abs(mode[0]) < VERTEX_TOLERANCE, (name, mode)

    def test_the_cubic_passes_through_both_ends_with_their_tangents(self):
        # A quarter circle of radius 1 in (u, mu), sampled 0.3 rad apart; an
        # independent reference: the circle itself.
        angle = 0.3
        lower = ContinuationState(
            np.array([1.0]),
            0.0,
            np.array([0.0, 1.0]),
            0.0,
        )
        upper = ContinuationState(
            np.array([math.cos(angle)]),
            math.sin(angle),
            np.array([-math.sin(angle), math.cos(angle)]),
            angle,
        )
        chord = math.hypot(1.0 - math.cos(angle), math.sin(angle))
        start = _hermite(lower, upper, chord, 0.0)
        end = _hermite(lower, upper, chord, 1.0)
        assert np.allclose(start, [1.0, 0.0], atol=1.0e-15)
        assert np.allclose(end, [math.cos(angle), math.sin(angle)], atol=1.0e-15)
        middle = _hermite(lower, upper, chord, 0.5)
        chord_middle = 0.5 * (np.array([1.0, 0.0]) + end)
        # The cubic is off the circle by O(step^4), the chord by O(step^2).
        assert abs(np.linalg.norm(middle) - 1.0) < 1.0e-4
        assert abs(np.linalg.norm(chord_middle) - 1.0) > 1.0e-2


class TestWeightedArclength(TestCase):
    """DEQ-168: with the state scale set, a rescaled state is continued alike."""

    def test_the_branch_point_is_found_at_every_state_scale(self):
        # The corrector's residual tolerance has the field's units, so it is
        # rescaled too; left at 1e-9, sigma = 1e-3 stops 4.7e-6 off the vertex.
        coefficients = PITCHFORKS["against the cubic"]
        for sigma in (1.0e-3, 1.0e3, 1.0e6):
            branch = _nontrivial(
                coefficients,
                sigma,
                arclength_state_scale=sigma,
                newton_tolerance=NEWTON_TOLERANCE * sigma,
            )
            kinds = [point.kind for point in branch.special_points]
            assert kinds == ["pitchfork"], (sigma, kinds)
            assert abs(branch.special_points[0].parameter) < VERTEX_TOLERANCE

    def test_a_rescaled_branch_has_the_same_points(self):
        coefficients = PITCHFORKS["against the cubic"]
        plain = _nontrivial(coefficients)
        scaled = _nontrivial(coefficients, 1.0e3, arclength_state_scale=1.0e3)
        assert len(plain.points) == len(scaled.points)
        for one, other in zip(plain.points, scaled.points, strict=True):
            assert abs(one.parameter - other.parameter) < SCALE_TOLERANCE
            difference = np.asarray(other.state) / 1.0e3 - np.asarray(one.state)
            assert np.linalg.norm(difference) < SCALE_TOLERANCE

    def test_time_rescaling_needs_no_weight(self):
        coefficients = PITCHFORKS["against the cubic"]
        for scale in (1.0e-3, 1.0e3, 1.0e6):
            branch = _nontrivial(coefficients, scale=scale)
            kinds = [point.kind for point in branch.special_points]
            assert kinds == ["pitchfork"], (scale, kinds)

    def test_the_constraint_weights_the_state_only(self):
        tangent = np.array([3.0, 4.0, 2.0])
        step = CorrectionStep(
            np.array([1.0, 1.0]),
            1.0,
            np.zeros(2),
            0.0,
            tangent,
        )
        weighted = PseudoArclength(state_scale=10.0)
        assert abs(weighted.constraint(step) - (7.0 / 100.0 + 2.0)) < 1.0e-15
        assert np.allclose(weighted.border(step), [0.03, 0.04, 2.0], atol=1.0e-15)
        assert PseudoArclength().border(step) is tangent
        with self.assertRaises(ValueError):
            PseudoArclength(state_scale=0.0)

    def test_the_tangent_is_a_unit_vector_in_the_weighted_norm(self):
        branch = _nontrivial(
            PITCHFORKS["against the cubic"],
            1.0e3,
            arclength_state_scale=1.0e3,
        )
        tangent = np.asarray(branch.points[3].tangent, dtype=float)
        norm = float(tangent[:-1] @ tangent[:-1]) / 1.0e6 + tangent[-1] ** 2
        assert abs(norm - 1.0) < 1.0e-12
