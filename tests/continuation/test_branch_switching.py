"""Branch switching reaches the secondary branch, or refuses; never the old one.

The examples are chosen so the secondary branch is known in closed form:

* pitchfork ``x' = mu x - x^3``: trivial branch ``x = 0``, secondary ``mu = x^2``;
* transcritical ``x' = mu x - x^2``: trivial branch ``x = 0``, secondary ``x = mu``;
* a planar pitchfork in rotated coordinates ``a = x + y``, ``b = x - y`` with
  ``a' = mu a - a^3`` and ``b' = -b + a^2``: trivial branch ``x = y = 0``,
  secondary ``a^2 = mu``, ``b = mu``. Its extended Jacobian at the branch point
  is not zero, so the kernel and the left null vector are both non-trivial.

Every switch is run at the configuration's defaults: the kick is the default
``initial_step`` of 0.05, at which the old fixed-kick fallback returned the
trivial branch of the pitchfork.
"""

import math
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.continuation.continuation_point import ContinuationPoint
from discrecontinual_equations.continuation.corrector import NewtonSettings
from discrecontinual_equations.continuation.jacobian import FiniteDifferenceJacobian
from discrecontinual_equations.continuation.linear_solver import (
    LeastSquaresFallbackSolver,
)
from discrecontinual_equations.continuation.residual import EquationResidual
from discrecontinual_equations.continuation.switching import (
    BranchSwitchError,
    NullSpaceBranchSwitcher,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.test_continuer import (
    Pitchfork,
    Transcritical,
    X,
    Y,
    build_equation,
)

RESIDUAL_TOLERANCE = 1.0e-8
CURVE_TOLERANCE = 1.0e-6
# The secondary branch must be traced well away from the branch point.
FAR_FROM_BRANCH_POINT = 0.5


class RotatedPitchfork(DeterministicFunction):
    """A planar pitchfork whose symmetric direction is the diagonal ``x = y``."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        a = point[0] + point[1]
        b = point[0] - point[1]
        a_rate = mu * a - a**3
        b_rate = -b + a * a
        return [0.5 * (a_rate + b_rate), 0.5 * (a_rate - b_rate)]


class SkewCrossing(DeterministicFunction):
    """x' = (x - mu)(x - 2 mu): branches x = mu and x = 2 mu, not orthogonal."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x = point[0]
        return [(x - mu) * (x - 2.0 * mu)]


class TripleCrossing(DeterministicFunction):
    """x' = x^3 - x mu^2: three branches x = 0, x = mu, x = -mu cross at 0."""

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x = point[0]
        return [x**3 - x * mu * mu]


class SkewPlanar(DeterministicFunction):
    """x' = (x - mu)(x - 2 mu), y' = -y + x^2: a planar skew crossing.

    Primary branch x = mu, y = mu^2 (tangent (1, 0, 1)); secondary x = 2 mu,
    y = 4 mu^2 (tangent (2, 0, 1)). The left null vector is the x row only.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x, y = point[0], point[1]
        return [(x - mu) * (x - 2.0 * mu), -y + x * x]


class OneSidedTranscritical(DeterministicFunction):
    """x' = x (mu - 30 mu^2 - x): the secondary branch folds back at mu = 1/60.

    On the side mu > 0 the hyperplane normal to the secondary tangent through
    the default kick misses the branch, so only the other side can be reached.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        mu = self.parameters[0].value
        x = point[0]
        return [x * (mu - 30.0 * mu * mu - x)]


def _point(state, parameter, tangent) -> ContinuationPoint:
    return ContinuationPoint(
        arclength=0.0,
        state=list(state),
        parameter=parameter,
        tangent=list(tangent),
        eigenvalues=[(0.0, 0.0)] * len(state),
        unstable_dimension=0,
        stability="stable",
        measure=0.0,
    )


def _switcher(function_type, variables, kick=0.05) -> NullSpaceBranchSwitcher:
    residual = EquationResidual(build_equation(function_type, variables), 0, 0.0)
    return NullSpaceBranchSwitcher(
        residual=residual,
        jacobian=FiniteDifferenceJacobian(residual, 1.0e-7),
        linear_solver=LeastSquaresFallbackSolver(),
        kick=kick,
        singular_tolerance=1.0e-6,
        settings=NewtonSettings(1.0e-10, 12),
        direction=1,
    )


def _unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    return array / np.linalg.norm(array)


def _default_config(**overrides: float) -> ContinuationConfig:
    settings: dict = {
        "detectors": ["branch_point"],
        "initial_parameter": -2.0,
        "parameter_lower_bound": -2.0,
        "parameter_upper_bound": 2.0,
    }
    settings.update(overrides)
    return ContinuationConfig(**settings)


def _branch_point(function_type, variables, kinds):
    equation = build_equation(function_type, variables)
    continuer = ContinuerBuilder.build(_default_config(), equation)
    branch = continuer.solve(equation, initial_values=[0.0] * len(variables))
    located = [point for point in branch.special_points if point.kind in kinds]
    assert located, "the primary sweep must detect the branch point"
    return equation, continuer, located[0]


def _trace_from(equation, seed_state, seed_parameter, direction):
    config = _default_config(
        initial_parameter=float(seed_parameter),
        direction=direction,
    )
    continuer = ContinuerBuilder.build(config, equation)
    return continuer.solve(equation, initial_values=list(seed_state))


def _primary_distance(seed_state, seed_parameter, branch_point):
    """Distance of the seed from the primary branch's tangent line."""
    displacement = np.concatenate(
        [
            np.asarray(seed_state) - np.asarray(branch_point.state),
            [seed_parameter - branch_point.parameter],
        ],
    )
    tangent = np.asarray(branch_point.tangent)
    tangent = tangent / np.linalg.norm(tangent)
    return float(np.linalg.norm(displacement - (displacement @ tangent) * tangent))


class TestPitchforkSwitch(TestCase):
    def setUp(self):
        self.equation, self.continuer, self.branch_point = _branch_point(
            Pitchfork,
            [X()],
            ("pitchfork", "branch_point"),
        )

    def test_seed_lies_on_the_secondary_branch_not_the_trivial_one(self):
        state, parameter = self.continuer.seed_secondary_branch(self.branch_point)

        x = float(state[0])
        assert abs(x) > 0.5 * ContinuationConfig().initial_step
        assert abs(parameter - x * x) < CURVE_TOLERANCE
        assert abs(parameter * x - x**3) < RESIDUAL_TOLERANCE

    def test_continuing_from_the_seed_traces_the_secondary_branch(self):
        state, parameter = self.continuer.seed_secondary_branch(self.branch_point)

        branch = _trace_from(self.equation, state, parameter, direction=1)

        xs = np.array([point.state[0] for point in branch.points])
        mus = np.array([point.parameter for point in branch.points])
        assert np.max(np.abs(xs)) > FAR_FROM_BRANCH_POINT
        assert np.max(np.abs(mus - xs * xs)) < CURVE_TOLERANCE
        assert np.min(np.abs(xs)) > 0.0


class TestTranscriticalSwitch(TestCase):
    def setUp(self):
        self.equation, self.continuer, self.branch_point = _branch_point(
            Transcritical,
            [X()],
            ("transcritical", "branch_point"),
        )

    def test_seed_lies_on_the_secondary_branch_not_the_trivial_one(self):
        state, parameter = self.continuer.seed_secondary_branch(self.branch_point)

        x = float(state[0])
        assert abs(x) > 0.25 * ContinuationConfig().initial_step
        assert abs(parameter - x) < CURVE_TOLERANCE

    def test_continuing_from_the_seed_traces_the_secondary_branch(self):
        state, parameter = self.continuer.seed_secondary_branch(self.branch_point)

        branch = _trace_from(self.equation, state, parameter, direction=1)

        xs = np.array([point.state[0] for point in branch.points])
        mus = np.array([point.parameter for point in branch.points])
        assert np.max(np.abs(xs)) > FAR_FROM_BRANCH_POINT
        assert np.max(np.abs(mus - xs)) < CURVE_TOLERANCE

    def test_seed_is_on_the_side_continuation_leaves_by(self):
        # The secondary branch x = mu crosses the parameter transversally: with
        # direction +1 the seed must have mu > 0, so continuing from it moves
        # away from the branch point instead of back through it.
        for direction in (1, -1):
            equation = build_equation(Transcritical, [X()])
            config = _default_config(direction=direction)
            continuer = ContinuerBuilder.build(config, equation)
            branch_point = self.branch_point
            _state, parameter = continuer.seed_secondary_branch(branch_point)
            assert parameter * direction > 0.0, direction


class TestPlanarBranchPointSwitch(TestCase):
    def setUp(self):
        self.equation, self.continuer, self.branch_point = _branch_point(
            RotatedPitchfork,
            [X(), Y()],
            ("pitchfork", "transcritical", "branch_point"),
        )

    def test_branch_point_has_a_nonzero_extended_jacobian(self):
        # Guard on the example: the generic (non-zero Jacobian) path is exercised.
        real_parts = sorted(real for real, _imag in self.branch_point.eigenvalues)
        assert abs(real_parts[0] + 1.0) < 1.0e-6
        assert abs(real_parts[1]) < 1.0e-6

    def test_seed_lies_on_the_secondary_branch_not_the_trivial_one(self):
        state, parameter = self.continuer.seed_secondary_branch(self.branch_point)

        a = float(state[0] + state[1])
        b = float(state[0] - state[1])
        assert abs(a) > 0.25 * ContinuationConfig().initial_step
        assert abs(parameter - a * a) < CURVE_TOLERANCE
        assert abs(b - a * a) < CURVE_TOLERANCE

    def test_continuing_from_the_seed_traces_the_secondary_branch(self):
        state, parameter = self.continuer.seed_secondary_branch(self.branch_point)

        branch = _trace_from(self.equation, state, parameter, direction=1)

        states = np.array([point.state for point in branch.points])
        mus = np.array([point.parameter for point in branch.points])
        a = states[:, 0] + states[:, 1]
        assert np.max(np.abs(a)) > FAR_FROM_BRANCH_POINT
        assert np.max(np.abs(mus - a * a)) < CURVE_TOLERANCE


class TestNoSilentReturn(TestCase):
    """A switch either reaches a different branch or raises; it never lies."""

    def test_every_kick_size_reaches_the_secondary_branch_or_refuses(self):
        for step in (0.005, 0.02, 0.05, 0.1, 0.2, 0.4):
            equation = build_equation(Pitchfork, [X()])
            config = _default_config(initial_step=step)
            continuer = ContinuerBuilder.build(config, equation)
            branch = continuer.solve(equation, initial_values=[0.0])
            branch_point = branch.special_points[0]
            try:
                state, parameter = continuer.seed_secondary_branch(branch_point)
            except BranchSwitchError:
                continue
            assert abs(state[0]) > 0.5 * step * 0.5, step
            assert abs(parameter - state[0] ** 2) < CURVE_TOLERANCE, step

    def test_switching_at_a_regular_point_refuses(self):
        _equation, continuer, _ = _branch_point(
            Pitchfork,
            [X()],
            ("pitchfork", "branch_point"),
        )
        regular = ContinuationPoint(
            arclength=0.0,
            state=[0.0],
            parameter=-1.0,
            tangent=[0.0, 1.0],
            eigenvalues=[(-1.0, 0.0)],
            unstable_dimension=0,
            stability="stable",
            measure=0.0,
        )

        with pytest.raises(BranchSwitchError):
            continuer.seed_secondary_branch(regular)

    def test_seed_is_off_the_primary_branch(self):
        for function_type, variables, kinds in (
            (Pitchfork, [X()], ("pitchfork", "branch_point")),
            (Transcritical, [X()], ("transcritical", "branch_point")),
            (RotatedPitchfork, [X(), Y()], ("pitchfork", "transcritical")),
        ):
            _equation, continuer, branch_point = _branch_point(
                function_type,
                variables,
                kinds,
            )
            state, parameter = continuer.seed_secondary_branch(branch_point)
            kick = ContinuationConfig().initial_step
            distance = _primary_distance(state, parameter, branch_point)
            assert distance > 0.25 * kick * math.sqrt(0.5), function_type


class TestSecondaryDirection(TestCase):
    """The branching equation's second root against hand-derived tangents.

    Each reference tangent is differentiated by hand from the closed-form
    secondary branch, independently of the code under test.
    """

    def _assert_parallel(self, computed, expected):
        cosine = abs(float(_unit(computed) @ _unit(expected)))
        assert cosine > 1.0 - 1.0e-8, (computed, expected)

    def test_pitchfork_secondary_tangent_is_the_state_axis(self):
        # Branch mu = x^2: tangent (1, 0) at the origin.
        direction = _switcher(Pitchfork, [X()]).secondary_direction(
            np.zeros(2),
            np.array([0.0, 1.0]),
        )
        self._assert_parallel(direction, [1.0, 0.0])

    def test_transcritical_secondary_tangent_is_the_diagonal(self):
        # Branch x = mu: tangent (1, 1) / sqrt(2).
        direction = _switcher(Transcritical, [X()]).secondary_direction(
            np.zeros(2),
            np.array([0.0, 1.0]),
        )
        self._assert_parallel(direction, [1.0, 1.0])

    def test_skew_crossing_uses_both_roots_not_the_orthogonal_complement(self):
        # Primary x = mu, secondary x = 2 mu: tangent (2, 1), which is not the
        # unit vector orthogonal to the primary tangent (1, 1).
        direction = _switcher(SkewCrossing, [X()]).secondary_direction(
            np.zeros(2),
            _unit([1.0, 1.0]),
        )
        self._assert_parallel(direction, [2.0, 1.0])

    def test_skew_crossing_from_the_other_branch(self):
        direction = _switcher(SkewCrossing, [X()]).secondary_direction(
            np.zeros(2),
            _unit([2.0, 1.0]),
        )
        self._assert_parallel(direction, [1.0, 1.0])

    def test_rotated_pitchfork_secondary_tangent(self):
        # Branch a = s, b = mu = s^2: x = (a + b) / 2, y = (a - b) / 2, so the
        # tangent at the origin is (1/2, 1/2, 0).
        direction = _switcher(RotatedPitchfork, [X(), Y()]).secondary_direction(
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
        )
        self._assert_parallel(direction, [1.0, 1.0, 0.0])

    def test_skew_crossing_switch_lands_on_the_other_line(self):
        state, parameter = _switcher(SkewCrossing, [X()]).secondary_seed(
            ContinuationPoint(
                arclength=0.0,
                state=[0.0],
                parameter=0.0,
                tangent=list(_unit([1.0, 1.0])),
                eigenvalues=[(0.0, 0.0)],
                unstable_dimension=0,
                stability="stable",
                measure=0.0,
            ),
        )
        assert abs(state[0] - 2.0 * parameter) < CURVE_TOLERANCE
        assert abs(parameter) > 0.01

    def test_degenerate_crossing_refuses_rather_than_returning_x_zero(self):
        # All second derivatives vanish at the origin, so the branching equation
        # is degenerate (not a simple branch point). The kernel complement is
        # tried; whatever happens, the trivial branch x = 0 is never returned.
        switcher = _switcher(TripleCrossing, [X()])
        assert np.allclose(
            switcher.secondary_direction(np.zeros(2), np.array([0.0, 1.0])),
            [1.0, 0.0],
        ) or np.allclose(
            switcher.secondary_direction(np.zeros(2), np.array([0.0, 1.0])),
            [-1.0, 0.0],
        )
        try:
            state, parameter = switcher.secondary_seed(
                ContinuationPoint(
                    arclength=0.0,
                    state=[0.0],
                    parameter=0.0,
                    tangent=[0.0, 1.0],
                    eigenvalues=[(0.0, 0.0)],
                    unstable_dimension=0,
                    stability="stable",
                    measure=0.0,
                ),
            )
        except BranchSwitchError:
            return
        assert abs(abs(state[0]) - abs(parameter)) < CURVE_TOLERANCE
        assert abs(state[0]) > 0.01

    def test_planar_skew_crossing_projects_on_the_left_null_vector(self):
        direction = _switcher(SkewPlanar, [X(), Y()]).secondary_direction(
            np.zeros(3),
            _unit([1.0, 0.0, 1.0]),
        )
        self._assert_parallel(direction, [2.0, 0.0, 1.0])

    def test_inaccurate_primary_tangent_still_gives_the_exact_secondary(self):
        # In one dimension the kernel is the whole plane, so both roots of the
        # branching equation are exact even when q1 is 0.2 rad off the primary
        # tangent; the root farther from q1 is still the other branch.
        angle = math.atan2(1.0, 1.0) + 0.2
        tangent = np.array([math.cos(angle), math.sin(angle)])
        direction = _switcher(SkewCrossing, [X()]).secondary_direction(
            np.zeros(2),
            tangent,
        )
        self._assert_parallel(direction, [2.0, 1.0])


class TestRetries(TestCase):
    """The switch retries the other side and larger kicks before refusing."""

    def test_other_side_is_tried_when_the_first_misses_the_branch(self):
        state, parameter = _switcher(OneSidedTranscritical, [X()]).secondary_seed(
            _point([0.0], 0.0, [0.0, 1.0]),
        )
        assert parameter < 0.0
        assert abs(state[0] - (parameter - 30.0 * parameter**2)) < CURVE_TOLERANCE

    def test_larger_kicks_are_tried_from_an_inaccurately_located_point(self):
        # The point is 0.02 past the pitchfork; with a 0.005 kick the corrected
        # point (h, h^2) is farther than the kick from the prediction until the
        # kick reaches 0.02.
        state, parameter = _switcher(Pitchfork, [X()], kick=0.005).secondary_seed(
            _point([0.0], 0.02, [0.0, 1.0]),
        )
        assert abs(parameter - state[0] ** 2) < CURVE_TOLERANCE
        assert abs(state[0]) >= 0.02 - 1.0e-12

    def test_a_branch_far_from_the_point_is_not_accepted(self):
        # (0, -0.3) is a regular point; correcting along x = h reaches the
        # secondary branch at mu = h^2, farther than the kick from the
        # prediction, so it is not certified as the branch crossing here.
        with pytest.raises(BranchSwitchError):
            _switcher(Pitchfork, [X()]).secondary_seed(
                _point([0.0], -0.3, [0.0, 1.0]),
            )
