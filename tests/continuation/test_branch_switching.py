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
from discrecontinual_equations.continuation.switching import BranchSwitchError
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


class TestPlanarBranchPointSwitch(TestCase):
    def setUp(self):
        self.equation, self.continuer, self.branch_point = _branch_point(
            RotatedPitchfork,
            [X(), Y()],
            ("pitchfork", "transcritical", "branch_point"),
        )

    def test_branch_point_has_a_nonzero_extended_jacobian(self):
        # Guard on the example: the generic (non-zero Jacobian) path is exercised.
        assert abs(self.branch_point.parameter) < 1.0e-6

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
        equation, continuer, _ = _branch_point(
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
