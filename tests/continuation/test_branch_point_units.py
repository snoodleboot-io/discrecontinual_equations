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

from discrecontinual_equations.continuation import detection
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.continuation.continuation_state import (
    ContinuationState,
)
from discrecontinual_equations.continuation.detection import (
    DetectionContext,
    _branch_label,
    _branch_step,
    _null_pair,
    _vertex_offset,
)
from discrecontinual_equations.continuation.jacobian import (
    FiniteDifferenceJacobian,
    JacobianProvider,
)
from discrecontinual_equations.continuation.residual import DifferentiableResidual
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


def _continued(coefficients, scale=1.0, sigma=1.0, exact=True):
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


def _context(coefficients, sigma=1.0, exact=True, u0=0.0, mu_shift=0.0):
    """A point on the branch through ``u0`` (``u0 = 0`` is the trivial branch)."""
    r, al, be, k, d = coefficients
    v, w = 1.5 * d * u0 * u0, 0.5 * d * u0 * u0
    mu = -(r * u0 + al * v + be * w + k * u0 * u0) + mu_shift
    field = _Field(coefficients, 1.0, sigma)
    provider = (
        field if exact else FiniteDifferenceJacobian(field, _epsilon(sigma))
    )
    state = sigma * (M @ np.array([u0, v, w]))
    tangent = np.array([0.0, 0.0, 0.0, 1.0])
    return DetectionContext(ContinuationState(state, mu, tangent, 0.0), provider)


class TestRelativeValueStop(TestCase):
    """A relative value stop finds the same root whatever the function's units."""

    @staticmethod
    def _root(finder_class, slope: float, relative: bool) -> float:
        settings = RootFinderSettings(1.0e-8, 1.0e-12, 60, relative=relative)
        finder = finder_class(settings)
        located = finder.locate(lambda x: slope * (x - 0.3), 0.0, 1.0)
        assert located is not None
        return located

    def test_bisection_locates_the_root_at_every_scale(self):
        for slope in (1.0e-18, 1.0e-9, 1.0, 1.0e9):
            assert abs(self._root(Bisection, slope, True) - 0.3) < LOCATION_TOLERANCE

    def test_an_absolute_stop_accepts_the_first_midpoint_of_a_slow_function(self):
        assert abs(self._root(Bisection, 1.0e-9, False) - 0.5) < LOCATION_TOLERANCE

    def test_secant_locates_the_root_at_every_scale(self):
        for slope in (1.0e-18, 1.0e-9, 1.0, 1.0e9):
            assert abs(self._root(Secant, slope, True) - 0.3) < LOCATION_TOLERANCE

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
                    label = _branch_label(_context(coefficients, sigma, exact))
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

    def test_a_jacobian_that_does_not_move_grows_the_step(self):
        context = _context(PITCHFORKS["reflection symmetric"], 1.0e6)
        pair = _null_pair(context)
        assert pair is not None
        assert _branch_step(context, pair) > 1.0e4 * detection._BRANCH_STEP


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
