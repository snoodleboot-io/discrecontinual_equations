"""Finite-difference typical sizes and steps through the configs (DEQ-158, DEQ-166).

DEQ-158: ``typical`` (and ``parameter_typical``) set the step of a central
difference for a component that lives far below one, ``h_j = r max(|x_j|,
typical_j)`` (Dennis and Schnabel 1983, Sec. 5.4). They existed on the
finite-difference classes, but no config passed them, so a problem whose states
sit near ``1e-12`` was stepped as if they were of size one. Now
``ContinuationConfig`` and every stochastic solver config accept them, and
``AnalyticJacobian`` hands them to the cusp and generalized-Hopf residuals.

The oracles are independent of the code under test: the branch-point labels of
the DEQ-146 reference field (checked against a sympy center-manifold reduction
in ``test_branch_label_scale``), a closed-form Jacobian, and, for the stochastic
solvers, the same path integrated in the Ito form with the Stratonovich
correction ``b b' / 2`` written out by hand.

DEQ-166: ``Codim2Config.curve_epsilon`` was never read; it is refused with a
``DeprecationWarning``. ``derivative_epsilon`` sets the finite-difference
derivative provider's step.
"""

import math
import warnings
from unittest import mock

import numpy as np
import pytest
from pydantic import ValidationError

from discrecontinual_equations.continuation import codim2_builder
from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.codim2_builder import (
    Codim2Driver,
    CurveSeed,
)
from discrecontinual_equations.continuation.codim2_config import Codim2Config
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.continuation.derivative_provider import (
    FiniteDifferenceDerivatives,
    configured_provider,
)
from discrecontinual_equations.continuation.jacobian import (
    AnalyticJacobian,
    JacobianProvider,
)
from discrecontinual_equations.continuation.residual import DifferentiableResidual
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.finite_difference import ORDER_STEPS, central_column
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.solver.stochastic.ensemble import StochasticEnsemble
from discrecontinual_equations.solver.stochastic.euler_maruyama import (
    euler_maruyama_config,
    euler_maruyama_solver,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_config import (
    MilsteinConfig,
)
from discrecontinual_equations.solver.stochastic.milstein.milstein_solver import (
    MilsteinSolver,
)
from discrecontinual_equations.solver.stochastic.srk2.srk2_config import SRK2Config
from discrecontinual_equations.solver.stochastic.srk2.srk2_solver import SRK2Solver
from discrecontinual_equations.solver.stochastic.srk3.srk3_config import SRK3Config
from discrecontinual_equations.solver.stochastic.srk3.srk3_solver import SRK3Solver
from discrecontinual_equations.solver.stochastic.srk4.srk4_config import SRK4Config
from discrecontinual_equations.solver.stochastic.srk4.srk4_solver import SRK4Solver
from discrecontinual_equations.solver.stochastic.wiener import BrownianPath
from discrecontinual_equations.solver.stochastic.zhan_duan_li_li import (
    zhan_duan_li_li_config,
    zhan_duan_li_li_solver,
)
from discrecontinual_equations.variable import Variable
from tests.continuation import test_codim2 as scenarios
from tests.continuation.test_branch_point_units import (
    PITCHFORKS,
    TRANSCRITICAL,
    _equation,
    _Field,
)
from tests.continuation.test_curve_residual_steps import (
    _bautin,
    _cusp,
    _moved_steps,
)

EulerMaruyamaConfig = euler_maruyama_config.EulerMaruyamaConfig
EulerMaruyamaSolver = euler_maruyama_solver.EulerMaruyamaSolver
ZhanDuanLiLiConfig = zhan_duan_li_li_config.ZhanDuanLiLiConfig
ZhanDuanLiLiSolver = zhan_duan_li_li_solver.ZhanDuanLiLiSolver
# The DEQ-158 acceptance scale.
_SIGMA = 1.0e-12
# Identity between two computations of the same differences (two paths to one
# provider, or a typical size of one against none): only the order of a few
# operations may differ, so a few ulps, pinned well above them.
_SAME = 1.0e-10
_CANCELLED = (0.0, 1.0, -2.0, -0.5, 1.0)
# A central difference at the default relative step 1e-7 rounds by about
# eps / 1e-7 = 2.2e-9 relative: an allowance for it, not for truncation.
_ROUNDING = 1.0e-8
_LABELS = {
    **{name: (coefficients, "pitchfork") for name, coefficients in PITCHFORKS.items()},
    "transcritical": (TRANSCRITICAL, "transcritical"),
    "cancelled": (_CANCELLED, "branch_point"),
}


def _config(**settings) -> ContinuationConfig:
    """Continue ``x = 0`` from ``mu = -0.3`` through the branch point at ``0``."""
    return ContinuationConfig(
        continuation_parameter_index=0,
        initial_parameter=-0.3,
        direction=1,
        detectors=["branch_point"],
        maximum_points=40,
        initial_step=0.017,
        maximum_step=0.03,
        parameter_lower_bound=-1.0,
        parameter_upper_bound=0.3,
        jacobian="finite_difference_jacobian",
        **settings,
    )


def _jacobian(config: ContinuationConfig, residual) -> JacobianProvider:
    """The Jacobian provider the builder wires from ``config``."""
    continuer = ContinuerBuilder.assemble(config, residual)
    return continuer._components.jacobian  # noqa: SLF001


def _labels(coefficients, sigma: float, **settings) -> list[str]:
    continuer = ContinuerBuilder.assemble(
        _config(**settings),
        _Field(coefficients, 1.0, sigma),
    )
    branch = continuer.solve(_equation(), [0.0, 0.0, 0.0])
    return [point.kind for point in branch.special_points]


class TestContinuationAtStateScale:
    """The DEQ-146 field with its state scaled by ``1e-12``, default epsilon."""

    @pytest.mark.parametrize("name", list(_LABELS))
    def test_typical_through_the_config_labels_every_case(self, name):
        coefficients, label = _LABELS[name]
        assert _labels(coefficients, _SIGMA, typical=_SIGMA) == [label]

    def test_without_it_the_branch_point_is_lost(self):
        # Step 1e-7 on a state of 1e-12: the Jacobian is the field's chord over
        # a hundred thousand times the state, and no branch point is seen
        # (at 1e-6 the pitchforks are mislabelled transcritical instead).
        coefficients, _label = _LABELS["through the curvature"]
        assert _labels(coefficients, _SIGMA) == []

    def test_the_jacobian_is_accurate_at_that_scale(self):
        # The field is s M F(M^-1 x / sigma); at x = sigma M (u, v, w) its
        # Jacobian is M F'(u, v, w) M^-1, independent of sigma: the reference.
        coefficients = PITCHFORKS["against the cubic"]
        field = _Field(coefficients, 1.0, _SIGMA)
        state = _SIGMA * np.array([0.3, -0.2, 0.5])
        exact = field.state_jacobian(state, 0.1)
        errors = {}
        for typical in (None, _SIGMA):
            jacobian = _jacobian(_config(typical=typical), field)
            computed = jacobian.state_jacobian(state, 0.1)
            errors[typical] = float(np.max(np.abs(computed - exact)))
        # The rounding of a central difference at relative step 1e-7 is about
        # eps / 1e-7 = 2.2e-9 relative: measured 7.1e-10 with typical, 1.8e9
        # without it.
        scale = float(np.max(np.abs(exact)))
        assert errors[_SIGMA] <= _ROUNDING * scale
        assert errors[None] > 1.0e3 * scale

    def test_a_typical_size_of_one_is_the_default(self):
        coefficients = PITCHFORKS["against the cubic"]
        field = _Field(coefficients, 1.0, 1.0)
        state = np.array([0.3, -0.2, 0.5])
        jacobians = [
            _jacobian(_config(**settings), field).extended_jacobian(state, 0.1)
            for settings in ({}, {"typical": 1.0, "parameter_typical": 1.0})
        ]
        assert np.allclose(jacobians[0], jacobians[1], rtol=_SAME, atol=0.0)

    def test_the_parameter_typical_sets_the_parameter_step(self):
        # f = x - (p / 1e-8)^3, a parameter that lives near 1e-8: the exact
        # slope is -3 (p / 1e-8)^2 / 1e-8; a step of 1e-7 is ten times p.
        class Cubic(DifferentiableResidual):
            def evaluate(self, state, parameter):
                return np.array([state[0] - (parameter / 1.0e-8) ** 3])

            def extended_jacobian(self, state, parameter, epsilon, typical=None):
                raise NotImplementedError

        parameter = 2.0e-8
        exact = -3.0 * (parameter / 1.0e-8) ** 2 / 1.0e-8
        slopes = {}
        for typical in (None, 1.0e-8):
            jacobian = _jacobian(_config(parameter_typical=typical), Cubic())
            slopes[typical] = float(
                jacobian.parameter_derivative(np.zeros(1), parameter)[0],
            )
        # Measured 2.7e-10 with the typical size, 8.3 without.
        assert abs(slopes[1.0e-8] - exact) <= _ROUNDING * abs(exact)
        assert abs(slopes[None] - exact) > abs(exact)

    @pytest.mark.parametrize("bad", [0.0, -1.0, [1.0, 0.0, 1.0]])
    def test_a_non_positive_typical_size_is_refused(self, bad):
        with pytest.raises(ValidationError):
            ContinuationConfig(typical=bad)

    def test_a_non_positive_parameter_typical_is_refused(self):
        with pytest.raises(ValidationError):
            ContinuationConfig(parameter_typical=0.0)


class _Recording(DifferentiableResidual):
    """``x_i - p``, recording the typical sizes ``AnalyticJacobian`` passes."""

    def __init__(self) -> None:
        self.seen: list[object] = []

    def evaluate(self, state, parameter):
        return np.asarray(state, dtype=float) - parameter

    def extended_jacobian(self, state, parameter, epsilon, typical=None):
        del parameter, epsilon
        self.seen.append(None if typical is None else np.array(typical))
        size = np.asarray(state).size
        return np.hstack([np.eye(size), -np.ones((size, 1))])


class _Legacy(DifferentiableResidual):
    """A residual written before DEQ-158: no ``typical`` keyword."""

    def evaluate(self, state, parameter):
        return np.array([state[0] - parameter])

    def extended_jacobian(self, state, parameter, epsilon):
        del state, parameter, epsilon
        return np.array([[1.0, -1.0]])


class TestAnalyticJacobian:
    def test_the_config_reaches_the_residual(self):
        residual = _Recording()
        config = _config(typical=1.0e-8, parameter_typical=1.0e-9).model_copy(
            update={"jacobian": "analytic_jacobian", "detectors": []},
        )
        ContinuerBuilder.assemble(config, residual).solve(
            _equation(),
            [-0.3, -0.3, -0.3],
        )
        assert residual.seen
        for typical in residual.seen:
            assert np.array_equal(typical, [1.0e-8, 1.0e-8, 1.0e-8, 1.0e-9])

    def test_unset_sizes_call_the_residual_as_before(self):
        legacy = AnalyticJacobian(_Legacy(), 1.0e-5)
        assert np.array_equal(
            legacy.extended_jacobian(np.zeros(1), 0.0),
            [[1.0, -1.0]],
        )
        recording = _Recording()
        AnalyticJacobian(recording, 1.0e-5).extended_jacobian(np.zeros(1), 0.0)
        assert recording.seen == [None]

    @pytest.mark.parametrize("curve", ["cusp", "generalized_hopf"])
    def test_the_curves_step_their_parameters_by_their_typical_size(self, curve):
        # At scale 1e-8 the three parameters are of size 1e-8: each moves
        # epsilon max(|p|, typical) - the typical size, here twice the scale,
        # not one.
        scale = 1.0e-8
        epsilon = 1.0e-5
        build = _cusp if curve == "cusp" else _bautin
        residual, state, parameter, field = build(scale)
        unknowns = np.append(state, parameter)
        typical = np.full(unknowns.size, 2.0 * scale)
        field.seen.clear()
        AnalyticJacobian(
            residual,
            epsilon,
            typical=typical[:-1],
            parameter_typical=float(typical[-1]),
        ).extended_jacobian(state, parameter)
        point = tuple(float(value) for value in unknowns[-3:])
        moved = _moved_steps(field, point)
        for index, distances in enumerate(moved):
            expected = epsilon * max(abs(point[index]), 2.0 * scale)
            assert distances
            assert max(distances) == pytest.approx(expected, rel=1.0e-6), index

    def test_without_sizes_the_columns_are_unchanged(self):
        # The DEQ-159 columns, bit for bit, when no typical size is given.
        residual, state, parameter, _field = _cusp(1.0)
        before = residual.extended_jacobian(state, parameter, 1.0e-5)
        after = AnalyticJacobian(
            residual,
            1.0e-5,
            typical=1.0,
            parameter_typical=1.0,
        ).extended_jacobian(state, parameter)
        assert np.allclose(before, after, rtol=_SAME, atol=0.0)


class TestCentralColumnTypical:
    def test_the_step_floors_at_the_typical_size(self):
        seen: list[float] = []

        def rows(state, parameter):
            seen.append(float(state[0]))
            del parameter
            return np.array([np.sin(state[0] / 1.0e-12)])

        state = np.array([3.0e-13])
        column = central_column(rows, state, 0.0, 0, 1.0e-6, 1.0e-12)
        assert seen[0] - state[0] == pytest.approx(1.0e-18, rel=1.0e-3)
        exact = np.cos(0.3) / 1.0e-12
        assert abs(column[0] - exact) <= 1.0e-6 * abs(exact)


class _Cubic(StochasticFunction):
    """``dX = b(X) o dW`` with ``b = X^3 / S^2``: nonlinear at the state's scale."""

    scale = _SIGMA

    def __init__(self, variables, parameters, time=None):
        super().__init__(
            variables,
            parameters,
            [Variable(name=f"d{v.name}/dt", discretization=[]) for v in variables],
            time,
        )

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        return [0.0 * x for x in point]

    def diffusion(self, point, time=None):  # noqa: ARG002 (base signature)
        return [x * x * x / self.scale**2 for x in point]

    def eval_batch(self, points, time=None):  # noqa: ARG002 (base signature)
        return 0.0 * points

    def diffusion_batch(self, points, time=None):  # noqa: ARG002 (base signature)
        return points**3 / self.scale**2


class _CorrectedCubic(_Cubic):
    """The same SDE in Ito form: drift ``b b' / 2 = 1.5 X^5 / S^4``, by hand."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        return [1.5 * x**5 / self.scale**4 for x in point]

    def eval_batch(self, points, time=None):  # noqa: ARG002 (base signature)
        return 1.5 * points**5 / self.scale**4


def _stochastic_equation(function_type) -> DifferentialEquation:
    time = Variable(name="t", discretization=[])
    variables = [Variable(name="X", discretization=[])]
    return DifferentialEquation(
        variables=variables,
        time=time,
        parameters=[],
        derivative=function_type(variables, [], time),
    )


_SOLVERS = {
    "euler_maruyama": (EulerMaruyamaSolver, EulerMaruyamaConfig),
    "milstein": (MilsteinSolver, MilsteinConfig),
    "srk2": (SRK2Solver, SRK2Config),
    "srk3": (SRK3Solver, SRK3Config),
    "srk4": (SRK4Solver, SRK4Config),
    "zhan_duan_li_li": (ZhanDuanLiLiSolver, ZhanDuanLiLiConfig),
}
_HORIZON = 0.25
_STEP = 2.0**-6
_START = [0.5 * _SIGMA]
# The slope error at a step of eps^(1/3) S is about (6e-6)^2 relative, and it
# enters the path through the drift's b b' dt only: measured 8.4e-13 at worst
# over the six solvers (outputs/verification/deq-167/stochastic_scale.txt).
_AGREEMENT = 1.0e-10


def _final(solver_type, config) -> float:
    """The path's end value; ``inf`` when an unscaled run blows up on the way."""
    path = BrownianPath.sample(0.0, _HORIZON, _STEP / 4.0, 1, seed=7)
    function_type = _Cubic if config.calculus == "stratonovich" else _CorrectedCubic
    solver = solver_type(config, wiener=path)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # The unscaled runs overflow, and Milstein's commutativity check reads
        # their wrong slopes as non-commuting noise and says so.
        warnings.simplefilter("ignore")
        try:
            solver.solve(_stochastic_equation(function_type), _START)
        except OverflowError:
            return math.inf
    return float(solver.solution.results[0].discretization[-1])


def _close(value: float, reference: float, tolerance: float) -> bool:
    """``|value - reference| <= tolerance |reference|``; false for NaN or inf."""
    return bool(abs(value - reference) <= tolerance * abs(reference))


class TestStochasticSolversAtStateScale:
    """``b = X^3 / S^2`` at ``X ~ S = 1e-12``, Stratonovich against Ito by hand.

    The Ito form's drift ``b b' / 2 = 1.5 X^5 / S^4`` is written out, so the two
    forms are one SDE and, on one Brownian path, one path to the difference
    error of the slope a solver reads.
    """

    @pytest.mark.parametrize("name", list(_SOLVERS))
    def test_typical_through_the_config(self, name):
        solver_type, config_type = _SOLVERS[name]

        def config(calculus, typical=None):
            return config_type(
                start_time=0.0,
                end_time=_HORIZON,
                step_size=_STEP,
                calculus=calculus,
                typical=typical,
            )

        # The probe is the form that reads b': the Stratonovich one, except in
        # the theta scheme at its default theta = 1/2, whose Stratonovich step
        # needs no slope and whose Ito step subtracts b b' / 2. The reference is
        # the other form; Milstein reads b' in both, so it takes typical too.
        probe, other = (
            ("ito", "stratonovich")
            if name == "zhan_duan_li_li"
            else ("stratonovich", "ito")
        )
        reference = _final(solver_type, config(other, _SIGMA))
        scaled = _final(solver_type, config(probe, _SIGMA))
        unscaled = _final(solver_type, config(probe))
        assert _close(scaled, reference, _AGREEMENT)
        # A step of 6e-6 on a state of 5e-13 is no slope at all: the run is
        # off by orders of magnitude, or overflows.
        assert not _close(unscaled, reference, 1.0e-2)

    def test_the_ensemble(self):
        path = BrownianPath.sample(0.0, _HORIZON, _STEP / 4.0, 1, seed=7)
        finals = {}
        for calculus, typical, function_type in (
            ("ito", _SIGMA, _CorrectedCubic),
            ("stratonovich", _SIGMA, _Cubic),
            ("stratonovich", None, _Cubic),
        ):
            config = EulerMaruyamaConfig(
                start_time=0.0,
                end_time=_HORIZON,
                step_size=_STEP,
                calculus=calculus,
                typical=typical,
            )
            with np.errstate(all="ignore"):
                result = StochasticEnsemble(config, wiener=path).solve(
                    _stochastic_equation(function_type),
                    _START,
                    paths=1,
                )
            finals[(calculus, typical)] = float(result.final[0, 0])
        reference = finals[("ito", _SIGMA)]
        assert _close(finals[("stratonovich", _SIGMA)], reference, _AGREEMENT)
        assert not _close(finals[("stratonovich", None)], reference, 1.0e-2)

    @pytest.mark.parametrize("bad", [0.0, -1.0e-12, [1.0, -1.0]])
    def test_a_non_positive_typical_size_is_refused(self, bad):
        with pytest.raises(ValidationError):
            EulerMaruyamaConfig(typical=bad)


def _cusp_equation() -> DifferentialEquation:
    return scenarios._equation(  # noqa: SLF001
        scenarios.CuspNormalForm,
        1,
        -0.25,
        0.75,
    )


class TestCodim2Steps:
    """DEQ-166: ``curve_epsilon`` retired, ``derivative_epsilon`` wired."""

    def test_curve_epsilon_warns_and_is_dropped(self):
        with pytest.warns(DeprecationWarning, match="curve_epsilon"):
            config = Codim2Config(curve_epsilon=1.0e-3)
        assert "curve_epsilon" not in Codim2Config.model_fields
        assert "curve_epsilon" not in config.model_dump()

    def test_a_config_without_it_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Codim2Config()

    def test_the_epsilon_description_is_the_relative_step(self):
        description = str(
            Codim2Config.model_fields["finite_difference_epsilon"].description
        )
        assert "Relative step" in description
        assert "max(|x|, typical_x)" in description
        assert "Outer" not in description

    def test_derivative_epsilon_is_refused_for_automatic_differentiation(self):
        with pytest.raises(ValidationError, match="derivative_epsilon"):
            Codim2Config(derivative_epsilon=1.0e-3)

    @pytest.mark.parametrize("bad", [0.0, -1.0e-3])
    def test_a_non_positive_derivative_epsilon_is_refused(self, bad):
        with pytest.raises(ValidationError):
            Codim2Config(
                derivative_provider="finite_difference_derivatives",
                derivative_epsilon=bad,
            )

    def test_the_provider_takes_the_configured_step_for_every_order(self):
        provider = configured_provider("finite_difference_derivatives", 1.0e-3)
        assert isinstance(provider, FiniteDifferenceDerivatives)
        assert [provider.relative_step(m) for m in range(1, 5)] == [1.0e-3] * 4
        default = configured_provider("finite_difference_derivatives", None)
        assert isinstance(default, FiniteDifferenceDerivatives)
        assert [default.relative_step(m) for m in range(1, 5)] == list(ORDER_STEPS)

    def test_the_driver_builds_the_provider_from_the_config(self):
        seen: list[tuple[str, float | None]] = []

        def recording(key, epsilon):
            seen.append((key, epsilon))
            return configured_provider(key, epsilon)

        config = Codim2Config(
            derivative_provider="finite_difference_derivatives",
            derivative_epsilon=2.0e-4,
            continuation_parameter_index=1,
            second_parameter_index=0,
            curve="fold",
            codim2_detectors=["cusp"],
            initial_parameter=-0.25,
            direction=1,
            parameter_lower_bound=-0.35,
            parameter_upper_bound=0.35,
        )
        with mock.patch.object(codim2_builder, "configured_provider", recording):
            result = Codim2Driver.run(
                config,
                _cusp_equation(),
                CurveSeed(state=[0.5], parameter_a=0.75),
            )
        assert seen == [("finite_difference_derivatives", 2.0e-4)]
        assert [point.kind for point in result.points] == ["cusp"]
