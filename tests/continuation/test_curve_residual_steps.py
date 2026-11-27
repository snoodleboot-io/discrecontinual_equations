"""The differenced columns of the cusp and generalized-Hopf curves, at scale (DEQ-159).

``AnalyticJacobian`` hands its ``epsilon`` to a ``DifferentiableResidual``, and the
two such residuals - the cusp and the generalized-Hopf curve - difference their
three parameter columns centrally. A central difference at step ``h`` errs by the
truncation ``h^2 |d3 r / dp3| / 6`` plus the rounding ``~ eps |r| / h`` (Dennis
and Schnabel 1983, Sec. 5.4), both measured against the size of the parameter, so
the step is ``epsilon max(|p|, 1)``, the shared rule of DEQ-118. An absolute
``1e-5`` either side of a parameter of ``3e11`` (an ulp there is ``6e-5``) is no
move at all, and the column comes back as zero.

The fields are a cusp and a Bautin normal form written in scaled units ``Z = u / S
- c``, ``P = p / S``, so that at scale ``S`` the state and all three parameters are
of size ``S``. Their rows - the field, the fold or Hopf test function and the
codim-2 coefficient - are written out by hand below, as are the parameter columns,
and the hand-written rows are checked against the residual at every scale while the
columns are checked against central differences at ``S = 1`` (``TestReference``):
nothing at scale is checked against differences of the code under test. Every row
is linear or quadratic in the parameters, so a central difference is exact up to
rounding, and the tolerance is the rounding allowance.
"""

import numpy as np
import pytest

from discrecontinual_equations.continuation.cusp_curve import CuspCurveResidual
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.generalized_hopf_curve import (
    GeneralizedHopfCurveResidual,
)
from discrecontinual_equations.continuation.jacobian import (
    AnalyticJacobian,
    FiniteDifferenceJacobian,
)
from discrecontinual_equations.continuation.two_parameter_problem import (
    TwoParameterProblem,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.finite_difference import central_column
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable

_EPS = float(np.finfo(float).eps)
# The step a Codim3Config hands the residual (its finite_difference_epsilon).
_STEP = 1.0e-5
# The rows are at most quadratic in the parameters, so the difference error is the
# rounding eps |r| / h of each row: relative to the row's largest entry it is about
# eps / (_STEP * 0.3) ~ 1e-10 here. An absolute step errs by 1e-3 at S = 1e8 and
# by the whole column at S = 1e12.
_AGREEMENT = 1.0e-8
_SCALES = (1.0e8, 1.0e12)
# Parameters in scaled units: of order one, one negative so the sign is exercised.
_UNITS = (0.3, -0.4, 0.5)
_OMEGA = 1.5
_KAPPA = 2.4


class _Pa(Parameter, name="Pa", abbreviation="pa"):
    pass


class _Pb(Parameter, name="Pb", abbreviation="pb"):
    pass


class _Pc(Parameter, name="Pc", abbreviation="pc"):
    pass


class _U(Variable, name="u", abbreviation="u"):
    pass


class _T(Variable, name="t", abbreviation="t"):
    pass


class _Scaled(DeterministicFunction):
    """A field ``S F(u / S - c; p / S)``, recording the parameters it is called at."""

    scale = 1.0
    centre: tuple[float, ...] = ()
    seen: list[tuple[float, float, float]]

    def parameter_values(self) -> tuple[float, float, float]:
        pa, pb, pc = (parameter.value for parameter in self.parameters)
        self.seen.append((float(pa), float(pb), float(pc)))
        return pa, pb, pc


class _Cusp(_Scaled):
    """``f = pa + pb Z + pc Z^2 - S Z^3``, ``Z = u / S - 1/2``."""

    centre = (0.5,)

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        pa, pb, pc = self.parameter_values()
        z = point[0] / self.scale - self.centre[0]
        return [pa + pb * z + pc * z * z - self.scale * z * z * z]


class _Bautin(_Scaled):
    """A Hopf normal form with ``l1 = 2 pb / (S^3 omega)`` at ``Z = 0``.

    ``f1 = pc + pa^2 / S + pa Z1 - S w Z2 + pb Z1 |Z|^2`` and
    ``f2 = pb pc / S + S w Z1 + pa Z2 + pb Z2 |Z|^2``, ``Z = u / S - (3/4, -1/2)``.
    """

    centre = (0.75, -0.5)

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        pa, pb, pc = self.parameter_values()
        s, w = self.scale, _OMEGA
        z1 = point[0] / s - self.centre[0]
        z2 = point[1] / s - self.centre[1]
        radius = z1 * z1 + z2 * z2
        return [
            pc + pa * pa / s + pa * z1 - s * w * z2 + pb * z1 * radius,
            pb * pc / s + s * w * z1 + pa * z2 + pb * z2 * radius,
        ]


def _field(kind: type[_Scaled], scale: float) -> _Scaled:
    size = len(kind.centre)
    parameters = [_Pa(value=0.0), _Pb(value=0.0), _Pc(value=0.0)]
    field = kind(
        variables=[_U() for _ in range(size)],
        parameters=parameters,
        results=[_U() for _ in range(size)],
        time=None,
    )
    field.scale = scale
    field.seen = []
    return field


def _problem(field: _Scaled) -> TwoParameterProblem:
    size = len(field.centre)
    equation = DifferentialEquation(
        variables=[_U() for _ in range(size)],
        time=_T(),
        parameters=field.parameters,
        derivative=field,
    )
    return TwoParameterProblem(equation, 0, 1, 0.0, AutomaticDifferentiation())


def _parameters(scale: float) -> tuple[float, float, float]:
    pa, pb, pc = (unit * scale for unit in _UNITS)
    return pa, pb, pc


def _cusp(scale: float) -> tuple[CuspCurveResidual, np.ndarray, float, _Scaled]:
    """The cusp residual at ``u = 5 S / 4`` (``Z = 3/4``), unit borders."""
    field = _field(_Cusp, scale)
    pa, pb, pc = _parameters(scale)
    residual = CuspCurveResidual(
        _problem(field),
        1,
        np.ones(1),
        np.ones(1),
        field.parameters[2],
    )
    return residual, np.array([1.25 * scale, pa, pb]), pc, field


def _cusp_rows(scale: float, state: np.ndarray, pc: float) -> np.ndarray:
    """``(f, g, a)`` with unit borders: ``g = -f_u``, ``a = f_uu / 2``."""
    u, pa, pb = state
    z = u / scale - 0.5
    return np.array(
        [
            pa + pb * z + pc * z * z - scale * z**3,
            -(pb + 2.0 * pc * z - 3.0 * scale * z * z) / scale,
            (pc - 3.0 * scale * z) / scale**2,
        ],
    )


def _cusp_columns(scale: float, state: np.ndarray) -> np.ndarray:
    """``d (f, g, a) / d (pa, pb, pc)``."""
    z = state[0] / scale - 0.5
    return np.array(
        [
            [1.0, z, z * z],
            [0.0, -1.0 / scale, -2.0 * z / scale],
            [0.0, 0.0, 1.0 / scale**2],
        ],
    )


def _bautin(
    scale: float,
) -> tuple[GeneralizedHopfCurveResidual, np.ndarray, float, _Scaled]:
    """The generalized-Hopf residual at ``Z = 0`` off the curve, unit borders."""
    field = _field(_Bautin, scale)
    pa, pb, pc = _parameters(scale)
    residual = GeneralizedHopfCurveResidual(
        _problem(field),
        2,
        np.eye(2),
        np.eye(2),
        field.parameters[2],
    )
    state = np.array([0.75 * scale, -0.5 * scale, _KAPPA, pa, pb])
    return residual, state, pc, field


def _bautin_rows(scale: float, state: np.ndarray, pc: float) -> np.ndarray:
    """``(f, G_00, G_11, l1)`` at ``Z = 0``.

    ``A = [[a, -w], [w, a]]`` with ``a = pa / S``. With both borders the identity,
    the bordered solve gives ``G = -(A^2 + kappa I)``. ``q = p = (1, -i) / sqrt 2``
    for every ``a``, ``B = 0`` at ``Z = 0``, and the cubic ``pb Z |Z|^2`` has
    ``C(q, q, conj q) = 4 pb q / S^3``, so ``g21 = 4 pb / S^3`` and ``l1 = g21 /
    (2 omega)`` with ``omega = sqrt(kappa)``.
    """
    kappa, pa, pb = state[2:]
    a = pa / scale
    diagonal = -(a * a - _OMEGA**2 + kappa)
    return np.array(
        [
            pc + pa * pa / scale,
            pb * pc / scale,
            diagonal,
            diagonal,
            2.0 * pb / (scale**3 * np.sqrt(kappa)),
        ],
    )


def _bautin_columns(scale: float, state: np.ndarray, pc: float) -> np.ndarray:
    """``d (f, G_00, G_11, l1) / d (pa, pb, pc)``."""
    kappa, pa, pb = state[2:]
    a = pa / scale
    return np.array(
        [
            [2.0 * pa / scale, 0.0, 1.0],
            [0.0, pc / scale, pb / scale],
            [-2.0 * a / scale, 0.0, 0.0],
            [-2.0 * a / scale, 0.0, 0.0],
            [0.0, 2.0 / (scale**3 * np.sqrt(kappa)), 0.0],
        ],
    )


def _assert_rows_agree(computed: np.ndarray, expected: np.ndarray) -> None:
    """Each row within ``_AGREEMENT`` of that row's largest entry."""
    size = np.max(np.abs(expected), axis=1, keepdims=True)
    error = np.abs(computed - expected) / size
    assert np.all(error <= _AGREEMENT), error


def _moved_steps(field: _Scaled, point: tuple[float, ...]) -> list[set[float]]:
    """Every distance each of the three parameters was moved from ``point``."""
    moved: list[set[float]] = [set(), set(), set()]
    for seen in field.seen:
        for index in range(3):
            if seen[index] != point[index]:
                moved[index].add(abs(seen[index] - point[index]))
    return moved


_CURVES = {
    "cusp": (_cusp, _cusp_rows, lambda s, x, _p: _cusp_columns(s, x)),
    "generalized_hopf": (_bautin, _bautin_rows, _bautin_columns),
}


@pytest.mark.parametrize("curve", sorted(_CURVES))
class TestReference:
    """The hand-written rows and columns are what the residuals compute."""

    @pytest.mark.parametrize("scale", [1.0, *_SCALES])
    def test_rows_are_the_residual(self, curve, scale):
        build, rows, _columns = _CURVES[curve]
        residual, state, parameter, _field = build(scale)
        expected = rows(scale, state, parameter)
        _assert_rows_agree(residual.evaluate(state, parameter)[None, :], expected[None])

    def test_columns_are_its_differences_at_unit_scale(self, curve):
        # Fourth-order Richardson of the default central-difference Jacobian.
        build, _rows, columns = _CURVES[curve]
        residual, state, parameter, _field = build(1.0)
        coarse = FiniteDifferenceJacobian(residual, 1.0e-3)
        fine = FiniteDifferenceJacobian(residual, 5.0e-4)
        reference = (
            4.0 * fine.extended_jacobian(state, parameter)
            - coarse.extended_jacobian(state, parameter)
        ) / 3.0
        differenced = slice(state.size - 2, state.size + 1)
        _assert_rows_agree(
            reference[:, differenced],
            columns(1.0, state, parameter),
        )


@pytest.mark.parametrize("scale", _SCALES)
@pytest.mark.parametrize("curve", sorted(_CURVES))
class TestAnalyticJacobianAtScale:
    """``AnalyticJacobian`` over each curve, against the hand-written columns."""

    def test_parameter_columns(self, curve, scale):
        build, _rows, columns = _CURVES[curve]
        residual, state, parameter, _field = build(scale)
        jacobian = AnalyticJacobian(residual, _STEP).extended_jacobian(state, parameter)
        differenced = jacobian[:, state.size - 2 :]
        _assert_rows_agree(differenced, columns(scale, state, parameter))

    def test_parameter_derivative(self, curve, scale):
        build, _rows, columns = _CURVES[curve]
        residual, state, parameter, _field = build(scale)
        derivative = AnalyticJacobian(residual, _STEP).parameter_derivative(
            state,
            parameter,
        )
        expected = columns(scale, state, parameter)[:, -1]
        error = np.abs(derivative - expected)
        assert np.all(error <= _AGREEMENT * np.max(np.abs(expected))), error

    def test_third_parameter_is_restored(self, curve, scale):
        build, _rows, _columns = _CURVES[curve]
        residual, state, parameter, field = build(scale)
        AnalyticJacobian(residual, _STEP).extended_jacobian(state, parameter)
        assert field.parameters[2].value == parameter


@pytest.mark.parametrize("curve", sorted(_CURVES))
class TestSteps:
    """Each parameter steps ``epsilon max(|p|, 1)``: ``epsilon`` itself below one."""

    @pytest.mark.parametrize("scale", [1.0, *_SCALES])
    def test_jacobian_steps(self, curve, scale):
        build, _rows, _columns = _CURVES[curve]
        residual, state, parameter, field = build(scale)
        AnalyticJacobian(residual, _STEP).extended_jacobian(state, parameter)
        point = _parameters(scale)
        for moved, value in zip(_moved_steps(field, point), point, strict=True):
            expected = _STEP * max(abs(value), 1.0)
            assert moved, value
            assert all(
                abs(step - expected) <= 4.0 * _EPS * abs(value) for step in moved
            )

    def test_at_unit_scale_the_step_is_epsilon(self, curve):
        build, _rows, _columns = _CURVES[curve]
        residual, state, parameter, field = build(1.0)
        AnalyticJacobian(residual, _STEP).extended_jacobian(state, parameter)
        point = _parameters(1.0)
        for moved in _moved_steps(field, point):
            assert all(abs(step - _STEP) <= 4.0 * _EPS for step in moved)


class TestGeneralizedHopfSelection:
    """``adapt`` picks the pair of ``G`` entries from a Jacobian at the same rule."""

    @pytest.mark.parametrize("scale", _SCALES)
    def test_selection_steps_are_relative(self, scale):
        residual, state, parameter, field = _bautin(scale)
        residual.adapt(state, parameter)
        point = _parameters(scale)
        steps = _moved_steps(field, point)
        for moved, value in zip(steps, point, strict=True):
            expected = 1.0e-6 * abs(value)
            assert moved, value
            assert all(
                abs(step - expected) <= 4.0 * _EPS * abs(value) for step in moved
            )


class TestCentralColumn:
    """The shared central difference over one unknown of ``(state, parameter)``."""

    @staticmethod
    def _identity(state: np.ndarray, parameter: float) -> np.ndarray:
        return np.append(state, parameter)

    def test_divides_by_the_width_actually_stepped(self):
        # The rows are the unknowns themselves: their difference is exactly the
        # width the machine stepped, so the quotient is a unit vector exactly;
        # over 2 h it is off by up to eps |x| / h.
        state = np.array([0.1, -3.7, 1.0e8 + 0.3])
        parameter = -(1.0e8 + 0.7)
        for column in range(state.size + 1):
            computed = central_column(self._identity, state, parameter, column, 1.0e-7)
            assert np.array_equal(computed, np.eye(state.size + 1)[column]), column

    def test_step_is_epsilon_times_max_of_size_and_one(self):
        state = np.array([0.25, -5.0e9])
        parameter = 3.0e11
        calls: list[np.ndarray] = []

        def record(point: np.ndarray, value: float) -> np.ndarray:
            calls.append(np.append(point, value))
            return np.zeros(1)

        for column in range(3):
            central_column(record, state, parameter, column, 1.0e-6)
        unknowns = np.append(state, parameter)
        steps = [np.max(np.abs(call - unknowns)) for call in calls]
        expected = 1.0e-6 * np.repeat([1.0, 5.0e9, 3.0e11], 2)
        assert np.allclose(steps, expected, rtol=1.0e-9, atol=0.0)

    def test_leaves_the_state_unchanged(self):
        state = np.array([1.0, 2.0])
        central_column(self._identity, state, 0.5, 0, 1.0e-6)
        assert np.array_equal(state, [1.0, 2.0])


@pytest.mark.parametrize("curve", sorted(_CURVES))
def test_at_unit_scale_the_jacobian_is_the_absolute_steps_one(curve):
    # Every unknown here is of size one or less, so the relative step is the old
    # absolute one; only the division by the width stepped, rather than by 2 h,
    # differs, by about eps |p| / h ~ 1e-11 relative.
    build, _rows, _columns = _CURVES[curve]
    residual, state, parameter, _field = build(1.0)
    computed = AnalyticJacobian(residual, _STEP).extended_jacobian(state, parameter)
    absolute = []
    for column in range(state.size - 2, state.size + 1):
        if column < state.size:
            step = np.zeros(state.size)
            step[column] = _STEP
            upper = residual.evaluate(state + step, parameter)
            lower = residual.evaluate(state - step, parameter)
        else:
            upper = residual.evaluate(state, parameter + _STEP)
            lower = residual.evaluate(state, parameter - _STEP)
        absolute.append((upper - lower) / (2.0 * _STEP))
    expected = np.column_stack(absolute)
    differenced = computed[:, state.size - 2 :]
    scale = np.max(np.abs(expected), axis=1, keepdims=True)
    assert np.all(np.abs(differenced - expected) <= 1.0e-10 * scale)
