"""The finite differences outside ``jacobian_at``, on badly scaled states (DEQ-118).

The noise slopes the stochastic solvers read (``ItoCoefficients`` and its batched
twin), the continuation Jacobian (``FiniteDifferenceJacobian``) and the derivative
fallback (``FiniteDifferenceDerivatives``) all difference centrally. A central
difference at step ``h_j`` errs in entry ``(i, j)`` by the truncation
``h_j^2 |d3 f_i / dx_j3| / 6`` plus the rounding ``~ eps |f_i| / h_j`` (Dennis and
Schnabel 1983, Sec. 5.4), so a step that keeps both small follows the size of
component ``j``: ``h_j = r max(|x_j|, typical_j)``, with ``r`` the relative step.
An absolute step of ``1e-6`` on a component of size ``1e12`` (an ulp there is
``1.2e-4``) rounds away, and the difference comes back as zero.

The field is the polynomial of ``test_jacobian_step``, nonlinear in every
component at that component's own scale, and the reference Jacobian is written out
by hand and checked against exact rational differences first, so nothing here is
checked against the code under test.
"""

from fractions import Fraction
from typing import ClassVar

import numpy as np
import pytest

from discrecontinual_equations.continuation.derivative_provider import (
    FiniteDifferenceDerivatives,
)
from discrecontinual_equations.continuation.jacobian import FiniteDifferenceJacobian
from discrecontinual_equations.continuation.residual import ResidualFunction
from discrecontinual_equations.continuation.sparsity import SparsityPattern
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.finite_difference import (
    CENTRAL_STEP,
    step_scales,
    typical_sizes,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.function.stochastic import StochasticFunction
from discrecontinual_equations.solver.stochastic.coefficients import ItoCoefficients
from discrecontinual_equations.solver.stochastic.ensemble import BatchedCoefficients
from discrecontinual_equations.variable import Variable

_EPS = float(np.finfo(float).eps)
# Slack on the error model: the rounding term counts one ulp per evaluation and
# the true rounding of a few-term sum, and of the step itself, is a small
# multiple of that.
_SLACK = 4.0
# The state in units of each component's scale: of order one, with a negative
# entry so the sign of a component is exercised.
_UNITS = (0.7, -1.3, 0.9)
_EXACT_STEP = Fraction(1, 10**40)
_LARGE = (1, 10**8, 10**12)
_SMALL = (1, Fraction(1, 10**8), Fraction(1, 10**12))
# The continuation parameter of the residual below, and its own scale.
_PARAMETER = 3.0e8


class _X(Variable, name="x", abbreviation="x"):
    pass


class _T(Variable, name="t", abbreviation="t"):
    pass


def _polynomial(scales, point):
    """``f`` in scaled units ``u_j = x_j / s_j``, output ``i`` scaled by ``s_i``.

    ``f1`` carries ``x0`` unscaled: a unit coupling in a row of size ``s1``.
    Its terms are written so that the same code evaluates floats and Fractions.
    """
    s0, s1, s2 = scales
    u0, u1, u2 = point[0] / s0, point[1] / s1, point[2] / s2
    return [
        s0 * (u0 * u1 + u2 * u2),
        s1 * (u1 * u2 + u1 * u1) + point[0],
        s2 * (u1 * u1 * u1 + u0 * u0 * u2),
    ]


def _floats(scales) -> tuple[float, ...]:
    return tuple(float(s) for s in scales)


def _state(scales) -> np.ndarray:
    return np.array([float(u * s) for u, s in zip(_UNITS, scales, strict=True)])


def _analytic(scales, state: np.ndarray) -> np.ndarray:
    """The Jacobian of :func:`_polynomial`, by hand."""
    s0, s1, s2 = _floats(scales)
    u0, u1, u2 = state[0] / s0, state[1] / s1, state[2] / s2
    return np.array(
        [
            [u1, u0 * s0 / s1, 2.0 * u2 * s0 / s2],
            [1.0, u2 + 2.0 * u1, s1 * u1 / s2],
            [s2 * 2.0 * u0 * u2 / s0, s2 * 3.0 * u1 * u1 / s1, s2 * u0 * u0 / s2],
        ],
    )


def _magnitude(scales, state: np.ndarray) -> np.ndarray:
    """Sum of the absolute terms of each ``f_i``: what its rounding is relative to."""
    s0, s1, s2 = _floats(scales)
    u0, u1, u2 = abs(state[0] / s0), abs(state[1] / s1), abs(state[2] / s2)
    return np.array(
        [
            s0 * (u0 * u1 + u2 * u2),
            s1 * (u1 * u2 + u1 * u1) + abs(state[0]),
            s2 * (u1**3 + u0 * u0 * u2),
        ],
    )


def _third(scales) -> np.ndarray:
    """``|d3 f_i / dx_j3|``, by hand: only ``f2`` is cubic, and only in ``x1``."""
    s1, s2 = float(scales[1]), float(scales[2])
    third = np.zeros((3, 3))
    third[2, 1] = 6.0 * s2 / s1**3
    return third


def _allowance(scales, state: np.ndarray, relative: float) -> np.ndarray:
    """What a central difference at ``h_j = relative max(|x_j|, s_j)`` can reach.

    Truncation ``h^2 T / 6`` against rounding ``eps F / h``, with slack, plus an
    ulp of the entry itself.
    """
    step = relative * np.maximum(np.abs(state), np.array(_floats(scales)))
    truncation = step**2 * _third(scales) / 6.0
    rounding = _EPS * _magnitude(scales, state)[:, None] / step
    return _SLACK * (truncation + rounding) + _EPS * np.abs(_analytic(scales, state))


def _exact_jacobian(scales, state: np.ndarray) -> np.ndarray:
    """Central differences in exact rational arithmetic, with a step of ``1e-40``.

    The field is a cubic polynomial, so the only error is ``h^2 f''' / 6``, some
    ``1e-80`` relative: an independent check on :func:`_analytic`.
    """
    point = [Fraction(x) for x in state]
    columns = []
    for j in range(3):
        step = _EXACT_STEP * max(abs(point[j]), Fraction(1))
        up, down = list(point), list(point)
        up[j] += step
        down[j] -= step
        pairs = zip(_polynomial(scales, up), _polynomial(scales, down), strict=True)
        columns.append([float((a - b) / (2 * step)) for a, b in pairs])
    return np.array(columns).T


def _typical(scales, given: bool) -> dict[str, np.ndarray]:  # noqa: FBT001
    """The ``typical`` keyword, when one is given."""
    return {"typical": np.array(_floats(scales))} if given else {}


def _assert_within(computed: np.ndarray, scales, relative: float) -> None:
    state = _state(scales)
    error = np.abs(computed - _analytic(scales, state))
    allowance = _allowance(scales, state, relative)
    assert np.all(error <= allowance), error / allowance


# (scales, whether a typical size is given): large components need none, since
# their value is their scale; small ones need one, or the default of one swamps
# them.
_CASES = [
    pytest.param(_LARGE, False, id="1-1e8-1e12"),
    pytest.param(_SMALL, True, id="1-1e-8-1e-12-typical"),
]


class _Noise(StochasticFunction):
    """Vector noise ``b = f(x)``, and any drift: only the slopes are read."""

    scales: tuple = _LARGE

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        return [0.0] * len(point)

    def diffusion(self, point, time=None):  # noqa: ARG002 (base signature)
        return _polynomial(self.scales, point)


class _MatrixNoise(_Noise):
    """``G = [f, 2 f]``: two drivers, column ``k``'s Jacobian is ``(k + 1) J``."""

    def diffusion(self, point, time=None):  # noqa: ARG002 (base signature)
        column = _polynomial(self.scales, point)
        return [[value, 2.0 * value] for value in column]


class _SeparableNoise(_Noise):
    """Declared diagonal noise ``b_i = s_i (u_i^3 + u_i^2)``, ``u_i = x_i / s_i``."""

    diagonal_noise = True

    def diffusion(self, point, time=None):  # noqa: ARG002 (base signature)
        units = [x / s for x, s in zip(point, self.scales, strict=True)]
        return [s * (u**3 + u**2) for u, s in zip(units, self.scales, strict=True)]


def _separable_slopes(scales, state: np.ndarray) -> np.ndarray:
    units = state / np.array(_floats(scales))
    return 3.0 * units**2 + 2.0 * units


def _separable_allowance(scales, state: np.ndarray, relative: float) -> np.ndarray:
    size = np.array(_floats(scales))
    step = relative * np.maximum(np.abs(state), size)
    units = np.abs(state / size)
    magnitude = size * (units**3 + units**2)
    truncation = step**2 * (6.0 / size**2) / 6.0
    exact = np.abs(_separable_slopes(scales, state))
    return _SLACK * (truncation + _EPS * magnitude / step) + _EPS * exact


def _equation(kind: type[_Noise], scales) -> DifferentialEquation:
    function = kind(
        variables=[_X(), _X(), _X()],
        parameters=[],
        results=[_X(), _X(), _X()],
        time=None,
    )
    function.scales = scales
    return DifferentialEquation(
        variables=function.variables,
        time=_T(),
        parameters=[],
        derivative=function,
    )


class TestReference:
    """The hand-written Jacobian is right before anything is compared to it."""

    @pytest.mark.parametrize("scales", [_LARGE, _SMALL], ids=["large", "small"])
    def test_against_exact_rational_differences(self, scales):
        state = _state(scales)
        exact = _exact_jacobian(scales, state)
        assert np.allclose(_analytic(scales, state), exact, rtol=1.0e-14, atol=0.0)


@pytest.mark.parametrize(("scales", "typical"), _CASES)
class TestNoiseSlopes:
    """The slopes Milstein's correction and the commutativity check read."""

    def test_diffusion_jacobian(self, scales, typical):
        coefficients = ItoCoefficients(
            _equation(_Noise, scales),
            "ito",
            **_typical(scales, typical),
        )
        computed = coefficients.diffusion_jacobian(_state(scales), 0.0)
        _assert_within(computed, scales, CENTRAL_STEP)

    def test_column_jacobians_of_vector_noise(self, scales, typical):
        coefficients = ItoCoefficients(
            _equation(_Noise, scales),
            "ito",
            **_typical(scales, typical),
        )
        jacobians = coefficients.column_jacobians(_state(scales), 0.0)
        # Column k is b_k e_k: its Jacobian is row k of J and zeros.
        for k in range(3):
            rows = np.zeros((3, 3), dtype=bool)
            rows[k] = True
            assert np.all(jacobians[k][~rows] == 0.0)
        stacked = np.array([jacobians[k][k] for k in range(3)])
        _assert_within(stacked, scales, CENTRAL_STEP)

    def test_column_jacobians_of_matrix_noise(self, scales, typical):
        coefficients = ItoCoefficients(
            _equation(_MatrixNoise, scales),
            "ito",
            **_typical(scales, typical),
        )
        jacobians = coefficients.column_jacobians(_state(scales), 0.0)
        _assert_within(jacobians[0], scales, CENTRAL_STEP)
        _assert_within(jacobians[1] / 2.0, scales, CENTRAL_STEP)

    def test_diagonal_slopes_of_undeclared_noise(self, scales, typical):
        coefficients = ItoCoefficients(
            _equation(_Noise, scales),
            "ito",
            **_typical(scales, typical),
        )
        state = _state(scales)
        slopes = coefficients.diagonal_slopes(state, 0.0)
        error = np.abs(slopes - np.diag(_analytic(scales, state)))
        assert np.all(error <= np.diag(_allowance(scales, state, CENTRAL_STEP)))

    def test_diagonal_slopes_of_declared_noise(self, scales, typical):
        coefficients = ItoCoefficients(
            _equation(_SeparableNoise, scales),
            "ito",
            **_typical(scales, typical),
        )
        state = _state(scales)
        slopes = coefficients.diagonal_slopes(state, 0.0)
        assert coefficients.declared_diagonal
        error = np.abs(slopes - _separable_slopes(scales, state))
        assert np.all(error <= _separable_allowance(scales, state, CENTRAL_STEP))

    def test_batched_column_jacobians(self, scales, typical):
        coefficients = BatchedCoefficients(
            _equation(_MatrixNoise, scales),
            "ito",
            **_typical(scales, typical),
        )
        points = np.array([_state(scales), _state(scales)])
        jacobians = coefficients.column_jacobians(points, 0.0)
        for path in range(2):
            _assert_within(jacobians[path, 0], scales, CENTRAL_STEP)
            _assert_within(jacobians[path, 1] / 2.0, scales, CENTRAL_STEP)

    def test_batched_diagonal_slopes(self, scales, typical):
        state = _state(scales)
        points = np.array([state, state])
        for kind, exact, allowance in (
            (
                _Noise,
                np.diag(_analytic(scales, state)),
                np.diag(_allowance(scales, state, CENTRAL_STEP)),
            ),
            (
                _SeparableNoise,
                _separable_slopes(scales, state),
                _separable_allowance(scales, state, CENTRAL_STEP),
            ),
        ):
            coefficients = BatchedCoefficients(
                _equation(kind, scales),
                "ito",
                **_typical(scales, typical),
            )
            slopes = coefficients.diagonal_slopes(points, 0.0)
            assert np.all(np.abs(slopes - exact) <= allowance)

    def test_batched_slopes_are_the_per_path_slopes_bit_for_bit(self, scales, typical):
        # The ensemble's single-path hex-for-hex promise needs the same arithmetic.
        state = _state(scales)
        per_path = ItoCoefficients(
            _equation(_MatrixNoise, scales),
            "ito",
            **_typical(scales, typical),
        )
        batched = BatchedCoefficients(
            _equation(_MatrixNoise, scales),
            "ito",
            **_typical(scales, typical),
        )
        assert np.array_equal(
            batched.column_jacobians(state[None, :], 0.0)[0],
            per_path.column_jacobians(state, 0.0),
        )


class _Residual(ResidualFunction):
    """``f(x) + p^2 (1, 1, 1)``: ``df/dp = 2 p`` in every row, exactly quadratic."""

    def __init__(self, scales) -> None:
        self.scales = scales

    def evaluate(self, state: np.ndarray, parameter: float) -> np.ndarray:
        field = np.array(_polynomial(self.scales, list(state)), dtype=float)
        return field + parameter * parameter


@pytest.mark.parametrize(("scales", "typical"), _CASES)
class TestContinuationJacobian:
    def test_state_jacobian(self, scales, typical):
        provider = FiniteDifferenceJacobian(
            _Residual(scales),
            CENTRAL_STEP,
            **_typical(scales, typical),
        )
        computed = provider.state_jacobian(_state(scales), 0.0)
        _assert_within(computed, scales, CENTRAL_STEP)

    def test_parameter_derivative(self, scales, typical):
        # An ulp of 3e8 is 6e-8: an absolute step of 1e-7 is a third rounding.
        provider = FiniteDifferenceJacobian(
            _Residual(scales),
            1.0e-7,
            **_typical(scales, typical),
        )
        step = 1.0e-7 * _PARAMETER
        magnitude = _magnitude(scales, _state(scales)) + _PARAMETER**2
        allowance = _SLACK * _EPS * magnitude / step
        # The step follows |lambda|: a negative parameter is as large as a positive.
        for parameter in (_PARAMETER, -_PARAMETER):
            derivative = provider.parameter_derivative(_state(scales), parameter)
            assert np.all(np.abs(derivative - 2.0 * parameter) <= allowance), parameter


@pytest.mark.parametrize(("scales", "typical"), _CASES)
class TestDerivativeFallback:
    def test_jacobian(self, scales, typical):
        provider = FiniteDifferenceDerivatives(
            CENTRAL_STEP,
            **_typical(scales, typical),
        )
        computed = provider.jacobian(_Field.make(scales), _state(scales), 0.0)
        _assert_within(computed, scales, CENTRAL_STEP)

    def test_second_derivative_along_the_largest_component(self, scales, typical):
        # f0 is s0 u2^2 in x2: its Taylor coefficient of order two along e2 is
        # s0 / s2^2, and the central second difference of a quadratic leaves only
        # rounding, eps F / step^2, at a step of 1e-4 max(|x2|, typical_2).
        state = _state(scales)
        provider = FiniteDifferenceDerivatives(
            1.0e-4,
            **_typical(scales, typical),
        )
        direction = np.array([0.0, 0.0, 1.0])
        coefficients = provider.taylor(_Field.make(scales), state, direction, 2, 0.0)
        s0, _, s2 = _floats(scales)
        expected = s0 / s2**2
        step = 1.0e-4 * max(abs(state[2]), s2)
        allowance = _SLACK * _EPS * _magnitude(scales, state)[0] / step**2
        error = abs(coefficients[2][0].real - expected)
        assert error <= allowance + _EPS * expected


class _Field(DeterministicFunction):
    scales: tuple = _LARGE

    @classmethod
    def make(cls, scales) -> "_Field":
        field = cls(
            variables=[_X(), _X(), _X()],
            parameters=[],
            results=[_X(), _X(), _X()],
            time=None,
        )
        field.scales = scales
        return field

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        return _polynomial(self.scales, point)


class TestStepRule:
    """The step itself: ``r max(|x_j|, typical_j)``, the user's ``r``."""

    def test_at_unit_scale_the_relative_step_is_the_absolute_one(self):
        # The documented meaning of a configured epsilon is kept where the
        # state is of size one or less: there the step is epsilon itself.
        state = np.array([0.0, -0.25, 1.0, -1.0])
        assert np.array_equal(1.0e-7 * step_scales(state, None), np.full(4, 1.0e-7))

    def test_the_step_grows_with_the_component(self):
        state = np.array([3.0, -5.0e9, 0.25])
        expected = np.array([3.0, 5.0e9, 1.0])
        assert np.array_equal(step_scales(state, None), expected)

    def test_the_typical_size_floors_the_step(self):
        state = np.array([1.0e-12, 0.0, 4.0])
        typical = np.array([1.0e-12, 2.0, 1.0])
        assert np.array_equal(step_scales(state, typical), [1.0e-12, 2.0, 4.0])

    def test_batched_points_take_one_typical_per_component(self):
        points = np.array([[0.5, 3.0], [-7.0, 0.0]])
        assert np.array_equal(
            step_scales(points, np.array([1.0, 2.0])),
            [[1.0, 3.0], [7.0, 2.0]],
        )

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.array([1.0, 0.0, 1.0])])
    def test_a_non_positive_typical_size_is_refused(self, bad):
        with pytest.raises(ValueError, match="typical"):
            typical_sizes(bad, 3)

    def test_slopes_divide_by_the_step_actually_taken(self):
        # b(x) = x: the difference of b is exactly the width the machine stepped,
        # so dividing by that width gives one exactly; 2 h does not, off 1e+8.
        coefficients = ItoCoefficients(_equation(_IdentityNoise, _LARGE), "ito")
        state = np.array([0.1, -3.7, 1.0e8 + 0.3])
        assert np.array_equal(coefficients.diffusion_jacobian(state, 0.0), np.eye(3))
        assert np.array_equal(coefficients.diagonal_slopes(state, 0.0), np.ones(3))
        provider = FiniteDifferenceJacobian(_IdentityResidual(), 1.0e-7)
        assert np.array_equal(provider.state_jacobian(state, 0.0), np.eye(3))

    def test_an_absolute_step_would_round_away(self):
        # The failure this replaces: 1e-6 either side of 0.9e12 is no move at all.
        state = 0.9e12
        assert (state + 1.0e-6) - (state - 1.0e-6) == 0.0
        coefficients = ItoCoefficients(_equation(_Noise, _LARGE), "ito")
        slopes = coefficients.diffusion_jacobian(_state(_LARGE), 0.0)
        assert np.all(np.isfinite(slopes))
        assert slopes[2, 2] != 0.0


class _IdentityNoise(_Noise):
    def diffusion(self, point, time=None):  # noqa: ARG002 (base signature)
        return list(point)


class _IdentityResidual(ResidualFunction):
    def evaluate(self, state: np.ndarray, parameter: float) -> np.ndarray:
        del parameter
        return np.array(state, dtype=float)


class _DeclaredIdentityNoise(_IdentityNoise):
    diagonal_noise = True


class _ParameterResidual(ResidualFunction):
    def evaluate(self, state: np.ndarray, parameter: float) -> np.ndarray:
        return np.full(state.size, parameter)


class _Spy(DeterministicFunction):
    """``f(x) = x``, recording every point it is evaluated at."""

    points: list[list[float]]

    @classmethod
    def make(cls) -> "_Spy":
        field = cls(
            variables=[_X(), _X(), _X()],
            parameters=[],
            results=[_X(), _X(), _X()],
            time=None,
        )
        field.points = []
        return field

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        self.points.append(list(point))
        return list(point)


class TestStepActuallyTaken:
    """Every quotient over the width stepped, on every path through the code."""

    _STATE = np.array([0.1, -3.7, 1.0e8 + 0.3])

    def test_declared_diagonal_slopes(self):
        equation = _equation(_DeclaredIdentityNoise, _LARGE)
        per_path = ItoCoefficients(equation, "ito")
        batched = BatchedCoefficients(equation, "ito")
        assert np.array_equal(per_path.diagonal_slopes(self._STATE, 0.0), np.ones(3))
        assert per_path.declared_diagonal
        points = np.array([self._STATE, -self._STATE])
        assert np.array_equal(batched.diagonal_slopes(points, 0.0), np.ones((2, 3)))
        assert batched.declared_diagonal

    def test_parameter_derivative(self):
        provider = FiniteDifferenceJacobian(_ParameterResidual(), 1.0e-7)
        derivative = provider.parameter_derivative(np.zeros(2), 1.0e8 + 0.3)
        assert np.array_equal(derivative, np.ones(2))


class TestDirectionalStep:
    """``FiniteDifferenceDerivatives`` keeps a configured epsilon's meaning."""

    def test_at_unit_scale_the_stencil_is_the_absolute_one(self):
        # Every component the direction moves is below its typical size, so the
        # samples are x + k epsilon d exactly, whatever the direction's length.
        spy = _Spy.make()
        point = np.array([0.5, -0.25, 3.0e6])
        direction = np.array([0.5, 0.25, 0.0])
        FiniteDifferenceDerivatives(1.0e-3).taylor(spy, point, direction, 4, 0.0)
        sampled = {tuple(p) for p in spy.points}
        expected = {tuple(point + k * (1.0e-3 * direction)) for k in (-2, -1, 0, 1, 2)}
        assert sampled == expected
        assert FiniteDifferenceDerivatives(1.0e-3).step(point, direction) == 1.0e-3

    def test_the_step_follows_the_component_the_direction_moves_most(self):
        # d = (2, 1e-12): x1 = 4e12 moves eps s_1 per unit of max|d| = 2.
        point = np.array([0.5, 4.0e12])
        direction = np.array([2.0, 1.0e-12])
        step = FiniteDifferenceDerivatives(1.0e-6).step(point, direction)
        # max |d_j| / s_j: 2 / 1 against 1e-12 / 4e12; the first wins.
        assert step == 1.0e-6
        step = FiniteDifferenceDerivatives(1.0e-6).step(point, np.array([0.0, 2.0]))
        assert step == pytest.approx(1.0e-6 * 4.0e12, rel=1.0e-15)


def _separable(scales, point):
    """``f_i = s_i (u_i^3 + u_i^2)``: row ``i`` reads ``x_i`` alone."""
    return [
        s * ((x / s) ** 3 + (x / s) ** 2) for x, s in zip(point, scales, strict=True)
    ]


class _SeparableResidual(ResidualFunction):
    def __init__(self, scales) -> None:
        self.scales = scales

    def evaluate(self, state: np.ndarray, parameter: float) -> np.ndarray:
        del parameter
        return np.array(_separable(self.scales, list(state)), dtype=float)


class _SeparableField(_Field):
    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        return _separable(self.scales, point)


@pytest.mark.parametrize(("scales", "typical"), _CASES)
class TestGroupedColumns:
    """Curtis-Powell-Reid groups move each column by its own step."""

    _ROWS: ClassVar[list[list[int]]] = [[0], [1], [2]]

    def test_continuation_jacobian(self, scales, typical):
        dense = FiniteDifferenceJacobian(
            _SeparableResidual(scales),
            CENTRAL_STEP,
            **_typical(scales, typical),
        )
        grouped = FiniteDifferenceJacobian(
            _SeparableResidual(scales),
            CENTRAL_STEP,
            sparsity=self._ROWS,
            **_typical(scales, typical),
        )
        state = _state(scales)
        expected = dense.state_jacobian(state, 0.0)
        assert np.array_equal(grouped.state_jacobian(state, 0.0), expected)
        slopes = _separable_slopes(scales, state)
        error = np.abs(np.diag(expected) - slopes)
        assert np.all(error <= _separable_allowance(scales, state, CENTRAL_STEP))

    def test_derivative_fallback(self, scales, typical):
        provider = FiniteDifferenceDerivatives(
            CENTRAL_STEP,
            **_typical(scales, typical),
        )
        field = _SeparableField.make(scales)
        state = _state(scales)
        pattern = SparsityPattern.from_rows(self._ROWS, 3)
        dense = provider.jacobian(field, state, 0.0)
        assert np.array_equal(provider.jacobian(field, state, 0.0, pattern), dense)
        error = np.abs(np.diag(dense) - _separable_slopes(scales, state))
        assert np.all(error <= _separable_allowance(scales, state, CENTRAL_STEP))
