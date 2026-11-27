"""The Jacobian every Lyapunov estimator reads, on badly scaled states.

``jacobian_at`` forward-differences a field. Its error in entry ``(i, j)`` is the
truncation ``h_j |d2 f_i / dx_j2| / 2`` plus the rounding ``~ 2 eps |f_i| / h_j``
(Dennis and Schnabel 1983, Sec. 5.4), so the step that balances them scales with
the size of component ``j`` - with its value, or with its typical magnitude where
the value is near zero. The states here have components that differ by ``1e8`` and
``1e12``, in both directions, and every entry is held to the accuracy a step of
that size allows. The reference Jacobians are written out by hand and checked
first against exact rational differences, so nothing here is checked against the
code under test.
"""

import time
from fractions import Fraction
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.jacobian_field import JacobianField
from discrecontinual_equations.continuation.lyapunov import (
    LyapunovSettings,
    lyapunov_spectrum,
)
from discrecontinual_equations.continuation.noise import (
    field_at,
    fields_at,
    jacobian_at,
    jacobians_along,
)
from discrecontinual_equations.continuation.region_analysis import describe_region
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import LorenzField, State

_EPS = float(np.finfo(float).eps)
# Slack on the error model: the rounding term counts one ulp per evaluation and
# the true rounding of a few-term sum is a small multiple of that.
_SLACK = 4.0
# The state in units of each component's scale: of order one, with a negative
# entry so the step's sign is exercised.
_UNITS = (0.7, -1.3, 0.9)
_EXACT_STEP = Fraction(1, 10**40)


class _Scaled(DeterministicFunction):
    """A polynomial field that is nonlinear in each component at that one's scale.

    In scaled units ``u_j = x_j / s_j`` (output ``i`` also scaled by ``s_i``)::

        f0 = s0 (u0 u1 + u2^2)
        f1 = s1 (u1 u2 + u1^2) + x0
        f2 = s2 (u1^3 + u0^2 u2)

    ``df1/dx0 = 1`` sits in a row of size ``s1``: with ``s1 = 1e8`` it is the
    entry a forward difference cannot resolve, the one the DEQ-41 lane hit.
    The scales are integers or ``Fraction``s so the field can be evaluated exactly.
    """

    scales: tuple[Fraction | int, ...] = (1, 1, 1)

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        s0, s1, s2 = self.scales
        u0, u1, u2 = point[0] / s0, point[1] / s1, point[2] / s2
        return [
            s0 * (u0 * u1 + u2 * u2),
            s1 * (u1 * u2 + u1 * u1) + point[0],
            s2 * (u1 * u1 * u1 + u0 * u0 * u2),
        ]


class _Large(_Scaled):
    scales = (1, 10**8, 10**12)


class _Small(_Scaled):
    scales = (1, Fraction(1, 10**8), Fraction(1, 10**12))


def _field(kind: type[_Scaled]) -> _Scaled:
    return kind(
        variables=[State(), State(), State()],
        parameters=[],
        results=[State(), State(), State()],
        time=None,
    )


def _state(kind: type[_Scaled]) -> np.ndarray:
    return np.array([float(u * s) for u, s in zip(_UNITS, kind.scales, strict=True)])


def _units(kind: type[_Scaled], state: np.ndarray) -> list[float]:
    return [x / float(s) for x, s in zip(state, kind.scales, strict=True)]


def _analytic(kind: type[_Scaled], state: np.ndarray) -> np.ndarray:
    """The Jacobian, by hand."""
    s0, s1, s2 = (float(s) for s in kind.scales)
    u0, u1, u2 = _units(kind, state)
    return np.array(
        [
            [u1 * s0 / s0, u0 * s0 / s1, 2.0 * u2 * s0 / s2],
            [1.0, s1 * (u2 + 2.0 * u1) / s1, s1 * u1 / s2],
            [s2 * 2.0 * u0 * u2 / s0, s2 * 3.0 * u1 * u1 / s1, s2 * u0 * u0 / s2],
        ],
    )


def _curvature(kind: type[_Scaled], state: np.ndarray) -> np.ndarray:
    """``|d2 f_i / dx_j2|``, by hand."""
    s0, s1, s2 = (float(s) for s in kind.scales)
    _, u1, u2 = _units(kind, state)
    return np.abs(
        np.array(
            [
                [0.0, 0.0, 2.0 * s0 / s2**2],
                [0.0, 2.0 * s1 / s1**2, 0.0],
                [2.0 * s2 * u2 / s0**2, 6.0 * s2 * u1 / s1**2, 0.0],
            ],
        ),
    )


def _third(kind: type[_Scaled]) -> np.ndarray:
    """``|d3 f_i / dx_j3|``, by hand: only ``f2`` is cubic, and only in ``x1``."""
    s1, s2 = float(kind.scales[1]), float(kind.scales[2])
    third = np.zeros((3, 3))
    third[2, 1] = 6.0 * s2 / s1**3
    return third


def _magnitude(kind: type[_Scaled], state: np.ndarray) -> np.ndarray:
    """Sum of the absolute terms of each ``f_i``: what its rounding is relative to."""
    s0, s1, s2 = (float(s) for s in kind.scales)
    u0, u1, u2 = (abs(u) for u in _units(kind, state))
    return np.array(
        [
            s0 * (u0 * u1 + u2 * u2),
            s1 * (u1 * u2 + u1 * u1) + abs(state[0]),
            s2 * (u1**3 + u0 * u0 * u2),
        ],
    )


def _forward_allowance(kind: type[_Scaled], state: np.ndarray) -> np.ndarray:
    """What a forward difference stepped at each component's own scale can reach.

    ``h_j = sqrt(eps) s_j`` balances truncation against rounding when the field
    varies on the scale ``s_j``; at that step the error is
    ``sqrt(eps) (s_j M_ij / 2 + 2 F_i / s_j)``.
    """
    scale = np.array([float(s) for s in kind.scales])
    curvature = _curvature(kind, state)
    magnitude = _magnitude(kind, state)[:, None]
    root = np.sqrt(_EPS)
    return root * (scale * curvature / 2.0 + 2.0 * magnitude / scale)


def _central_allowance(kind: type[_Scaled], state: np.ndarray) -> np.ndarray:
    """The same for a central difference at ``h_j = eps^(1/3) s_j``.

    Truncation ``h^2 T / 6`` against rounding ``eps F / h``.
    """
    scale = np.array([float(s) for s in kind.scales])
    magnitude = _magnitude(kind, state)[:, None]
    cube = _EPS ** (1.0 / 3.0)
    return cube**2 * (scale**2 * _third(kind) / 6.0 + magnitude / scale)


def _exact_jacobian(kind: type[_Scaled], state: np.ndarray) -> np.ndarray:
    """Central differences in exact rational arithmetic, with a step of ``1e-40``.

    The field is a cubic polynomial, so the only error is ``h^2 f''' / 6``, some
    ``1e-80`` relative: an independent check on the hand-written Jacobian.
    """
    field = _field(kind)
    point = [Fraction(x) for x in state]
    columns = []
    for j in range(3):
        step = _EXACT_STEP * max(abs(point[j]), Fraction(1))
        up, down = list(point), list(point)
        up[j] += step
        down[j] -= step
        high, low = field.eval(up), field.eval(down)
        pairs = zip(high, low, strict=True)
        columns.append([float((a - b) / (2 * step)) for a, b in pairs])
    return np.array(columns).T


class TestReference(TestCase):
    """The hand-written Jacobian is right before anything is compared to it."""

    def test_large_scales(self):
        state = _state(_Large)
        exact = _exact_jacobian(_Large, state)
        assert np.allclose(_analytic(_Large, state), exact, rtol=1.0e-14, atol=0.0)

    def test_small_scales(self):
        state = _state(_Small)
        exact = _exact_jacobian(_Small, state)
        assert np.allclose(_analytic(_Small, state), exact, rtol=1.0e-14, atol=0.0)


class TestForwardStep(TestCase):
    """Every entry to the accuracy a forward difference allows."""

    def _check(self, kind: type[_Scaled], typical: np.ndarray | None) -> None:
        state = _state(kind)
        computed = jacobian_at(_field(kind), state, typical=typical)
        analytic = _analytic(kind, state)
        allowance = _SLACK * _forward_allowance(kind, state) + _EPS * np.abs(analytic)
        error = np.abs(computed - analytic)
        assert np.all(error <= allowance), error / allowance

    def test_components_1e8_and_1e12_larger(self):
        # Large components need no typical size: their value is their scale.
        self._check(_Large, None)

    def test_components_1e8_and_1e12_smaller(self):
        # Small components need one, or the default of one swamps them.
        self._check(_Small, np.array([float(s) for s in _Small.scales]))

    def test_without_a_typical_size_small_components_are_swamped(self):
        # The step floors at the typical size, one by default, and a step of
        # 1.5e-8 on a component of size 1e-12 is all truncation: f0 is quadratic
        # in x2 at that scale.
        state = _state(_Small)
        computed = jacobian_at(_field(_Small), state)
        error = np.abs(computed - _analytic(_Small, state))
        assert error[0, 2] > 1.0e3 * _SLACK * _forward_allowance(_Small, state)[0, 2]

    def test_a_non_positive_typical_size_is_refused(self):
        state = _state(_Large)
        with pytest.raises(ValueError, match="typical"):
            jacobian_at(_field(_Large), state, typical=np.array([1.0, 0.0, 1.0]))
        with pytest.raises(ValueError, match="typical"):
            jacobian_at(_field(_Large), state, typical=np.array([1.0, -1.0, 1.0]))

    def test_a_scalar_typical_size_applies_to_every_component(self):
        state = _state(_Large)
        field = _field(_Large)
        scalar = jacobian_at(field, state, typical=2.0)
        vector = jacobian_at(field, state, typical=np.full(3, 2.0))
        assert np.array_equal(scalar, vector)

    def test_the_step_actually_taken_is_divided_by(self):
        # f(x) = x: the difference of the field is exactly the step the machine
        # took, so dividing by that step gives the identity exactly. Dividing by
        # the nominal step instead is off by up to sqrt(eps) relative.
        state = np.array([0.1, -3.7, 1.0e8 + 0.3])
        for central in (False, True):
            computed = jacobian_at(_Identity.make(), state, central=central)
            assert np.array_equal(computed, np.eye(3))

    def test_the_step_moves_away_from_zero(self):
        # Dennis and Schnabel step by sign(x_j) h_j, so a one-sided difference
        # never crosses zero from a component's side of it.
        spy = _Identity.make()
        state = np.array([2.0, -2.0, 0.0])
        jacobian_at(spy, state)
        moved = [np.array(point) - state for point in spy.points[1:]]
        assert moved[0][0] > 0.0
        assert moved[1][1] < 0.0
        assert moved[2][2] > 0.0

    def test_step_is_sqrt_eps_times_scale(self):
        spy = _Identity.make()
        state = np.array([3.0, 0.25, -5.0e9])
        jacobian_at(spy, state, typical=np.array([1.0, 1.0e-3, 1.0]))
        steps = [abs(np.array(point) - state).max() for point in spy.points[1:]]
        expected = np.sqrt(_EPS) * np.array([3.0, 0.25, 5.0e9])
        assert np.allclose(steps, expected, rtol=1.0e-6)

    def test_typical_size_defaults_to_one(self):
        # Dennis and Schnabel's typx, IMSL's XSCALE and UNCMIN's sx all default
        # to one: a component below one in size is stepped as if it were one.
        spy = _Identity.make()
        state = np.array([0.0, 0.25, -1.0e-6])
        jacobian_at(spy, state)
        steps = [abs(np.array(point) - state).max() for point in spy.points[1:]]
        assert np.allclose(steps, np.sqrt(_EPS), rtol=1.0e-6)


class TestCentralStep(TestCase):
    """A central difference trades two evaluations per column for ``eps^(2/3)``."""

    def _check(self, kind: type[_Scaled], typical: np.ndarray | None) -> None:
        state = _state(kind)
        computed = jacobian_at(_field(kind), state, typical=typical, central=True)
        analytic = _analytic(kind, state)
        allowance = _SLACK * _central_allowance(kind, state) + _EPS * np.abs(analytic)
        error = np.abs(computed - analytic)
        assert np.all(error <= allowance), error / allowance

    def test_components_1e8_and_1e12_larger(self):
        self._check(_Large, None)

    def test_components_1e8_and_1e12_smaller(self):
        self._check(_Small, np.array([float(s) for s in _Small.scales]))

    def test_resolves_the_entry_forward_differences_cannot(self):
        # df1/dx0 = 1 in a row of size 1e8: forward rounding is up to
        # eps 1e8 / 1.5e-8 ~ 1.5, central eps^(2/3) 1e8 ~ 4e-3. At one state either
        # can round luckily, so take the worst over many.
        generator = np.random.default_rng(3)
        scales = np.array([float(s) for s in _Large.scales])
        forward, central = [], []
        for _ in range(50):
            state = generator.uniform(0.5, 1.5, 3) * scales
            forward.append(abs(jacobian_at(_field(_Large), state)[1, 0] - 1.0))
            difference = jacobian_at(_field(_Large), state, central=True)[1, 0]
            central.append(abs(difference - 1.0))
        assert max(forward) > 0.1
        assert max(central) < 1.0e-2

    def test_step_is_cube_root_eps_times_scale(self):
        spy = _Identity.make()
        state = np.array([3.0, 0.25, -5.0e9])
        jacobian_at(spy, state, typical=np.array([1.0, 1.0e-3, 1.0]), central=True)
        # No evaluation at the state itself: every point is a step.
        steps = [abs(np.array(point) - state).max() for point in spy.points]
        cube = _EPS ** (1.0 / 3.0)
        expected = cube * np.array([3.0, 3.0, 0.25, 0.25, 5.0e9, 5.0e9])
        assert np.allclose(steps, expected, rtol=1.0e-6)


class _Identity(DeterministicFunction):
    """``f(x) = x``, recording every point it is evaluated at."""

    points: list[list[float]]

    @classmethod
    def make(cls) -> "_Identity":
        field = cls(
            variables=[State(), State(), State()],
            parameters=[],
            results=[State(), State(), State()],
            time=None,
        )
        field.points = []
        return field

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        self.points.append(list(point))
        return list(point)


def _lorenz() -> LorenzField:
    return LorenzField(
        variables=[State(), State(), State()],
        parameters=[],
        results=[State(), State(), State()],
        time=None,
    )


def _lorenz_jacobian(state: np.ndarray) -> np.ndarray:
    x, y, z = state
    return np.array(
        [[-10.0, 10.0, 0.0], [28.0 - z, -1.0, -x], [y, x, -8.0 / 3.0]],
    )


class TestJacobianField(TestCase):
    """A field that carries its own Jacobian, which ``jacobian_at`` then returns."""

    def test_evaluates_as_the_field_it_wraps(self):
        field = _field(_Large)
        wrapped = JacobianField(field, lambda state: _analytic(_Large, state))
        state = _state(_Large)
        assert wrapped.eval(point=list(state)) == field.eval(point=list(state))
        assert wrapped.variables is field.variables
        assert wrapped.parameters is field.parameters
        assert wrapped.time is field.time

    def test_supplied_jacobian_is_used_without_differencing(self):
        spy = _Identity.make()
        supplied = np.arange(9.0).reshape(3, 3)
        wrapped = JacobianField(spy, lambda _state: supplied)
        assert np.array_equal(jacobian_at(wrapped, np.ones(3)), supplied)
        assert spy.points == []

    def test_taylor_jets_by_default(self):
        # Exact to rounding even in the entry no difference can resolve.
        for kind in (_Large, _Small):
            state = _state(kind)
            computed = jacobian_at(JacobianField(_field(kind)), state)
            analytic = _analytic(kind, state)
            scale = np.array([float(s) for s in kind.scales])
            bound = 8.0 * _EPS * _magnitude(kind, state)[:, None] / scale
            assert np.all(np.abs(computed - analytic) <= bound + _EPS * abs(analytic))
        assert jacobian_at(JacobianField(_field(_Large)), _state(_Large))[1, 0] == 1.0

    def test_jets_agree_with_the_closed_form_on_lorenz(self):
        state = np.array([1.5, -2.0, 24.0])
        jets = jacobian_at(JacobianField(_lorenz()), state)
        assert np.array_equal(jets, _lorenz_jacobian(state))


def _lorenz_top_over_t_1000() -> float:
    spectrum = lyapunov_spectrum(
        JacobianField(_lorenz(), _lorenz_jacobian),
        np.array([1.0, 1.0, 1.0]),
        count=1,
        settings=LyapunovSettings(dt=0.01, horizon=100000, transient=1000),
    )
    return spectrum.top


class TestLorenzWithAnalyticJacobian(TestCase):
    """Lorenz's top exponent over ``T = 1000``, which CI can now afford."""

    def test_top_exponent_at_t_1000(self):
        # Viswanath (1998) gives 0.9056; T = 1000 holds it to about a percent.
        assert abs(_lorenz_top_over_t_1000() - 0.9056) < 1.0e-2

    def test_analytic_and_differenced_jacobians_give_the_same_exponent(self):
        settings = LyapunovSettings(dt=0.01, horizon=10000, transient=1000)
        start = np.array([1.0, 1.0, 1.0])
        differenced = lyapunov_spectrum(_lorenz(), start, count=1, settings=settings)
        analytic = lyapunov_spectrum(
            JacobianField(_lorenz(), _lorenz_jacobian),
            start,
            count=1,
            settings=settings,
        )
        assert abs(differenced.top - analytic.top) < 1.0e-6


def _batch(kind: type[_Scaled]) -> np.ndarray:
    """Rows on both sides of zero, one with a component exactly at zero."""
    state = _state(kind)
    rows = [state * factor for factor in (1.0, -0.3, 2.5, 1.0e-3)]
    rows.append(state * np.array([0.0, 1.0, -1.0]))
    return np.array(rows)


class _Summing(DeterministicFunction):
    """``f_i = x_i * sum(x)``, summed by the builtin ``sum``."""

    @classmethod
    def make(cls) -> "_Summing":
        return cls(
            variables=[State(), State(), State()],
            parameters=[],
            results=[State(), State(), State()],
            time=None,
        )

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        total = sum(value * value for value in point)
        return [value * total for value in point]


class TestBatchedJacobians(TestCase):
    """``jacobians_along`` takes ``jacobian_at``'s steps, row by row, to the bit."""

    def _check(self, kind: type[_Scaled], **options) -> None:
        field = _field(kind)
        states = _batch(kind)
        batched = jacobians_along(field, states, **options)
        for row, state in enumerate(states):
            pointwise = jacobian_at(field, state, **options)
            np.testing.assert_array_equal(batched[row], pointwise)

    def test_forward_rows_equal_the_pointwise_jacobian(self):
        self._check(_Large)

    def test_forward_rows_with_typical_sizes(self):
        self._check(_Small, typical=np.array([float(s) for s in _Small.scales]))

    def test_central_rows_equal_the_pointwise_jacobian(self):
        self._check(_Large, central=True)
        self._check(_Small, typical=1.0e-9, central=True)

    def test_a_base_in_hand_changes_nothing(self):
        field = _field(_Large)
        states = _batch(_Large)
        base = fields_at(field, states)
        np.testing.assert_array_equal(
            jacobians_along(field, states, base),
            jacobians_along(field, states),
        )
        for state in states:
            # ``base`` is still the third positional argument, as callers pass it.
            np.testing.assert_array_equal(
                jacobian_at(field, state, field_at(field, state)),
                jacobian_at(field, state),
            )

    def test_a_field_that_sums_agrees_batched_and_pointwise(self):
        # The builtin sum compensates Python floats but not NumPy scalars, so the
        # batched and single-state paths must hand a field the same kind of float.
        field = _Summing.make()
        states = np.array([[0.1, 0.2, 0.3], [-1.7, 2.9, 1.0e-3], [3.0, -0.7, 0.11]])
        base = fields_at(field, states)
        jacobians = jacobians_along(field, states, base)
        for row, state in enumerate(states):
            np.testing.assert_array_equal(base[row], field_at(field, state))
            np.testing.assert_array_equal(jacobians[row], jacobian_at(field, state))

    def test_a_jacobian_field_answers_every_row_itself(self):
        wrapped = JacobianField(_field(_Large))
        states = _batch(_Large)
        batched = jacobians_along(wrapped, states)
        for row, state in enumerate(states):
            np.testing.assert_array_equal(batched[row], jacobian_at(wrapped, state))


# An equilibrium far from the origin, at a power of two so every evaluation of
# the linear field below is exact: an ulp there is 2^-18, about 4e-6, so an
# absolute step of 1e-7 rounds away and Newton would be handed a zero Jacobian.
_FAR = float(2**34)


class _FarFocus(DeterministicFunction):
    """A stable focus at ``(2^34, -2^34)``: eigenvalues ``-1 +- i``."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        u, v = point[0] - _FAR, point[1] + _FAR
        return [-u - v, u - v]


class TestRegionAnalysisUsesTheSharedStep(TestCase):
    """``region_analysis`` classifies with ``jacobian_at``, not an absolute step."""

    def test_finds_and_classifies_an_equilibrium_far_from_the_origin(self):
        field = _FarFocus(
            variables=[State(), State()],
            parameters=[],
            results=[State(), State()],
            time=None,
        )
        seed = np.array([_FAR + 1024.0, -_FAR - 512.0])
        assert describe_region(field, [seed]) == "stable focus"


@pytest.mark.timing
def test_lorenz_top_exponent_at_t_1000_takes_under_30_seconds():
    # Wall time, as CI sees it; about 8 s of CPU on one thread.
    start = time.perf_counter()
    _lorenz_top_over_t_1000()
    assert time.perf_counter() - start < 30.0
