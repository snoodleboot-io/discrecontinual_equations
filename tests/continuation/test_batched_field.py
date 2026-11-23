"""One-pass field values and Jacobians: the numbers of a state at a time (DEQ-75)."""

import math
from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.batched_field import (
    field_jacobians,
    field_values,
)
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.periodic_orbit import (
    PeriodicOrbit,
    PeriodicOrbitSolution,
    _hermite_positions,
)
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import Alpha, HopfField, SnicField, State

_AUTODIFF = AutomaticDifferentiation()


class _EveryOperation(DeterministicFunction):
    """Each operation the duals mirror, so a wrong term in any one shows."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        a = self.parameters[0].value
        x, y, z = point[0], point[1], point[2]
        return [
            x * y - z / (1.0 + x * x) + a * x**3,
            2.0 - y + (x - 0.5) * z**-2 + 3.0 / (2.0 + y * y),
            -z + x / y + (y + 1.5) ** 4 - 1.25,
            0.75,
        ]


class _Branching(DeterministicFunction):
    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x, y = point[0], point[1]
        if x * x + y * y < 1.0e-12:
            return [0.0, 0.0]
        return [y, -x]


class _MathModule(DeterministicFunction):
    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        return [math.sin(point[0]), point[1]]


class _RowVector(DeterministicFunction):
    """Silently wrong on arrays: right only when the state count is the dimension."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        rotation = np.array([[0.0, 1.0], [-2.0, 0.5]])
        return list(np.asarray(point, dtype=float).T @ rotation)


class _Smallest(DeterministicFunction):
    """Silently wrong on arrays after the first state: a reduction over them.

    ``x.min()`` of one state's coordinate is the coordinate; of the batch it is
    the smallest over every state, right for the state that holds it and wrong
    for the rest.
    """

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x, y = point[0], point[1]
        return [y, -x.min()]


def _function(kind: type, dimension: int, value: float = 0.3) -> DeterministicFunction:
    return kind(
        variables=[State() for _ in range(dimension)],
        parameters=[Alpha(value=value)],
        results=[State() for _ in range(dimension)],
        time=None,
    )


def _states(count: int, dimension: int) -> np.ndarray:
    generator = np.random.default_rng(75)
    return 0.4 + generator.uniform(0.2, 1.5, size=(count, dimension))


def _single_values(function: DeterministicFunction):
    return lambda state: np.array(
        function.eval(point=list(state), time=None),
        dtype=float,
    )


def _single_jacobian(function: DeterministicFunction):
    return lambda state: _AUTODIFF.jacobian(function, state, 0.0)


class TestFieldValues(TestCase):
    def test_values_equal_a_state_at_a_time_bit_for_bit(self) -> None:
        function = _function(_EveryOperation, 3)
        states = _states(50, 3)
        values = field_values(function, states, _single_values(function))
        expected = np.array([_single_values(function)(state) for state in states])
        assert values is not None
        np.testing.assert_array_equal(values, expected)

    def test_a_constant_component_is_broadcast(self) -> None:
        function = _function(_EveryOperation, 3)
        values = field_values(function, _states(7, 3), _single_values(function))
        assert values is not None
        np.testing.assert_array_equal(values[:, 3], np.full(7, 0.75))

    def test_a_field_that_branches_on_its_state_is_refused(self) -> None:
        function = _function(_Branching, 2)
        assert field_values(function, _states(9, 2), _single_values(function)) is None

    def test_a_math_module_field_is_refused(self) -> None:
        function = _function(_MathModule, 2)
        assert field_values(function, _states(9, 2), _single_values(function)) is None

    def test_a_silently_wrong_field_is_caught_by_the_spot_check(self) -> None:
        function = _function(_RowVector, 2)
        # Two states in two dimensions: the row-vector product runs on arrays
        # and returns the transposed answer, which only the comparison sees.
        assert field_values(function, _states(2, 2), _single_values(function)) is None

    def test_the_last_state_is_checked_as_well_as_the_first(self) -> None:
        function = _function(_Smallest, 2)
        states = _states(9, 2)
        states = states[np.argsort(states[:, 0])]
        # Right in the first row, which holds the smallest x; wrong after it.
        assert field_values(function, states, _single_values(function)) is None

    def test_one_state_is_left_to_the_caller(self) -> None:
        function = _function(HopfField, 2)
        assert field_values(function, _states(1, 2), _single_values(function)) is None


class TestFieldJacobians(TestCase):
    def test_jacobians_equal_the_taylor_jets_bit_for_bit(self) -> None:
        function = _function(_EveryOperation, 3)
        states = _states(64, 3)
        jacobians = field_jacobians(function, states, _single_jacobian(function))
        expected = np.array([_single_jacobian(function)(state) for state in states])
        assert jacobians is not None
        assert jacobians.shape == (64, 4, 3)
        np.testing.assert_array_equal(jacobians, expected)

    def test_jacobians_match_central_differences(self) -> None:
        # An oracle that shares no code with the duals or the jets.
        function = _function(_EveryOperation, 3)
        states = _states(5, 3)
        jacobians = field_jacobians(function, states, _single_jacobian(function))
        assert jacobians is not None
        values = _single_values(function)
        step = 1.0e-6
        for state, jacobian in zip(states, jacobians, strict=True):
            for j in range(3):
                shift = np.zeros(3)
                shift[j] = step
                column = (values(state + shift) - values(state - shift)) / (2 * step)
                np.testing.assert_allclose(jacobian[:, j], column, rtol=1e-7, atol=1e-8)

    def test_a_fractional_power_is_refused(self) -> None:
        function = _function(SnicField, 2)
        assert (
            field_jacobians(function, _states(9, 2), _single_jacobian(function)) is None
        )

    def test_a_branching_field_is_refused(self) -> None:
        function = _function(_Branching, 2)
        assert (
            field_jacobians(function, _states(9, 2), _single_jacobian(function)) is None
        )


class TestBatchedMonodromy(TestCase):
    """The monodromy built from one batch of propagators, against an oracle."""

    def _hopf(
        self, value: float, intervals: int
    ) -> tuple[PeriodicOrbit, PeriodicOrbitSolution]:
        function = _function(HopfField, 2, value)
        grid = np.linspace(0.0, 1.0, intervals + 1)
        radius = math.sqrt(value)
        states = np.column_stack(
            [radius * np.cos(2 * math.pi * grid), radius * np.sin(2 * math.pi * grid)],
        )
        return PeriodicOrbit(function, intervals, 1, 0.0), PeriodicOrbitSolution(
            states,
            2 * math.pi,
        )

    def test_the_product_equals_carrying_the_transition_through(self) -> None:
        orbit, solution = self._hopf(0.5, 40)
        transition = np.eye(2)
        # The pre-DEQ-75 loop, written out: Y carried through every RK4 stage.
        states, period = solution.states, solution.period
        fields = [orbit._field(state) for state in states]  # noqa: SLF001
        slopes = period * np.array(fields)
        fractions = np.linspace(0.0, 1.0, 9)
        for i in range(40):
            # The mesh's own widths: a position one ulp off moves a forward
            # difference with step 1e-7 by about 1e-9, which is not rounding.
            step = float(orbit.mesh[i + 1] - orbit.mesh[i])
            positions = _hermite_positions(
                states[i : i + 2],
                slopes[i : i + 2],
                step,
                fractions,
            )
            jacobian = orbit._jacobian  # noqa: SLF001
            matrices = [period * jacobian(position) for position in positions]
            substep = step / 4
            for k in range(4):
                start, middle, end = matrices[2 * k : 2 * k + 3]
                k1 = start @ transition
                k2 = middle @ (transition + 0.5 * substep * k1)
                k3 = middle @ (transition + 0.5 * substep * k2)
                k4 = end @ (transition + substep * k3)
                transition = transition + substep / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        np.testing.assert_allclose(orbit.monodromy(solution), transition, atol=4e-15)

    def test_the_multipliers_are_the_hopf_cycles(self) -> None:
        # Exact: 1 and exp(-2 mu T) = exp(-4 pi mu) on r' = r (mu - r^2).
        orbit, solution = self._hopf(0.25, 80)
        multipliers = np.sort(orbit.floquet_multipliers(solution).real)
        np.testing.assert_allclose(
            multipliers,
            [math.exp(-math.pi), 1.0],
            rtol=1e-6,
        )
