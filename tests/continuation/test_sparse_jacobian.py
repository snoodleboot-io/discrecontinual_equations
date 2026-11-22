"""Sparse field Jacobians by column grouping (DEQ-53 item 2).

The grouping is checked against its definition (no row holds two columns of one
group) and against SciPy's independent ``group_columns`` run in the same column
order; the grouped Jacobians against the dense column-by-column ones (bit for bit),
against analytic Jacobians and against SciPy's ``approx_derivative`` with the same
sparsity, on a unidirectional ring and on a periodic two-dimensional lattice.
"""

from unittest import TestCase

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative, group_columns

from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.continuation.jacobian import FiniteDifferenceJacobian
from discrecontinual_equations.continuation.residual import EquationResidual
from discrecontinual_equations.continuation.sparsity import SparsityPattern
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.systems import FerroelectricRing
from discrecontinual_equations.variable import Variable

_EPSILON = 1.0e-7
_SIDE = 5
_FD_TOLERANCE = 1.0e-6


class Coupling(Parameter, name="Coupling", abbreviation="lam"):
    pass


class Gain(Parameter, name="Gain", abbreviation="a"):
    pass


class Bias(Parameter, name="Bias", abbreviation="eps"):
    pass


class Site(Variable, name="Site", abbreviation="x"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class Counted:
    calls = 0


class CountingRing(FerroelectricRing):
    def eval(self, point, time=None):
        Counted.calls += 1
        return super().eval(point, time)


class AllenCahnLattice(DeterministicFunction):
    """Periodic square lattice, ``x' = a x - x^3 + lam (neighbours - 4x) + eps``."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        Counted.calls += 1
        coupling = self.parameters[0].value
        gain = self.parameters[1].value
        bias = self.parameters[2].value
        side = round(len(point) ** 0.5)
        out = []
        for row in range(side):
            for column in range(side):
                here = point[row * side + column]
                neighbours = (
                    point[((row + 1) % side) * side + column]
                    + point[((row - 1) % side) * side + column]
                    + point[row * side + (column + 1) % side]
                    + point[row * side + (column - 1) % side]
                )
                out.append(
                    gain * here
                    - here * here * here
                    + coupling * (neighbours - 4.0 * here)
                    + bias,
                )
        return out


class ProductCoupling(DeterministicFunction):
    """``x_i' = x_i x_(i+1)``: the coupling vanishes where the neighbour is zero."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        count = len(point)
        return [point[i] * point[(i + 1) % count] for i in range(count)]


def _equation(function_type, cells, coupling=0.3, gain=1.0, bias=0.05):
    parameters = [Coupling(value=coupling), Gain(value=gain), Bias(value=bias)]
    variables = [Site() for _ in range(cells)]
    derivative = function_type(
        variables=variables,
        parameters=parameters,
        results=[Site() for _ in range(cells)],
        time=None,
    )
    return DifferentialEquation(
        variables=variables,
        time=Time(),
        parameters=parameters,
        derivative=derivative,
    )


def _ring_rows(cells):
    return [[i, (i + 1) % cells] for i in range(cells)]


def _lattice_rows(side):
    return [
        [
            row * side + column,
            ((row + 1) % side) * side + column,
            ((row - 1) % side) * side + column,
            row * side + (column + 1) % side,
            row * side + (column - 1) % side,
        ]
        for row in range(side)
        for column in range(side)
    ]


def _ring_analytic(state, coupling, gain):
    cells = state.size
    jacobian = np.zeros((cells, cells))
    for i in range(cells):
        jacobian[i, i] = gain - 3.0 * state[i] ** 2
        jacobian[i, (i + 1) % cells] -= coupling
    return jacobian


def _lattice_analytic(state, coupling, gain, side):
    jacobian = np.zeros((side * side, side * side))
    for i, dependencies in enumerate(_lattice_rows(side)):
        jacobian[i, i] = gain - 3.0 * state[i] ** 2 - 4.0 * coupling
        for neighbour in dependencies[1:]:
            jacobian[i, neighbour] += coupling
    return jacobian


def _state(size, seed=3):
    return np.random.default_rng(seed).uniform(-1.2, 1.2, size)


class TestColumnGroups(TestCase):
    def test_groups_are_structurally_orthogonal_on_random_patterns(self):
        generator = np.random.default_rng(7)
        for _ in range(25):
            mask = generator.random((30, 24)) < 0.12
            pattern = SparsityPattern(mask)
            for group in range(pattern.group_count):
                members = mask[:, pattern.groups == group]
                assert members.sum(axis=1).max(initial=0) <= 1

    def test_groups_match_scipy_greedy_in_the_same_column_order(self):
        generator = np.random.default_rng(11)
        for _ in range(25):
            mask = generator.random((40, 40)) < 0.08
            ours = SparsityPattern(mask).groups
            reference = group_columns(mask, order=np.arange(40))
            np.testing.assert_array_equal(ours, reference)

    def test_ring_needs_two_groups_when_even_three_when_odd(self):
        for cells, expected in [(4, 2), (6, 2), (64, 2), (5, 3), (7, 3), (65, 3)]:
            pattern = SparsityPattern.from_rows(_ring_rows(cells), cells)
            assert pattern.group_count == expected, cells

    def test_lattice_groups_are_few_and_independent_of_size(self):
        # The five-point stencil needs at least five groups (a row's non-zeros);
        # the greedy sweep in column order uses 8-10 at every size measured.
        counts = {
            side: SparsityPattern.from_rows(
                _lattice_rows(side),
                side * side,
            ).group_count
            for side in (4, 6, 8, 10, 16, 20)
        }
        assert all(5 <= count <= 10 for count in counts.values()), counts

    def test_dense_pattern_is_one_column_per_group(self):
        pattern = SparsityPattern(np.ones((4, 4), dtype=bool))
        np.testing.assert_array_equal(pattern.groups, np.arange(4))

    def test_expand_places_each_compressed_entry_in_its_column(self):
        mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
        pattern = SparsityPattern(mask)
        np.testing.assert_array_equal(pattern.groups, [0, 0, 1])
        compressed = np.array([[2.0, 3.0], [5.0, 7.0]])
        np.testing.assert_array_equal(
            pattern.expand(compressed),
            [[2.0, 0.0, 3.0], [0.0, 5.0, 0.0]],
        )

    def test_directions_seed_each_group(self):
        mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
        np.testing.assert_array_equal(
            SparsityPattern(mask).directions(),
            [[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )

    def test_detection_keeps_an_entry_seen_only_at_the_given_state(self):
        state = np.array([0.5, -0.25])

        def jacobian(point):
            entry = -1.0 if np.array_equal(point, state) else 0.0
            return np.array([[0.0, entry], [0.0, 0.0]])

        mask = SparsityPattern.detect(jacobian, state).mask
        np.testing.assert_array_equal(mask, [[False, True], [False, False]])

    def test_a_pattern_must_be_a_matrix(self):
        with pytest.raises(ValueError, match="two-dimensional"):
            SparsityPattern(np.ones(3, dtype=bool))


class TestFiniteDifferenceGrouping(TestCase):
    def _check(self, function_type, cells, rows, analytic):
        equation = _equation(function_type, cells)
        residual = EquationResidual(equation, 0, 0.0)
        state = _state(cells)
        dense = FiniteDifferenceJacobian(residual, _EPSILON).state_jacobian(state, 0.3)
        Counted.calls = 0
        provider = FiniteDifferenceJacobian(residual, _EPSILON, sparsity=rows)
        grouped = provider.state_jacobian(state, 0.3)
        calls = Counted.calls
        pattern = SparsityPattern.from_rows(rows, cells)

        np.testing.assert_array_equal(grouped, dense)
        np.testing.assert_allclose(grouped, analytic(state), atol=_FD_TOLERANCE)
        reference = approx_derivative(
            lambda x: residual.evaluate(x, 0.3),
            state,
            method="3-point",
            rel_step=_EPSILON,
            sparsity=(pattern.mask, pattern.groups),
        ).toarray()
        np.testing.assert_allclose(grouped, reference, atol=_FD_TOLERANCE)
        return calls, pattern

    def test_ring_grouped_differences_equal_the_dense_ones(self):
        calls, pattern = self._check(
            CountingRing,
            9,
            _ring_rows(9),
            lambda state: _ring_analytic(state, 0.3, 1.0),
        )
        assert calls == 2 * pattern.group_count == 6

    def test_lattice_grouped_differences_equal_the_dense_ones(self):
        calls, pattern = self._check(
            AllenCahnLattice,
            _SIDE * _SIDE,
            _lattice_rows(_SIDE),
            lambda state: _lattice_analytic(state, 0.3, 1.0, _SIDE),
        )
        assert calls == 2 * pattern.group_count < 2 * _SIDE * _SIDE

    def test_detected_pattern_is_the_ring_and_is_found_once(self):
        cells = 8
        equation = _equation(CountingRing, cells)
        residual = EquationResidual(equation, 0, 0.0)
        provider = FiniteDifferenceJacobian(residual, _EPSILON, sparsity="detect")
        dense = FiniteDifferenceJacobian(residual, _EPSILON)
        first = provider.state_jacobian(_state(cells), 0.3)
        Counted.calls = 0
        second = provider.state_jacobian(_state(cells, seed=4), 0.3)
        calls = Counted.calls

        expected = SparsityPattern.from_rows(_ring_rows(cells), cells).mask
        np.testing.assert_array_equal(provider._pattern.mask, expected)  # noqa: SLF001
        np.testing.assert_array_equal(first, dense.state_jacobian(_state(cells), 0.3))
        np.testing.assert_array_equal(
            second,
            dense.state_jacobian(_state(cells, seed=4), 0.3),
        )
        assert calls == 2 * 2

    def test_detection_sees_a_coupling_that_vanishes_at_the_first_point(self):
        cells = 6
        equation = _equation(ProductCoupling, cells)
        residual = EquationResidual(equation, 0, 0.0)
        provider = FiniteDifferenceJacobian(residual, _EPSILON, sparsity="detect")
        provider.state_jacobian(np.zeros(cells), 0.3)

        expected = SparsityPattern.from_rows(_ring_rows(cells), cells).mask
        np.testing.assert_array_equal(provider._pattern.mask, expected)  # noqa: SLF001

    def test_a_declared_pattern_object_is_used_as_given(self):
        cells = 6
        equation = _equation(CountingRing, cells)
        residual = EquationResidual(equation, 0, 0.0)
        pattern = SparsityPattern.from_rows(_ring_rows(cells), cells)
        provider = FiniteDifferenceJacobian(residual, _EPSILON, sparsity=pattern)
        state = _state(cells)
        np.testing.assert_array_equal(
            provider.state_jacobian(state, 0.3),
            FiniteDifferenceJacobian(residual, _EPSILON).state_jacobian(state, 0.3),
        )


class TestAutomaticDifferentiationGrouping(TestCase):
    def test_lattice_grouped_jets_equal_the_dense_ones(self):
        side = 6
        equation = _equation(AllenCahnLattice, side * side)
        state = _state(side * side)
        pattern = SparsityPattern.from_rows(_lattice_rows(side), side * side)
        provider = AutomaticDifferentiation()
        Counted.calls = 0
        grouped = provider.jacobian(equation.derivative, state, 0.0, sparsity=pattern)
        calls = Counted.calls
        dense = provider.jacobian(equation.derivative, state, 0.0)

        np.testing.assert_array_equal(grouped, dense)
        np.testing.assert_allclose(
            grouped,
            _lattice_analytic(state, 0.3, 1.0, side),
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        assert calls == pattern.group_count

    def test_ring_grouped_jets_equal_the_analytic_jacobian(self):
        cells = 11
        equation = _equation(CountingRing, cells)
        state = _state(cells)
        pattern = SparsityPattern.from_rows(_ring_rows(cells), cells)
        grouped = AutomaticDifferentiation().jacobian(
            equation.derivative,
            state,
            0.0,
            sparsity=pattern,
        )
        np.testing.assert_allclose(
            grouped,
            _ring_analytic(state, 0.3, 1.0),
            rtol=1.0e-13,
            atol=1.0e-13,
        )


class TestSparseContinuation(TestCase):
    def _branch(self, sparsity):
        cells = 12
        equation = _equation(CountingRing, cells, coupling=0.1, bias=0.0)
        config = ContinuationConfig(
            continuation_parameter_index=2,
            initial_parameter=0.0,
            direction=-1,
            detectors=["fold", "branch_point"],
            maximum_points=15,
            initial_step=0.05,
            maximum_step=0.05,
            jacobian_sparsity=sparsity,
        )
        Counted.calls = 0
        branch = ContinuerBuilder.build(config, equation).solve(
            equation,
            [float(np.sqrt(0.9))] * cells,
        )
        return branch, Counted.calls

    def test_declared_and_detected_sparsity_reproduce_the_dense_branch(self):
        dense, dense_calls = self._branch("dense")
        declared, declared_calls = self._branch(_ring_rows(12))
        detected, detected_calls = self._branch("detect")

        reference = [p.model_dump() for p in dense.points + dense.special_points]
        assert [p.model_dump() for p in declared.points + declared.special_points] == (
            reference
        )
        assert [p.model_dump() for p in detected.points + detected.special_points] == (
            reference
        )
        assert declared_calls < dense_calls / 2
        assert declared_calls < detected_calls < dense_calls
