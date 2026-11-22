"""One Jacobian per accepted equilibrium point beyond the corrector's own (DEQ-53).

The corrector builds a Jacobian at each Newton iterate; the converged point then
needs one more, shared by the tangent, the branch-point and Hopf test functions and
the point's stability. These tests count field evaluations between consecutive
corrector calls on a ring of ``n`` cells: one extended Jacobian is ``2n + 2``
evaluations, and before the reuse the same stretch cost ``8n + 4``.
"""

from unittest import TestCase

import numpy as np

from discrecontinual_equations.continuation.builder import ContinuerBuilder
from discrecontinual_equations.continuation.continuation_config import (
    ContinuationConfig,
)
from discrecontinual_equations.continuation.corrector import (
    CorrectionResult,
    Corrector,
)
from discrecontinual_equations.continuation.jacobian import (
    FiniteDifferenceJacobian,
    MemoizedJacobian,
)
from discrecontinual_equations.continuation.residual import ResidualFunction
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.systems import FerroelectricRing
from discrecontinual_equations.variable import Variable

_CELLS = 6
_EPSILON = 1.0e-7


class Coupling(Parameter, name="Coupling", abbreviation="lam"):
    pass


class Gain(Parameter, name="Gain", abbreviation="a"):
    pass


class Bias(Parameter, name="Bias", abbreviation="eps"):
    pass


class Cell(Variable, name="Cell", abbreviation="x"):
    pass


class Time(Variable, name="Time", abbreviation="t"):
    pass


class CountingRing(FerroelectricRing):
    """The ferroelectric ring, counting its evaluations."""

    calls = 0

    def eval(self, point, time=None):
        CountingRing.calls += 1
        return super().eval(point, time)


class RecordingCorrector(Corrector):
    """Delegate to the wired corrector, noting the evaluation count around each call."""

    def __init__(self, inner: Corrector) -> None:
        self._inner = inner
        self.calls: list[tuple[int, int, bool]] = []

    def correct(self, predicted_state, predicted_parameter, tangent):
        before = CountingRing.calls
        result: CorrectionResult = self._inner.correct(
            predicted_state,
            predicted_parameter,
            tangent,
        )
        self.calls.append((before, CountingRing.calls, result.converged))
        return result


class CountingResidual(ResidualFunction):
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, state: np.ndarray, parameter: float) -> np.ndarray:
        self.calls += 1
        return np.array([parameter * state[0] - state[1] ** 3, state[0] * state[1]])


def _ring(cells: int) -> DifferentialEquation:
    parameters = [Coupling(value=0.1), Gain(value=1.0), Bias(value=0.0)]
    variables = [Cell() for _ in range(cells)]
    derivative = CountingRing(
        variables=variables,
        parameters=parameters,
        results=[Cell() for _ in range(cells)],
        time=None,
    )
    return DifferentialEquation(
        variables=variables,
        time=Time(),
        parameters=parameters,
        derivative=derivative,
    )


def _config(detectors: list[str]) -> ContinuationConfig:
    return ContinuationConfig(
        continuation_parameter_index=2,
        initial_parameter=0.0,
        direction=1,
        detectors=detectors,
        maximum_points=12,
        initial_step=0.05,
        maximum_step=0.05,
    )


def _evaluations_between_corrections(detectors: list[str]) -> list[int]:
    equation = _ring(_CELLS)
    config = _config(detectors)
    base = ContinuerBuilder.build(config, equation)
    recorder = RecordingCorrector(base._components.corrector)  # noqa: SLF001
    wired = base._components.model_copy(update={"corrector": recorder})  # noqa: SLF001
    continuer = type(base)(solver_config=config, components=wired)
    continuer.solve(equation, [float(np.sqrt(0.9))] * _CELLS)
    gaps = []
    for earlier, later in zip(recorder.calls, recorder.calls[1:], strict=False):
        _start, end, converged = earlier
        if converged:
            gaps.append(later[0] - end)
    return gaps


class TestJacobianReuse(TestCase):
    def test_one_jacobian_per_point_with_every_detector(self):
        # Given the default detectors on a branch with no bifurcation
        # When the continuation runs
        gaps = _evaluations_between_corrections(["fold", "branch_point", "hopf"])

        # Then each accepted point costs one extended Jacobian after its correction
        assert len(gaps) >= 10
        assert set(gaps) == {2 * _CELLS + 2}

    def test_detectors_add_no_field_evaluations(self):
        without = _evaluations_between_corrections([])
        with_all = _evaluations_between_corrections(["fold", "branch_point", "hopf"])

        assert with_all == without

    def test_branch_is_bit_identical_to_unshared_jacobians(self):
        # Given the same continuation with and without the retained Jacobians
        equation = _ring(_CELLS)
        config = _config(["fold", "branch_point", "hopf"])
        shared = ContinuerBuilder.build(config, equation).solve(
            equation,
            [float(np.sqrt(0.9))] * _CELLS,
        )
        plain = _plain_branch(config, equation)

        # Then every stored number matches exactly
        assert [p.model_dump() for p in shared.points] == [
            p.model_dump() for p in plain.points
        ]


def _plain_branch(config: ContinuationConfig, equation: DifferentialEquation):
    """The same continuation with a memo that never retains (the base no-op scope)."""
    original = MemoizedJacobian.retained
    MemoizedJacobian.retained = FiniteDifferenceJacobian.retained  # type: ignore[method-assign]
    try:
        return ContinuerBuilder.build(config, equation).solve(
            equation,
            [float(np.sqrt(0.9))] * _CELLS,
        )
    finally:
        MemoizedJacobian.retained = original  # type: ignore[method-assign]


class TestMemoizedJacobian(TestCase):
    def setUp(self) -> None:
        self.residual = CountingResidual()
        self.inner = FiniteDifferenceJacobian(self.residual, _EPSILON)
        self.memo = MemoizedJacobian(self.inner)
        self.state = np.array([0.3, -0.7])

    def test_outside_a_retained_scope_every_call_differentiates(self):
        self.memo.state_jacobian(self.state, 0.5)
        self.memo.state_jacobian(self.state, 0.5)

        assert self.residual.calls == 2 * 2 * 2

    def test_inside_a_retained_scope_a_repeat_is_free_and_identical(self):
        with self.memo.retained():
            first = self.memo.extended_jacobian(self.state, 0.5)
            calls = self.residual.calls
            again = self.memo.extended_jacobian(self.state, 0.5)
            state_part = self.memo.state_jacobian(self.state, 0.5)

        assert self.residual.calls == calls == 2 * 2 + 2
        np.testing.assert_array_equal(again, first)
        np.testing.assert_array_equal(state_part, first[:, :-1])
        np.testing.assert_array_equal(
            first,
            self.inner.extended_jacobian(self.state, 0.5),
        )

    def test_a_different_point_or_parameter_is_differentiated_afresh(self):
        with self.memo.retained():
            self.memo.state_jacobian(self.state, 0.5)
            self.memo.state_jacobian(self.state, 0.25)
            self.memo.state_jacobian(self.state + 1.0e-12, 0.5)

        assert self.residual.calls == 3 * 2 * 2

    def test_the_memory_is_dropped_when_the_scope_closes(self):
        with self.memo.retained():
            self.memo.state_jacobian(self.state, 0.5)
        with self.memo.retained():
            self.memo.state_jacobian(self.state, 0.5)

        assert self.residual.calls == 2 * 2 * 2

    def test_a_caller_mutating_its_copy_does_not_corrupt_the_memory(self):
        with self.memo.retained():
            first = self.memo.state_jacobian(self.state, 0.5)
            expected = first.copy()
            first[:] = 0.0
            again = self.memo.state_jacobian(self.state, 0.5)

        np.testing.assert_array_equal(again, expected)

    def test_old_points_are_forgotten_beyond_the_capacity(self):
        with self.memo.retained():
            for shift in range(MemoizedJacobian.capacity + 1):
                self.memo.state_jacobian(self.state + shift, 0.5)
            calls = self.residual.calls
            self.memo.state_jacobian(self.state, 0.5)

        assert self.residual.calls == calls + 2 * 2

    def test_a_recently_used_point_outlives_newer_unused_ones(self):
        with self.memo.retained():
            for shift in range(MemoizedJacobian.capacity):
                self.memo.state_jacobian(self.state + shift, 0.5)
            self.memo.state_jacobian(self.state, 0.5)
            self.memo.state_jacobian(self.state + 100.0, 0.5)
            calls = self.residual.calls
            self.memo.state_jacobian(self.state, 0.5)

        assert self.residual.calls == calls

    def test_the_last_capacity_points_are_all_remembered(self):
        with self.memo.retained():
            for shift in range(MemoizedJacobian.capacity):
                self.memo.state_jacobian(self.state + shift, 0.5)
            calls = self.residual.calls
            for shift in range(MemoizedJacobian.capacity):
                self.memo.state_jacobian(self.state + shift, 0.5)

        assert self.residual.calls == calls
