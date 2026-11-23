"""The tangent off the corrector's factorisation, and AUTO's step control (DEQ-75)."""

import math
from itertools import pairwise
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CycleSeed,
    _adapted_step,
    _Arc,
    classify_transition,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from tests.continuation.fields import (
    Alpha,
    FoldOfCyclesField,
    HopfField,
    State,
    Time,
)


def _equation(field_type: type, value: float) -> DifferentialEquation:
    parameters = [Alpha(value=value)]
    return DifferentialEquation(
        variables=[State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=field_type(
            variables=[State(), State()],
            parameters=parameters,
            results=[State(), State()],
            time=None,
        ),
    )


def _circle(radius: float, intervals: int) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, intervals + 1)
    return np.column_stack(
        [radius * np.cos(2 * math.pi * grid), radius * np.sin(2 * math.pi * grid)],
    )


def _outer(mu: float) -> float:
    return math.sqrt((1.0 + math.sqrt(1.0 + 4.0 * mu)) / 2.0)


class _FreshTangent(CycleContinuation):
    """The pre-DEQ-75 tangent: rebuilt and refactored at the converged point."""

    __slots__ = []

    def _tangent(self, unknowns, nodes, previous):
        self._factored = None
        return super()._tangent(unknowns, nodes, previous)


class _CountingReuse(CycleContinuation):
    """Counts the bordered matrices assembled, each of which is factored."""

    __slots__ = ["assemblies"]

    def _bordered(self, *args):
        self.assemblies = getattr(self, "assemblies", 0) + 1
        return super()._bordered(*args)


class _CountingFresh(_CountingReuse):
    __slots__ = []

    def _tangent(self, unknowns, nodes, previous):
        self._factored = None
        return super()._tangent(unknowns, nodes, previous)


class TestAdaptedStep(TestCase):
    """AUTO-07p's ADPTDS with ITNW = 5, entry by entry."""

    def test_the_table(self) -> None:
        assert _adapted_step(0.1, 0) == pytest.approx(0.2)
        assert _adapted_step(0.1, 1) == pytest.approx(0.2)
        assert _adapted_step(0.1, 2) == pytest.approx(0.15)
        assert _adapted_step(0.1, 3) == pytest.approx(0.1)
        assert _adapted_step(0.1, 4) == pytest.approx(0.1)
        assert _adapted_step(0.1, 5) == pytest.approx(0.05)
        assert _adapted_step(0.1, 9) == pytest.approx(0.05)


class TestTangentReuse(TestCase):
    def _trace(self, kind: type, scheme: str) -> list:
        mu = 0.5
        continuation = kind(
            _equation(HopfField, mu),
            0,
            40,
            phase_index=1,
            phase_value=0.0,
            collocation=scheme,
        )
        seed = CycleSeed(_circle(math.sqrt(mu), 40), 2.0 * math.pi, mu)
        # A maximum equal to the step: the adaptive path, which reuses the
        # factorisation, at a step that never grows.
        points, _ = continuation.trace(seed, 0.1, 20, max_arclength=0.1)
        return points

    def test_the_reused_tangent_keeps_every_point_on_the_branch(self) -> None:
        # The tangent only orients the step; each point is still a cycle of
        # radius sqrt(mu) and period 2 pi to the discretisation's error.
        for scheme in ("trapezoidal", "hermite_simpson"):
            reused = self._trace(CycleContinuation, scheme)
            fresh = self._trace(_FreshTangent, scheme)
            assert len(reused) == len(fresh) == 21
            for a, b in zip(reused, fresh, strict=True):
                # Same branch, same spacing to well under the step.
                assert abs(a.parameter - b.parameter) < 1e-6
                assert abs(a.solution.period - b.solution.period) < 1e-6
            for point in reused[1:]:
                radius = np.linalg.norm(point.solution.states, axis=1)
                assert np.max(np.abs(radius - math.sqrt(point.parameter))) < 5e-3

    def test_the_reuse_saves_a_factorisation_per_step(self) -> None:
        counts = {}
        for kind in (_CountingReuse, _CountingFresh):
            continuation = kind(
                _equation(HopfField, 0.5),
                0,
                40,
                phase_index=1,
                phase_value=0.0,
            )
            seed = CycleSeed(_circle(math.sqrt(0.5), 40), 2.0 * math.pi, 0.5)
            continuation.trace(seed, 0.1, 10, max_arclength=0.1)
            counts[kind] = continuation.assemblies
        # One corrector iteration a step on this branch, so the fresh tangent
        # assembles and factors a second matrix every step and the reuse none.
        assert counts[_CountingFresh] - counts[_CountingReuse] == 10


class TestFixedStepUnchanged(TestCase):
    def test_the_fixed_step_rebuilds_its_tangent_bit_for_bit(self) -> None:
        traces = []
        for kind in (CycleContinuation, _FreshTangent):
            continuation = kind(
                _equation(HopfField, 0.5),
                0,
                40,
                phase_index=1,
                phase_value=0.0,
            )
            seed = CycleSeed(_circle(math.sqrt(0.5), 40), 2.0 * math.pi, 0.5)
            traces.append(continuation.trace(seed, 0.1, 15)[0])
        for a, b in zip(*traces, strict=True):
            assert a.parameter == b.parameter
            assert a.solution.period == b.solution.period
            np.testing.assert_array_equal(a.solution.states, b.solution.states)


class TestAdaptiveStep(TestCase):
    def _continuation(self, mu: float) -> CycleContinuation:
        return CycleContinuation(
            _equation(FoldOfCyclesField, mu),
            0,
            40,
            phase_index=1,
            phase_value=0.0,
        )

    def test_fewer_points_over_the_same_arclength(self) -> None:
        mu = -0.1
        seed = CycleSeed(_circle(_outer(mu), 40), 2.0 * math.pi, mu)
        fixed, _ = self._continuation(mu).trace(seed, 0.01, 120, direction=-1.0)
        adaptive, _ = self._continuation(mu).trace(
            seed,
            0.01,
            120,
            direction=-1.0,
            max_arclength=0.08,
        )
        assert len(adaptive) < len(fixed) / 2
        # Both spend the same arclength, so they end at the same place on the
        # branch, within the last step's reach.
        assert adaptive[-1].parameter == pytest.approx(fixed[-1].parameter, abs=2e-3)
        assert adaptive[-1].amplitude == pytest.approx(fixed[-1].amplitude, abs=2e-3)

    def test_the_fold_is_located_as_the_fixed_step_locates_it(self) -> None:
        mu = -0.1
        seed = CycleSeed(_circle(_outer(mu), 40), 2.0 * math.pi, mu)
        _, fixed = self._continuation(mu).trace(seed, 0.01, 200, direction=-1.0)
        _, adaptive = self._continuation(mu).trace(
            seed,
            0.01,
            200,
            direction=-1.0,
            max_arclength=0.08,
        )
        assert (
            [b.kind for b in adaptive]
            == [b.kind for b in fixed]
            == [
                "fold_of_cycles",
            ]
        )
        # The exact fold is at mu = -1/4; both brackets are one base step wide.
        assert abs(adaptive[0].parameter + 0.25) < 1e-4
        assert abs(adaptive[0].parameter - fixed[0].parameter) < 1e-4

    def test_the_step_never_exceeds_its_maximum(self) -> None:
        mu = -0.1
        seed = CycleSeed(_circle(_outer(mu), 40), 2.0 * math.pi, mu)
        points, _ = self._continuation(mu).trace(
            seed,
            0.01,
            60,
            direction=-1.0,
            max_arclength=0.03,
        )
        unknowns = [
            np.concatenate(
                [p.solution.states.ravel(), [p.solution.period, p.parameter]]
            )
            for p in points
        ]
        # The constraint measures each step along the tangent; the chord to
        # the corrected point is a little longer on a curving branch.
        gaps = [np.linalg.norm(b - a) for a, b in pairwise(unknowns)]
        assert max(gaps) < 0.03 * 1.05

    def test_the_limits_are_checked(self) -> None:
        mu = -0.1
        seed = CycleSeed(_circle(_outer(mu), 40), 2.0 * math.pi, mu)
        continuation = self._continuation(mu)
        with pytest.raises(ValueError, match="max_arclength"):
            continuation.trace(seed, 0.01, 5, max_arclength=0.005)
        with pytest.raises(ValueError, match="min_arclength"):
            continuation.trace(seed, 0.01, 5, min_arclength=0.001)
        with pytest.raises(ValueError, match="min_arclength"):
            continuation.trace(seed, 0.01, 5, max_arclength=0.1, min_arclength=0.02)


class TestReuseGuards(TestCase):
    def test_a_factorisation_with_another_border_is_not_reused(self) -> None:
        mu = 0.5
        continuation = CycleContinuation(
            _equation(HopfField, mu),
            0,
            30,
            phase_index=1,
            phase_value=0.0,
        )
        seed = CycleSeed(_circle(math.sqrt(mu), 30), 2.0 * math.pi, mu)
        points, _ = continuation.trace(seed, 0.05, 1, max_arclength=0.05)
        nodes = 31
        unknowns = np.concatenate(
            [points[-1].solution.states.ravel(), [points[-1].solution.period, mu]],
        )
        border = np.zeros(unknowns.size)
        border[-1] = 1.0
        arc = _Arc(unknowns, border, 0.0)
        continuation._correct(unknowns + 1e-4, nodes, arc)  # noqa: SLF001
        other = np.ones(unknowns.size) / math.sqrt(unknowns.size)
        reused = continuation._tangent(unknowns, nodes, other)  # noqa: SLF001
        continuation._factored = None  # noqa: SLF001
        fresh = continuation._tangent(unknowns, nodes, other)  # noqa: SLF001
        np.testing.assert_array_equal(reused, fresh)


class TestCrossingRetaken(TestCase):
    def test_a_crossing_is_bracketed_at_the_base_step(self) -> None:
        mu = -0.1
        seed = CycleSeed(_circle(_outer(mu), 40), 2.0 * math.pi, mu)
        continuation = CycleContinuation(
            _equation(FoldOfCyclesField, mu),
            0,
            40,
            phase_index=1,
            phase_value=0.0,
        )
        points, _ = continuation.trace(
            seed,
            0.01,
            200,
            direction=-1.0,
            max_arclength=0.16,
        )
        brackets = [
            (a, b) for a, b in pairwise(points) if classify_transition(a, b) is not None
        ]
        assert len(brackets) == 1
        lower, upper = brackets[0]
        gap = np.linalg.norm(
            np.concatenate(
                [
                    upper.solution.states.ravel() - lower.solution.states.ravel(),
                    [
                        upper.solution.period - lower.solution.period,
                        upper.parameter - lower.parameter,
                    ],
                ],
            ),
        )
        # Retaken at the base step: the chord is the step, a little curved.
        assert gap < 0.01 * 1.05


class _Recording(CycleContinuation):
    """Records each correction's step and phase reference against its anchor."""

    __slots__ = ["calls", "fail_above", "slow"]

    def _correct(self, prediction, nodes, arc):
        reference = self._phase_reference
        self.calls = [
            *getattr(self, "calls", []),
            (
                arc.arclength,
                arc.anchor.copy(),
                None if reference is None else reference[0].copy(),
            ),
        ]
        if arc.arclength > getattr(self, "fail_above", math.inf):
            self._factored = None
            self._iterations = 0
            return None
        corrected = super()._correct(prediction, nodes, arc)
        if getattr(self, "slow", False):
            self._iterations = 5
        return corrected


class TestStepRecovery(TestCase):
    """The failure, slow-convergence and retake paths of the adaptive step."""

    def _recording(self, phase: str = "point") -> _Recording:
        return _Recording(
            _equation(FoldOfCyclesField, -0.1),
            0,
            40,
            phase_index=1,
            phase_value=0.0,
            phase=phase,
        )

    def _seed(self) -> CycleSeed:
        return CycleSeed(_circle(_outer(-0.1), 40), 2.0 * math.pi, -0.1)

    def test_a_failed_long_step_is_halved_and_retried(self) -> None:
        fixed, _ = self._recording().trace(self._seed(), 0.01, 40, direction=-1.0)
        continuation = self._recording()
        continuation.fail_above = 0.03
        adaptive, _ = continuation.trace(
            self._seed(),
            0.01,
            40,
            direction=-1.0,
            max_arclength=0.08,
        )
        lengths = [call[0] for call in continuation.calls if call[0] > 0.0]
        assert max(lengths) > 0.03  # it did try, and fail
        # Every failure is retried shorter, so the budget is still spent.
        assert adaptive[-1].parameter == pytest.approx(fixed[-1].parameter, abs=2e-3)

    def test_slow_convergence_never_takes_the_step_below_its_minimum(self) -> None:
        continuation = self._recording()
        continuation.slow = True
        continuation.trace(
            self._seed(),
            0.01,
            12,
            direction=-1.0,
            max_arclength=0.08,
        )
        # The seed's own correction has no step; the last is cut to the budget.
        lengths = [call[0] for call in continuation.calls if call[0] > 0.0]
        # Five iterations halve the step; min_arclength (the base step) holds it.
        assert min(lengths[:-1]) == pytest.approx(0.01)

    def test_a_retaken_step_corrects_against_the_kept_points_phase(self) -> None:
        continuation = self._recording("integral")
        continuation.trace(
            self._seed(),
            0.01,
            200,
            direction=-1.0,
            max_arclength=0.16,
        )
        # The seed's own correction, before any reference, is not a step.
        steps = [call for call in continuation.calls if call[0] > 0.0]
        retaken = [
            index
            for index, (a, b) in enumerate(pairwise(steps))
            if np.array_equal(a[1], b[1])
        ]
        assert retaken  # the fold's crossing was stepped over once, and retaken
        for _, anchor, reference in steps:
            # The phase reference is always the anchor's own orbit: a
            # discarded correction does not leave its orbit behind.
            np.testing.assert_array_equal(reference, anchor[: reference.size])
