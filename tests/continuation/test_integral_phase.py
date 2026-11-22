"""AUTO's integral phase condition and weighted arclength, as options (DEQ-56).

The phase condition and the arclength norm are Doedel's: the integral
``int_0^1 <x(t) - x_hat(t), x_hat'(t)> dt = 0`` against the previous cycle, and
the pseudo-arclength measured as ``int_0^1 <dx, x_hat_dot> dt`` plus the
parameter term (the period's weight is zero, AUTO's default for periodic
solutions), each integral by the active scheme's own quadrature.
"""

import math
from unittest import TestCase

import numpy as np
import pytest

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CycleSeed,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import Alpha, HopfField, State, Time

# The shifted cycle's fixed radius squared: radius one half.
_RADIUS_SQUARED = 0.25


class ShiftedHopfField(DeterministicFunction):
    """A Hopf cycle of radius 1/2 about ``(p, 0)``: ``p`` only moves it sideways.

    With ``u = x - p``: ``u' = a u - y - (u^2 + y^2) u`` and
    ``y' = u + a y - (u^2 + y^2) y``, ``a = 1/4``. Every cycle on the branch is
    the same circle, period ``2 pi``, multipliers ``1`` and ``exp(-4 pi a)``.
    Pinning ``x_0 = 0`` works only while the circle crosses ``x = 0``, that is
    for ``|p| < 1/2``.
    """

    def eval(
        self,
        point: list[float],
        time: float | None = None,  # noqa: ARG002 (base signature)
    ) -> list[float]:
        shift = self.parameters[0].value
        u, y = point[0] - shift, point[1]
        square = u * u + y * y
        return [
            _RADIUS_SQUARED * u - y - square * u,
            u + _RADIUS_SQUARED * y - square * y,
        ]


def _equation(field: type, value: float) -> DifferentialEquation:
    parameters = [Alpha(value=value)]
    return DifferentialEquation(
        variables=[State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=field(
            variables=[State(), State()],
            parameters=parameters,
            results=[State(), State()],
            time=None,
        ),
    )


def _circle(radius: float, intervals: int, centre: float = 0.0) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, intervals + 1)
    return np.column_stack(
        [
            centre + radius * np.cos(2.0 * math.pi * grid),
            radius * np.sin(2.0 * math.pi * grid),
        ],
    )


class TestQuadrature(TestCase):
    """The weights the integrals use, against integrals known in closed form."""

    @staticmethod
    def _scheme(name: str, points: int | None):
        continuation = CycleContinuation(
            _equation(HopfField, 0.25),
            0,
            7,
            collocation=name,
            collocation_points=points,
        )
        return continuation._collocation  # noqa: SLF001

    def test_gauss_legendre_integrates_its_own_polynomials_on_a_graded_mesh(self):
        mesh = np.array([0.0, 0.05, 0.2, 0.3, 0.55, 0.6, 0.9, 1.0])
        for points in (1, 2, 3, 4):
            scheme = self._scheme("gauss_legendre", points)
            grid, weights = scheme.grid(mesh), scheme.weights(mesh)
            for power in range(points + 1):
                exact = 1.0 / (power + 1)
                assert abs(weights @ grid**power - exact) < 1e-14, (points, power)

    def test_one_wide_schemes_use_the_trapezoidal_rule(self):
        mesh = np.array([0.0, 0.1, 0.4, 0.5, 1.0])
        for name in ("trapezoidal", "hermite_simpson"):
            weights = self._scheme(name, None).weights(mesh)
            np.testing.assert_allclose(weights, [0.05, 0.2, 0.2, 0.3, 0.25])


class TestPhaseCondition(TestCase):
    """The point condition loses the branch where the integral one keeps it."""

    def _trace(self, phase: str, scheme: str = "hermite_simpson"):
        intervals = 40
        continuation = CycleContinuation(
            _equation(ShiftedHopfField, 0.0),
            0,
            intervals,
            phase_index=0,
            phase_value=0.0,
            collocation=scheme,
            phase=phase,
            arclength="weighted",
        )
        seed = CycleSeed(_circle(0.5, intervals), 2.0 * math.pi, 0.0)
        points, _ = continuation.trace(seed, 0.05, 40)
        return points

    def test_point_condition_cannot_pass_the_edge_of_the_circle(self):
        """It reaches ``p = 1/2``, where ``x = 0`` touches the circle, and turns back.

        The point condition has no solution past there, so pseudo-arclength
        rounds an artificial fold - the cycle itself goes on unchanged.
        """
        points = self._trace("point")
        furthest = max(point.parameter for point in points)
        assert furthest < 0.5 + 1e-3
        assert points[-1].parameter < furthest - 0.1

    def test_integral_condition_carries_the_cycle_past_it(self):
        for scheme in ("trapezoidal", "hermite_simpson", "gauss_legendre"):
            points = self._trace("integral", scheme)
            assert max(point.parameter for point in points) > 1.0, scheme
            for point in points:
                states = point.solution.states
                radius = np.hypot(states[:, 0] - point.parameter, states[:, 1])
                assert np.max(np.abs(radius - 0.5)) < 1e-3, scheme
                # Trapezoidal at 40 intervals is 2e-3 off the period: its own error.
                assert abs(point.solution.period / (2.0 * math.pi) - 1.0) < 3e-3, scheme

    def test_each_point_is_the_next_ones_reference(self):
        """AUTO takes the condition against the previous solution, not the seed."""
        intervals = 20
        continuation = CycleContinuation(
            _equation(ShiftedHopfField, 0.0),
            0,
            intervals,
            phase_index=0,
            phase="integral",
            arclength="weighted",
        )
        seed = CycleSeed(_circle(0.5, intervals), 2.0 * math.pi, 0.0)
        points, _ = continuation.trace(seed, 0.05, 3)
        reference = continuation._phase_reference  # noqa: SLF001
        assert reference is not None
        np.testing.assert_array_equal(
            reference[0],
            points[-1].solution.states.ravel(),
        )

    def test_rejects_an_unknown_phase_condition(self):
        with pytest.raises(ValueError, match="phase"):
            CycleContinuation(_equation(HopfField, 0.25), 0, 10, phase="nearest")


class TestArclength(TestCase):
    """One weighted step spans the same parameter under every scheme."""

    @staticmethod
    def _span(arclength: str, scheme: str, points: int | None) -> float:
        mu, intervals = 0.25, 24
        continuation = CycleContinuation(
            _equation(HopfField, mu),
            0,
            intervals,
            phase_index=1,
            collocation=scheme,
            collocation_points=points,
            arclength=arclength,
        )
        seed = CycleSeed(_circle(math.sqrt(mu), intervals), 2.0 * math.pi, mu)
        trace, _ = continuation.trace(seed, 0.05, 1)
        return trace[1].parameter - trace[0].parameter

    def _spans(self, arclength: str) -> list[float]:
        return [
            self._span(arclength, scheme, points)
            for scheme, points in (
                ("trapezoidal", None),
                ("hermite_simpson", None),
                ("gauss_legendre", 2),
                ("gauss_legendre", 3),
                ("gauss_legendre", 4),
            )
        ]

    def test_weighted_steps_agree_within_five_percent(self):
        spans = self._spans("weighted")
        assert max(spans) / min(spans) < 1.05, spans

    def test_euclidean_steps_do_not(self):
        """The default shrinks with the unknowns, about as ``1/sqrt(m)``."""
        spans = self._spans("euclidean")
        assert max(spans) / min(spans) > 1.5, spans

    def test_rejects_an_unknown_norm(self):
        with pytest.raises(ValueError, match="arclength"):
            CycleContinuation(_equation(HopfField, 0.25), 0, 10, arclength="l1")
