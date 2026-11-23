"""The batched Dormand-Prince pass that carries a frame's manifold branches.

Every check is against something the code under test did not compute: the
tableau as published, closed-form flows, and scipy's own per-branch RK45, which
the batch is meant to reproduce row by row.
"""

import math
from unittest import TestCase

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.common import select_initial_step

from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable
from discrecontinual_equations.webplot import manifold_flow
from discrecontinual_equations.webplot.manifold_flow import flow_branches

# Dormand and Prince (1980), Table 2: the fifth-order weights b and the
# embedded fourth-order weights b*.
_FIFTH = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
_FOURTH = [
    5179 / 57600,
    0,
    7571 / 16695,
    393 / 640,
    -92097 / 339200,
    187 / 2100,
    1 / 40,
]


class Mu(Parameter, name="Mu", abbreviation="a"):
    pass


class State(Variable, name="State", abbreviation="v"):
    pass


class Decay(DeterministicFunction):
    """x' = -x, y' = -2 y: x = x0 e^-t, y = y0 e^-2t."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        return [-point[0], -2.0 * point[1]]


class Drift(DeterministicFunction):
    """x' = 1, y' = -y: x crosses any wall at a known time; x' is a constant."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        return [1.0, -point[1]]


class Rotation(DeterministicFunction):
    """A spiral, x' = a x - y, y' = x + a y, written with math so it cannot
    take arrays."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        a = self.parameters[0].value
        x, y = point[0], point[1]
        return [math.fsum([a * x, -y]), math.fsum([x, a * y])]


class Sly(DeterministicFunction):
    """x' = -x for a number, but twice that for an array: it broadcasts without
    complaint and gets the columns wrong."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x, y = point[0], point[1]
        factor = 1.0 if isinstance(x, float) else 2.0
        return [-factor * x, -y]


class Onset(DeterministicFunction):
    """x' = 1, y' = -k(x) (y - 1/2) with k jumping from 1 to 1000 near x = 1.

    The jump is where steps are rejected hard enough to be cut by the largest
    allowed factor."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        x, y = point[0], point[1]
        rate = 1.0 + 500.0 * (1.0 + np.tanh(40.0 * (x - 1.0)))
        return [1.0, -rate * (y - 0.5)]


class Jerk(DeterministicFunction):
    """x' = y, y' = z, z' = -a z - y + x - x^2."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        a = self.parameters[0].value
        x, y, z = point[0], point[1], point[2]
        return [y, z, -a * z - y + x - x * x]


def _function(kind, dimension, values=()):
    return kind(
        variables=[State() for _ in range(dimension)],
        parameters=[Mu(value=value) for value in values],
        results=[State() for _ in range(dimension)],
        time=None,
    )


def _rk45(function, start, sense, bounds, duration, margin, samples):  # noqa: PLR0913
    def rhs(_t, state):
        values = function.eval(point=[float(v) for v in state], time=None)
        return [sense * float(v) for v in values]

    def leaving(_t, state):
        return min(
            min(value - (lo - margin), (hi + margin) - value)
            for value, (lo, hi) in zip(state, bounds, strict=True)
        )

    leaving.terminal = True
    solution = solve_ivp(
        rhs,
        [0.0, duration],
        start,
        rtol=1.0e-8,
        atol=1.0e-10,
        events=leaving,
        dense_output=True,
    )
    return solution.sol(np.linspace(0.0, solution.t[-1], samples)).T


class TestTableau(TestCase):
    def test_the_step_weights_are_the_fifth_order_ones(self):
        np.testing.assert_allclose(manifold_flow.B, _FIFTH[:6], rtol=0, atol=1e-16)

    def test_the_error_weights_are_the_two_orders_apart(self):
        difference = np.array(_FOURTH) - np.array(_FIFTH)
        np.testing.assert_allclose(manifold_flow.E, difference, rtol=0, atol=1e-16)

    def test_each_stage_is_taken_at_its_node(self):
        np.testing.assert_allclose(
            manifold_flow.A.sum(axis=1),
            manifold_flow.C,
            rtol=0,
            atol=1e-15,
        )

    def test_the_interpolant_starts_on_the_slope_and_ends_on_the_step(self):
        p = manifold_flow.P
        np.testing.assert_array_equal(p[:, 0], [1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(p.sum(axis=1), _FIFTH, rtol=0, atol=1e-15)


class TestClosedForms(TestCase):
    def test_a_decay_is_drawn_to_its_exponential_at_every_sample(self):
        [curve] = flow_branches(
            _function(Decay, 2),
            [np.array([1.0, 0.5])],
            [1.0],
            ((-5.0, 5.0), (-5.0, 5.0)),
            duration=4.0,
            margin=0.2,
            samples=41,
        )
        times = np.linspace(0.0, 4.0, 41)
        exact = np.column_stack([np.exp(-times), 0.5 * np.exp(-2.0 * times)])
        np.testing.assert_allclose(np.array(curve), exact, rtol=0, atol=2e-8)

    def test_a_backward_row_runs_the_flow_in_reverse(self):
        [curve] = flow_branches(
            _function(Decay, 2),
            [np.array([0.1, 0.0])],
            [-1.0],
            ((-5.0, 5.0), (-5.0, 5.0)),
            duration=2.0,
            margin=0.2,
            samples=21,
        )
        times = np.linspace(0.0, 2.0, 21)
        np.testing.assert_allclose(
            np.array(curve)[:, 0],
            0.1 * np.exp(times),
            rtol=1e-8,
        )

    def test_a_row_stops_where_it_crosses_the_widened_wall(self):
        forward, backward = flow_branches(
            _function(Drift, 2),
            [np.array([0.0, 0.3]), np.array([0.0, 0.3])],
            [1.0, -1.0],
            ((-1.0, 1.0), (-1.0, 1.0)),
            duration=60.0,
            margin=0.2,
            samples=7,
        )
        # x' = 1 reaches 1.2 at t = 1.2; y = 0.3 e^-t on the way.
        times = np.linspace(0.0, 1.2, 7)
        np.testing.assert_allclose(np.array(forward)[:, 0], times, atol=1e-12)
        np.testing.assert_allclose(
            np.array(forward)[:, 1],
            0.3 * np.exp(-times),
            atol=1e-9,
        )
        np.testing.assert_allclose(np.array(backward)[:, 0], -times, atol=1e-12)
        np.testing.assert_allclose(
            np.array(backward)[:, 1],
            0.3 * np.exp(times),
            atol=1e-9,
        )

    def test_rows_of_one_batch_keep_their_own_stopping_times(self):
        near, far = flow_branches(
            _function(Drift, 2),
            [np.array([0.9, 0.0]), np.array([-0.9, 0.0])],
            [1.0, 1.0],
            ((-1.0, 1.0), (-1.0, 1.0)),
            duration=60.0,
            margin=0.2,
            samples=3,
        )
        np.testing.assert_allclose(np.array(near)[:, 0], [0.9, 1.05, 1.2], atol=1e-12)
        np.testing.assert_allclose(np.array(far)[:, 0], [-0.9, 0.15, 1.2], atol=1e-12)

    def test_a_row_that_starts_outside_stops_where_it_comes_back_in(self):
        # solve_ivp's event fires on a sign change either way, so a branch
        # started beyond the widened wall stops on reaching it.
        [curve] = flow_branches(
            _function(Drift, 2),
            [np.array([-2.0, 0.0])],
            [1.0],
            _BOX2,
            duration=60.0,
            margin=0.2,
            samples=5,
        )
        np.testing.assert_allclose(
            np.array(curve)[:, 0],
            np.linspace(-2.0, -1.2, 5),
            atol=1e-12,
        )

    def test_nothing_to_carry_is_nothing_drawn(self):
        assert flow_branches(_function(Decay, 2), [], [], _BOX2, 1.0, 0.2, 5) == []


_BOX2 = ((-1.0, 1.0), (-1.0, 1.0))


class TestAgainstPerBranchRK45(TestCase):
    """Row by row the batch is scipy's RK45 with its box event, to rounding."""

    def test_a_fan_of_jerk_branches(self):
        function = _function(Jerk, 3, [0.7])
        bounds = ((-0.6, 1.6), (-0.9, 0.9), (-0.9, 0.9))
        rng = np.random.default_rng(7)
        starts = [
            np.array(centre) + 1.0e-3 * rng.standard_normal(3)
            for centre in ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            for _ in range(5)
        ]
        senses = [(-1.0) ** k for k in range(len(starts))]
        curves = flow_branches(function, starts, senses, bounds, 60.0, 0.2, 360)
        for start, sense, curve in zip(starts, senses, curves, strict=True):
            reference = _rk45(function, start, sense, bounds, 60.0, 0.2, 360)
            np.testing.assert_allclose(np.array(curve), reference, rtol=0, atol=1e-11)

    def test_a_sudden_stiff_onset_rejects_and_recovers_as_rk45_does(self):
        function = _function(Onset, 2)
        box = ((-1.0, 3.0), (-2.0, 2.0))
        starts = [np.array([0.0, 0.0]), np.array([0.5, 1.0])]
        curves = flow_branches(function, starts, [1.0, 1.0], box, 60.0, 0.2, 200)
        for start, curve in zip(starts, curves, strict=True):
            reference = _rk45(function, start, 1.0, box, 60.0, 0.2, 200)
            np.testing.assert_allclose(np.array(curve), reference, rtol=0, atol=1e-11)

    def test_the_first_step_of_each_row_is_scipys(self):
        function = _function(Jerk, 3, [0.7])
        rng = np.random.default_rng(3)
        states = np.vstack(
            [
                1.0e-3 * rng.standard_normal((3, 3)),
                [[1.0, 0.0, 0.0], [0.3, -0.2, 0.1], [0.0, 0.0, 0.0]],
            ],
        )
        senses = np.array([1.0, -1.0, 1.0, -1.0, 1.0, 1.0])

        def slope(rows):
            values = [function.eval(point=list(row), time=None) for row in rows]
            return senses[:, None] * np.array(values, dtype=float)

        slopes = slope(states)
        for duration in (60.0, 1.0e-4):
            steps = manifold_flow._initial_step(  # noqa: SLF001 (the rule alone)
                slope,
                states,
                slopes,
                duration,
                (1.0e-8, 1.0e-10),
            )
            for k, state in enumerate(states):
                expected = select_initial_step(
                    lambda _t, y, k=k: slope(np.tile(y, (len(states), 1)))[k],
                    0.0,
                    state,
                    duration,
                    np.inf,
                    slopes[k],
                    1.0,
                    4,
                    1.0e-8,
                    1.0e-10,
                )
                assert steps[k] == pytest.approx(expected, rel=1e-12)

    def test_a_field_that_cannot_take_arrays_is_taken_a_point_at_a_time(self):
        function = _function(Rotation, 2, [0.3])
        starts = [np.array([0.01, 0.0]), np.array([0.0, -0.02])]
        curves = flow_branches(function, starts, [1.0, 1.0], _BOX2, 60.0, 0.2, 50)
        for start, curve in zip(starts, curves, strict=True):
            reference = _rk45(function, start, 1.0, _BOX2, 60.0, 0.2, 50)
            np.testing.assert_allclose(np.array(curve), reference, rtol=0, atol=1e-11)

    def test_a_field_whose_columns_disagree_with_its_points_is_not_trusted(self):
        function = _function(Sly, 2)
        starts = [np.array([0.5, 0.5]), np.array([-0.5, 0.25])]
        curves = flow_branches(function, starts, [1.0, 1.0], _BOX2, 3.0, 0.2, 11)
        times = np.linspace(0.0, 3.0, 11)
        for start, curve in zip(starts, curves, strict=True):
            np.testing.assert_allclose(
                np.array(curve),
                np.outer(np.exp(-times), start),
                atol=1e-8,
            )
