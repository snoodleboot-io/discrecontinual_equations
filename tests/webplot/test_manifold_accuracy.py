"""The drawn manifolds against a tight-tolerance reference, to within a pixel.

Each branch :func:`saddle_manifolds` returns starts on the manifold's chart and
is carried by the flow until it leaves the box. The reference here takes the
branch's own first point and carries it again with DOP853 at a tolerance four
orders tighter, stopped by the same box and resampled at the same count of
evenly spaced times. A branch that strays further than about one display pixel
(2e-3 units at normal zoom) from that reference is drawn wrong.

The fields are the films' own: the Shilnikov jerk, whose two equilibria are
saddle-foci with a curve and a surface each, and the Bogdanov-Takens normal
form, a planar saddle.
"""

from unittest import TestCase

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.function.deterministic import DeterministicFunction
from discrecontinual_equations.parameter import Parameter
from discrecontinual_equations.variable import Variable
from discrecontinual_equations.webplot.stage_builder import saddle_manifolds

_PIXEL = 2.0e-3
_MARGIN = 0.2
_SAMPLES = 360
_TIME = 60.0


class Mu(Parameter, name="Mu", abbreviation="a"):
    pass


class State(Variable, name="State", abbreviation="v"):
    pass


class Jerk(DeterministicFunction):
    """x' = y, y' = z, z' = -a z - y + x - x^2: saddle-foci at x = 0 and x = 1."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        a = self.parameters[0].value
        x, y, z = point[0], point[1], point[2]
        return [y, z, -a * z - y + x - x * x]


class BogdanovTakens(DeterministicFunction):
    """x' = y, y' = b1 + b2 y + x^2 + x y: a saddle at x = +sqrt(-b1)."""

    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        b1, b2 = self.parameters[0].value, self.parameters[1].value
        x, y = point[0], point[1]
        return [y, b1 + b2 * y + x * x + x * y]


def _function(kind, values, dimension):
    return kind(
        variables=[State() for _ in range(dimension)],
        parameters=[Mu(value=value) for value in values],
        results=[State() for _ in range(dimension)],
        time=None,
    )


def _reference(function, start, bounds, forward):
    sense = 1.0 if forward else -1.0

    def rhs(_t, state):
        return [sense * float(v) for v in function.eval(point=list(state), time=None)]

    def leaving(_t, state):
        return min(
            min(value - (lo - _MARGIN), (hi + _MARGIN) - value)
            for value, (lo, hi) in zip(state, bounds, strict=True)
        )

    leaving.terminal = True
    solution = solve_ivp(
        rhs,
        [0.0, _TIME],
        start,
        method="DOP853",
        rtol=1.0e-12,
        atol=1.0e-14,
        events=leaving,
        dense_output=True,
    )
    return solution.sol(np.linspace(0.0, solution.t[-1], _SAMPLES)).T


def _worst(function, equilibrium, bounds):
    manifolds = saddle_manifolds(function, np.asarray(equilibrium, float), bounds)
    worst = 0.0
    for manifold in manifolds:
        curve = np.asarray(manifold.curve)
        forward = manifold.kind.startswith("unstable")
        reference = _reference(function, curve[0], bounds, forward)
        worst = max(worst, float(np.abs(curve - reference).max()))
    return manifolds, worst


class TestManifoldsMatchTheReference(TestCase):
    def test_the_jerk_saddle_foci_within_a_pixel(self):
        function = _function(Jerk, [0.7], 3)
        bounds = ((-0.6, 1.6), (-0.9, 0.9), (-0.9, 0.9))
        for equilibrium in ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]):
            manifolds, worst = _worst(function, equilibrium, bounds)
            kinds = {manifold.kind for manifold in manifolds}
            assert kinds in (
                {"unstable", "stable-surface"},
                {"stable", "unstable-surface"},
            )
            assert worst < _PIXEL

    def test_the_bogdanov_takens_saddle_within_a_pixel(self):
        function = _function(BogdanovTakens, [-0.25, 0.5], 2)
        bounds = ((-1.5, 1.5), (-1.5, 1.5))
        manifolds, worst = _worst(function, [0.5, 0.0], bounds)
        assert sorted(m.kind for m in manifolds) == [
            "stable",
            "stable",
            "unstable",
            "unstable",
        ]
        assert worst < _PIXEL
