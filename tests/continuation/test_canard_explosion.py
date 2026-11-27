"""The van der Pol canard explosion: cycle continuation against shooting.

The cycle branch through the explosion comes from :class:`CycleContinuation`.
It is checked here against a continuation of the same family written for this
test alone, by shooting, which shares nothing with it but the equation:

* the family is parametrised by the cycle's leftmost point ``m``. There
  ``x' = 0``, so the point is ``(m, m^3/3 - m)`` exactly, and ``m`` runs
  monotonically from the Hopf (near 1) to the relaxation oscillation (near -2)
  while ``a`` stays flat to ten figures;
* from that point the cycle is integrated forward and backward to its crossing
  of ``x = a`` below the equilibrium, and ``a`` is the root of the mismatch in
  ``y`` there. Each half is integrated in the direction it contracts - forward
  along the jump and the attracting branch, backward along the repelling one -
  so no exponential growth is ever integrated, which is what makes canards
  computable by shooting at all;
* the period is the two halves' times added, the amplitude the rightmost
  point (where ``x'`` falls through zero) less ``m``.

The slow-manifold canard of :mod:`.slow_manifold` is a third, independent
reading of where the explosion sits.
"""

from unittest import TestCase

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleContinuation,
    CycleSeed,
)
from discrecontinual_equations.continuation.slow_manifold import (
    CanardSearch,
    FenichelExpansion,
    SlowManifold,
)
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.deterministic import DeterministicFunction
from tests.continuation.fields import Alpha, State, Time
from tests.continuation.test_slow_manifold import VanDerPol

_EPSILON = 0.05
_SEED_A = 0.999
_INTERVALS = 160
_STEPS = 150


class Forced(DeterministicFunction):
    def eval(self, point, time=None):  # noqa: ARG002 (base signature)
        a = self.parameters[0].value
        x, y = point[0], point[1]
        return [y - x * x * x / 3.0 + x, _EPSILON * (a - x)]


def _critical(x):
    return x**3 / 3.0 - x


def _rhs(a, sign):
    def rhs(_t, u):
        return [sign * (u[1] - _critical(u[0])), sign * _EPSILON * (a - u[0])]

    return rhs


def _half(m, a, sign):
    """From the leftmost point to the crossing of ``x = a`` below the focus."""

    def crossing(_t, u):
        return u[0] - a

    def turning(_t, u):
        return u[1] - _critical(u[0])

    crossing.terminal = True
    crossing.direction = -1.0 if sign > 0 else 1.0
    # The rightmost point is where x' = y - x^3/3 + x falls through zero.
    turning.direction = -1.0
    return solve_ivp(
        _rhs(a, sign),
        (0.0, 1000.0),
        [m, _critical(m)],
        method="DOP853",
        rtol=1.0e-13,
        atol=1.0e-14,
        events=(crossing, turning),
    )


def _mismatch(m, a):
    return float(
        _half(m, a, 1.0).y_events[0][0][1] - _half(m, a, -1.0).y_events[0][0][1]
    )


def shoot(m, guess, width=1.0e-7):
    """``(a, period, amplitude)`` of the cycle whose leftmost x is ``m``."""
    low, high = guess - width, guess + width
    while _mismatch(m, low) * _mismatch(m, high) > 0.0:
        width *= 4.0
        low, high = guess - width, guess + width
    a = brentq(lambda value: _mismatch(m, value), low, high, xtol=1e-16, rtol=1e-15)
    forward, backward = _half(m, a, 1.0), _half(m, a, -1.0)
    period = forward.t_events[0][0] + backward.t_events[0][0]
    rightmost = max(event[0] for event in forward.y_events[1])
    return a, period, float(rightmost - m)


def _extremes(point, a):
    """The orbit's smallest and largest ``x``, off its Hermite interpolant.

    The nodes alone miss an extremum by up to half a node spacing; the cubic
    through two neighbouring nodes and their slopes (``T f``, known exactly)
    finds it to fourth order, which is what makes ``m`` and the amplitude
    comparable with shooting outside the flat core of the explosion.
    """
    states = point.solution.states
    period = point.solution.period
    mesh = point.mesh
    slopes = np.array([_rhs(a, 1.0)(0.0, u)[0] for u in states]) * period
    x = states[:, 0]
    fine = np.linspace(0.0, 1.0, 201)
    h00 = 2 * fine**3 - 3 * fine**2 + 1
    h10 = fine**3 - 2 * fine**2 + fine
    h01 = -2 * fine**3 + 3 * fine**2
    h11 = fine**3 - fine**2
    found = []
    for pick in (np.argmin, np.argmax):
        i = int(pick(x[:-1]))
        values = []
        for k in (i - 1, i):
            j = k % (len(x) - 1)
            width = mesh[j + 1] - mesh[j]
            values.append(
                h00 * x[j]
                + h10 * width * slopes[j]
                + h01 * x[j + 1]
                + h11 * width * slopes[j + 1],
            )
        found.append(np.concatenate(values))
    return float(found[0].min()), float(found[1].max())


def _seed():
    a = _SEED_A
    rhs = _rhs(a, 1.0)
    settled = solve_ivp(rhs, (0.0, 20000.0), [a + 0.01, -2.0 / 3.0], rtol=1e-11)

    def upward(_t, u):
        return u[1] + 2.0 / 3.0

    upward.direction = 1.0
    orbit = solve_ivp(
        rhs,
        (0.0, 400.0),
        settled.y[:, -1],
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
        events=upward,
        dense_output=True,
    )
    start, finish = orbit.t_events[0][-2], orbit.t_events[0][-1]
    states = orbit.sol(np.linspace(start, finish, _INTERVALS + 1)).T
    return CycleSeed(states, finish - start, a)


def _branch():
    parameters = [Alpha(value=_SEED_A)]
    eq = DifferentialEquation(
        variables=[State(), State()],
        time=Time(),
        parameters=parameters,
        derivative=Forced(
            variables=[State(), State()],
            parameters=parameters,
            results=[State(), State()],
            time=None,
        ),
    )
    continuation = CycleContinuation(
        eq, 0, _INTERVALS, phase_index=1, phase_value=-2.0 / 3.0
    )
    points, _ = continuation.trace(_seed(), 1.0, _STEPS, direction=-1.0)
    return points


class TestCanardExplosion(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points = _branch()
        cls.extremes = [_extremes(p, p.parameter) for p in cls.points]
        cls.leftmost = [low for low, _ in cls.extremes]

    def test_the_branch_crosses_the_whole_explosion(self):
        assert max(self.leftmost) > 0.8  # a Hopf-sized cycle
        assert min(self.leftmost) < -1.7  # a relaxation oscillation

    def test_period_amplitude_and_parameter_match_shooting(self):
        # Every point of the branch inside the explosion, and either side of it.
        chosen = [i for i, m in enumerate(self.leftmost) if -1.75 < m < 0.85][::6]
        assert len(chosen) >= 8
        guess = _SEED_A
        worst = {"a": 0.0, "a_tail": 0.0, "period": 0.0, "amplitude": 0.0}
        for i in chosen:
            point = self.points[i]
            a, period, amplitude = shoot(self.leftmost[i], guess)
            guess = a
            low, high = self.extremes[i]
            spread = high - low
            key = "a" if self.leftmost[i] > -1.66 else "a_tail"
            worst[key] = max(worst[key], abs(point.parameter - a))
            worst["period"] = max(
                worst["period"],
                abs(point.solution.period - period) / period,
            )
            worst["amplitude"] = max(worst["amplitude"], abs(spread - amplitude))
        # Through the explosion a is flat in m and agrees to the collocation's
        # own error; in the relaxation tail it steepens and m's readout limits it.
        assert worst["a"] < 1.0e-9, worst
        assert worst["a_tail"] < 5.0e-8, worst
        assert worst["period"] < 1.0e-4, worst
        assert worst["amplitude"] < 1.0e-3, worst

    def test_the_explosion_sits_on_the_slow_manifold_canard(self):
        system = VanDerPol(_EPSILON)
        expansion = FenichelExpansion(system, 3)
        attracting = SlowManifold(system, expansion.at(2.0 / 3.0, 2.0, 0.9935))
        repelling = SlowManifold(system, expansion.at(0.0, 0.0, 0.9935))
        canard = CanardSearch(attracting, repelling, 1.0).locate(0.993, 0.994)
        # The cycles that follow S_r from the fold past x = -0.3 all sit within
        # the explosion's exponentially thin window around the canard: about
        # 1e-10 wide (shooting puts m = -0.3 at a_c + 8e-11), plus the
        # collocation's own offset of the same size.
        inside = [
            p.parameter
            for p, m in zip(self.points, self.leftmost, strict=True)
            if -1.2 < m < -0.3
        ]
        assert len(inside) >= 3
        assert max(abs(a - canard.parameter) for a in inside) < 3.0e-10
