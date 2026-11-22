"""Carry every manifold branch of a frame along the flow in one array pass.

A frame's saddles hand the stage up to a few dozen starts, and each used to be
its own ``solve_ivp`` with a Python right-hand side, so nearly all of a 3-D
film's build was interpreter overhead per step. Here the starts are rows of
one ``(branches, n)`` array, stepped together by the Dormand-Prince 5(4) pair
with a step size, an acceptance and a stopping time of each row's own. The
field is evaluated once per stage for every row that is still moving, on the
columns of the array, wherever the field's arithmetic broadcasts; a field that
does not (a ``math`` call, a branch on a value) is evaluated one row at a
time instead, with the stepping still shared.

Row by row it is the integrator ``solve_ivp(method="RK45")`` runs, so the
branches are the ones a per-branch solve would draw, to rounding: the
coefficients and the Shampine dense output of [1]_ and [2]_, the initial step
of [3]_ (II.4), the same safety factor and growth bounds, the same RMS error
norm, a row stopped where its box event's sign changes and the crossing found
by Brent's method on its interpolant, and the branch resampled at evenly spaced
times from that interpolant.

References
----------
.. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
       formulae", J. Comput. Appl. Math. 6(1), 19-26, 1980.
.. [2] L. F. Shampine, "Some practical Runge-Kutta formulas", Math. Comp.
       46(173), 135-150, 1986.
.. [3] E. Hairer, S. P. Norsett, G. Wanner, "Solving Ordinary Differential
       Equations I: Nonstiff Problems", 2nd ed., Springer, 1993.
"""

from collections.abc import Callable, Sequence
from itertools import pairwise

import numpy as np
from scipy.optimize import brentq

from discrecontinual_equations.function.deterministic import DeterministicFunction

# The Dormand-Prince 5(4) tableau [1]: nodes, stage weights, the fifth-order
# weights the step is taken with, and the difference to the embedded
# fourth-order weights (the seventh stage is the first of the next step).
C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0])
A = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1 / 5, 0.0, 0.0, 0.0, 0.0],
        [3 / 40, 9 / 40, 0.0, 0.0, 0.0],
        [44 / 45, -56 / 15, 32 / 9, 0.0, 0.0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0.0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    ],
)
B = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
E = np.array(
    [-71 / 57600, 0.0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40],
)
# Shampine's free fourth-order interpolant [2]: row j weights stage j in the
# coefficients of x, x^2, x^3, x^4 over the step.
P = np.array(
    [
        [
            1.0,
            -8048581381 / 2820520608,
            8663915743 / 2820520608,
            -12715105075 / 11282082432,
        ],
        [0.0, 0.0, 0.0, 0.0],
        [
            0.0,
            131558114200 / 32700410799,
            -68118460800 / 10900136933,
            87487479700 / 32700410799,
        ],
        [
            0.0,
            -1754552775 / 470086768,
            14199869525 / 1410260304,
            -10690763975 / 1880347072,
        ],
        [
            0.0,
            127303824393 / 49829197408,
            -318862633887 / 49829197408,
            701980252875 / 199316789632,
        ],
        [0.0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
        [0.0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423],
    ],
)
_STAGES = 6
_ORDER = 4  # of the error estimate: the error goes as h ** (order + 1)
_EXPONENT = -1.0 / (_ORDER + 1)
_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 10.0
_EPS = float(np.finfo(float).eps)
# How closely the field evaluated on whole columns must agree with the field
# evaluated a point at a time before the columns are trusted. Elementwise
# arithmetic agrees exactly; a power may differ in its last bit.
_AGREEMENT = 1.0e-12


class _Field:
    """The field at every row of a ``(rows, n)`` array, times each row's sense.

    The columns are handed to ``eval`` as the coordinates, which a field
    written in plain arithmetic evaluates for every row at once. Whether it
    does is settled on the starts (at least two rows, so a one-element array
    cannot pass for a number): a field that raises, returns the wrong shape or
    disagrees with its own per-point values is evaluated row by row instead.
    """

    def __init__(self, function: DeterministicFunction, probe: np.ndarray) -> None:
        self._function = function
        self._columns = self._agrees(np.concatenate([probe, probe[:1]]))

    def __call__(self, states: np.ndarray, senses: np.ndarray) -> np.ndarray:
        values = self._on_columns(states) if self._columns else self._by_row(states)
        return senses[:, None] * values

    def _agrees(self, states: np.ndarray) -> bool:
        try:
            columns = self._on_columns(states)
        except (TypeError, ValueError, AttributeError, IndexError):
            return False
        return bool(
            np.allclose(columns, self._by_row(states), rtol=_AGREEMENT, atol=0.0),
        )

    def _on_columns(self, states: np.ndarray) -> np.ndarray:
        values = self._function.eval(point=list(states.T), time=None)
        if len(values) != states.shape[1]:
            message = "the field has a different dimension from the state"
            raise ValueError(message)
        result = np.empty_like(states)
        for column, value in enumerate(values):
            # A constant component broadcasts down its column.
            result[:, column] = value
        return result

    def _by_row(self, states: np.ndarray) -> np.ndarray:
        return np.array(
            [
                self._function.eval(point=[float(v) for v in state], time=None)
                for state in states
            ],
            dtype=float,
        )


def _rms(values: np.ndarray) -> np.ndarray:
    """The root-mean-square of each row."""
    return np.sqrt(np.einsum("ij,ij->i", values, values) / values.shape[1])


def _leaving(states: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """How far inside the widened box each row is: negative once it is out."""
    return np.minimum(states - low, high - states).min(axis=1)


def _initial_step(
    slope: Callable[[np.ndarray], np.ndarray],
    states: np.ndarray,
    slopes: np.ndarray,
    duration: float,
    tolerance: tuple[float, float],
) -> np.ndarray:
    """The first step of each row, by the rule of [3] (II.4)."""
    rtol, atol = tolerance
    scale = atol + np.abs(states) * rtol
    d0 = _rms(states / scale)
    d1 = _rms(slopes / scale)
    small = (d0 < 1.0e-5) | (d1 < 1.0e-5)  # noqa: PLR2004 (the rule's own cut)
    h0 = np.where(small, 1.0e-6, 0.01 * d0 / np.where(small, 1.0, d1))
    h0 = np.minimum(h0, duration)
    d2 = _rms((slope(states + h0[:, None] * slopes) - slopes) / scale) / h0
    flat = (d1 <= 1.0e-15) & (d2 <= 1.0e-15)  # noqa: PLR2004 (the rule's own cut)
    largest = np.maximum(np.maximum(d1, d2), 1.0e-300)
    h1 = np.where(
        flat,
        np.maximum(1.0e-6, h0 * 1.0e-3),
        (0.01 / largest) ** (1.0 / (_ORDER + 1)),
    )
    return np.minimum(np.minimum(100.0 * h0, h1), duration)


class _Steps:
    """Every accepted step of every row, kept for the dense output.

    A step is its row, where it starts, its size, the state it starts from and
    its seven stages; the interpolant's coefficients are formed from the
    stages once, when the rows are resampled.
    """

    def __init__(self) -> None:
        self._parts: list[tuple[np.ndarray, ...]] = []

    def add(self, *step: np.ndarray) -> None:
        """One iteration's accepted steps: rows, starts, sizes, states, stages."""
        self._parts.append(step)

    def by_row(self, rows: int) -> list[tuple[np.ndarray, ...]]:
        """Each row's steps in order, as starts, sizes, states and ``K^T P``."""
        if not self._parts:
            return [(np.empty(0),) * 4 for _ in range(rows)]
        owner = np.concatenate([part[0] for part in self._parts])
        order = np.argsort(owner, kind="stable")
        starts, sizes, states = (
            np.concatenate([part[k] for part in self._parts])[order] for k in (1, 2, 3)
        )
        stages = np.concatenate([part[4] for part in self._parts], axis=1)[:, order]
        weights = _weights(stages)
        edges = np.searchsorted(owner[order], np.arange(rows + 1))
        return [
            (starts[a:b], sizes[a:b], states[a:b], weights[a:b])
            for a, b in pairwise(edges)
        ]


def _weights(stages: np.ndarray) -> np.ndarray:
    """``K^T P`` for each step: ``(steps, n, 4)`` from ``(7, steps, n)`` stages."""
    return np.einsum("srn,sk->rnk", stages, P)


def _interpolate(segment: tuple[np.ndarray, ...], times: np.ndarray) -> np.ndarray:
    """A row's dense output at ``times``, picking the step each time falls in."""
    starts, sizes, states, weights = segment
    knots = np.concatenate([starts[:1], starts + sizes])
    index = np.clip(np.searchsorted(knots, times, side="left") - 1, 0, len(starts) - 1)
    x = (times - starts[index]) / sizes[index]
    powers = np.cumprod(np.repeat(x[:, None], weights.shape[2], axis=1), axis=1)
    return states[index] + sizes[index][:, None] * np.einsum(
        "tnk,tk->tn",
        weights[index],
        powers,
    )


def _crossing(
    segment: tuple[np.ndarray, ...],
    low: np.ndarray,
    high: np.ndarray,
) -> float:
    """Where in its last step a row crosses the box, by Brent's method."""
    start, size = float(segment[0][0]), float(segment[1][0])

    def margin(time: float) -> float:
        return float(_leaving(_interpolate(segment, np.array([time])), low, high)[0])

    return float(brentq(margin, start, start + size, xtol=4 * _EPS, rtol=4 * _EPS))


def _combine(weights: np.ndarray, stages: np.ndarray) -> np.ndarray:
    """``sum_j weights[j] * stages[j]`` over the leading axis."""
    count = weights.shape[0]
    return (weights @ stages[:count].reshape(count, -1)).reshape(stages.shape[1:])


def flow_branches(  # noqa: PLR0913, PLR0915 (one integrator, its settings spelled out)
    function: DeterministicFunction,
    starts: Sequence[np.ndarray],
    senses: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    duration: float,
    margin: float,
    samples: int,
    tolerance: tuple[float, float] = (1.0e-8, 1.0e-10),
) -> list[list[list[float]]]:
    """Each start carried along the flow until it leaves the widened box.

    ``senses`` is +1 for a row carried forward in time and -1 for one carried
    backward. A row stops where it crosses the edge of ``bounds`` widened by
    ``margin`` - or, like ``solve_ivp``'s sign-change event, crosses back in -
    or at ``duration``, and is returned as ``samples`` states at evenly spaced
    times from its start to its stop. ``tolerance`` is ``(rtol, atol)``.
    """
    if not starts:
        return []
    rtol, atol = tolerance
    begin = np.array([np.asarray(start, dtype=float) for start in starts])
    rows = begin.shape[0]
    low = np.array([lo for lo, _ in bounds], dtype=float) - margin
    high = np.array([hi for _, hi in bounds], dtype=float) + margin
    field = _Field(function, begin)

    # The rows still moving, compacted as rows stop: which row each is, its
    # time, state, slope, next step size, box margin, and whether its last
    # attempt was rejected (no growth on the step after a rejection) or it is
    # starting a fresh step (only then is a tiny step raised to the minimum).
    row = np.arange(rows)
    sense = np.asarray(senses, dtype=float)
    t = np.zeros(rows)
    y = begin.copy()
    f = field(y, sense)
    h_abs = _initial_step(
        lambda states: field(states, sense),
        y,
        f,
        duration,
        tolerance,
    )
    g = _leaving(y, low, high)
    rejected = np.zeros(rows, dtype=bool)
    fresh = np.ones(rows, dtype=bool)

    stop = np.zeros(rows)
    steps = _Steps()
    while row.size:
        min_step = 10.0 * np.abs(np.nextafter(t, np.inf) - t)
        h_abs = np.where(fresh, np.maximum(h_abs, min_step), h_abs)
        stuck = h_abs < min_step
        t_new = np.minimum(t + h_abs, duration)
        h = t_new - t
        stages = np.empty((_STAGES + 1, *y.shape))
        stages[0] = f
        for s in range(1, _STAGES):
            stages[s] = field(y + h[:, None] * _combine(A[s, :s], stages), sense)
        y_new = y + h[:, None] * _combine(B, stages)
        stages[_STAGES] = field(y_new, sense)
        scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
        error = _rms(_combine(E, stages) * h[:, None] / scale)
        accepted = (error < 1.0) & ~stuck
        with np.errstate(divide="ignore"):
            factor = np.where(error == 0.0, np.inf, _SAFETY * error**_EXPONENT)
        grow = np.minimum(_MAX_FACTOR, factor)
        grow = np.where(rejected, np.minimum(1.0, grow), grow)
        h_abs = h * np.where(accepted, grow, np.maximum(_MIN_FACTOR, factor))
        rejected = ~accepted
        fresh = accepted

        steps.add(
            row[accepted], t[accepted], h[accepted], y[accepted], stages[:, accepted]
        )
        # Any change of sign ends the row, so a row still moving has the
        # sign it started with and ``g`` never needs updating.
        g_new = _leaving(y_new, low, high)
        crossed = accepted & (
            ((g <= 0.0) & (g_new >= 0.0)) | ((g >= 0.0) & (g_new <= 0.0))
        )
        ended = (accepted & (t_new >= duration)) | crossed | stuck
        # A row too stuck to step ends where it got to, as solve_ivp's does.
        stop[row[ended]] = np.where(accepted[ended], t_new[ended], t[ended])
        for at in np.flatnonzero(crossed):
            last = (
                t[at : at + 1],
                h[at : at + 1],
                y[at : at + 1],
                _weights(stages[:, at : at + 1]),
            )
            stop[row[at]] = _crossing(last, low, high)

        t = np.where(accepted, t_new, t)
        y = np.where(accepted[:, None], y_new, y)
        f = np.where(accepted[:, None], stages[_STAGES], f)
        if ended.any():
            keep = ~ended
            row, sense, t, y, f, g = (
                row[keep],
                sense[keep],
                t[keep],
                y[keep],
                f[keep],
                g[keep],
            )
            h_abs, rejected, fresh = h_abs[keep], rejected[keep], fresh[keep]

    branches = []
    for index, segment in enumerate(steps.by_row(rows)):
        if len(segment[0]) == 0:
            states = np.repeat(begin[index : index + 1], samples, axis=0)
        else:
            states = _interpolate(segment, np.linspace(0.0, stop[index], samples))
        branches.append([[float(v) for v in state] for state in states])
    return branches
