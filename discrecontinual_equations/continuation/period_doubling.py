"""Follow a period-doubling cascade: locate each doubling, switch, and go on.

:class:`CycleContinuation` detects a period-doubling on a branch of cycles but
stays on that branch. The cycle born there has twice the period, so to follow it
the mesh has to double too - the same number of intervals per turn round the
orbit - and the continuation has to be started on the new branch rather than on
the old cycle traversed twice, which solves the doubled system just as well.
:class:`PeriodDoublingCascade` does both, level after level, and reads each
doubling's parameter off the branch to the accuracy a cascade needs: the gaps
between successive doublings shrink by Feigenbaum's ``delta = 4.6692...``, so by
the fifth the gap is a few parts in ten thousand of the first, and the linear
estimate :func:`classify_transition` makes between two traced points is too
coarse to take ratios of.

Each doubling is located by inverse interpolation: the parameter as a cubic in
the flip multiplier through the four traced points nearest ``-1``, evaluated at
``-1``. The flip multiplier is smooth along the branch, so this is fourth order
in the parameter step where the linear estimate is second.

The switch onto the doubled branch uses the flip multiplier's eigenvector. At a
period-doubling the monodromy has an eigenvalue ``-1``, whose eigenvector ``v``
returns as ``-v`` after one period and as ``v`` after two; the doubled cycle
bifurcates along that direction. The seed for it is the flow from ``x_0 + eps v``
over two periods, sampled on the doubled mesh, solved at a parameter just past
the doubling. Both the doubled cycle and the old cycle traversed twice solve
that system, and they are told apart by how far the seed's two halves differ: an
old cycle's halves coincide. The eigenvector comes from the Hermite monodromy of
the cycle (:meth:`PeriodicOrbit.monodromy`), and the flow from scipy's DOP853.
"""

import warnings
from collections.abc import Sequence

import numpy as np
from scipy.integrate import solve_ivp

from discrecontinual_equations.continuation.cycle_continuation import (
    CycleBifurcation,
    CycleContinuation,
    CyclePoint,
    CycleSeed,
    CycleSeedWarning,
    ResolvedBranch,
    classify_transition,
    resolved_branch,
)
from discrecontinual_equations.continuation.periodic_orbit import (
    PeriodicOrbit,
    PeriodicOrbitSolution,
)
from discrecontinual_equations.differential_equation import DifferentialEquation

# Where the old cycle's halves are told from the doubled cycle's: a doubled
# cycle that differs from its own half-period shift by less than this fraction
# of its size is the old cycle again.
_DISTINCT = 1.0e-3
_FLOQUET_TOLERANCE = 2.0e-2
_INTERPOLATION_POINTS = 4
_INTEGRATION_TOLERANCE = 1.0e-10
# A multiplier is real when its imaginary part is this small beside it.
_REAL = 1.0e-9
# The kicks tried along the flip eigenvector, as fractions of the cycle's size,
# and the steps past the doubling, as fractions of the trace's parameter step.
# Largest first: the doubled cycle stands off the old one as the square root of
# the distance past the doubling, with a large constant on Rossler - 1.6 at
# 0.03 past the first, where kicks of a tenth of the size and less converged
# back onto the old cycle and three tenths found the new one.
_KICKS = (3.0e-1, 1.0e-1, 3.0e-2)
_OFFSETS = (0.5, 0.25, 1.0)
_STEPS = 200
_POINTS_PER_GAP = 16
# The most the arclength is rescaled between two traces of one branch, how
# far the spacing may stray from its target before it is, and how many traces
# a branch gets.
_RESCALE = 3.0
_SPACING_SLACK = 2.0
_TRACE_ATTEMPTS = 4
# Before two gaps are known, how much the next is expected to shrink: a step
# size, not a measurement, so the round number serves.
_GAP_RATIO_PRIOR = 4.0


class CascadeLevel:
    """One branch of a cascade: cycles of one period, up to where it doubles.

    ``multiple`` is the period in units of the first branch's (1, 2, 4, ...),
    ``intervals`` the mesh it was traced on, and ``doubling`` the located
    parameter of its period-doubling, ``None`` where the trace did not reach one.
    """

    __slots__ = ["_doubling", "_intervals", "_multiple", "_points"]

    def __init__(
        self,
        multiple: int,
        intervals: int,
        points: list[CyclePoint],
        doubling: float | None,
    ) -> None:
        self._multiple = multiple
        self._intervals = intervals
        self._points = points
        self._doubling = doubling

    @property
    def multiple(self) -> int:
        """The period, in units of the cascade's first branch."""
        return self._multiple

    @property
    def intervals(self) -> int:
        """Mesh intervals the branch was traced on."""
        return self._intervals

    @property
    def points(self) -> list[CyclePoint]:
        """The traced cycles, from the branch's start to past its doubling."""
        return self._points

    @property
    def doubling(self) -> float | None:
        """Parameter of this branch's period-doubling, by inverse interpolation."""
        return self._doubling

    @property
    def bifurcations(self) -> list[CycleBifurcation]:
        """Every transition :func:`classify_transition` finds along the branch."""
        return [
            transition
            for lower, upper in zip(self._points, self._points[1:], strict=False)
            if (transition := classify_transition(lower, upper)) is not None
        ]

    def resolved(self, tolerance: float = _FLOQUET_TOLERANCE) -> ResolvedBranch:
        """The stretch of this branch whose multipliers can be believed."""
        return resolved_branch(self._points, tolerance)


def flip_multiplier(point: CyclePoint) -> float:
    """The real multiplier nearest ``-1``: the one a period-doubling sends through it.

    ``nan`` when the cycle has no real multiplier on the negative axis.
    """
    multipliers = point.multipliers
    real = multipliers[
        (np.abs(multipliers.imag) <= _REAL * np.abs(multipliers))
        & (multipliers.real < 0.0)
    ].real
    if real.size == 0:
        return float("nan")
    return float(real[np.argmin(np.abs(real + 1.0))])


def locate_doubling(points: Sequence[CyclePoint]) -> float | None:
    """The parameter where the flip multiplier crosses ``-1``, to fourth order.

    The first crossing along ``points`` is bracketed, and the parameter is fitted
    as a cubic in the flip multiplier through the four points nearest it (fewer
    where the branch has fewer). Interpolating the parameter in the multiplier,
    rather than the multiplier in the parameter, makes the answer an evaluation
    instead of a root-find. ``None`` when the flip multiplier never crosses.
    """
    flips = np.array([flip_multiplier(point) for point in points]) + 1.0
    crossing = next(
        (
            index
            for index in range(len(points) - 1)
            if np.isfinite(flips[index])
            and np.isfinite(flips[index + 1])
            and flips[index] * flips[index + 1] <= 0.0
            and flips[index] != flips[index + 1]
        ),
        None,
    )
    if crossing is None:
        return None
    start = max(0, crossing - _INTERPOLATION_POINTS // 2 + 1)
    stop = min(len(points), start + _INTERPOLATION_POINTS)
    start = max(0, stop - _INTERPOLATION_POINTS)
    chosen = [index for index in range(start, stop) if np.isfinite(flips[index])]
    parameters = np.array([points[index].parameter for index in chosen])
    excess = flips[chosen]
    coefficients = np.polyfit(excess, parameters, len(chosen) - 1)
    return float(np.polyval(coefficients, 0.0))


class PeriodDoublingCascade:
    """Follow a cascade of period-doublings, doubling the mesh at each switch.

    ``intervals`` is the first branch's mesh; the branch of period ``2^k`` times
    the first is traced on ``2^k`` times as many, so every branch has the same
    resolution per turn. The remaining arguments are :class:`CycleContinuation`'s,
    with Gauss-Legendre collocation the default here: a cascade is read off the
    flip multiplier, and that is the scheme whose multipliers carry an error
    bound (see :attr:`CyclePoint.floquet_error`). Measured on Rossler against
    Poincare-map shooting, at twenty intervals per turn its flip multiplier was
    6.8e-8 off on the period-1 cycle and 1.4e-5 on the period-16, under a bound
    of 1.5e-6 and 6.9e-5; Hermite-Simpson on the same unknowns, eighty nodes per
    turn, was 1.4e-7 and 4.2e-3 off while its trivial multiplier read 1.2e-7
    and 9.8e-5 - low by forty-three times on the long orbit.
    """

    __slots__ = [
        "_collocation",
        "_collocation_points",
        "_equation",
        "_intervals",
        "_parameter_index",
        "_phase_index",
        "_phase_value",
    ]

    def __init__(  # noqa: PLR0913 (CycleContinuation's own arguments)
        self,
        equation: DifferentialEquation,
        parameter_index: int,
        intervals: int,
        phase_index: int = 0,
        phase_value: float = 0.0,
        *,
        collocation: str = "gauss_legendre",
        collocation_points: int | None = None,
    ) -> None:
        self._equation = equation
        self._parameter_index = parameter_index
        self._intervals = intervals
        self._phase_index = phase_index
        self._phase_value = phase_value
        self._collocation = collocation
        self._collocation_points = collocation_points

    def continuation(self, level: int) -> CycleContinuation:
        """The continuation for the branch of period ``2^level``: its mesh doubled."""
        return CycleContinuation(
            self._equation,
            self._parameter_index,
            self._intervals * 2**level,
            self._phase_index,
            self._phase_value,
            collocation=self._collocation,
            collocation_points=self._collocation_points,
        )

    def follow(  # noqa: PLR0913 (the trace's own dials)
        self,
        seed: CycleSeed,
        arclength: float,
        levels: int,
        *,
        steps: int = _STEPS,
        points_per_gap: int = _POINTS_PER_GAP,
        direction: float = 1.0,
    ) -> list[CascadeLevel]:
        """Trace ``levels`` branches, switching at each period-doubling.

        ``seed`` is a cycle on the first branch and ``arclength`` the first
        branch's step; ``steps`` is the most any one branch is traced for, and
        ``direction`` the first branch's sense in the parameter, as for
        :meth:`CycleContinuation.trace`. Each branch is traced until its
        doubling is crossed and kept to two points past it, so the cubic has
        points on both sides. Every later branch is traced away from the
        doubling it was born at, on whichever side its seed was found: past it
        for a supercritical doubling, before it for a subcritical one.

        The step is the cascade's own business after the first branch, because
        the gap it has to resolve shrinks by about ``delta`` per level while one
        arclength step covers less and less parameter as the mesh grows. Each
        later branch is stepped so that its predicted gap - the last gap over
        the last measured ratio, or over four before a ratio is known - spans
        ``points_per_gap`` points; the second branch, whose gap nothing predicts
        yet, keeps the first's parameter spacing. The arclength that gives that
        spacing is carried over from the previous branch and corrected on the
        branch itself (see :meth:`_trace_to_doubling`). The cubic's error is
        fourth order in the spacing relative to the gap, so a fixed count per
        gap keeps every doubling located to about the same relative accuracy.

        Stops early - returning the levels it has - where a branch never
        doubles within its steps, or where no seed on the doubled branch is
        found.
        """
        cascade: list[CascadeLevel] = []
        current: CycleSeed | None = seed
        heading = direction
        spacing: float | None = None
        stand_off: float | None = None
        for level in range(levels):
            if current is None:
                break
            intervals = self._intervals * 2**level
            target = self._target_spacing(cascade, spacing, points_per_gap)
            if target is not None and spacing is not None:
                arclength = arclength * target / spacing
            points, arclength = self._trace_to_doubling(
                self.continuation(level),
                current,
                heading,
                arclength,
                steps,
                target,
                points_per_gap + _INTERPOLATION_POINTS,
            )
            doubling = locate_doubling(points)
            cascade.append(CascadeLevel(2**level, intervals, points, doubling))
            if doubling is None or level + 1 == levels:
                break
            spacing = _median_spacing(points)
            switched = self._doubled_seed(points, doubling, level + 1, stand_off)
            current = None
            if switched is not None:
                current, stand_off = switched
                heading = float(np.sign(current.parameter - doubling))
        return cascade

    @staticmethod
    def _target_spacing(
        cascade: list[CascadeLevel],
        spacing: float | None,
        points_per_gap: int,
    ) -> float | None:
        """The parameter spacing the next branch should be traced at.

        ``None`` for the first branch, which takes its caller's arclength; the
        previous branch's spacing for the second; and the predicted gap over
        ``points_per_gap`` after that.
        """
        doublings = [level.doubling for level in cascade if level.doubling is not None]
        if len(doublings) < 2:  # noqa: PLR2004 (a gap needs two doublings)
            return spacing
        gaps = np.diff(doublings)
        ratio = gaps[-2] / gaps[-1] if gaps.size > 1 else _GAP_RATIO_PRIOR
        return float(gaps[-1] / ratio / points_per_gap)

    def _trace_to_doubling(  # noqa: PLR0913 (one branch's whole trace)
        self,
        continuation: CycleContinuation,
        seed: CycleSeed,
        direction: float,
        arclength: float,
        steps: int,
        spacing: float | None,
        budget: int,
    ) -> tuple[list[CyclePoint], float]:
        """Trace one branch from ``seed`` until a period-doubling is crossed.

        Returns the points, up to two past the doubling, and the arclength they
        were traced at. The branch is traced from ``seed`` for ``budget`` steps
        and, where that falls short, traced again from ``seed`` with as many
        steps as the distance covered says it needs (twice as many where it
        crossed nothing), up to ``steps``. With a target ``spacing``, a trace
        whose parameter step - the median over its second half - is more than
        twice or less than half the target is traced again at the arclength
        rescaled by the ratio, by at most a factor of three. The second half,
        because the parameter barely moves at first: the doubled cycle leaves
        its birth as the square root of the distance past it.

        Always again from the seed, never on from a traced point. That was a
        workaround: re-seeding :class:`CycleContinuation` solved the seed under
        trapezoidal collocation first, which failed on long orbits traced under
        Gauss-Legendre, or carried a period-2 cycle to its other crossing of
        the section. Since DEQ-106 a trace restarts from any of its points
        (:meth:`CycleSeed.from_point`), but retracing from the seed is kept so
        that every level's points are the ones the cascade always gave.
        """
        points: list[CyclePoint] = []
        past = _INTERPOLATION_POINTS // 2
        for _ in range(_TRACE_ATTEMPTS):
            points, _ = continuation.trace(seed, arclength, budget, direction)
            if len(points) <= budget:  # the branch was lost, or never found
                break
            crossed = _first_doubling(points)
            scale = 1.0
            if spacing is not None:
                measured = _median_spacing(points[len(points) // 2 :])
                if measured > 0.0:
                    scale = min(_RESCALE, max(1.0 / _RESCALE, spacing / measured))
            on_spacing = 1.0 / _SPACING_SLACK <= scale <= _SPACING_SLACK
            if crossed is not None and on_spacing and len(points) > crossed + past:
                break
            covered = abs(points[-1].parameter - seed.parameter)
            if crossed is not None:
                covered = abs(points[crossed + 1].parameter - seed.parameter)
            reach = covered if crossed is not None else 2.0 * covered
            step = _median_spacing(points[len(points) // 2 :]) * scale
            needed = int(np.ceil(reach / step)) + past + 1 if step > 0.0 else budget
            if budget >= steps and (crossed is None or on_spacing):
                break
            arclength = arclength * scale
            budget = min(
                steps, max(needed, budget if crossed is not None else 2 * budget)
            )
        crossed = _first_doubling(points)
        if crossed is not None:
            return points[: crossed + 1 + past], arclength
        return points, arclength

    def _cycle_seed(
        self,
        start: np.ndarray,
        period: float,
        parameter: float,
        intervals: int,
    ) -> CycleSeed | None:
        """The flow from ``start`` over ``period``, as a seed on ``intervals``."""
        states = self._flow(start, period, parameter, intervals)
        if states is None:
            return None
        return CycleSeed(states, period, parameter)

    def _doubled_seed(
        self,
        points: list[CyclePoint],
        doubling: float,
        level: int,
        stand_off: float | None,
    ) -> tuple[CycleSeed, float] | None:
        """A seed on the doubled branch near ``doubling``, and its stand-off.

        The traced point nearest the doubling gives the eigenvector; the flow is
        taken from it at the trial parameter, so the seed's first half is the
        old cycle displaced along ``v`` and its second half the same displaced
        along ``-v``.

        The doubled cycle stands off the old one as ``k sqrt(|c - c_pd|)``,
        and ``stand_off`` is the ``k`` the last switch measured, so the kick
        tried first is the one that lands on it; the fixed kicks are the
        fallback, and all there is at the first switch. That ordering is what
        makes a switch cheap: a kick that misses costs the seed solve its full
        forty iterations of dense Newton, 46 s on 80 Rossler intervals and 204
        s on 160 when measured, against 5 s for one that lands. Returns the
        measured ``k`` with the seed.
        """
        nearest = min(points, key=lambda point: abs(point.parameter - doubling))
        spacing = _median_spacing(points)
        direction = self._flip_direction(nearest)
        continuation = self.continuation(level)
        intervals = self._intervals * 2**level
        size = nearest.amplitude
        for offset in _OFFSETS:
            for sign in (1.0, -1.0):
                distance = offset * spacing
                parameter = doubling + sign * distance
                kicks = list(_KICKS)
                if stand_off is not None:
                    predicted = 0.5 * stand_off * np.sqrt(distance) / size
                    kicks = [predicted, 3.0 * predicted, predicted / 3.0, *kicks]
                for kick in kicks:
                    trial = self._cycle_seed(
                        nearest.solution.states[0] + kick * size * direction,
                        2.0 * nearest.solution.period,
                        parameter,
                        intervals,
                    )
                    if trial is None:
                        continue
                    with warnings.catch_warnings():
                        # Most kicks find no cycle; that is the search, not news.
                        warnings.simplefilter("ignore", CycleSeedWarning)
                        solved, _ = continuation.trace(trial, 0.0, 0)
                    if solved and _halves_differ(solved[0], size):
                        return trial, _start_gap(solved[0]) / np.sqrt(distance)
        return None

    def _flip_direction(self, point: CyclePoint) -> np.ndarray:
        """The unit eigenvector of the monodromy for the multiplier nearest ``-1``."""
        self._equation.derivative.parameters[self._parameter_index].value = (
            point.parameter
        )
        orbit = PeriodicOrbit(
            self._equation.derivative,
            point.solution.states.shape[0] - 1,
            self._phase_index,
            self._phase_value,
        )
        orbit.set_mesh(point.mesh)
        monodromy = orbit.monodromy(
            PeriodicOrbitSolution(point.solution.states, point.solution.period),
        )
        values, vectors = np.linalg.eig(monodromy)
        vector = np.real(vectors[:, np.argmin(np.abs(values + 1.0))])
        return vector / np.linalg.norm(vector)

    def _flow(
        self,
        start: np.ndarray,
        duration: float,
        parameter: float,
        intervals: int,
    ) -> np.ndarray | None:
        """The flow from ``start`` sampled at ``intervals + 1`` uniform times."""
        field = self._equation.derivative
        field.parameters[self._parameter_index].value = parameter

        def rhs(_time: float, state: np.ndarray) -> list[float]:
            return field.eval(point=list(state), time=None)

        times = np.linspace(0.0, duration, intervals + 1)
        solution = solve_ivp(
            rhs,
            (0.0, duration),
            start,
            method="DOP853",
            t_eval=times,
            rtol=_INTEGRATION_TOLERANCE,
            atol=_INTEGRATION_TOLERANCE,
        )
        if not solution.success:
            return None
        return np.asarray(solution.y.T, dtype=float)


def _first_doubling(points: Sequence[CyclePoint]) -> int | None:
    """Index of the point just before the first period-doubling, if any."""
    for index in range(len(points) - 1):
        transition = classify_transition(points[index], points[index + 1])
        if transition is not None and transition.kind == "period_doubling":
            return index
    return None


def _median_spacing(points: Sequence[CyclePoint]) -> float:
    """The median parameter step between consecutive points."""
    return float(np.median(np.abs(np.diff([point.parameter for point in points]))))


def _half_gap(point: CyclePoint) -> float:
    """Largest distance between a doubled-mesh cycle and its half-period shift."""
    states = point.solution.states
    half = (states.shape[0] - 1) // 2
    return float(np.max(np.linalg.norm(states[:half] - states[half:-1], axis=1)))


def _start_gap(point: CyclePoint) -> float:
    """Distance between a doubled-mesh cycle's start and its half-period point.

    Twice the doubled cycle's displacement from the old one where the kick is
    applied, which the largest gap over the orbit is not: the flip
    eigenfunction's size varies round the orbit.
    """
    states = point.solution.states
    half = (states.shape[0] - 1) // 2
    return float(np.linalg.norm(states[0] - states[half]))


def _halves_differ(point: CyclePoint, size: float) -> bool:
    """Whether a cycle on a doubled mesh is a new cycle, not an old one twice.

    The two halves of an old cycle traversed twice coincide; a doubled cycle's
    differ by about twice its distance from the old one.
    """
    return point.amplitude > 0.0 and _half_gap(point) > _DISTINCT * size


def feigenbaum_ratios(doublings: Sequence[float]) -> list[float]:
    """Successive gap ratios ``(c_k - c_(k-1)) / (c_(k+1) - c_k)``.

    For a cascade in the Feigenbaum universality class these converge to
    ``delta = 4.669201609...``.
    """
    gaps = np.diff(np.asarray(doublings, dtype=float))
    return [float(gaps[k] / gaps[k + 1]) for k in range(gaps.size - 1)]
