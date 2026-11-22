"""Slow manifolds and canards of a planar fast-slow system, by boundary-value problem.

A planar fast-slow system on the fast time is

    x' = f(x, y, eps),    y' = eps g(x, y, eps),

with ``x`` fast and ``y`` slow. Where the critical manifold ``f(x, y, 0) = 0`` is
normally hyperbolic - ``f_x != 0`` - Fenichel's theorem gives a slow manifold
``x = h(y, eps)`` an ``O(eps)`` distance from it: attracting where ``f_x < 0``,
repelling where ``f_x > 0``. A repelling one cannot be computed by integrating
forward, and an attracting one cannot be followed past a fold where it turns
repelling. Both can be computed as boundary-value problems, which is the approach
of Desroches et al., "Mixed-mode oscillations with multiple time scales", SIAM
Review 54 (2012), section 8.2, problem (8.1):

    u' = T F(u),    u(0) in L,    u(1) in Sigma,

an orbit segment of total time ``T`` from a set ``L`` near the slow manifold to a
cross-section ``Sigma``; ``T < 0`` for a repelling manifold, which is integrated
backward where it attracts. In the plane ``L`` is a point and ``Sigma`` a line, so
the three boundary conditions fix ``T`` and the segment is isolated.

**The start point is Fenichel's expansion, not the critical manifold.** The slow
manifold is the center manifold of the layer problem with ``eps' = 0`` added, so
``h`` has an expansion ``h_0(y) + eps h_1(y) + ...`` found one order at a time from
the invariance equation ``f(h, y, eps) = eps h_y g(h, y, eps)``: each order is a
homological equation whose operator is multiplication by the layer eigenvalue
``f_x(h_0, y, 0)``, invertible exactly where the manifold is normally hyperbolic.
:class:`FenichelExpansion` solves it in truncated two-variable Taylor series (in
the slow offset and in ``eps``), so it needs only the arithmetic of the field. A
segment started on the order-``N`` expansion is ``O(eps^(N+1))`` off the manifold
there and the flow shrinks that at an exponential rate, so the start is no longer
the limit on accuracy.

**A family, not a single solve.** No integration supplies the first segment of a
repelling manifold, so each manifold is continued as a family of orbit segments:
the section starts on the start point itself, where the segment is a single point
with ``T = 0``, and is moved to its target in steps, each segment the seed of the
next (the homotopy of section 8.2). :meth:`SlowManifold.follow` then continues the
finished segment in the system parameter.

**A canard is where the two families meet** (section 8.3). With both manifolds
brought to one section through the fold, the signed gap between their crossings
is a smooth function of the parameter whose zero is the maximal canard, and the
two segments joined there are the seed for the concatenated problem (8.6)-(8.7) -
``u(0) = L_a``, ``u(1) = L_r``, with ``T`` and the parameter both free - which
:class:`MaximalCanard` solves and continues in ``eps``.

The boundary-value problems are handed to SciPy's ``solve_bvp`` (Kierzenka and
Shampine's fourth-order collocation with residual control), whose adaptive mesh
places nodes in the ``O(eps)`` layer at a segment's start and around the fold.
The periodic-orbit collocation elsewhere in the package is for cycles and stays
there.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_bvp

from discrecontinual_equations.continuation.bivariate import Bivariate

# What a fast-slow field is evaluated on: floats, arrays of collocation points,
# or truncated Taylor series in the slow offset and eps.
Value = float | np.ndarray | Bivariate

_SLOW_AXIS = 0
_EPSILON_AXIS = 1
_NEWTON_ITERATIONS = 50
_NEWTON_TOLERANCE = 1.0e-14
_DEFAULT_ORDER = 3
_DEFAULT_TOLERANCE = 1.0e-9
_DEFAULT_MAX_NODES = 20_000
# The family's first section sits this far from the start point, as a fraction of
# the whole sweep; later steps grow by _GROWTH after a success and halve after a
# failure, and a sweep that needs a step smaller than _SMALLEST of the first gives
# up rather than creep.
_FIRST_STEP = 0.02
_GROWTH = 1.5
_SMALLEST = 1.0e-3
_LARGEST = 0.25
_FIRST_NODES = 12
# A seed mesh is resampled to this many points before every solve.
_SEED_NODES = 600
_HALF = 0.5
_BISECTIONS = 6
_SECANT_ITERATIONS = 40
_GAP_TOLERANCE = 1.0e-15
_FAST, _SLOW = 0, 1


class SlowManifoldError(RuntimeError):
    """A slow-manifold or canard computation that did not converge."""


class FastSlowSystem(ABC):
    """``x' = fast(x, y, eps, p)``, ``y' = eps slow(x, y, eps, p)`` on the fast time.

    ``fast`` and ``slow`` must use only ``+ - *`` and integer powers of their
    arguments (and of floats), so that they can be evaluated on arrays of
    collocation points and on :class:`~.bivariate.Bivariate` series alike.
    ``epsilon`` is the time-scale ratio and ``parameter`` the one system
    parameter a family or canard is continued in.
    """

    __slots__ = ["epsilon"]

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    @abstractmethod
    def fast(self, x: Value, y: Value, epsilon: Value, parameter: float) -> Value:
        """The fast component ``f``."""

    @abstractmethod
    def slow(self, x: Value, y: Value, epsilon: Value, parameter: float) -> Value:
        """The slow component ``g``, before the factor ``eps``."""

    def field(self, state: np.ndarray, parameter: float) -> np.ndarray:
        """``(f, eps g)`` at ``state``, which may be ``(2,)`` or ``(2, m)``."""
        x, y = state[_FAST], state[_SLOW]
        rate = self.fast(x, y, self.epsilon, parameter)
        drift = self.epsilon * self.slow(x, y, self.epsilon, parameter)
        first, second = np.broadcast_arrays(np.asarray(rate), np.asarray(drift))
        return np.array([first, second], dtype=float)


@dataclass(frozen=True, slots=True)
class FenichelPoint:
    """The slow manifold over one value of the slow variable, to a given order.

    ``coefficients[k]`` is ``h_k(slow)``, so ``fast`` is their sum against powers
    of ``epsilon``. ``rate`` is the layer eigenvalue ``f_x(h_0, slow, 0)``:
    negative on an attracting branch, positive on a repelling one.
    """

    slow: float
    epsilon: float
    coefficients: tuple[float, ...]
    rate: float

    @property
    def fast(self) -> float:
        """``h(slow, eps)`` summed to the expansion's order."""
        return float(
            sum(c * self.epsilon**k for k, c in enumerate(self.coefficients)),
        )

    @property
    def state(self) -> np.ndarray:
        """The point ``(h(slow, eps), slow)``."""
        return np.array([self.fast, self.slow])

    @property
    def attracting(self) -> bool:
        """Whether the branch attracts in forward time."""
        return self.rate < 0.0


class FenichelExpansion:
    """Fenichel's slow manifold ``x = h(y, eps)`` as a series, one order at a time.

    On the series ring in ``(s, eps)`` with ``y = y_0 + s``, truncated at total
    degree ``order``, the invariance equation's residual is
    ``R(h) = f(h, y, eps) - eps h_s g(h, y, eps)``. Order ``eps^k`` of ``R = 0`` is
    the homological equation ``f_x(h_0(y), y, 0) h_k(y) = (known from h_0 ..
    h_(k-1))``, whose operator is multiplication by the layer eigenvalue as a
    *function* of ``y`` - invertible exactly where the manifold is normally
    hyperbolic. So the expansion is built in two stages:

    1. ``h_0(s)``, the critical manifold, by the chord iteration
       ``h <- h - f(h, y, 0) / f_x(h_0, y_0, 0)``; the constant divisor misses
       only terms one power of ``s`` higher, so each pass fixes one more.
    2. ``h <- h - R(h) / f_x(h_0(s), y_0 + s, 0)``, dividing by the layer
       eigenvalue as a series in ``s``. What that misses - ``f_x(h)`` against
       ``f_x(h_0)``, the nonlinearity, and ``eps h_s g`` - is each one power of
       ``eps`` higher, so each pass fixes one more order and ``order`` passes,
       from ``h_0`` exact at ``eps^0``, reach ``eps^order``.

    A constant divisor in the second stage is not enough: ``eps h_s`` keeps the
    total degree, so the error it leaves is never pushed past the truncation.
    """

    __slots__ = ["_order", "_system"]

    def __init__(self, system: FastSlowSystem, order: int = _DEFAULT_ORDER) -> None:
        if order < 0:
            message = f"order must be non-negative, got {order}"
            raise ValueError(message)
        self._system = system
        self._order = order

    @property
    def order(self) -> int:
        """The highest power of ``eps`` kept."""
        return self._order

    def critical(self, slow: float, guess: float, parameter: float) -> float:
        """The critical manifold's ``x`` over ``slow``, by Newton from ``guess``."""
        x = guess
        for _ in range(_NEWTON_ITERATIONS):
            value, slope = self._linear(x, slow, parameter)
            if slope == 0.0:
                break
            step = value / slope
            x -= step
            if abs(step) <= _NEWTON_TOLERANCE * max(1.0, abs(x)):
                return x
        message = f"no critical point near x = {guess} over y = {slow}"
        raise SlowManifoldError(message)

    def at(self, slow: float, guess: float, parameter: float) -> FenichelPoint:
        """The expansion over ``slow``, on the branch of the critical manifold
        nearest ``guess``."""
        base = self.critical(slow, guess, parameter)
        _, rate = self._linear(base, slow, parameter)
        if rate == 0.0:
            message = f"the critical manifold folds at ({base}, {slow})"
            raise SlowManifoldError(message)
        series = self.series(base, slow, parameter, rate)
        coefficients = tuple(
            float(complex(series.coefficient(0, k)).real)
            for k in range(self._order + 1)
        )
        return FenichelPoint(slow, self._system.epsilon, coefficients, rate)

    def series(
        self,
        base: float,
        slow: float,
        parameter: float,
        rate: float,
    ) -> Bivariate:
        """``h`` as a series in ``(y - slow, eps)`` to total degree ``order``."""
        order = self._order
        system = self._system
        offset = Bivariate.seed(slow, _SLOW_AXIS, order)
        epsilon = Bivariate.seed(0.0, _EPSILON_AXIS, order)
        # h_0(s): the critical manifold over y_0 + s, one power of s per pass,
        # since f_x(h_0(s)) differs from the constant divisor only at order s;
        # the constant term is exact already, so ``order`` passes reach s^order.
        critical = Bivariate.constant(base, order)
        for _ in range(order):
            critical = critical - _series(
                system.fast(critical, offset, 0.0, parameter),
                order,
            ) * (1.0 / rate)
        critical = _truncate_epsilon(critical, 0)
        # f_x(h_0(s), y_0 + s, 0) as a series in s: the eps-coefficient of
        # f(h_0(s) + eps, ...), eps being free until the main iteration.
        shifted = _series(
            system.fast(critical + epsilon, offset, 0.0, parameter),
            order,
        )
        layer = _shift_epsilon(shifted)
        inverse = _reciprocal(layer, rate, order)
        graph = critical
        for _ in range(order):
            graph = graph - self._residual(graph, offset, epsilon, parameter) * inverse
        return graph

    def residual(self, point: FenichelPoint, parameter: float) -> Bivariate:
        """The invariance residual of ``point``'s series: zero through its order."""
        series = self.series(point.coefficients[0], point.slow, parameter, point.rate)
        order = self._order
        return self._residual(
            series,
            Bivariate.seed(point.slow, _SLOW_AXIS, order),
            Bivariate.seed(0.0, _EPSILON_AXIS, order),
            parameter,
        )

    def _residual(
        self,
        graph: Bivariate,
        offset: Bivariate,
        epsilon: Bivariate,
        parameter: float,
    ) -> Bivariate:
        system = self._system
        rate = system.fast(graph, offset, epsilon, parameter)
        drift = system.slow(graph, offset, epsilon, parameter)
        order = graph.order
        return _series(rate, order) - epsilon * graph.derivative(
            _SLOW_AXIS,
        ) * _series(drift, order)

    def _linear(self, x: float, slow: float, parameter: float) -> tuple[float, float]:
        """``f(x, slow, 0)`` and ``f_x`` there, exactly, from a first-order series."""
        value = _series(
            self._system.fast(Bivariate.seed(x, 0, 1), slow, 0.0, parameter),
            1,
        )
        return (
            float(complex(value.coefficient(0, 0)).real),
            float(complex(value.coefficient(1, 0)).real),
        )


def _truncate_epsilon(series: Bivariate, power: int) -> Bivariate:
    """Only the terms of ``series`` with ``eps`` to at most ``power``."""
    return Bivariate(
        {key: value for key, value in series.coefficients.items() if key[1] <= power},
        series.order,
    )


def _shift_epsilon(series: Bivariate) -> Bivariate:
    """The coefficient of ``eps^1`` of ``series``, as a series in ``s`` alone."""
    return Bivariate(
        {
            (key[0], 0): value
            for key, value in series.coefficients.items()
            if key[1] == 1
        },
        series.order,
    )


def _reciprocal(series: Bivariate, constant: float, order: int) -> Bivariate:
    """``1 / series`` for a series whose constant term is ``constant != 0``.

    Written ``constant (1 + q)`` with ``q`` vanishing at the origin, the inverse
    is the geometric series in ``-q``, which terminates at the truncation.
    """
    unit = Bivariate.constant(1.0, order)
    tail = series * (1.0 / constant) - unit
    term, total = unit, unit
    for _ in range(order):
        term = -(term * tail)
        total = total + term
    return total * (1.0 / constant)


def _series(value: Value, order: int) -> Bivariate:
    """``value`` as a series of ``order``: a field term may come back a float."""
    if isinstance(value, Bivariate):
        return value
    return Bivariate.constant(float(np.asarray(value)), order)


@dataclass(frozen=True, slots=True)
class Section:
    """The cross-section ``{u[index] = value}`` an orbit segment ends on."""

    index: int
    value: float


@dataclass(frozen=True, slots=True)
class OrbitSegment:
    """One solved orbit segment: ``u(s)`` on ``s in [0, 1]`` and its total time.

    ``mesh`` and ``states`` (one row per mesh point) are the collocation solution;
    ``time`` is ``T``, negative for a segment integrated backward. ``residual``
    is the largest relative collocation residual ``solve_bvp`` reported.
    """

    mesh: np.ndarray
    states: np.ndarray
    time: float
    parameter: float
    residual: float

    @property
    def start(self) -> np.ndarray:
        """``u(0)``."""
        return self.states[0]

    @property
    def end(self) -> np.ndarray:
        """``u(1)``, on the section."""
        return self.states[-1]

    @property
    def times(self) -> np.ndarray:
        """The mesh in unscaled time, from 0 to ``T``."""
        return self.mesh * self.time


@dataclass(frozen=True, slots=True)
class BoundarySettings:
    """How tightly ``solve_bvp`` solves, and how many nodes it may use."""

    tolerance: float = _DEFAULT_TOLERANCE
    max_nodes: int = _DEFAULT_MAX_NODES


def _coarsened(seed: OrbitSegment) -> OrbitSegment:
    """``seed`` on at most ``_SEED_NODES`` points, placed by arclength.

    ``solve_bvp`` only ever adds nodes, so a family whose every member seeds the
    next would carry each member's mesh forward and double it again; the seed is
    a guess, so it is resampled first and the solver refines what it needs.
    Half the nodes follow the orbit's arclength (the jumps and the start layer)
    and half are uniform in rescaled time (the slow crawl).
    """
    if len(seed.mesh) <= _SEED_NODES:
        return seed
    steps = np.linalg.norm(np.diff(seed.states, axis=0), axis=1)
    arclength = np.concatenate([[0.0], np.cumsum(steps)])
    monitor = _HALF * arclength / arclength[-1] + _HALF * seed.mesh
    targets = np.linspace(0.0, 1.0, _SEED_NODES)
    mesh = np.interp(targets, monitor, seed.mesh)
    states = np.column_stack(
        [np.interp(mesh, seed.mesh, seed.states[:, k]) for k in range(2)],
    )
    return OrbitSegment(mesh, states, seed.time, seed.parameter, seed.residual)


def _solve(
    rhs: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    conditions: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    seed: OrbitSegment,
    unknowns: Sequence[float],
    settings: BoundarySettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    seed = _coarsened(seed)
    solution = solve_bvp(
        rhs,
        conditions,
        seed.mesh,
        seed.states.T,
        p=np.asarray(unknowns, dtype=float),
        tol=settings.tolerance,
        max_nodes=settings.max_nodes,
    )
    if not solution.success:
        message = f"solve_bvp failed: {solution.message}"
        raise SlowManifoldError(message)
    return (
        solution.x,
        solution.y.T,
        solution.p,
        float(np.max(solution.rms_residuals)),
    )


class SlowManifold:
    """One slow manifold of ``system``, as orbit segments of problem (8.1).

    ``start`` is the Fenichel point the segments leave from; whether the manifold
    attracts or repels is read off its layer eigenvalue, and fixes the sign of
    ``T``. Every segment ends on a section ``{u[index] = value}``.
    """

    __slots__ = ["_index", "_settings", "_start", "_system"]

    def __init__(
        self,
        system: FastSlowSystem,
        start: FenichelPoint,
        index: int = _FAST,
        settings: BoundarySettings | None = None,
    ) -> None:
        self._system = system
        self._start = start
        self._index = index
        self._settings = settings or BoundarySettings()

    @property
    def start(self) -> FenichelPoint:
        """The Fenichel point every segment starts on."""
        return self._start

    @property
    def sign(self) -> float:
        """``+1`` forward along an attracting manifold, ``-1`` backward along a
        repelling one: the sign of ``T``."""
        return 1.0 if self._start.attracting else -1.0

    def solve(self, seed: OrbitSegment, section: float) -> OrbitSegment:
        """The segment from the start point to ``{u[index] = section}``.

        ``seed`` is any nearby segment of the family - its mesh, states, time and
        parameter are the initial guess - and the parameter it carries is the
        one solved at.
        """
        system, parameter, point = self._system, seed.parameter, self._start.state
        index = self._index

        def rhs(_s: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
            return p[0] * system.field(u, parameter)

        def conditions(
            left: np.ndarray,
            right: np.ndarray,
            _p: np.ndarray,
        ) -> np.ndarray:
            return np.array(
                [left[0] - point[0], left[1] - point[1], right[index] - section],
            )

        mesh, states, unknowns, residual = _solve(
            rhs,
            conditions,
            seed,
            [seed.time],
            self._settings,
        )
        return OrbitSegment(mesh, states, float(unknowns[0]), parameter, residual)

    def first(self, parameter: float, section: float) -> OrbitSegment:
        """A short segment leaving the start point, a step towards ``section``.

        At ``T = 0`` the segment is the start point itself; a straight line along
        the field for the time the section's coordinate needs is the seed, and
        the solve bends it onto the flow.
        """
        point = self._start.state
        velocity = self._system.field(point, parameter)
        rate = float(velocity[self._index])
        if rate == 0.0:
            message = "the start point is an equilibrium; no segment leaves it"
            raise SlowManifoldError(message)
        distance = _FIRST_STEP * (section - point[self._index])
        time = distance / rate
        if time * self.sign <= 0.0:
            message = (
                "the section lies against the flow from the start point: "
                f"T would be {time} on a manifold of sign {self.sign}"
            )
            raise SlowManifoldError(message)
        mesh = np.linspace(0.0, 1.0, _FIRST_NODES)
        states = point[None, :] + np.outer(mesh * time, velocity)
        seed = OrbitSegment(mesh, states, time, parameter, float("nan"))
        return self.solve(seed, point[self._index] + distance)

    def sweep(
        self,
        parameter: float,
        section: float,
        members: int | None = None,
    ) -> list[OrbitSegment]:
        """The family of segments whose sections run from the start to ``section``.

        Each member's end point lies on the manifold, so the family sweeps it
        out; the last member is the whole manifold up to ``section``. The step
        between sections adapts: it grows after a converged member and halves
        after a failed one. ``members`` asks for that many evenly spaced
        sections instead, with the same halving where one fails.
        """
        origin = float(self._start.state[self._index])
        span = section - origin
        family = [self.first(parameter, origin + _FIRST_STEP * span)]
        reached = _FIRST_STEP
        step = (1.0 - _FIRST_STEP) / members if members else _FIRST_STEP
        while reached < 1.0:
            trial = min(step, 1.0 - reached)
            try:
                member = self.solve(family[-1], origin + (reached + trial) * span)
            except SlowManifoldError:
                step = trial / 2.0
                if step < _SMALLEST:
                    message = (
                        f"the family stalled at section "
                        f"{origin + reached * span} on its way to {section}"
                    )
                    raise SlowManifoldError(message) from None
                continue
            family.append(member)
            reached += trial
            if not members:
                step = min(trial * _GROWTH, _LARGEST)
        return family

    def follow(
        self,
        segment: OrbitSegment,
        parameters: Sequence[float],
        section: float,
    ) -> list[OrbitSegment]:
        """Continue ``segment`` through ``parameters``, each solve seeding the next.

        The start point stays where it was computed: moving it with the
        parameter would change the segment only by the start point's own
        distance from the manifold, which the flow shrinks exponentially.
        """
        followed = []
        seed = segment
        for value in parameters:
            moved = OrbitSegment(
                seed.mesh,
                seed.states,
                seed.time,
                float(value),
                seed.residual,
            )
            seed = self.solve(moved, section)
            followed.append(seed)
        return followed


@dataclass(frozen=True, slots=True)
class Canard:
    """A maximal canard: the parameter, and the orbit joining the two manifolds.

    ``gap`` is the attracting crossing of the shared section minus the repelling
    one, at the located parameter; ``attracting`` and ``repelling`` are the two
    segments that meet there.
    """

    parameter: float
    gap: float
    attracting: OrbitSegment
    repelling: OrbitSegment
    iterations: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)


class CanardSearch:
    """Where the attracting and repelling families meet on a shared section.

    Both manifolds are brought to ``section`` (the fast coordinate through the
    fold) and the signed distance between their crossings, measured along the
    slow coordinate, is the gap whose zero is the maximal canard. Krupa and
    Szmolyan (SIAM J. Math. Anal. 33, 2001, Theorem 3.1) show the gap has a
    nonzero derivative in the parameter there, so a secant iteration converges
    superlinearly.
    """

    __slots__ = ["_attracting", "_repelling", "_section"]

    def __init__(
        self,
        attracting: SlowManifold,
        repelling: SlowManifold,
        section: float,
    ) -> None:
        if not attracting.start.attracting or repelling.start.attracting:
            message = "the first manifold must attract and the second repel"
            raise ValueError(message)
        self._attracting = attracting
        self._repelling = repelling
        self._section = section

    def segments(self, parameter: float) -> tuple[OrbitSegment, OrbitSegment]:
        """Both manifolds swept to the section at ``parameter``."""
        attracting = self._attracting.sweep(parameter, self._section)[-1]
        repelling = self._repelling.sweep(parameter, self._section)[-1]
        return attracting, repelling

    def gap(
        self,
        attracting: OrbitSegment,
        repelling: OrbitSegment,
    ) -> float:
        """The attracting crossing's slow coordinate minus the repelling one's."""
        return float(attracting.end[_SLOW] - repelling.end[_SLOW])

    def locate(self, first: float, second: float) -> Canard:
        """The canard parameter, by secant from ``first`` and ``second``.

        Each iterate continues the previous iterate's segments in the parameter
        rather than sweeping afresh, so the families are swept only twice.
        """
        attracting, repelling = self.segments(first)
        previous = (first, self.gap(attracting, repelling))
        history = [previous]
        current_parameter = second
        for iteration in range(1, _SECANT_ITERATIONS + 1):
            attracting = self._attracting.follow(
                attracting,
                [current_parameter],
                self._section,
            )[-1]
            repelling = self._repelling.follow(
                repelling,
                [current_parameter],
                self._section,
            )[-1]
            current = (current_parameter, self.gap(attracting, repelling))
            history.append(current)
            if abs(current[1]) <= _GAP_TOLERANCE or current[1] == previous[1]:
                return Canard(
                    current[0],
                    current[1],
                    attracting,
                    repelling,
                    iteration,
                    history,
                )
            slope = (current[1] - previous[1]) / (current[0] - previous[0])
            following = current[0] - current[1] / slope
            if following == current[0]:
                return Canard(
                    current[0],
                    current[1],
                    attracting,
                    repelling,
                    iteration,
                    history,
                )
            previous, current_parameter = current, following
        message = f"the canard search did not converge: {history[-3:]}"
        raise SlowManifoldError(message)


@dataclass(frozen=True, slots=True)
class CanardOrbit:
    """A maximal canard as one orbit from ``L_a`` to ``L_r``: problem (8.6)-(8.7)."""

    segment: OrbitSegment
    epsilon: float

    @property
    def parameter(self) -> float:
        """The canard's parameter value."""
        return self.segment.parameter


class MaximalCanard:
    """The maximal canard as one boundary-value problem, continued in ``eps``.

    The orbit runs from the attracting manifold's start point to the repelling
    manifold's - ``u(0) = L_a``, ``u(1) = L_r`` - with its total time and the
    parameter both free: four conditions for two equations and two unknowns.
    It is seeded by the two halves a :class:`CanardSearch` found, joined at the
    section, and needs nothing else; at each new ``eps`` both start points are
    recomputed from the Fenichel expansion at the previous canard's parameter.
    """

    __slots__ = ["_expansion", "_guesses", "_settings", "_system"]

    def __init__(
        self,
        system: FastSlowSystem,
        expansion: FenichelExpansion,
        guesses: tuple[tuple[float, float], tuple[float, float]],
        settings: BoundarySettings | None = None,
    ) -> None:
        self._system = system
        self._expansion = expansion
        self._guesses = guesses
        self._settings = settings or BoundarySettings()

    @staticmethod
    def join(canard: Canard) -> OrbitSegment:
        """The two halves of ``canard`` as one forward orbit from ``L_a`` to ``L_r``.

        The attracting half runs forward for ``T_a``; the repelling half was
        solved backward from ``L_r`` for ``|T_r|``, so it is reversed and appended.
        """
        forward, backward = canard.attracting, canard.repelling
        total = forward.time - backward.time
        first = forward.mesh * forward.time / total
        second = (forward.time - backward.time * (1.0 - backward.mesh[::-1])) / total
        mesh = np.concatenate([first, second[1:]])
        states = np.vstack([forward.states, backward.states[::-1][1:]])
        return OrbitSegment(mesh, states, total, canard.parameter, float("nan"))

    def solve(self, seed: OrbitSegment) -> CanardOrbit:
        """The canard at the system's current ``eps``, from ``seed``."""
        system = self._system
        left, right = self._ends(seed.parameter)

        def rhs(_s: np.ndarray, u: np.ndarray, p: np.ndarray) -> np.ndarray:
            return p[0] * system.field(u, float(p[1]))

        def conditions(
            start: np.ndarray,
            finish: np.ndarray,
            _p: np.ndarray,
        ) -> np.ndarray:
            return np.concatenate([start - left, finish - right])

        mesh, states, unknowns, residual = _solve(
            rhs,
            conditions,
            seed,
            [seed.time, seed.parameter],
            self._settings,
        )
        segment = OrbitSegment(
            mesh,
            states,
            float(unknowns[0]),
            float(unknowns[1]),
            residual,
        )
        return CanardOrbit(segment, system.epsilon)

    def follow(
        self,
        seed: OrbitSegment,
        epsilons: Sequence[float],
    ) -> list[CanardOrbit]:
        """Continue the canard through ``epsilons``, each solve seeding the next.

        ``seed`` is the canard at the system's current ``eps``. A step that fails
        is retried through its midpoint, up to ``_BISECTIONS`` times, so a
        coarse list of targets does not have to be a safe one.
        """
        orbits = []
        current = self._system.epsilon
        for value in epsilons:
            orbit = self._reach(seed, current, float(value), _BISECTIONS)
            orbits.append(orbit)
            seed, current = orbit.segment, float(value)
        return orbits

    def _reach(
        self,
        seed: OrbitSegment,
        start: float,
        target: float,
        depth: int,
    ) -> CanardOrbit:
        self._system.epsilon = target
        try:
            return self.solve(seed)
        except SlowManifoldError:
            if depth == 0:
                raise
        middle = _HALF * (start + target)
        halfway = self._reach(seed, start, middle, depth - 1)
        return self._reach(halfway.segment, middle, target, depth - 1)

    def _ends(self, parameter: float) -> tuple[np.ndarray, np.ndarray]:
        attracting, repelling = self._guesses
        expansion = self._expansion
        return (
            expansion.at(attracting[_SLOW], attracting[_FAST], parameter).state,
            expansion.at(repelling[_SLOW], repelling[_FAST], parameter).state,
        )
