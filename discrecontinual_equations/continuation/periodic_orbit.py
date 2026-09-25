"""Limit cycles by a periodic boundary-value problem, with Floquet multipliers.

A limit cycle is a closed orbit ``x(t + T) = x(t)`` of unknown period ``T``. On the
rescaled interval ``s in [0, 1]`` it satisfies ``x'(s) = T f(x(s))`` with the
periodic boundary condition ``x(1) = x(0)`` and one phase condition to remove the
time-translation freedom. The trajectory is discretised by the trapezoidal rule and
the augmented system - node states plus the period - is solved by Newton's method.

Stability is read from the Floquet multipliers: the eigenvalues of the monodromy
matrix obtained by integrating the variational equation ``Y'(s) = T J(x(s)) Y(s)``,
``Y(0) = I`` over one period. One multiplier is always the trivial ``+1`` along the
orbit; the others govern the cycle's stability and its bifurcations (a real ``+1``
is a fold of cycles, ``-1`` a period-doubling, a complex conjugate pair on the unit
circle a Neimark-Sacker torus bifurcation).
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.function.function import Function

_TOLERANCE = 1.0e-11
_ITERATIONS = 60
_STEP = 1.0e-7
_VARIATIONAL_SUBSTEPS = 8
_PERIOD_FLOOR_FRACTION = 0.1
_BACKTRACK_STEPS = 20
_BACKTRACK_FACTOR = 0.5
# How closely two successive mesh resolutions must agree on the period before the
# coarser of them is called resolved.
_RESOLUTION_TOLERANCE = 1.0e-3
# How many times the mesh may be doubled while searching for that agreement. Each
# doubling is a whole continuation, so this bounds an expensive search rather than
# expressing a numerical limit.
_RESOLUTION_DOUBLINGS = 3
# Consecutive agreements required before a level is certified. Before convergence is
# asymptotic, successive meshes can agree while both are still wrong. Under the
# earlier mesh monitor that happened on van der Pol at mu = 12 (a 200/400 agreement
# of 1.13e-3 against a 400-node error of 1.21e-3). The current monitor converges
# cleanly there and every single agreement measured was conservative, 2.6x to 4.6x,
# but a field the monitor handles less well can still show it, so the second
# confirmation stays.
_RESOLUTION_CONFIRMATIONS = 2


class PeriodicOrbitSolution:
    """A computed limit cycle: node states on ``[0, 1]`` and the period."""

    __slots__ = ["_period", "_states"]

    def __init__(self, states: np.ndarray, period: float) -> None:
        self._states = states
        self._period = period

    @property
    def states(self) -> np.ndarray:
        """State at each mesh node, shape ``(nodes, dimension)``."""
        return self._states

    @property
    def period(self) -> float:
        """The period of the cycle."""
        return self._period

    def component(self, index: int) -> np.ndarray:
        """One state component around the cycle."""
        return self._states[:, index]

    def amplitude(self, index: int) -> float:
        """Peak-to-mean amplitude of one component around the cycle."""
        column = self._states[:, index]
        return float(np.max(column) - np.mean(column))


class ResolutionSettings:
    """Controls for the mesh-resolution search: how close, and how far to look."""

    __slots__ = ["doublings", "tolerance"]

    def __init__(
        self,
        tolerance: float = _RESOLUTION_TOLERANCE,
        doublings: int = _RESOLUTION_DOUBLINGS,
    ) -> None:
        self.tolerance = tolerance
        self.doublings = doublings


class ResolvedCycle:
    """A cycle together with the mesh resolution that was measured to resolve it.

    ``estimate`` is the relative period shift between ``intervals`` and the next
    coarser mesh tried, so it is a measured accuracy claim rather than an assumption.
    """

    __slots__ = ["estimate", "intervals", "period", "states"]

    def __init__(
        self,
        states: np.ndarray,
        period: float,
        intervals: int,
        estimate: float,
    ) -> None:
        self.states = states
        self.period = period
        self.intervals = intervals
        self.estimate = estimate


class ResolutionLevel:
    """One mesh resolution tried during a study, exactly as it was measured.

    ``period`` is ``None`` when the continuation stalled at this resolution.
    ``agreement`` is the relative period shift from the previous level that reached
    the target, or ``None`` when there is no such level to compare against.
    """

    __slots__ = ["agreement", "intervals", "period"]

    def __init__(
        self,
        intervals: int,
        period: float | None,
        agreement: float | None,
    ) -> None:
        self.intervals = intervals
        self.period = period
        self.agreement = agreement


class ResolutionStudy:
    """Every resolution a search tried, and the one it could certify, if any.

    The levels are the measurement; ``resolved`` is a judgement drawn from them. A
    study is returned even when nothing is certified, because the table of periods
    against node count is exactly what is needed to decide what to try next.
    """

    __slots__ = ["levels", "resolved"]

    def __init__(
        self,
        levels: list[ResolutionLevel],
        resolved: ResolvedCycle | None,
    ) -> None:
        self.levels = levels
        self.resolved = resolved

    @property
    def best(self) -> ResolutionLevel | None:
        """The finest resolution that reached the target, certified or not."""
        reached = [level for level in self.levels if level.period is not None]
        return reached[-1] if reached else None


def _certified(levels: list[ResolutionLevel], tolerance: float) -> float | None:
    """The estimate that certifies the finest level so far, or ``None``.

    The last ``_RESOLUTION_CONFIRMATIONS`` levels must each have reached the target
    and agreed with the level before them to ``tolerance``. A stall, or the first
    level, carries no agreement, so a chain broken by either cannot certify.

    The estimate is the largest of those agreements. Extrapolating from the observed
    convergence rate instead - three-level Richardson - was measured and rejected:
    under the earlier mesh monitor convergence was not geometric, and the
    extrapolated error came out 4x to 33x smaller than the true one. Taking the
    largest agreement needs no assumption about the rate.
    """
    recent = levels[-_RESOLUTION_CONFIRMATIONS:]
    if len(recent) < _RESOLUTION_CONFIRMATIONS:
        return None
    agreements = [level.agreement for level in recent]
    if any(agreement is None or agreement >= tolerance for agreement in agreements):
        return None
    return max(agreement for agreement in agreements if agreement is not None)


class PeriodicOrbit:
    """Solve for a limit cycle by a periodic boundary-value problem."""

    __slots__ = ["_function", "_intervals", "_mesh", "_phase_index", "_phase_value"]

    def __init__(
        self,
        function: Function,
        intervals: int,
        phase_index: int = 0,
        phase_value: float = 0.0,
    ) -> None:
        self._function = function
        self._intervals = intervals
        self._phase_index = phase_index
        self._phase_value = phase_value
        self._mesh = np.linspace(0.0, 1.0, intervals + 1)

    @property
    def mesh(self) -> np.ndarray:
        """Node positions on ``[0, 1]`` (uniform unless set otherwise)."""
        return self._mesh

    def set_mesh(self, mesh: np.ndarray) -> None:
        """Replace the node distribution on ``[0, 1]`` (endpoints must be 0 and 1)."""
        self._mesh = np.asarray(mesh, dtype=float)

    def _field(self, state: np.ndarray) -> np.ndarray:
        return np.array(self._function.eval(point=list(state), time=None), dtype=float)

    def _residual(
        self,
        unknowns: np.ndarray,
        nodes: int,
        dimension: int,
    ) -> np.ndarray:
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        period = float(unknowns[-1])
        # Each interior node is the right end of one interval and the left end of
        # the next, so evaluating per node rather than per interval halves the
        # field evaluations - and the line search repeats this up to twenty times
        # per Newton iteration.
        fields = np.array([self._field(states[i]) for i in range(nodes)])
        steps = np.diff(self._mesh[: self._intervals + 1]).astype(float)
        weights = _HALF * period * steps
        collocation = (
            states[1:] - states[:-1] - weights[:, None] * (fields[:-1] + fields[1:])
        )
        blocks = [collocation.ravel(), states[-1] - states[0]]
        phase = states[0, self._phase_index] - self._phase_value
        blocks.append(np.array([phase]))
        return np.concatenate(blocks)

    def estimate_period_error(
        self,
        states: np.ndarray,
        period: float,
    ) -> float | None:
        """Estimate the relative period error by re-solving at doubled resolution.

        A converged solve proves the *discrete* collocation system was satisfied. It
        says nothing about whether the mesh resolves the orbit, and an under-resolved
        mesh converges perfectly well to the wrong cycle - on van der Pol at
        ``mu = 40``, a 200-node continuation reaches the target and reports a period
        12% wrong. Nothing in the residual gate can catch that, so this is the only
        way to know whether a returned cycle is worth trusting.

        The solution is interpolated onto a mesh with twice as many nodes and
        re-solved from there; the period shift is the estimate. ``None`` means the
        refined solve failed, which is a verdict rather than the absence of one: a
        solution too coarse to seed a finer mesh is not resolved.

        The estimate is a single comparison between two meshes, so it is not a bound.
        On van der Pol it measured 2.6x to 4.6x the true error at ``mu = 12``, but
        before convergence is asymptotic two meshes can agree while both are still
        wrong - under an earlier mesh monitor it was 7% optimistic. Treat it as a
        screen for cycles that are badly off, and use :meth:`continue_to_resolved`,
        which requires two consecutive agreements, where the number matters.

        Costs one extra solve at double the nodes: about a fifth of a continuation
        where the cycle is resolved, and far less where it is not, since an
        under-resolved solution fails to refine almost immediately.

        An inter-node collocation defect is the obvious cheaper alternative and does
        *not* work: an adapted mesh concentrates nodes where the field is largest, so
        that defect tracks stiffness rather than error and does not separate a good
        cycle from a bad one.
        """
        nodes = self._intervals + 1
        if states.shape[0] != nodes:
            message = (
                f"states has {states.shape[0]} nodes, but this solver has "
                f"{nodes}; pass the solution this solver returned"
            )
            raise ValueError(message)
        if not np.isfinite(period) or period <= 0.0:
            message = f"period must be finite and positive, got {period}"
            raise ValueError(message)
        coarse = self._mesh
        fine_mesh = np.interp(
            np.linspace(0.0, 1.0, 2 * self._intervals + 1),
            np.linspace(0.0, 1.0, nodes),
            coarse,
        )
        fine_states = np.column_stack(
            [
                np.interp(fine_mesh, coarse, states[:, index])
                for index in range(states.shape[1])
            ],
        )
        refined = type(self)(
            self._function,
            2 * self._intervals,
            phase_index=self._phase_index,
            phase_value=self._phase_value,
        )
        refined.set_mesh(fine_mesh)
        solution = self._refined_solve(refined, fine_states, period)
        if solution is None or not np.isfinite(solution.period):
            return None
        if solution.period == 0.0:
            return None
        return abs(solution.period - period) / abs(solution.period)

    def _refined_solve(
        self,
        refined: "PeriodicOrbit",
        states: np.ndarray,
        period: float,
    ) -> PeriodicOrbitSolution | None:
        """Solve on the refined mesh. Overridden where ``solve`` resets the mesh."""
        return refined.solve(states, period)

    def solve(
        self,
        seed_states: np.ndarray,
        seed_period: float,
    ) -> PeriodicOrbitSolution | None:
        """Refine a limit cycle from seed node states and a seed period."""
        nodes = self._intervals + 1
        dimension = seed_states.shape[1]
        unknowns = np.concatenate(
            [seed_states.astype(float).flatten(), [float(seed_period)]],
        )
        for _ in range(_ITERATIONS):
            residual = self._residual(unknowns, nodes, dimension)
            if np.linalg.norm(residual) < _TOLERANCE:
                break
            jacobian = np.empty((residual.size, unknowns.size))
            for j in range(unknowns.size):
                shifted = unknowns.copy()
                shifted[j] += _STEP
                jacobian[:, j] = (
                    self._residual(shifted, nodes, dimension) - residual
                ) / _STEP
            update, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
            unknowns = unknowns + update
        if np.linalg.norm(self._residual(unknowns, nodes, dimension)) >= _TOLERANCE:
            return None
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        return PeriodicOrbitSolution(states, float(unknowns[-1]))

    def _jacobian(self, state: np.ndarray) -> np.ndarray:
        base = self._field(state)
        dimension = state.size
        columns = np.empty((dimension, dimension))
        for j in range(dimension):
            shifted = state.copy()
            shifted[j] += _STEP
            columns[:, j] = (self._field(shifted) - base) / _STEP
        return columns

    def monodromy(self, solution: PeriodicOrbitSolution) -> np.ndarray:
        """Integrate the variational equation over one period, returning ``Y(1)``."""
        states = solution.states
        period = solution.period
        dimension = states.shape[1]
        transition = np.eye(dimension)
        for i in range(self._intervals):
            step = float(self._mesh[i + 1] - self._mesh[i])
            substep = step / _VARIATIONAL_SUBSTEPS
            start = states[i]
            velocity = (states[i + 1] - states[i]) / step
            for k in range(_VARIATIONAL_SUBSTEPS):
                position = start + velocity * (k * substep)
                matrix = period * self._jacobian(position)
                k1 = matrix @ transition
                k2 = matrix @ (transition + 0.5 * substep * k1)
                k3 = matrix @ (transition + 0.5 * substep * k2)
                k4 = matrix @ (transition + substep * k3)
                transition = transition + substep / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return transition

    def floquet_multipliers(self, solution: PeriodicOrbitSolution) -> np.ndarray:
        """Floquet multipliers: eigenvalues of the monodromy matrix."""
        return np.linalg.eigvals(self.monodromy(solution))


class RobustPeriodicOrbit(PeriodicOrbit):
    """A limit-cycle solver hardened against collapse to the trivial period.

    The periodic boundary-value problem always admits the spurious solution in which
    every node coincides and the period is zero; a plain Newton iteration can slide
    into it when the seed is poor or the cycle is near an infinite-period (SNIC)
    bifurcation. This solver keeps the period above a floor and backtracks any step
    that fails to reduce the residual, so it converges to the genuine cycle in cases
    where the plain solver fails.

    (An adaptive mesh was investigated as the alternative remedy and rejected: for
    smooth cycles the uniform time mesh is already near-optimal, and both arc-length
    and curvature monitors degraded the period. The obstacle near a SNIC is this
    Newton collapse, not mesh resolution.)
    """

    __slots__ = ()

    def solve(
        self,
        seed_states: np.ndarray,
        seed_period: float,
    ) -> PeriodicOrbitSolution | None:
        """Refine a limit cycle with period-positivity damping and backtracking."""
        nodes = self._intervals + 1
        dimension = seed_states.shape[1]
        floor = _PERIOD_FLOOR_FRACTION * float(seed_period)
        unknowns = np.concatenate(
            [seed_states.astype(float).flatten(), [float(seed_period)]],
        )
        for _ in range(_ITERATIONS):
            residual = self._residual(unknowns, nodes, dimension)
            norm = float(np.linalg.norm(residual))
            if norm < _TOLERANCE:
                break
            jacobian = np.empty((residual.size, unknowns.size))
            for j in range(unknowns.size):
                shifted = unknowns.copy()
                shifted[j] += _STEP
                jacobian[:, j] = (
                    self._residual(shifted, nodes, dimension) - residual
                ) / _STEP
            update, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
            unknowns = self._guarded_step(
                unknowns,
                update,
                (nodes, dimension),
                norm,
                floor,
            )
        if np.linalg.norm(self._residual(unknowns, nodes, dimension)) >= _TOLERANCE:
            return None
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        return PeriodicOrbitSolution(states, float(unknowns[-1]))

    def _guarded_step(
        self,
        unknowns: np.ndarray,
        update: np.ndarray,
        shape: tuple[int, int],
        norm: float,
        floor: float,
    ) -> np.ndarray:
        nodes, dimension = shape
        scale = 1.0
        period, change = unknowns[-1], update[-1]
        if change < 0.0 and period + change < floor:
            scale = max(0.0, (period - floor) / (-change))
        for _ in range(_BACKTRACK_STEPS):
            candidate = unknowns + scale * update
            if candidate[-1] > floor and (
                np.linalg.norm(self._residual(candidate, nodes, dimension)) < norm
            ):
                return candidate
            scale *= _BACKTRACK_FACTOR
        return unknowns + scale * update


_MIDPOINT_WEIGHT = 4.0
_SIXTH = 6.0
_EIGHTH = 8.0


class HermiteSimpsonOrbit(PeriodicOrbit):
    """A limit-cycle solver with fourth-order Hermite-Simpson collocation.

    The base solver uses the trapezoidal rule (second order). Hermite-Simpson adds a
    collocation condition at each interval midpoint, taking the midpoint state from
    the cubic Hermite interpolant of the endpoints and their derivatives. The result
    converges at fourth order, so it reaches a given accuracy with far fewer nodes -
    an alternative to mesh adaptation for demanding cycles.
    """

    __slots__ = ()

    def _residual(
        self,
        unknowns: np.ndarray,
        nodes: int,
        dimension: int,
    ) -> np.ndarray:
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        period = float(unknowns[-1])
        blocks = []
        for i in range(self._intervals):
            step = float(self._mesh[i + 1] - self._mesh[i])
            here = period * self._field(states[i])
            ahead = period * self._field(states[i + 1])
            midpoint = 0.5 * (states[i] + states[i + 1]) + (step / _EIGHTH) * (
                here - ahead
            )
            middle = period * self._field(midpoint)
            blocks.append(
                states[i + 1]
                - states[i]
                - (step / _SIXTH) * (here + _MIDPOINT_WEIGHT * middle + ahead),
            )
        blocks.append(states[-1] - states[0])
        phase = states[0, self._phase_index] - self._phase_value
        blocks.append(np.array([phase]))
        return np.concatenate(blocks)


_HALF = 0.5
_AUTODIFF = AutomaticDifferentiation()


class AnalyticPeriodicOrbit(PeriodicOrbit):
    """A limit-cycle solver whose Newton Jacobian is assembled analytically.

    The base solver builds the Newton Jacobian by finite-differencing the whole
    residual, which costs O(N^2) field evaluations and carries finite-difference
    noise. This solver assembles the same (banded) Jacobian block by block from the
    exact state Jacobian of the field, obtained by forward-mode automatic
    differentiation (Taylor jets). The cost drops to O(N) field-Jacobian evaluations
    and the Jacobian is exact, which is the prerequisite for an affordable adaptive
    mesh and for analytic Floquet analysis. The field must be analytic (polynomial or
    rational); transcendental fields should use the base solver.
    """

    __slots__ = ()

    def _sparse_jacobian(
        self,
        unknowns: np.ndarray,
        nodes: int,
        dimension: int,
    ) -> coo_matrix:
        """The same Jacobian as :meth:`_analytic_jacobian`, assembled sparsely.

        Trapezoidal collocation makes this matrix bordered almost-block-diagonal:
        a band from the interval blocks, one dense column for the period, the
        periodicity rows coupling the first and last nodes, and the phase row. It
        is well under 1% nonzero, so materialising it densely costs O(N^2) memory
        and solving it densely costs O(N^3). Assembling the triplets directly
        avoids ever forming the dense array.
        """
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        period = float(unknowns[-1])
        size = unknowns.size
        identity = np.eye(dimension)
        intervals = self._intervals
        derivatives = np.array(
            [_AUTODIFF.jacobian(self._function, states[i], 0.0) for i in range(nodes)],
        )
        fields = np.array([self._field(states[i]) for i in range(nodes)])
        steps = np.diff(self._mesh[: intervals + 1]).astype(float)
        weights = _HALF * period * steps

        # Collocation blocks, built by broadcasting rather than per interval: the
        # block against node i is -I - w Df(x_i) and against node i+1 is
        # I - w Df(x_i+1), with one dense entry per row for the period.
        left = -identity[None] - weights[:, None, None] * derivatives[:-1]
        right = identity[None] - weights[:, None, None] * derivatives[1:]
        span = np.arange(dimension)
        starts = np.arange(intervals) * dimension
        block_rows = np.broadcast_to(
            (starts[:, None] + span)[:, :, None],
            (intervals, dimension, dimension),
        )
        left_columns = np.broadcast_to(
            (starts[:, None] + span)[:, None, :],
            (intervals, dimension, dimension),
        )
        right_columns = left_columns + dimension
        period_rows = starts[:, None] + span
        period_values = -_HALF * steps[:, None] * (fields[:-1] + fields[1:])

        periodic = intervals * dimension + span
        rows = np.concatenate(
            [
                block_rows.ravel(),
                block_rows.ravel(),
                period_rows.ravel(),
                periodic,
                periodic,
                [intervals * dimension + dimension],
            ],
        )
        columns = np.concatenate(
            [
                left_columns.ravel(),
                right_columns.ravel(),
                np.full(intervals * dimension, size - 1),
                intervals * dimension + span,
                span,
                [self._phase_index],
            ],
        )
        values = np.concatenate(
            [
                left.ravel(),
                right.ravel(),
                period_values.ravel(),
                np.ones(dimension),
                -np.ones(dimension),
                [1.0],
            ],
        )
        shape = (intervals * dimension + dimension + 1, size)
        return coo_matrix((values, (rows, columns)), shape=shape)

    def _newton_step(
        self,
        unknowns: np.ndarray,
        nodes: int,
        dimension: int,
        residual: np.ndarray,
    ) -> np.ndarray:
        """Solve one Newton system, falling back to a dense least squares.

        A sparse LU is the fast path. Where the Jacobian is singular it yields a
        non-finite update rather than raising, so the result is checked before it
        is trusted and the dense least-squares solve takes over when it is not.
        """
        sparse = self._sparse_jacobian(unknowns, nodes, dimension).tocsc()
        update = spsolve(sparse, -residual)
        if np.all(np.isfinite(update)):
            return update
        dense = self._analytic_jacobian(unknowns, nodes, dimension)
        fallback, *_ = np.linalg.lstsq(dense, -residual, rcond=None)
        return fallback

    def _analytic_jacobian(
        self,
        unknowns: np.ndarray,
        nodes: int,
        dimension: int,
    ) -> np.ndarray:
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        period = float(unknowns[-1])
        size = unknowns.size
        rows = self._intervals * dimension + dimension + 1
        jacobian = np.zeros((rows, size))
        identity = np.eye(dimension)
        for i in range(self._intervals):
            step = float(self._mesh[i + 1] - self._mesh[i])
            here = _AUTODIFF.jacobian(self._function, states[i], 0.0)
            ahead = _AUTODIFF.jacobian(self._function, states[i + 1], 0.0)
            row = i * dimension
            weight = _HALF * period * step
            jacobian[row : row + dimension, i * dimension : (i + 1) * dimension] = (
                -identity - weight * here
            )
            block = slice((i + 1) * dimension, (i + 2) * dimension)
            jacobian[row : row + dimension, block] = identity - weight * ahead
            jacobian[row : row + dimension, -1] = (
                -_HALF * step * (self._field(states[i]) + self._field(states[i + 1]))
            )
        periodic = self._intervals * dimension
        jacobian[periodic : periodic + dimension, self._intervals * dimension :][
            :,
            :dimension,
        ] = identity
        jacobian[periodic : periodic + dimension, 0:dimension] = -identity
        jacobian[periodic + dimension, self._phase_index] = 1.0
        return jacobian

    def solve(
        self,
        seed_states: np.ndarray,
        seed_period: float,
    ) -> PeriodicOrbitSolution | None:
        """Refine a limit cycle using an exact, analytically-assembled Jacobian."""
        nodes = self._intervals + 1
        dimension = seed_states.shape[1]
        unknowns = np.concatenate(
            [seed_states.astype(float).flatten(), [float(seed_period)]],
        )
        for _ in range(_ITERATIONS):
            residual = self._residual(unknowns, nodes, dimension)
            if np.linalg.norm(residual) < _TOLERANCE:
                break
            update = self._newton_step(unknowns, nodes, dimension, residual)
            unknowns = unknowns + update
        if np.linalg.norm(self._residual(unknowns, nodes, dimension)) >= _TOLERANCE:
            return None
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        return PeriodicOrbitSolution(states, float(unknowns[-1]))


_ADAPTIVE_TOLERANCE = 1.0e-10
_ADAPTIVE_RESIDUAL = 1.0e-8
_ADAPTIVE_ITERATIONS = 80
_MESH_SWEEPS = 15
_MESH_BLEND = 0.6
_MONITOR_FLOOR_FRACTION = 0.15
_MONITOR_SMOOTHING = 3
# Node density follows the cube root of curvature. A gentler target than
# curvature itself: the mesh moves in steps a warm-started Newton solve survives,
# so adaptation actually happens instead of silently failing and leaving the mesh
# where it was.
_MONITOR_EXPONENT = 1.0 / 3.0
_QUARTER = 0.25
_CONTINUATION_INITIAL_STEP = 1.5
# The smallest parameter step worth attempting before declaring the continuation
# stalled. Refining below this does not get past a genuine barrier: on van der Pol
# a 100x smaller floor advanced the stall point by only 0.12 in mu while costing
# two orders of magnitude more solves.
_CONTINUATION_MIN_STEP = 0.1
# How close to ``target`` counts as having arrived.
_CONTINUATION_TOLERANCE = 1.0e-6
_CONTINUATION_SHRINK = 0.5
_CONTINUATION_GROW = 1.3
_CONTINUATION_MAX_STEP = 2.0


class AdaptivePeriodicOrbit(AnalyticPeriodicOrbit):
    """A limit-cycle solver that adapts its mesh to resolve relaxation oscillations.

    For a smooth cycle the uniform time mesh is already near-optimal, but a stiff
    relaxation oscillation (van der Pol at large ``mu``) spends almost all of its
    period on slow branches and crosses the fast jumps in a vanishing time, which a
    uniform mesh cannot resolve - the period error is then first order and large.

    This solver equidistributes a curvature monitor so nodes concentrate at the
    jumps, moving the mesh gradually (each sweep a partial step toward the
    equidistributed target) and re-solving with a damped, analytically-differentiated
    Newton iteration. Gradual movement keeps the interpolated solution a good seed;
    the damping lets Newton converge from it. On stiff cycles this reaches accuracy a
    uniform mesh needs far more nodes to match.
    """

    __slots__ = ()

    def _damped_solve(
        self,
        seed_states: np.ndarray,
        seed_period: float,
    ) -> tuple[np.ndarray, float] | None:
        nodes = self._intervals + 1
        dimension = seed_states.shape[1]
        floor = _PERIOD_FLOOR_FRACTION * float(seed_period)
        unknowns = np.concatenate(
            [seed_states.astype(float).flatten(), [float(seed_period)]],
        )
        for _ in range(_ADAPTIVE_ITERATIONS):
            residual = self._residual(unknowns, nodes, dimension)
            norm = float(np.linalg.norm(residual))
            if norm < _ADAPTIVE_TOLERANCE:
                break
            update = self._newton_step(unknowns, nodes, dimension, residual)
            scale = 1.0
            period, change = unknowns[-1], update[-1]
            if change < 0.0 and period + change < floor:
                scale = max(0.0, (period - floor) / (-change))
            accepted = False
            for _ in range(_BACKTRACK_STEPS):
                candidate = unknowns + scale * update
                accept = candidate[-1] > floor and (
                    np.linalg.norm(self._residual(candidate, nodes, dimension)) < norm
                )
                if accept:
                    accepted = True
                    break
                scale *= _BACKTRACK_FACTOR
            if not accepted:
                # No descent direction here: every backtrack was rejected, so the
                # step left is ~2^-20 of the Newton step and committing it would
                # leave the iterate in place. Stop and let the residual gate below
                # decide - the iterate may already be good enough to accept.
                break
            unknowns = unknowns + scale * update
        if (
            np.linalg.norm(self._residual(unknowns, nodes, dimension))
            >= _ADAPTIVE_RESIDUAL
        ):
            return None
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        return states, float(unknowns[-1])

    def _equidistribute(self, states: np.ndarray) -> np.ndarray:
        """Move the mesh part of the way toward equidistributing the monitor.

        The monitor is the cube root of the curvature, taken as a true second
        derivative on the current non-uniform mesh. It used to be the raw second
        difference of neighbouring states, which is curvature times the local
        spacing squared - a quantity that shrinks wherever nodes are already dense,
        so the mesh read its own refinement as a reason to move nodes away. Its
        target was also sharp enough that the Newton re-solve failed on most
        sweeps, and a failed sweep leaves the mesh untouched while ``solve`` still
        reports success. Continuing van der Pol to ``mu = 40`` on 400 nodes, 75% of
        solves failed and the answer was 17% wrong; with this monitor none fail and
        the error falls at second order as nodes are added.
        """
        mesh = self._mesh
        count = mesh.size
        spacing = np.diff(mesh)
        slopes = (states[1:] - states[:-1]) / spacing[:, None]
        curvature = np.zeros(count)
        curvature[1:-1] = np.linalg.norm(
            2.0 * (slopes[1:] - slopes[:-1]) / (spacing[1:] + spacing[:-1])[:, None],
            axis=1,
        )
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]
        monitor = curvature**_MONITOR_EXPONENT
        for _ in range(_MONITOR_SMOOTHING):
            monitor[1:-1] = (
                _QUARTER * monitor[:-2] + _HALF * monitor[1:-1] + _QUARTER * monitor[2:]
            )
        monitor = monitor + _MONITOR_FLOOR_FRACTION * float(monitor.mean()) + _TOLERANCE
        cumulative = np.concatenate(
            [[0.0], np.cumsum(_HALF * (monitor[:-1] + monitor[1:]) * np.diff(mesh))],
        )
        cumulative /= cumulative[-1]
        target = np.interp(np.linspace(0.0, 1.0, count), cumulative, mesh)
        return (1.0 - _MESH_BLEND) * mesh + _MESH_BLEND * target

    def solve(
        self,
        seed_states: np.ndarray,
        seed_period: float,
        *,
        warm_start: bool = False,
    ) -> PeriodicOrbitSolution | None:
        """Refine a limit cycle, adapting the mesh to concentrate at fast jumps.

        With ``warm_start`` the current mesh is kept as the starting mesh instead of
        being reset to uniform; :meth:`continue_to` uses this to carry the adapted
        mesh from one continuation step to the next.
        """
        if not warm_start:
            self.set_mesh(np.linspace(0.0, 1.0, self._intervals + 1))
        result = self._damped_solve(seed_states, seed_period)
        if result is None:
            return None
        states, period = result
        for _ in range(_MESH_SWEEPS):
            previous = self._mesh.copy()
            target = self._equidistribute(states)
            reseeded = np.column_stack(
                [
                    np.interp(target, previous, states[:, j])
                    for j in range(states.shape[1])
                ],
            )
            self.set_mesh(target)
            stepped = self._damped_solve(reseeded, period)
            if stepped is None:
                # The next sweep would compute the same target from the same
                # states and fail the same way, so there is nothing to retry.
                self.set_mesh(previous)
                break
            states, period = stepped
        return PeriodicOrbitSolution(states, period)

    def _refined_solve(
        self,
        refined: "PeriodicOrbit",
        states: np.ndarray,
        period: float,
    ) -> PeriodicOrbitSolution | None:
        """Warm-start, so the refined mesh survives instead of being reset."""
        return refined.solve(states, period, warm_start=True)

    def continue_to_resolved(
        self,
        parameter_index: int,
        target: float,
        seed_states: np.ndarray,
        seed_period: float,
        settings: ResolutionSettings | None = None,
    ) -> ResolutionStudy:
        """Continue to ``target``, raising the mesh until the period stops moving.

        :meth:`continue_to` answers "did it arrive"; this answers "at what resolution
        is the answer trustworthy", which is the question a caller who does not
        already know the required node count actually has. The continuation is rerun
        at successively doubled meshes until two of them agree on the period to
        ``settings.tolerance``, and the resolution that achieved it is reported
        alongside the cycle.

        Doubling the *mesh* rather than refining the *solution* is deliberate and
        measured. An under-resolved cycle cannot be rescued by refining it - at
        ``mu = 40`` a 400-node solution fails outright when interpolated onto 800
        nodes - and solving cold at a higher resolution is no better, landing on a
        different cycle entirely (period 91.6 against a true 66.5). Only continuation
        reliably reaches a stiff cycle, so each resolution is continued from the
        start rather than lifted from the answer below it.

        Always returns the full study: every resolution tried, whether it reached the
        target, its period, and its agreement with the level before. ``resolved`` is
        set only once two consecutive levels agree to ``settings.tolerance``; one
        agreement is not enough, because before convergence is asymptotic two meshes
        can agree while both are still wrong. ``resolved.estimate`` is the larger of
        those two agreements. When nothing is certified the levels still show how far
        the period was from settling, which is what decides the next attempt.

        This costs one full continuation per resolution tried. Use
        :meth:`estimate_period_error` where a cycle is already in hand and only needs
        checking.
        """
        controls = settings if settings is not None else ResolutionSettings()
        parameter = self._function.parameters[parameter_index]
        original = float(parameter.value)
        nodes = seed_states.shape[0]
        coarse = np.linspace(0.0, 1.0, nodes)
        intervals = self._intervals
        levels: list[ResolutionLevel] = []
        previous: float | None = None
        try:
            for _ in range(controls.doublings + 1):
                parameter.value = original
                fine = np.linspace(0.0, 1.0, intervals + 1)
                seed = np.column_stack(
                    [
                        np.interp(fine, coarse, seed_states[:, index])
                        for index in range(seed_states.shape[1])
                    ],
                )
                orbit = type(self)(
                    self._function,
                    intervals,
                    phase_index=self._phase_index,
                    phase_value=self._phase_value,
                )
                solution = orbit.continue_to(
                    parameter_index,
                    target,
                    seed,
                    seed_period,
                )
                if solution is None:
                    # A stall says nothing about the period, so the chain of
                    # agreements restarts at the next level that does arrive.
                    levels.append(ResolutionLevel(intervals, None, None))
                    previous = None
                    intervals *= 2
                    continue
                agreement = (
                    None
                    if previous is None
                    else abs(solution.period - previous) / abs(solution.period)
                )
                levels.append(ResolutionLevel(intervals, solution.period, agreement))
                estimate = _certified(levels, controls.tolerance)
                if estimate is not None:
                    cycle = ResolvedCycle(
                        solution.states,
                        solution.period,
                        intervals,
                        estimate,
                    )
                    return ResolutionStudy(levels, cycle)
                previous = solution.period
                intervals *= 2
        finally:
            parameter.value = original
        return ResolutionStudy(levels, None)

    def continue_to(
        self,
        parameter_index: int,
        target: float,
        seed_states: np.ndarray,
        seed_period: float,
    ) -> PeriodicOrbitSolution | None:
        """Continue a relaxation cycle in a stiffness parameter to ``target``.

        Adaptive step size: the parameter advances toward ``target`` in steps that
        shrink on a failed solve and grow on success, warm-starting each step from
        the previous adapted mesh and solution. This reaches far stiffer cycles than
        a cold solve, whose Newton iteration fails from a uniform seed at large
        stiffness.

        Returns ``None`` if the continuation stalls - if refining the step below
        ``_CONTINUATION_MIN_STEP`` still will not solve. A returned solution is
        always the cycle *at* ``target``; a stalled continuation never reports the
        cycle it reached along the way as though it had arrived.
        """
        parameter = self._function.parameters[parameter_index]
        start = self.solve(seed_states, seed_period)
        if start is None:
            return None
        states, period = start.states, start.period
        value = float(parameter.value)
        direction = 1.0 if target >= value else -1.0
        step = _CONTINUATION_INITIAL_STEP
        while direction * (target - value) > _CONTINUATION_TOLERANCE:
            saved = self.mesh.copy()
            trial = value + direction * min(step, abs(target - value))
            parameter.value = trial
            candidate = self.solve(states, period, warm_start=True)
            if candidate is None:
                self.set_mesh(saved)
                parameter.value = value
                step *= _CONTINUATION_SHRINK
                if step < _CONTINUATION_MIN_STEP:
                    return None
                continue
            states, period = candidate.states, candidate.period
            value = trial
            step = min(step * _CONTINUATION_GROW, _CONTINUATION_MAX_STEP)
        parameter.value = value
        return PeriodicOrbitSolution(states, period)
