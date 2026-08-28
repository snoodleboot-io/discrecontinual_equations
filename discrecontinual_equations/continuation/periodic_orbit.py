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
        mesh = self._mesh
        count = mesh.size
        monitor = np.zeros(count)
        monitor[1:-1] = np.linalg.norm(
            states[2:] - 2.0 * states[1:-1] + states[:-2],
            axis=1,
        )
        monitor[0] = monitor[1]
        monitor[-1] = monitor[-2]
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
                self.set_mesh(previous)
                continue
            states, period = stepped
        return PeriodicOrbitSolution(states, period)

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
