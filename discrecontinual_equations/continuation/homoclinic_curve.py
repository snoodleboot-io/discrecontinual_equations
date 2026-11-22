"""Continue a saddle homoclinic orbit in two parameters (a global codim-1 curve).

The single-orbit projection boundary-value problem in :mod:`connecting_orbit` solves
a homoclinic for a fixed field. Continuing it turns the homoclinic into a curve in a
two-parameter plane - the global bifurcation curve that, near a Bogdanov-Takens
point, emanates tangent to the Hopf curve.

The augmented system carries the orbit nodes, the saddle equilibrium, and one active
continuation parameter as unknowns; the other parameter is swept. Its equations are
the trapezoidal collocation of the orbit, the equilibrium condition, the projection
boundary conditions onto the (recomputed) stable and unstable eigenspaces of the
saddle, and a phase condition. The square system is solved by Newton's method, and
each swept point is seeded from the previous one.

The Newton matrix is block-banded in the nodes and bordered by the equilibrium,
boundary and phase rows and the parameter column - HomCont's structure (Champneys,
Kuznetsov & Sandstede, IJBC 6, 1996) - so it is assembled sparse and factored with
a sparse LU, in time and memory linear in the nodes.
"""

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu

from discrecontinual_equations.continuation.connecting_orbit import (
    MeshSpec,
    OrbitSolution,
    _collocation_triplets,
)
from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.differential_equation import DifferentialEquation

_TOLERANCE = 1.0e-10
_ITERATIONS = 80
_STEP = 1.0e-7
_REAL_FLOOR = 1.0e-9
_HALF = 0.5
# Column ordering for the sparse LU; see ``connecting_orbit._factor`` for why.
_ORDERING = "COLAMD"
_AUTODIFF = AutomaticDifferentiation()


class HomoclinicCurvePoint:
    """A solved point on the homoclinic curve: its two parameters and orbit."""

    __slots__ = ["_continuation", "_equilibrium", "_orbit", "_sweep"]

    def __init__(
        self,
        continuation: float,
        sweep: float,
        orbit: OrbitSolution,
        equilibrium: np.ndarray,
    ) -> None:
        self._continuation = continuation
        self._sweep = sweep
        self._orbit = orbit
        self._equilibrium = np.asarray(equilibrium, dtype=float)

    @property
    def continuation(self) -> float:
        """Value of the active continuation parameter on the curve."""
        return self._continuation

    @property
    def sweep(self) -> float:
        """Value of the swept parameter."""
        return self._sweep

    @property
    def orbit(self) -> OrbitSolution:
        """The homoclinic orbit at this point."""
        return self._orbit

    @property
    def equilibrium(self) -> np.ndarray:
        """The saddle equilibrium the orbit connects to."""
        return self._equilibrium


class HomoclinicCurve:
    """Trace a saddle homoclinic bifurcation curve in two parameters."""

    __slots__ = [
        "_continuation_index",
        "_equation",
        "_mesh",
        "_sweep_index",
        "_time",
    ]

    def __init__(
        self,
        equation: DifferentialEquation,
        continuation_index: int,
        sweep_index: int,
        mesh: MeshSpec,
        time: float = 0.0,
    ) -> None:
        self._equation = equation
        self._continuation_index = continuation_index
        self._sweep_index = sweep_index
        self._mesh = mesh
        self._time = time

    def _field(
        self,
        point: np.ndarray,
        continuation: float,
        sweep: float,
    ) -> np.ndarray:
        parameters = self._equation.derivative.parameters
        parameters[self._continuation_index].value = float(continuation)
        parameters[self._sweep_index].value = float(sweep)
        return np.array(
            self._equation.derivative.eval(point=list(point), time=self._time),
            dtype=float,
        )

    def _field_jacobian(
        self,
        point: np.ndarray,
        continuation: float,
        sweep: float,
    ) -> np.ndarray:
        base = self._field(point, continuation, sweep)
        dimension = point.size
        columns = np.empty((dimension, dimension))
        for j in range(dimension):
            shifted = point.copy()
            shifted[j] += _STEP
            columns[:, j] = (self._field(shifted, continuation, sweep) - base) / _STEP
        return columns

    def _residual(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        nodes: int,
        dimension: int,
        sweep: float,
    ) -> np.ndarray:
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        equilibrium = unknowns[nodes * dimension : nodes * dimension + dimension]
        continuation = float(unknowns[-1])
        blocks = []
        for i in range(nodes - 1):
            step = times[i + 1] - times[i]
            here = self._field(states[i], continuation, sweep)
            ahead = self._field(states[i + 1], continuation, sweep)
            blocks.append(states[i + 1] - states[i] - 0.5 * step * (here + ahead))
        blocks.append(
            self._tail(states[0], states[-1], equilibrium, continuation, sweep),
        )
        centre = nodes // 2
        phase = states[centre, self._mesh.phase_index] - self._mesh.phase_value
        blocks.append(np.array([phase]))
        return np.concatenate(blocks)

    def _tail(
        self,
        departure_state: np.ndarray,
        arrival_state: np.ndarray,
        equilibrium: np.ndarray,
        continuation: float,
        sweep: float,
    ) -> np.ndarray:
        """The equilibrium condition and the projection boundary conditions.

        These are the only rows that see the saddle's eigenvectors, and the only
        ones besides the phase condition that reach across the mesh: they touch
        the two end nodes, the equilibrium and the continuation parameter.
        """
        jacobian = self._field_jacobian(equilibrium, continuation, sweep)
        values, vectors = np.linalg.eig(jacobian)
        left = np.linalg.inv(vectors)
        stable = [i for i in range(len(values)) if values[i].real < -_REAL_FLOOR]
        unstable = [i for i in range(len(values)) if values[i].real > _REAL_FLOOR]
        departure = departure_state - equilibrium
        arrival = arrival_state - equilibrium
        boundary = [(left[i] @ departure).real for i in stable]
        boundary += [(left[i] @ arrival).real for i in unstable]
        return np.concatenate(
            [self._field(equilibrium, continuation, sweep), np.array(boundary)],
        )

    def _jacobian(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        dimension: int,
        sweep: float,
        residual: np.ndarray,
    ) -> csr_matrix:
        """Assemble the bordered block-banded Newton matrix, in O(N) field work.

        This is the defining system HomCont continues a homoclinic orbit to a
        saddle with (Champneys, Kuznetsov & Sandstede 1996): collocation, the
        equilibrium condition, Beyn's projection boundary conditions, a phase
        condition, and one free system parameter. Its Jacobian has three parts:

        * the collocation rows, banded in the nodes exactly as for a single
          connecting orbit, from one field Jacobian per node, plus the parameter
          column ``-(h/2) (f_a(x_i) + f_a(x_{i+1}))``;
        * the equilibrium and boundary rows, which touch only the two end nodes,
          the equilibrium and the parameter, and whose dependence on the saddle's
          eigenvectors is differenced - as HomCont does - in ``3n + 1``
          evaluations that do not grow with the mesh;
        * the phase row, a single unit entry.

        The dense finite-difference matrix it replaces took one full residual per
        unknown: ``O(N^2)`` field evaluations and ``O(N^2)`` memory per iteration.
        """
        size = unknowns.size
        nodes = times.size
        states = unknowns[: nodes * dimension].reshape(nodes, dimension)
        continuation = float(unknowns[-1])
        derivatives = self._node_jacobians(states, continuation, sweep)
        row, column, value = _collocation_triplets(derivatives, times)
        bands = (nodes - 1) * dimension
        sensitivity = np.array(
            [
                self._parameter_sensitivity(state, continuation, sweep)
                for state in states
            ],
        )
        weights = _HALF * np.diff(times)[:, None]
        parameter = -weights * (sensitivity[:-1] + sensitivity[1:])
        rows = [row, np.arange(bands)]
        columns = [column, np.full(bands, size - 1)]
        values = [value, parameter.ravel()]
        tail_rows = bands + np.arange(residual.size - 1 - bands)
        base = residual[bands:-1]
        for index in self._tail_columns(nodes, dimension, size):
            shifted = unknowns.copy()
            shifted[index] += _STEP
            moved = self._tail(
                shifted[:dimension],
                shifted[bands : bands + dimension],
                shifted[nodes * dimension : nodes * dimension + dimension],
                float(shifted[-1]),
                sweep,
            )
            rows.append(tail_rows)
            columns.append(np.full(tail_rows.size, index))
            values.append((moved - base) / _STEP)
        rows.append(np.array([residual.size - 1]))
        columns.append(
            np.array([(nodes // 2) * dimension + self._mesh.phase_index]),
        )
        values.append(np.array([1.0]))
        return csr_matrix(
            (
                np.concatenate(values),
                (np.concatenate(rows), np.concatenate(columns)),
            ),
            shape=(residual.size, size),
        )

    @staticmethod
    def _tail_columns(nodes: int, dimension: int, size: int) -> list[int]:
        """Unknowns the tail rows depend on: both end nodes, equilibrium, parameter."""
        last = (nodes - 1) * dimension
        equilibrium = nodes * dimension
        return [
            *range(dimension),
            *range(last, last + dimension),
            *range(equilibrium, equilibrium + dimension),
            size - 1,
        ]

    def _node_jacobians(
        self,
        states: np.ndarray,
        continuation: float,
        sweep: float,
    ) -> np.ndarray:
        """The field Jacobian at every node, exact where the field allows it."""
        parameters = self._equation.derivative.parameters
        parameters[self._continuation_index].value = float(continuation)
        parameters[self._sweep_index].value = float(sweep)
        try:
            return np.array(
                [
                    _AUTODIFF.jacobian(self._equation.derivative, state, self._time)
                    for state in states
                ],
            )
        except (TypeError, ValueError, AttributeError):
            return np.array(
                [self._field_jacobian(state, continuation, sweep) for state in states],
            )

    def _parameter_sensitivity(
        self,
        point: np.ndarray,
        continuation: float,
        sweep: float,
    ) -> np.ndarray:
        """Forward difference of the field in the continuation parameter."""
        base = self._field(point, continuation, sweep)
        return (self._field(point, continuation + _STEP, sweep) - base) / _STEP

    @staticmethod
    def _step(jacobian: csr_matrix, residual: np.ndarray) -> np.ndarray:
        """Newton step from a sparse LU of the square bordered system.

        Falls back to the dense least-squares solve the dense Newton used when the
        matrix is singular or not square, which happens only for a degenerate
        saddle whose boundary-condition count does not match the unknowns.
        """
        if jacobian.shape[0] == jacobian.shape[1]:
            try:
                step = splu(csc_matrix(jacobian), permc_spec=_ORDERING).solve(
                    -residual,
                )
            except RuntimeError:  # exactly singular: use the dense fallback
                step = np.full(residual.size, np.nan)
            if np.all(np.isfinite(step)):
                return step
        fallback, *_ = np.linalg.lstsq(jacobian.toarray(), -residual, rcond=None)
        return fallback

    def _newton(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        nodes: int,
        dimension: int,
        sweep: float,
    ) -> np.ndarray | None:
        for _ in range(_ITERATIONS):
            residual = self._residual(unknowns, times, nodes, dimension, sweep)
            if np.linalg.norm(residual) < _TOLERANCE:
                return unknowns
            jacobian = self._jacobian(
                unknowns,
                times,
                dimension,
                sweep,
                residual,
            )
            unknowns = unknowns + self._step(jacobian, residual)
        residual = self._residual(unknowns, times, nodes, dimension, sweep)
        if np.linalg.norm(residual) < _TOLERANCE:
            return unknowns
        return None

    def solve_point(
        self,
        sweep: float,
        orbit_seed: np.ndarray,
        equilibrium_seed: np.ndarray,
        continuation_seed: float,
    ) -> HomoclinicCurvePoint | None:
        """Solve one point of the curve at a fixed swept-parameter value."""
        nodes = self._mesh.intervals + 1
        dimension = orbit_seed.shape[1]
        times = np.linspace(-self._mesh.half_length, self._mesh.half_length, nodes)
        unknowns = np.concatenate(
            [
                orbit_seed.astype(float).flatten(),
                np.asarray(equilibrium_seed, dtype=float),
                [float(continuation_seed)],
            ],
        )
        solved = self._newton(unknowns, times, nodes, dimension, sweep)
        if solved is None:
            return None
        states = solved[: nodes * dimension].reshape(nodes, dimension)
        equilibrium = solved[nodes * dimension : nodes * dimension + dimension]
        continuation = float(solved[-1])
        return HomoclinicCurvePoint(
            continuation,
            sweep,
            OrbitSolution(times, states),
            equilibrium,
        )

    def trace(
        self,
        sweep_values: list[float],
        orbit_seed: np.ndarray,
        equilibrium_seed: np.ndarray,
        continuation_seed: float,
    ) -> list[HomoclinicCurvePoint]:
        """Trace the curve across ``sweep_values``, seeding each from the last."""
        points: list[HomoclinicCurvePoint] = []
        orbit = orbit_seed.astype(float)
        equilibrium = np.asarray(equilibrium_seed, dtype=float)
        continuation = float(continuation_seed)
        for sweep in sweep_values:
            point = self.solve_point(sweep, orbit, equilibrium, continuation)
            if point is None:
                break
            points.append(point)
            orbit = point.orbit.states
            continuation = point.continuation
            equilibrium = point.equilibrium
        return points
