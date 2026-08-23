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
"""

import numpy as np

from discrecontinual_equations.continuation.connecting_orbit import (
    MeshSpec,
    OrbitSolution,
)
from discrecontinual_equations.differential_equation import DifferentialEquation

_TOLERANCE = 1.0e-10
_ITERATIONS = 80
_STEP = 1.0e-7
_REAL_FLOOR = 1.0e-9


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
        blocks.append(self._field(equilibrium, continuation, sweep))
        jacobian = self._field_jacobian(equilibrium, continuation, sweep)
        values, vectors = np.linalg.eig(jacobian)
        left = np.linalg.inv(vectors)
        stable = [i for i in range(len(values)) if values[i].real < -_REAL_FLOOR]
        unstable = [i for i in range(len(values)) if values[i].real > _REAL_FLOOR]
        departure = states[0] - equilibrium
        arrival = states[-1] - equilibrium
        boundary = [(left[i] @ departure).real for i in stable]
        boundary += [(left[i] @ arrival).real for i in unstable]
        blocks.append(np.array(boundary))
        centre = nodes // 2
        phase = states[centre, self._mesh.phase_index] - self._mesh.phase_value
        blocks.append(np.array([phase]))
        return np.concatenate(blocks)

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
            jacobian = np.empty((residual.size, unknowns.size))
            for j in range(unknowns.size):
                shifted = unknowns.copy()
                shifted[j] += _STEP
                perturbed = self._residual(shifted, times, nodes, dimension, sweep)
                jacobian[:, j] = (perturbed - residual) / _STEP
            step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
            unknowns = unknowns + step
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
