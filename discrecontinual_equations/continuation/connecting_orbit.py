"""Connecting orbits by a truncated projection boundary-value problem.

A homoclinic orbit leaves an equilibrium along its unstable manifold and returns
along its stable manifold. On a long finite interval it is approximated by a
trajectory whose departure end lies in the unstable eigenspace and whose arrival
end lies in the stable eigenspace - the projection boundary conditions - with a
phase condition to remove the time-translation freedom. The trajectory is
discretised by the trapezoidal rule and the resulting system is solved by
Gauss-Newton. The projection conditions use the left eigenvectors of the Jacobian,
the linear approximation of the manifolds parameterised exactly in ``manifold.py``.

This solves for a single connecting orbit of a fixed field; continuing it in a
parameter (the codim-1 global bifurcation) is the next step and is described in
``ROADMAP.md``. Real (saddle) spectra are supported; spiral connections need the
complex-eigenvector projection and are deferred.
"""

from abc import ABC, abstractmethod

import numpy as np

from discrecontinual_equations.continuation.derivative_provider import (
    AutomaticDifferentiation,
)
from discrecontinual_equations.function.function import Function

_IMAGINARY = 1.0e-9
_DEFAULT_TOLERANCE = 1.0e-9
_DEFAULT_ITERATIONS = 60
_STEP = 1.0e-7
# The finite-difference Jacobian carries noise of order ``_STEP``; singular
# directions below this cutoff are that noise, and solving along them turns a
# converged iterate into a divergent one. ``lstsq`` truncates them instead.
_RCOND = 1.0e-8
# The collocation system is overdetermined by the phase condition, so its
# least-squares minimum is nonzero and ``_DEFAULT_TOLERANCE`` is unreachable.
# Stop once repeated iterations stop improving on the best residual seen.
_STAGNATION_LIMIT = 3
_HALF = 0.5
_AUTODIFF = AutomaticDifferentiation()


class MeshSpec:
    """The finite-interval mesh and phase condition for a connecting orbit."""

    __slots__ = ["half_length", "intervals", "phase_index", "phase_value"]

    def __init__(
        self,
        half_length: float,
        intervals: int,
        phase_index: int = 0,
        phase_value: float = 0.0,
    ) -> None:
        self.half_length = half_length
        self.intervals = intervals
        self.phase_index = phase_index
        self.phase_value = phase_value


class OrbitSolution:
    """A computed connecting orbit: mesh times and the state at each node."""

    __slots__ = ["_states", "_times"]

    def __init__(self, times: np.ndarray, states: np.ndarray) -> None:
        self._times = times
        self._states = states

    @property
    def times(self) -> np.ndarray:
        """Mesh times from ``-T`` to ``+T``."""
        return self._times

    @property
    def states(self) -> np.ndarray:
        """State at each mesh node, shape ``(nodes, dimension)``."""
        return self._states

    def component(self, index: int) -> np.ndarray:
        """One state component along the orbit."""
        return self._states[:, index]


class ConnectingOrbit(ABC):
    """Solve for an orbit connecting an equilibrium to itself or another.

    Subclasses supply the projection boundary conditions via :meth:`_boundary`; the
    trapezoidal collocation, phase condition, and Gauss-Newton solve are shared.
    """

    __slots__ = ["_function", "_mesh"]

    def solve(self, seed: np.ndarray) -> OrbitSolution:
        """Refine ``seed`` to the connecting orbit by Gauss-Newton iteration.

        The iterate with the smallest residual is the one returned, not the last
        one computed: past the least-squares minimum the finite-difference Jacobian
        is dominated by noise and a further step degrades an already-converged
        orbit. Iteration stops early once ``_STAGNATION_LIMIT`` successive steps
        fail to improve on the best residual seen.
        """
        nodes = self._mesh.intervals + 1
        times = np.linspace(-self._mesh.half_length, self._mesh.half_length, nodes)
        unknowns = seed.astype(float).flatten()
        shape = seed.shape
        best = unknowns.copy()
        best_residual = float("inf")
        stagnant = 0
        analytic = True
        for _ in range(_DEFAULT_ITERATIONS):
            residual = self._residual(unknowns, times, shape)
            norm = float(np.linalg.norm(residual))
            if norm < best_residual:
                best, best_residual = unknowns.copy(), norm
                stagnant = 0
            else:
                stagnant += 1
                if stagnant >= _STAGNATION_LIMIT:
                    break
            if norm < _DEFAULT_TOLERANCE:
                break
            jacobian = self._jacobian(unknowns, times, shape, residual, analytic)
            if jacobian is None:
                analytic = False
                jacobian = self._numerical_jacobian(unknowns, times, shape, residual)
            step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=_RCOND)
            unknowns = unknowns + step
        return OrbitSolution(times, best.reshape(shape))

    def _residual(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        shape: tuple[int, int],
    ) -> np.ndarray:
        states = unknowns.reshape(shape)
        nodes = shape[0]
        blocks = [self._collocation(states, times, i) for i in range(nodes - 1)]
        blocks.append(self._boundary(states))
        blocks.append(np.array([self._phase(states)]))
        return np.concatenate(blocks)

    def _collocation(
        self,
        states: np.ndarray,
        times: np.ndarray,
        i: int,
    ) -> np.ndarray:
        step = times[i + 1] - times[i]
        here = np.array(self._function.eval(point=list(states[i]), time=None))
        ahead = np.array(self._function.eval(point=list(states[i + 1]), time=None))
        return states[i + 1] - states[i] - 0.5 * step * (here + ahead)

    def _phase(self, states: np.ndarray) -> float:
        centre = states.shape[0] // 2
        return states[centre, self._mesh.phase_index] - self._mesh.phase_value

    def _jacobian(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        shape: tuple[int, int],
        residual: np.ndarray,
        analytic: bool,  # noqa: FBT001
    ) -> np.ndarray | None:
        """The exact Jacobian, or ``None`` when the field cannot be differentiated.

        Automatic differentiation evaluates the field on Taylor jets, which a
        transcendental or otherwise non-analytic field may reject. That is not an
        error: the caller falls back to finite differences for the rest of the solve.
        """
        if not analytic:
            return None
        try:
            return self._analytic_jacobian(unknowns, times, shape, residual.size)
        except (TypeError, ValueError, AttributeError):
            return None

    def _analytic_jacobian(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        shape: tuple[int, int],
        rows: int,
    ) -> np.ndarray:
        """Assemble the exact Jacobian block by block, in O(N) field Jacobians.

        Trapezoidal collocation couples only neighbouring nodes, so the collocation
        rows are banded: block ``i`` holds ``-I - (h/2) Df(x_i)`` against node ``i``
        and ``I - (h/2) Df(x_{i+1})`` against node ``i+1``. Each node's field
        Jacobian is needed by the two intervals that meet there, so it is evaluated
        once per node rather than once per interval. The boundary and phase rows are
        affine in the states they touch, so their exact rows come from unit
        displacements.

        This replaces a dense finite-difference construction costing one full
        residual per unknown, and removes the finite-difference noise that made the
        smallest singular directions meaningless.
        """
        nodes, dimension = shape
        states = unknowns.reshape(shape)
        jacobian = np.zeros((rows, unknowns.size))
        identity = np.eye(dimension)
        derivatives = [
            _AUTODIFF.jacobian(self._function, states[i], 0.0) for i in range(nodes)
        ]
        for i in range(nodes - 1):
            weight = _HALF * float(times[i + 1] - times[i])
            row = i * dimension
            here = slice(i * dimension, (i + 1) * dimension)
            ahead = slice((i + 1) * dimension, (i + 2) * dimension)
            jacobian[row : row + dimension, here] = -identity - weight * derivatives[i]
            jacobian[row : row + dimension, ahead] = (
                identity - weight * derivatives[i + 1]
            )
        self._affine_rows(unknowns, shape, jacobian, (nodes - 1) * dimension)
        return jacobian

    def _affine_rows(
        self,
        unknowns: np.ndarray,
        shape: tuple[int, int],
        jacobian: np.ndarray,
        offset: int,
    ) -> None:
        """Fill the boundary and phase rows, which are affine in the states."""
        nodes, dimension = shape
        states = unknowns.reshape(shape)
        base = self._boundary(states)
        touched = (0, nodes - 1)
        for node in touched:
            for component in range(dimension):
                shifted = states.copy()
                shifted[node, component] += 1.0
                column = node * dimension + component
                jacobian[offset : offset + base.size, column] = (
                    self._boundary(shifted) - base
                )
        centre = nodes // 2
        phase_column = centre * dimension + self._mesh.phase_index
        jacobian[offset + base.size, phase_column] = 1.0

    def _numerical_jacobian(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        shape: tuple[int, int],
        residual: np.ndarray,
    ) -> np.ndarray:
        columns = np.empty((residual.size, unknowns.size))
        for j in range(unknowns.size):
            shifted = unknowns.copy()
            shifted[j] += _STEP
            columns[:, j] = (self._residual(shifted, times, shape) - residual) / _STEP
        return columns

    @abstractmethod
    def _boundary(self, states: np.ndarray) -> np.ndarray:
        """Projection boundary conditions at the two ends of the orbit."""
        raise NotImplementedError


def _split_eigenspaces(
    jacobian: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Eigenvalues, left eigenvectors, and stable/unstable indices of a Jacobian."""
    values, vectors = np.linalg.eig(jacobian)
    left = np.linalg.inv(vectors)
    stable = [i for i in range(len(values)) if values[i].real < -_IMAGINARY]
    unstable = [i for i in range(len(values)) if values[i].real > _IMAGINARY]
    return values, left, stable, unstable


def _projection_conditions(
    left: np.ndarray,
    values: np.ndarray,
    indices: list[int],
    displacement: np.ndarray,
) -> list[float]:
    """Real scalar conditions forcing ``displacement`` out of an eigenspace.

    A real eigenvalue contributes one condition (the real projection); a complex
    conjugate pair contributes two (real and imaginary parts of one representative),
    so a spiral eigenspace is fully constrained rather than collapsed to a line.
    """
    conditions: list[float] = []
    for index in indices:
        if values[index].imag < -_IMAGINARY:
            continue  # skip the negative-frequency partner of a conjugate pair
        projection = left[index] @ displacement
        conditions.append(projection.real)
        if values[index].imag > _IMAGINARY:
            conditions.append(projection.imag)
    return conditions


class HomoclinicOrbit(ConnectingOrbit):
    """A homoclinic orbit found by a truncated projection boundary-value problem."""

    __slots__ = ["_equilibrium", "_left", "_stable", "_unstable", "_values"]

    def __init__(
        self,
        function: Function,
        equilibrium: np.ndarray,
        jacobian: np.ndarray,
        mesh: MeshSpec,
    ) -> None:
        self._function = function
        self._equilibrium = equilibrium
        self._mesh = mesh
        self._values, self._left, self._stable, self._unstable = _split_eigenspaces(
            jacobian,
        )

    def _boundary(self, states: np.ndarray) -> np.ndarray:
        departure = states[0] - self._equilibrium
        arrival = states[-1] - self._equilibrium
        left = _projection_conditions(self._left, self._values, self._stable, departure)
        right = _projection_conditions(
            self._left,
            self._values,
            self._unstable,
            arrival,
        )
        return np.array(left + right)


class Terminus:
    """An endpoint saddle of a heteroclinic orbit: its position and Jacobian."""

    __slots__ = ["equilibrium", "jacobian"]

    def __init__(self, equilibrium: np.ndarray, jacobian: np.ndarray) -> None:
        self.equilibrium = equilibrium
        self.jacobian = jacobian


class HeteroclinicOrbit(ConnectingOrbit):
    """An orbit connecting one saddle to a different saddle.

    It leaves the ``source`` along that saddle's unstable manifold and arrives at the
    ``target`` along its stable manifold. The departure end is projected onto the
    source's stable left eigenvectors (forcing it into the unstable eigenspace) and
    the arrival end onto the target's unstable left eigenvectors.
    """

    __slots__ = [
        "_source",
        "_source_left",
        "_source_stable",
        "_source_values",
        "_target",
        "_target_left",
        "_target_unstable",
        "_target_values",
    ]

    def __init__(
        self,
        function: Function,
        source: Terminus,
        target: Terminus,
        mesh: MeshSpec,
    ) -> None:
        self._function = function
        self._mesh = mesh
        self._source = source.equilibrium
        self._target = target.equilibrium
        self._source_values, self._source_left, self._source_stable, _ = (
            _split_eigenspaces(source.jacobian)
        )
        self._target_values, self._target_left, _, self._target_unstable = (
            _split_eigenspaces(target.jacobian)
        )

    def _boundary(self, states: np.ndarray) -> np.ndarray:
        departure = states[0] - self._source
        arrival = states[-1] - self._target
        left = _projection_conditions(
            self._source_left,
            self._source_values,
            self._source_stable,
            departure,
        )
        right = _projection_conditions(
            self._target_left,
            self._target_values,
            self._target_unstable,
            arrival,
        )
        return np.array(left + right)
