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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr

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
# Convergence tolerance for the iterative least squares. Tight enough that the
# step matches a dense factorisation to well below the accuracy of the orbit.
_LSMR_TOLERANCE = 1.0e-13
_LSMR_ITERATIONS = 10000
# lsmr's istop value meaning it hit maxiter without converging.
_LSMR_ITERATION_LIMIT = 7
_HALF = 0.5
# How far the truncation estimate extends the interval, as a fraction of the
# half-length. Truncation error decays like ``exp(-2 lambda T)``, so a fraction
# rather than a fixed time keeps the reduction comparable across problems whose
# natural time scales differ by orders of magnitude. A quarter buys roughly a
# decade of reduction on the saddle measured in the docstrings below.
_TRUNCATION_EXTENSION = 0.25
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


class OrbitResolution:
    """Measured error estimates for a connecting orbit, one per resolution parameter.

    A truncated connecting orbit has two independent resolutions and either can be
    the inadequate one: the mesh spacing, which sets the collocation error, and the
    half-length, which sets how much of the orbit's approach to the equilibrium is
    cut off. ``discretisation`` and ``truncation`` are measured separately because a
    single number cannot express both, and reporting only one certifies orbits that
    are badly wrong in the other. Measured on the exact homoclinic
    ``x = 1.5 sech^2(t/2)``:

    * at spacing ``h = 0.25``, extending the half-length from 15 to 20 shifts the
      orbit by 5.2e-13 while its true error is 7.7e-3 - truncation is genuinely
      finished there, and alone it would claim ten orders of magnitude more accuracy
      than the orbit has;
    * at half-length 5, doubling the intervals past 160 shifts the orbit by about
      5e-6 while its true error is stuck at 2.8e-4 - the boundary is the limit, and
      the discretisation estimate alone is 50x optimistic.

    ``estimate`` is therefore the larger of the two, and ``limited_by`` names which
    one it was, since that is what says where to spend the next solve.
    """

    __slots__ = ["discretisation", "truncation"]

    def __init__(self, discretisation: float, truncation: float) -> None:
        self.discretisation = discretisation
        self.truncation = truncation

    @property
    def estimate(self) -> float:
        """The larger of the two components: neither alone bounds the error."""
        return max(self.discretisation, self.truncation)

    @property
    def limited_by(self) -> str:
        """Which resolution parameter to improve first, ``"spacing"`` or ``"length"``.

        ``"spacing"`` means halve the mesh spacing, which divides the discretisation
        error by four. ``"length"`` means extend the half-length; truncation error
        falls like ``exp(-2 lambda T)``, so ``ln(4) / (2 lambda)`` more time buys the
        same factor of four, where ``lambda`` is the leading eigenvalue magnitude.
        """
        if self.truncation > self.discretisation:
            return "length"
        return "spacing"


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
            unknowns = unknowns + self._least_squares(jacobian, residual)
        return OrbitSolution(times, best.reshape(shape))

    def estimate_orbit_error(self, solution: OrbitSolution) -> OrbitResolution:
        """Estimate the orbit's error in both resolution parameters by re-solving.

        A converged solve proves the *discrete* system was satisfied on *this* mesh
        over *this* interval. It says nothing about whether either is adequate, and
        :meth:`solve` cannot tell the caller: it returns its best iterate whatever
        that is, and the least-squares minimum is nonzero by construction, so neither
        a failure signal nor the residual size carries the information.

        Two solves, compared against the orbit in hand:

        * **discretisation** - re-solve on a mesh with twice the intervals over the
          same interval, seeded by interpolation, and take the largest state shift at
          the shared nodes. Second-order convergence makes this ``E - E/4``, so it
          runs about 0.75x the *coarse* error and 3x the *refined* one. Measured on
          the exact homoclinic it was 0.75x at every level from 80 to 640 intervals,
          so unlike the cycle estimator it is a usable bound rather than a screen.
        * **truncation** - re-solve on an interval longer by
          ``_TRUNCATION_EXTENSION``, at the same spacing so the original nodes are a
          subset, seeded by holding the end states, and take the largest shift on the
          overlap. Holding the end states needs no knowledge of the equilibria, so it
          serves homoclinic and heteroclinic orbits alike.

        ``estimate`` is the larger; see :class:`OrbitResolution` for why reporting
        either alone certifies orbits that are badly wrong.

        Neither component needs a failure signal. A refined solve that lands
        somewhere useless shows up as a *large* shift, which is the correct verdict.

        Costs two solves, one at double the intervals and one slightly longer.

        ``intervals`` must be even. The phase condition pins the centre node, and on
        an odd mesh the doubled mesh's centre node sits half a step away, so the two
        orbits come out translated relative to each other and the shift measures that
        translation instead of the error.
        """
        mesh = self._mesh
        nodes = mesh.intervals + 1
        states = solution.states
        if states.shape[0] != nodes:
            message = (
                f"solution has {states.shape[0]} nodes, but this solver has "
                f"{nodes}; pass the solution this solver returned"
            )
            raise ValueError(message)
        if mesh.intervals % 2 != 0:
            message = (
                f"intervals must be even to compare meshes, got {mesh.intervals}; "
                f"the phase condition pins the centre node and an odd mesh has no "
                f"matching centre when doubled"
            )
            raise ValueError(message)
        return OrbitResolution(
            self._discretisation_shift(solution),
            self._truncation_shift(solution),
        )

    def _solve_on(self, mesh: MeshSpec, seed: np.ndarray) -> OrbitSolution:
        """Solve the same problem on a different mesh, leaving this solver unchanged.

        The boundary conditions depend on the two end states and not on how many
        nodes lie between them, so swapping the mesh is enough to re-pose the problem
        at another resolution - no subclass needs to know how to rebuild itself.
        """
        saved = self._mesh
        self._mesh = mesh
        try:
            return self.solve(seed)
        finally:
            self._mesh = saved

    def _discretisation_shift(self, solution: OrbitSolution) -> float:
        """Largest state shift from doubling the intervals over the same interval."""
        mesh = self._mesh
        states = solution.states
        coarse_times = solution.times
        fine_times = np.linspace(
            -mesh.half_length,
            mesh.half_length,
            2 * mesh.intervals + 1,
        )
        seed = np.column_stack(
            [
                np.interp(fine_times, coarse_times, states[:, index])
                for index in range(states.shape[1])
            ],
        )
        refined = self._solve_on(
            MeshSpec(
                mesh.half_length,
                2 * mesh.intervals,
                mesh.phase_index,
                mesh.phase_value,
            ),
            seed,
        )
        return float(np.max(np.abs(refined.states[::2] - states)))

    def _truncation_shift(self, solution: OrbitSolution) -> float:
        """Largest state shift on the overlap from extending the interval.

        The extension is a whole number of mesh steps, so the original nodes are a
        subset of the longer mesh and the overlap needs no interpolation.
        """
        mesh = self._mesh
        states = solution.states
        spacing = 2.0 * mesh.half_length / mesh.intervals
        extra = max(1, round(_TRUNCATION_EXTENSION * mesh.half_length / spacing))
        seed = np.vstack(
            [
                np.repeat(states[:1], extra, axis=0),
                states,
                np.repeat(states[-1:], extra, axis=0),
            ],
        )
        longer = self._solve_on(
            MeshSpec(
                mesh.half_length + extra * spacing,
                mesh.intervals + 2 * extra,
                mesh.phase_index,
                mesh.phase_value,
            ),
            seed,
        )
        overlap = longer.states[extra : extra + states.shape[0]]
        return float(np.max(np.abs(overlap - states)))

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

    def _least_squares(self, jacobian: np.ndarray, residual: np.ndarray) -> np.ndarray:
        """Solve the overdetermined Newton system for a step.

        The collocation system is under 1% nonzero, and a dense singular-value
        factorisation of it is by far the most expensive part of an iteration.
        ``lsmr`` works on the sparse matrix directly and reaches the same step.

        Forming the normal equations instead would be faster still, but squares
        the condition number, and the conditioning here varies by several orders
        of magnitude along the iteration - not a trade worth making inside a
        solver whose callers supply their own fields. The dense factorisation
        remains as a fallback for the case where the iterative solve does not
        return a usable step.
        """
        sparse = csr_matrix(jacobian)
        step, stop, *_ = lsmr(
            sparse,
            -residual,
            atol=_LSMR_TOLERANCE,
            btol=_LSMR_TOLERANCE,
            maxiter=_LSMR_ITERATIONS,
        )
        if stop != _LSMR_ITERATION_LIMIT and np.all(np.isfinite(step)):
            return step
        fallback, *_ = np.linalg.lstsq(jacobian, -residual, rcond=_RCOND)
        return fallback

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
