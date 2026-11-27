"""Connecting orbits by a truncated projection boundary-value problem.

A homoclinic orbit leaves an equilibrium along its unstable manifold and returns
along its stable manifold. On a long finite interval it is approximated by a
trajectory whose departure end lies in the unstable eigenspace and whose arrival
end lies in the stable eigenspace - the projection boundary conditions - with a
phase condition to remove the time-translation freedom. The trajectory is
discretised by the trapezoidal rule and the resulting system is solved by
Gauss-Newton. The projection conditions use the left eigenvectors of the Jacobian,
the linear approximation of the manifolds parameterised exactly in ``manifold.py``.
These are Beyn's projection boundary conditions (IMA J. Numer. Anal. 10, 1990), as
HomCont uses them (Champneys, Kuznetsov & Sandstede, IJBC 6, 1996).

The Newton matrix is block-banded - each collocation row couples two neighbouring
nodes - and bordered by the boundary and phase rows, which touch only the two end
nodes and the centre. It is assembled sparse from per-node field Jacobians and
factored with a sparse LU, so an iteration costs time and memory linear in the
number of nodes.

This solves for a single connecting orbit of a fixed field; continuing it in a
parameter (the codim-1 global bifurcation) is the next step and is described in
``ROADMAP.md``. Real (saddle) spectra are supported; spiral connections need the
complex-eigenvector projection and are deferred.
"""

import warnings
from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import bmat, csc_matrix, csr_matrix
from scipy.sparse import identity as sparse_identity
from scipy.sparse.linalg import SuperLU, lsmr, splu

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
# Convergence tolerance for the iterative least squares, the fallback when the
# sparse factorisation fails. Tight enough that the step matches a dense
# factorisation to well below the accuracy of the orbit.
_LSMR_TOLERANCE = 1.0e-13
_LSMR_ITERATIONS = 10000
# lsmr's istop value meaning it hit maxiter without converging.
_LSMR_ITERATION_LIMIT = 7
_HALF = 0.5
# Column ordering for the sparse LU of the Newton system; see ``_factor``.
_ORDERING = "COLAMD"
# Inverse-iteration steps for the smallest singular value of the Newton matrix,
# which sets the augmented system's scale. The scale need only be right to within
# a small factor: one step was measured within 4x and two within 30% of the true
# value, and with either the step error matched QR's from cond(J) = 1e2 to 1e8.
_SCALE_STEPS = 2
_ROOT_HALF = 0.5**0.5
# How far the truncation estimate extends the interval, as a fraction of the
# half-length. Truncation error decays like ``exp(-2 lambda T)``, so a fraction
# rather than a fixed time keeps the reduction comparable across problems whose
# natural time scales differ by orders of magnitude. A quarter buys roughly a
# decade of reduction on the saddle measured in the docstrings below.
_TRUNCATION_EXTENSION = 0.25
# Multiple of an interval's truncation error its residual may reach; the exact
# orbit, sampled, reaches 1.24 of it (see CollocationResidual).
_RESIDUAL_SAFETY = 2.0
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


class CollocationResidual:
    """How far a path is from satisfying the trapezoidal rule, and how far it may be.

    ``largest`` is the largest component of ``x_{i+1} - x_i - (h/2)(f_i + f_{i+1})``
    over every interval. A converged solve of a consistent discrete system drives
    it to the Newton tolerance; one that has no solution - because the orbit it
    asks for does not exist - stops at a least-squares minimum that is not.

    Each interval is held to its own allowance: ``_RESIDUAL_SAFETY`` times the
    trapezoidal rule's local truncation error ``h^3 |x'''| / 12``, read off the
    largest third difference of the path among the (up to three) stencils that
    cover the interval, plus the Newton tolerance as a floor - a residual the
    solver itself calls converged cannot refute a path. ``worst`` is the largest
    ratio of an interval's residual to its allowance, and the path is a
    trajectory when it is at most 1. ``tolerance`` is the largest allowance, for
    scale.

    Per interval, because the truncation error varies by orders of magnitude
    along a connection: near a saddle ``x'''`` is 50 to 250 times smaller than
    mid-transition, and a test against the largest third difference anywhere
    passes a defect there as large as the whole orbit's error estimate. With a
    safety factor, because the exact orbit sampled on the mesh leaves a residual
    1.005 (``h = 0.2``) to 1.24 (``h = 1.6``) times its largest third difference
    over 12, measured on the Guckenheimer-Holmes connection; per interval it
    stays within 0.62 of the allowance. As in COLSYS and bvp4c, the error is
    controlled where it is made.

    Measured on that connection at ``h = 0.2``: genuine orbits leave 1e-15 to
    3e-10, a path with no orbit behind it 1.9e-3, against a largest allowance of
    6e-5. A leak ``eps x1^2 x2^2`` out of the plane is refused from
    ``eps = 1e-3`` (3e-4 at ``h = 0.1``): the threshold falls as ``h^2``.

    The resolution estimate cannot stand in for this. It measures how much a path
    moves when the mesh changes, and a non-trajectory converged on one mesh can
    sit still on the next: the path above had an estimate of 1.9e-4.
    """

    __slots__ = ["largest", "tolerance", "worst"]

    def __init__(self, largest: float, tolerance: float, worst: float) -> None:
        self.largest = largest
        self.tolerance = tolerance
        self.worst = worst

    @property
    def is_trajectory(self) -> bool:
        """Whether every interval's residual is within its allowance."""
        return self.worst <= 1.0


class ConnectingOrbit(ABC):
    """Solve for an orbit connecting an equilibrium to itself or another.

    Subclasses supply the projection boundary conditions via :meth:`_boundary`; the
    trapezoidal collocation, phase condition, and Gauss-Newton solve are shared.
    """

    __slots__ = ["_function", "_mesh"]
    _function: Function
    _mesh: MeshSpec

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
                jacobian = self._numerical_jacobian(
                    unknowns,
                    times,
                    shape,
                    residual.size,
                )
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

    def collocation_residual(self, solution: OrbitSolution) -> CollocationResidual:
        """The path's largest trapezoidal residual, against its truncation error.

        Only the collocation rows are measured: the boundary and phase rows say
        where the path starts, ends and is pinned, not whether it is a trajectory.
        Any path with the solver's dimension can be measured, not only one this
        solver returned. See :class:`CollocationResidual`.
        """
        states, times = solution.states, solution.times
        rows = [self._collocation(states, times, i) for i in range(len(times) - 1)]
        residuals = np.max(np.abs(np.array(rows)), axis=1)
        third = np.max(np.abs(np.diff(states, 3, axis=0)), axis=1, initial=0.0)
        covering = np.array(
            [
                np.max(third[max(0, i - 2) : i + 1], initial=0.0)
                for i in range(len(residuals))
            ],
        )
        allowance = _RESIDUAL_SAFETY * covering / 12.0 + _DEFAULT_TOLERANCE
        return CollocationResidual(
            float(np.max(residuals)),
            float(np.max(allowance)),
            float(np.max(residuals / allowance)),
        )

    def _shift(self, moved: np.ndarray, original: np.ndarray) -> float:
        """Largest component of the change between two paths at shared nodes.

        A solver posed in other coordinates overrides this to measure the change
        in the coordinates its caller reads the orbit in.
        """
        return float(np.max(np.abs(moved - original)))

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
        return self._shift(refined.states[::2], states)

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
        return self._shift(overlap, states)

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

    def _least_squares(self, jacobian: csr_matrix, residual: np.ndarray) -> np.ndarray:
        """Solve the overdetermined Newton system for a step, by a sparse direct solve.

        The least-squares step ``x`` minimising ``|J x + r|`` is the lower half of the
        solution of the augmented system

            [ I    J ] [ s ]   [ -r ]
            [ J^T  0 ] [ x ] = [  0 ],

        whose first row defines the residual ``s = -r - J x`` and whose second is
        the normal equations ``J^T s = 0``, without forming ``J^T J``. The identity
        block is scaled, ``alpha I`` with ``alpha = sigma_min(J) / sqrt(2)``, and
        that scaling is what keeps the condition number near ``cond(J)``: unscaled,
        with ``sigma_min < 1`` as it is here (about ``1/N``), the augmented matrix is
        conditioned like ``J^T J``. Measured on synthetic problems, unscaled forward
        errors were 4e-7 at ``cond(J) = 1e6`` and 1e-3 at ``1e8``, against QR's 2e-9
        and 3e-5; scaled, they match QR's (Bjorck, BIT 7, 1967). The scale comes
        from ``_SCALE_STEPS`` steps of inverse iteration through an unscaled
        factorisation, so a step costs two sparse factorisations, both in fill
        linear in the nodes because only the boundary and phase rows reach across
        the block-banded collocation.

        This replaces ``lsmr``, whose iteration count grew with the nodes. It
        remains the fallback for a factorisation that is singular or non-finite,
        with the dense least-squares solve behind it; both announce themselves with
        a ``RuntimeWarning``.
        """
        rows, columns = jacobian.shape
        right = np.concatenate([-residual, np.zeros(columns)])
        reason = "non-finite solution (near-singular factorisation)"
        try:
            scale = _ROOT_HALF * _smallest_singular_value(jacobian)
            solution = _factor(_augmented(jacobian, scale)).solve(right)
        except RuntimeError:
            reason = "exactly singular factorisation"
            solution = np.full(rows + columns, np.nan)
        step = solution[rows:]
        if np.all(np.isfinite(step)):
            return step
        warnings.warn(
            f"connecting-orbit Newton step: {reason}; falling back to lsmr",
            RuntimeWarning,
            stacklevel=2,
        )
        step, stop, *_ = lsmr(
            jacobian,
            -residual,
            atol=_LSMR_TOLERANCE,
            btol=_LSMR_TOLERANCE,
            maxiter=_LSMR_ITERATIONS,
        )
        if stop != _LSMR_ITERATION_LIMIT and np.all(np.isfinite(step)):
            return step
        warnings.warn(
            "connecting-orbit Newton step: lsmr did not converge after a singular "
            "factorisation; falling back to a dense least-squares solve",
            RuntimeWarning,
            stacklevel=2,
        )
        fallback, *_ = np.linalg.lstsq(jacobian.toarray(), -residual, rcond=_RCOND)
        return fallback

    def _jacobian(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        shape: tuple[int, int],
        residual: np.ndarray,
        analytic: bool,  # noqa: FBT001
    ) -> csr_matrix | None:
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
    ) -> csr_matrix:
        """Assemble the exact Jacobian block by block, in O(N) field Jacobians.

        Trapezoidal collocation couples only neighbouring nodes, so the collocation
        rows are banded: block ``i`` holds ``-I - (h/2) Df(x_i)`` against node ``i``
        and ``I - (h/2) Df(x_{i+1})`` against node ``i+1``. Each node's field
        Jacobian is needed by the two intervals that meet there, so it is evaluated
        once per node rather than once per interval. The boundary and phase rows are
        affine in the states they touch, so their exact rows come from unit
        displacements.

        The blocks go straight into a sparse matrix: nothing of size ``N^2`` is
        ever allocated.
        """
        states = unknowns.reshape(shape)
        derivatives = np.array(
            [_AUTODIFF.jacobian(self._function, state, 0.0) for state in states],
        )
        return self._assemble(unknowns, times, shape, rows, derivatives)

    def _numerical_jacobian(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        shape: tuple[int, int],
        rows: int,
    ) -> csr_matrix:
        """The same sparse assembly, with each node's field Jacobian differenced.

        The collocation rows see the field only through the node Jacobians, so
        differencing the field at each node - ``dimension`` extra evaluations per
        node - fills the same banded pattern as automatic differentiation, in O(N)
        work rather than one full residual per unknown.
        """
        states = unknowns.reshape(shape)
        derivatives = np.array([self._differenced(state) for state in states])
        return self._assemble(unknowns, times, shape, rows, derivatives)

    def _differenced(self, state: np.ndarray) -> np.ndarray:
        """Forward-difference Jacobian of the field at one node."""
        base = np.array(self._function.eval(point=list(state), time=None))
        columns = np.empty((state.size, state.size))
        for j in range(state.size):
            shifted = state.copy()
            shifted[j] += _STEP
            columns[:, j] = (
                np.array(self._function.eval(point=list(shifted), time=None)) - base
            ) / _STEP
        return columns

    def _assemble(
        self,
        unknowns: np.ndarray,
        times: np.ndarray,
        shape: tuple[int, int],
        rows: int,
        derivatives: np.ndarray,
    ) -> csr_matrix:
        """Collocation blocks from the node Jacobians, bordered by the affine rows."""
        nodes, dimension = shape
        row, column, value = _collocation_triplets(derivatives, times)
        border = self._affine_triplets(unknowns, shape, (nodes - 1) * dimension)
        return csr_matrix(
            (
                np.concatenate([value, border[2]]),
                (np.concatenate([row, border[0]]), np.concatenate([column, border[1]])),
            ),
            shape=(rows, unknowns.size),
        )

    def _affine_triplets(
        self,
        unknowns: np.ndarray,
        shape: tuple[int, int],
        offset: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The boundary and phase rows, which are affine in the states, as triplets."""
        nodes, dimension = shape
        states = unknowns.reshape(shape)
        base = self._boundary(states)
        rows: list[np.ndarray] = []
        columns: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for node in (0, nodes - 1):
            for component in range(dimension):
                shifted = states.copy()
                shifted[node, component] += 1.0
                rows.append(offset + np.arange(base.size))
                columns.append(np.full(base.size, node * dimension + component))
                values.append(self._boundary(shifted) - base)
        centre = nodes // 2
        rows.append(np.array([offset + base.size]))
        columns.append(np.array([centre * dimension + self._mesh.phase_index]))
        values.append(np.array([1.0]))
        return np.concatenate(rows), np.concatenate(columns), np.concatenate(values)

    @abstractmethod
    def _boundary(self, states: np.ndarray) -> np.ndarray:
        """Projection boundary conditions at the two ends of the orbit."""
        raise NotImplementedError


def _collocation_triplets(
    derivatives: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse triplets of the trapezoidal collocation rows, from node Jacobians.

    ``derivatives[i]`` is the field Jacobian at node ``i``. Interval ``i`` owns rows
    ``i*n .. i*n+n-1`` and holds ``-I - (h_i/2) Df(x_i)`` against node ``i`` and
    ``I - (h_i/2) Df(x_{i+1})`` against node ``i+1``: two ``n x n`` blocks per
    interval, ``2 N n^2`` nonzeros in all.
    """
    nodes, dimension, _ = derivatives.shape
    weights = _HALF * np.diff(times)[:, None, None]
    identity = np.eye(dimension)
    here = -identity - weights * derivatives[:-1]
    ahead = identity - weights * derivatives[1:]
    first = np.arange(nodes - 1)[:, None, None] * dimension
    rows = np.broadcast_to(first + np.arange(dimension)[None, :, None], here.shape)
    columns = np.broadcast_to(first + np.arange(dimension)[None, None, :], here.shape)
    return (
        np.concatenate([rows.ravel(), rows.ravel()]),
        np.concatenate([columns.ravel(), (columns + dimension).ravel()]),
        np.concatenate([here.ravel(), ahead.ravel()]),
    )


def _augmented(jacobian: csr_matrix, scale: float = 1.0) -> csc_matrix:
    """The augmented matrix ``[[scale I, J], [J^T, 0]]`` of the least-squares problem.

    Its lower solution block is the least-squares solution for any positive
    ``scale``; the scale changes only the conditioning.
    """
    rows = jacobian.shape[0]
    return bmat(
        [[scale * sparse_identity(rows), jacobian], [jacobian.T, None]],
        format="csc",
    )


def _smallest_singular_value(jacobian: csr_matrix) -> float:
    """Estimate ``sigma_min(J)`` by inverse iteration on ``J^T J``, never forming it.

    With the unscaled augmented matrix, the right-hand side ``[0, v]`` gives
    ``x = -(J^T J)^{-1} v``, so ``|x|`` grows like ``1/sigma_min^2``. The solves
    lose accuracy as ``cond(J)^2``, but the error lies along the smallest singular
    direction - the one being sought - so the estimate stays good. Raises
    ``RuntimeError`` if the matrix is exactly singular, as the step solve would.
    """
    rows, columns = jacobian.shape
    factor = _factor(_augmented(jacobian))
    vector = 1.0 + 0.5 * np.cos(np.arange(columns))
    vector /= np.linalg.norm(vector)
    growth = 1.0
    for _ in range(_SCALE_STEPS):
        image = factor.solve(np.concatenate([np.zeros(rows), vector]))[rows:]
        growth = float(np.linalg.norm(image))
        if not np.isfinite(growth) or growth == 0.0:
            message = "inverse iteration produced no finite direction"
            raise RuntimeError(message)
        vector = image / growth
    return growth**-0.5


def _factor(matrix: csc_matrix) -> SuperLU:
    """Sparse LU with a column ordering that keeps the bordered band's fill linear.

    The ordering matters more than anything else here. On the augmented saddle
    system ``COLAMD`` holds ``nnz(L + U) / nnz(A)`` at 2.6 from N = 200 to 3200;
    a minimum-degree ordering of ``A^T + A`` lets it grow with N, to 322 at 3200.
    """
    return splu(matrix, permc_spec=_ORDERING)


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
