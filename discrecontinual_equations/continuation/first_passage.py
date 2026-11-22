"""Mean first-passage and residence times, from the grid and from an ensemble.

A residence-time sensor is read through the time its state spends in each well of a
bistable potential before the noise carries it over the barrier: a target field tilts
the wells, the two mean residence times part, and their difference is the
measurement. The density of :mod:`.fokker_planck` says where the mass sits and the
quasipotential of :mod:`.quasipotential` how hard each crossing is on the
logarithmic scale; neither is a time. This module supplies the time, by two routes
that share no code and are checked against each other.

*The grid.* For ``dx = f dt + G dW`` with ``D = G G^T`` (Ito; a Stratonovich drift
is shifted to its Ito form first, as the density solve does), the mean time
``tau(x)`` for a path started at ``x`` to reach a target set ``T`` solves the
backward Kolmogorov equation

    ``L tau = -1`` off ``T``,  ``tau = 0`` on ``T``,
    ``L = f . grad + (1/2) D : grad grad``,

with a reflecting (zero normal derivative) condition on the walls of the box. ``L``
is the adjoint of the forward operator whose null vector
:class:`~.fokker_planck.StationaryFokkerPlanck` solves for, and it is built here
*as* the adjoint: the same exponentially fitted finite-volume fluxes are assembled,
and the generator is minus the transpose of the divergence they make. Two things
follow without further argument. The reflecting wall of the density solve - no flux
through the box faces - becomes the Neumann wall of the backward problem; and with
a diagonal ``D`` every off-diagonal entry of the generator is a non-negative rate
and every row sums to zero, so the discrete problem is *exactly* the mean
first-passage problem of a continuous-time Markov chain on the cells: positive, and
monotone in the target, at any noise level. Exponential fitting is what keeps it
accurate at small noise, where the time is exponentially large and is set in the
thin layer around the saddle that a central difference would smear.

*The ensemble.* :class:`EnsembleFirstPassage` steps paths with
:class:`~...solver.stochastic.ensemble.StochasticEnsemble` until each has entered
the target, and :class:`EnsembleResidence` runs paths for a long time and cuts them
into residences: a residence in a well starts when the path enters that well's set
and ends when it enters the other's, so the two sets act as a hysteretic readout
and the barrier region's dithering is not counted as switching. By the strong Markov
property the residences of one path are independent, and the mean residence in a
well is the mean first-passage time from the entry point to the other set - the
grid quantity, evaluated where the path came in. Each estimate comes with the
standard error of its mean and a normal confidence interval: the samples are close
to exponential, many hundreds are used, and the interval is the one the comparison
with the grid is made against.

Euler-Maruyama checks the target only at the step times, so a path that enters and
leaves within one step is missed, and the scheme's own weak error is first order;
the time comes out long by an amount proportional to the step. On the planar well
of the tests at ``sigma^2 = 1/2`` (grid time 11.41), 8000 paths give 11.65, 11.55
and 11.47 (standard error 0.11) at ``dt = 0.01``, ``0.005`` and ``0.0025``, and the
linear extrapolation to ``dt = 0`` is 11.39. The tests use ``dt = 0.0025``, where
the bias is below the Monte-Carlo error they assert against.

*Closed forms.* :func:`eyring_kramers_time` evaluates the small-noise asymptotics,
in the library's normalisation (``dx = f dt + G dW``, no ``eps``, quasipotential
``U`` with the density ``~ exp(-U)``; see :mod:`.quasipotential`). They rest on
three assumptions that the grid does not need, and the names say which:

* *small noise*: the formulas are leading-order Laplace asymptotics. Measured
  here the relative error is ``O(1 / Delta U)``: against the exact double
  integral below Kramers' time is short by 12.5%, 5.4% and 2.6% at
  ``Delta U = 4, 8, 16``, and on the planar wells the grid / closed-form excess
  halves with each halving of the variance. Berglund (arXiv 1106.5799) states
  ``1 + O(eps^{1/2})`` in one dimension (his Eq. (3.15)) and, for gradient
  systems in any dimension, the bound ``O(eps^{1/2} |log eps|^{3/2})`` of Bovier,
  Eckhoff, Gayrard and Klein (J. Eur. Math. Soc. 6 (2004); his Eq. (3.35));
* *overdamped*: the state is the position of a first-order (Smoluchowski)
  equation - no inertia, no velocity variable - and ``D`` is non-singular; and
* *a single non-degenerate saddle between two point attractors*, with the
  quasipotential smooth along the most probable path. Where the basins are
  separated by a saddle *cycle* instead - the three-cell ring's rotating wave is
  one - there is no such formula: the prefactor oscillates with ``log eps``
  (Day's cycling; Berglund, Section 4.3) and only the grid and the ensemble
  give the time.

In one dimension, for ``dx = -V'(x) dt + sigma dW`` started at ``x`` with a
reflecting wall at ``a`` and the target ``[b, inf)``, the exact time is the double
integral

    ``tau(x) = (2 / sigma^2) int_x^b dy int_a^y exp(2 (V(y) - V(z)) / sigma^2) dz``

(Berglund's Eq. (3.14) with ``eps = sigma^2 / 2``, the reflection of its
target), and Laplace's method on it gives Kramers' time, Berglund's Eq. (1.8),
``tau ~ (2 pi / sqrt(V''(m) |V''(s)|)) exp(2 Delta V / sigma^2)`` - the
overdamped limit of Kramers (1940) as reviewed by Hanggi, Talkner and Borkovec
(Rev. Mod. Phys. 62, 251 (1990)); the formula here was checked against
Berglund's statement and the exact integral, not against those two. The ticket wrote
the exponent as ``Delta V / eps`` and the integral with ``2 / eps``: those are
different noise conventions (``sqrt(2 eps)`` and ``sqrt(eps)`` respectively); with
the library's ``sigma`` both read ``2 Delta V / sigma^2 = Delta U``, which is what
:func:`eyring_kramers_time` takes.

*Measured.* The tests repeat these.

  - The one-dimensional double well against the exact integral, relative error
    at 101, 201, 401 and 801 cells: 2.12e-3, 5.31e-4, 1.33e-4, 3.33e-5 at
    ``sigma^2 = 1/16`` and 2.8e-4 ... 4.4e-6 at ``1/2`` - order 2.00 at every
    noise level, with an error that grows only linearly as the noise falls.
  - Planar gradient well (transverse curvature 2 at the minima, 1 at the saddle),
    grid / Eyring-Kramers at 161 cells: 1.361, 1.215, 1.092, 1.042 at
    ``sigma^2 = 1/4 ... 1/32``; 161 and 321 cells agree to 1e-3.
  - Non-Gibbs transverse field (Bouchet-Reygner ``integral F dt = +-1/2``), grid /
    closed form at 281 cells, ``sigma^2 = 1/8, 1/16, 1/32``: 1.162, 1.073, 1.033
    from the left well and 1.290, 1.120, 1.054 from the right. Without the
    non-Gibbs factor the closed form is out by ``e^(+-1/2)``.
  - Ensemble against grid, within three standard errors: first passages
    (2000 paths) and residences (300 paths, both wells, two tilts).

*Limits.* Four, each measured.

  - *Oblique drift.* Along an axis the fitted flux is exact for a locally
    constant drift; across the cells at an angle it adds a crosswind diffusion of
    order ``|f| h``, first order, and significant once it is comparable to
    ``sigma^2``. With a divergence-free rotation at ``sigma^2 = 1/32`` the grid /
    closed-form ratio is 0.816, 0.928, 0.983 at 161, 241, 361 cells: the time
    comes out short (the grid is hotter
    than the system), and the remedy is resolution, ``h |f| << sigma^2``.
  - *Round-off.* The restricted generator's condition number is ``|L| max tau``
    and grows with the time itself; :attr:`GridFirstPassage.round_off` reports
    the resulting bound, 1.3e-5 on the planar well at ``Delta U = 16``
    (``tau ~ 3e7``) and 4% at ``Delta U = 24``; beyond 1% the solve refuses.
  - *Entry states.* A mean residence is ``tau`` averaged over where the
    residences began, not ``tau`` at the attractor. At small noise the two agree
    (``tau`` is flat in the well to exponentially small terms); at the moderate
    noise a sensor runs at they do not - 3% apart on the planar well at
    ``sigma^2 = 1/2`` - and :class:`ResidenceTracker` keeps the entry states so
    the right one can be compared.
  - *Censoring.* Residences still open when a run stops are the long ones, and
    dropping them biases the mean short by a fraction of order
    ``tau / horizon``; :meth:`ResidenceTracker.stop_opening` lets every residence
    that began before the horizon run to its end instead (see there for the
    sizes on the ring).
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, bicgstab, spilu, splu

from discrecontinual_equations.continuation.fokker_planck import (
    DensityGrid,
    StationaryFokkerPlanck,
    _check_elliptic,
    _Discretisation,
    _face_cells,
)
from discrecontinual_equations.continuation.noise import NoiseMatrix
from discrecontinual_equations.continuation.stochastic import NoiseConvention
from discrecontinual_equations.differential_equation import DifferentialEquation
from discrecontinual_equations.function.function import Function
from discrecontinual_equations.solver.solver_config import StochasticConfig
from discrecontinual_equations.solver.stochastic.ensemble import StochasticEnsemble

_HALF = 0.5
# Two-sided 95% normal quantile, the default interval of every ensemble estimate.
_NORMAL_95 = 1.959963984540054
_DEFAULT_CHUNK_STEPS = 400
# After the horizon, how much longer (in horizons) open residences may run on.
_DRAIN_FACTOR = 10.0
_ORDERING = "MMD_AT_PLUS_A"
_ITERATIVE_DIMENSION = 3
_ILU_DROP = 1.0e-4
_ILU_FILL = 20.0
_ITERATIVE_TOLERANCE = 1.0e-10
_ITERATIVE_STEPS = 2000
_EPSILON = float(np.finfo(float).eps)
_ROUND_OFF_LIMIT = 1.0e-2
_UNVISITED, _FIRST, _SECOND, _CLOSED = -1, 0, 1, 2


class _ForwardOperator(StationaryFokkerPlanck):
    """The divergence operator of the density solve, assembled but not solved.

    The stencil is :class:`~.fokker_planck.StationaryFokkerPlanck`'s, reached through
    its own face coefficients and assembler, so that the backward problem here and
    the forward one there are adjoint by construction rather than by two codes kept
    in step. A test pins it: the stationary density that class returns is the null
    vector of this operator.
    """

    __slots__: list[str] = []

    def operator(self) -> csr_matrix:
        """``M`` with ``M p = div J`` on every cell, duplicates summed."""
        grid = self._grid
        points = grid.points()
        cells = self._diffusion(points)
        _check_elliptic(cells)
        builder = _Discretisation(grid, cells)
        for axis in range(grid.dimension):
            low, high = _face_cells(grid, axis)
            faces = _HALF * (points[low.ravel()] + points[high.ravel()])
            drift, diffusion = self._face_coefficients(faces, axis)
            builder.add_axis(axis, low, high, drift, diffusion)
        rows, columns, values = builder.triplets()
        return coo_matrix(
            (values, (rows, columns)),
            shape=(grid.size, grid.size),
        ).tocsr()


class GridFirstPassage:
    """Mean first-passage time to a target set, sampled on a :class:`DensityGrid`.

    ``values`` is shaped like the grid and is zero on the target. :meth:`at`
    interpolates multilinearly between cell centres, which is accurate where the
    time is flat - inside a well, where it is nearly constant - and is the reason a
    start point should be a well bottom or a point on the way into one, not a point
    on the barrier, where the time changes by order one across a cell.
    """

    __slots__ = ["_grid", "_interpolator", "_round_off", "_target", "_values"]

    def __init__(
        self,
        grid: DensityGrid,
        values: np.ndarray,
        target: np.ndarray,
        round_off: float = 0.0,
    ) -> None:
        self._grid = grid
        self._round_off = float(round_off)
        self._values = np.asarray(values, dtype=float).reshape(grid.shape)
        self._target = np.asarray(target, dtype=bool).reshape(grid.shape)
        self._interpolator = RegularGridInterpolator(
            tuple(grid.axis(axis) for axis in range(grid.dimension)),
            self._values,
            bounds_error=False,
            fill_value=np.nan,
        )

    @property
    def grid(self) -> DensityGrid:
        """The grid the time lives on."""
        return self._grid

    @property
    def values(self) -> np.ndarray:
        """``tau``, shaped like the grid."""
        return self._values

    @property
    def target(self) -> np.ndarray:
        """The target cells, as a mask shaped like the grid."""
        return self._target

    @property
    def round_off(self) -> float:
        """The relative error the linear solve can add: ``eps |L|_inf max tau``.

        That is machine precision times the condition number of the restricted
        generator, which here is exact rather than estimated: for this M-matrix
        with a constant right-hand side ``|L^{-1}|_inf`` is ``max tau`` itself. The
        condition therefore grows like the time - the smallest eigenvalue is the
        escape rate. On the planar double well at 161 cells it is 9.8e-12,
        9.6e-11, 4.7e-9, 1.3e-5, 3.9e-2 and 127 at ``Delta U = 1, 4, 8, 16, 24,
        32``; the solve refuses a result whose bound exceeds 1%, which is where
        the times stop being a measurement. (The diagonal-``D`` case is the
        M-matrix; a correlated ``D`` adds small entries of either sign and the
        bound is then an estimate.)
        """
        return self._round_off

    def at(self, state: Sequence[float] | np.ndarray) -> float:
        """``tau`` at a state, multilinearly interpolated; ``nan`` off the grid."""
        point = np.asarray(state, dtype=float)[None, :]
        return float(self._interpolator(point)[0])


class MeanFirstPassage:
    """The backward Kolmogorov problem ``L tau = -1`` on a grid, in any dimension.

    The drift, noise, convention and grid are those of
    :class:`~.fokker_planck.StationaryFokkerPlanck`, with the same requirement that
    ``D`` be non-singular on the whole grid; see the module docstring for the
    discretisation and why it is a Markov chain's own passage problem.

    ``iterative`` picks the linear solver. A sparse LU is exact to round-off and is
    the default in one and two dimensions; in three its fill grows like ``N^2`` for
    ``N`` cells and it is the wrong tool - on the three-cell ring's ``41^3`` grid it
    did not finish in 30 minutes on a loaded machine, against under a minute for
    BiCGSTAB preconditioned by an incomplete LU (drop tolerance ``1e-4``), which
    is the default from three dimensions on. The iteration is run to a relative
    residual of ``1e-10``; with the condition number bounded as in
    :attr:`GridFirstPassage.round_off`, that is a relative error in ``tau`` well
    below the discretisation's, and on the ring's ``31^3`` grid the two solvers
    agree to ``1e-11``.
    """

    __slots__ = ["_forward", "_generator", "_grid", "_iterative"]

    def __init__(
        self,
        drift: Function,
        noise: NoiseMatrix,
        convention: NoiseConvention,
        grid: DensityGrid,
        *,
        iterative: bool | None = None,
    ) -> None:
        self._grid = grid
        self._forward = _ForwardOperator(drift, noise, convention, grid)
        self._generator: csr_matrix | None = None
        self._iterative = (
            grid.dimension >= _ITERATIVE_DIMENSION if iterative is None else iterative
        )

    @property
    def grid(self) -> DensityGrid:
        """The grid the problem is posed on."""
        return self._grid

    def generator(self) -> csr_matrix:
        """The backward generator ``L = -M^T``, built once and kept.

        ``M p`` is the divergence of the probability flux, so ``dp/dt = -M p`` and
        the generator of the chain whose law that is acts on functions by the
        transpose. Its rows sum to zero exactly, because every flux enters ``M`` once
        with each sign in the same column.
        """
        if self._generator is None:
            self._generator = (-self._forward.operator().T).tocsr()
        return self._generator

    def solve(self, target: np.ndarray) -> GridFirstPassage:
        """The mean time to reach the cells where ``target`` is true.

        ``target`` is a boolean mask shaped like the grid (:meth:`Ball.on` and
        :meth:`HalfSpace.on` make one). Every cell must be able to reach it - a
        grid with no target, or a region of cells the drift and noise cannot
        leave, makes the system singular - and the solve raises rather than
        return infinities.
        """
        mask = np.asarray(target, dtype=bool).reshape(-1)
        if mask.size != self._grid.size:
            message = (
                f"the target mask has {mask.size} cells; the grid has "
                f"{self._grid.size}"
            )
            raise ValueError(message)
        if not mask.any():
            message = "the target set contains no grid cell"
            raise ValueError(message)
        free = np.flatnonzero(~mask)
        block = self.generator()[free][:, free].tocsc()
        times = np.zeros(self._grid.size)
        times[free] = (
            _iterative_solve(block) if self._iterative else _direct_solve(block)
        )
        largest = float(np.max(times))
        round_off = _EPSILON * float(abs(block).sum(axis=1).max()) * largest
        if (
            not np.all(np.isfinite(times))
            or float(np.min(times)) < -round_off * largest
            or round_off > _ROUND_OFF_LIMIT
        ):
            message = (
                "the first-passage solve did not return finite non-negative times: "
                "some cells cannot reach the target, or the times are beyond what "
                f"double precision resolves (round-off estimate {round_off:.2g})"
            )
            raise ValueError(message)
        np.maximum(times, 0.0, out=times)
        return GridFirstPassage(self._grid, times, mask, round_off)


def _direct_solve(block: csc_matrix) -> np.ndarray:
    """``block tau = -1`` by sparse LU.

    A minimum-degree ordering of ``A + A^T``: the generator's sparsity pattern is
    symmetric, and on the ring's ``31^3`` grid this factorises in 16 s against 27 s
    for the default column ordering.
    """
    return splu(block, permc_spec=_ORDERING).solve(-np.ones(block.shape[0]))


def _iterative_solve(block: csc_matrix) -> np.ndarray:
    """``block tau = -1`` by BiCGSTAB with an incomplete-LU preconditioner."""
    factor = spilu(block, drop_tol=_ILU_DROP, fill_factor=_ILU_FILL)
    preconditioner = LinearOperator(block.shape, factor.solve)
    right = -np.ones(block.shape[0])
    solution, info = bicgstab(
        block,
        right,
        M=preconditioner,
        rtol=_ITERATIVE_TOLERANCE,
        atol=0.0,
        maxiter=_ITERATIVE_STEPS,
    )
    if info != 0:
        message = (
            f"BiCGSTAB did not reach a relative residual of {_ITERATIVE_TOLERANCE:g} "
            f"(info {info}); solve with iterative=False or a coarser grid"
        )
        raise ValueError(message)
    return solution


class Ball:
    """The closed ball of ``radius`` about ``centre``: a target, or a well's set.

    The same set serves both routes, which is what makes their times comparable: called
    on a ``(k, n)`` array of states it answers which lie inside, for an ensemble, and
    :meth:`on` gives the grid cells whose centres lie inside, for the grid.
    """

    __slots__ = ["_centre", "_radius"]

    def __init__(self, centre: Sequence[float], radius: float) -> None:
        self._centre = np.asarray(centre, dtype=float)
        self._radius = float(radius)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Which rows of ``points`` lie in the ball."""
        offsets = np.asarray(points, dtype=float) - self._centre
        return np.einsum("ki,ki->k", offsets, offsets) <= self._radius**2

    def on(self, grid: DensityGrid) -> np.ndarray:
        """The cells of ``grid`` in the ball, as a mask shaped like the grid."""
        return self(grid.points()).reshape(grid.shape)


class HalfSpace:
    """The states with ``normal . x >= offset``, for an ensemble or a grid."""

    __slots__ = ["_normal", "_offset"]

    def __init__(self, normal: Sequence[float], offset: float) -> None:
        self._normal = np.asarray(normal, dtype=float)
        self._offset = float(offset)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Which rows of ``points`` lie in the half space."""
        return np.asarray(points, dtype=float) @ self._normal >= self._offset

    def on(self, grid: DensityGrid) -> np.ndarray:
        """The cells of ``grid`` in the half space, as a mask shaped like the grid."""
        return self(grid.points()).reshape(grid.shape)


def eyring_kramers_time(
    barrier: float,
    unstable_rate: float,
    saddle_hessian: np.ndarray | Sequence[Sequence[float]],
    minimum_hessian: np.ndarray | Sequence[Sequence[float]],
    nongibbs: float = 0.0,
) -> float:
    """The small-noise mean transition time between two point attractors.

    In the library's normalisation (``dx = f dt + G dW``, quasipotential ``U`` of the
    starting attractor ``m`` with the density ``~ exp(-U)``),

        ``tau ~ (2 pi / lambda_+) sqrt(|det H_s| / det H_m) exp(nongibbs) exp(dU)``

    with ``dU = U(s) - U(m)`` the ``barrier``, ``H_s`` and ``H_m`` the Hessians of
    ``U`` at the saddle and at the attractor, and ``lambda_+`` the positive
    eigenvalue of the drift's Jacobian at the saddle (``unstable_rate``). This is
    Bouchet and Reygner's Eq. (1.10) (Ann. Henri Poincare 17 (2016), arXiv
    1507.02104) with their ``V / eps`` read as ``U``: the ratio of determinants and
    ``lambda_+`` do not change under that rescaling, so neither does the
    prefactor. ``nongibbs`` is their ``integral F(rho_t) dt`` along the instanton
    from ``m`` to ``s``, with ``F = div l + <A, grad U>``, ``l = f + (1/2) D grad
    U`` and ``A_i = sum_j d_j D_ij / 2``; it vanishes when ``exp(-U)`` is
    stationary - for every gradient drift, and for a gradient drift plus a
    divergence-free rotation.

    For ``f = -grad V`` and ``G = sigma I``, ``U = 2 V / sigma^2``, ``lambda_+`` is
    the magnitude of the negative eigenvalue of ``Hess V`` at the saddle and the
    formula is the Eyring-Kramers law, Berglund's Eq. (1.9); in one dimension it
    is Kramers' ``2 pi / sqrt(V''(m) |V''(s)|) exp(2 Delta V / sigma^2)``.

    Assumptions: overdamped (first-order) dynamics with a non-singular ``D``; one
    non-degenerate saddle (one unstable direction, ``det H_s < 0``) between two
    point attractors, with ``U`` smooth along the instanton; and small noise - the
    relative error is ``O(1 / dU)``, so the formula is a limit, not a value, until
    ``dU`` is several units. It does not apply when the basins are separated by a
    saddle cycle, nor at a saddle that is degenerate (a pitchfork).
    """
    saddle = np.atleast_2d(np.asarray(saddle_hessian, dtype=float))
    minimum = np.atleast_2d(np.asarray(minimum_hessian, dtype=float))
    saddle_determinant = float(np.linalg.det(saddle))
    minimum_determinant = float(np.linalg.det(minimum))
    if saddle_determinant >= 0.0 or minimum_determinant <= 0.0:
        message = (
            "the Eyring-Kramers formula needs one unstable direction at the saddle "
            f"(det H_s < 0, got {saddle_determinant:.3g}) and a minimum "
            f"(det H_m > 0, got {minimum_determinant:.3g})"
        )
        raise ValueError(message)
    if unstable_rate <= 0.0:
        message = f"the unstable rate must be positive, got {unstable_rate}"
        raise ValueError(message)
    prefactor = (2.0 * math.pi / unstable_rate) * math.sqrt(
        abs(saddle_determinant) / minimum_determinant,
    )
    return prefactor * math.exp(nongibbs + barrier)


@dataclass(frozen=True)
class TimeSample:
    """Passage or residence times from an ensemble, and what they estimate.

    ``times`` holds one entry per completed passage or residence. A path that never
    completed one within the run is counted in ``censored`` and not in ``times``;
    a sample with censored paths underestimates the mean, and a caller comparing it
    with the grid should see ``censored == 0`` first. ``starts`` holds, for a
    residence, the state at which it began - the first sample inside the well's
    set - one row per entry of ``times``; it is empty for first passages, which all
    begin at the initial state.
    """

    times: np.ndarray
    """Completed times."""

    censored: int = 0
    """Paths still running when the run stopped."""

    starts: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    """The state each residence began at, ``(count, n)``."""

    @property
    def count(self) -> int:
        """Number of completed times."""
        return int(self.times.size)

    @property
    def mean(self) -> float:
        """Sample mean."""
        return float(np.mean(self.times))

    @property
    def standard_error(self) -> float:
        """Standard error of the mean, from the sample standard deviation."""
        return float(np.std(self.times, ddof=1) / math.sqrt(self.count))

    def confidence_interval(self, quantile: float = _NORMAL_95) -> tuple[float, float]:
        """Normal interval ``mean +- quantile * standard_error`` (95% by default)."""
        half = quantile * self.standard_error
        return self.mean - half, self.mean + half


def _first_true(hits: np.ndarray) -> np.ndarray:
    """Index of the first true entry along axis 0, ``-1`` where there is none."""
    found = hits.any(axis=0)
    first = np.argmax(hits, axis=0)
    return np.where(found, first, -1)


class EnsembleFirstPassage:
    """First-passage times to a target set, from an ensemble of paths.

    ``config`` sets the step, the scheme's calculus and the seed; its start and end
    times are ignored, because the run is chunked: paths are stepped
    ``chunk_steps`` at a time, those that entered ``target`` in the chunk are
    retired with the time of the first step that landed inside, and the rest go on
    until none remain or ``horizon`` is reached. One generator drives every chunk,
    so a run is reproducible from the seed.
    """

    __slots__ = ["_chunk", "_config", "_scheme"]

    def __init__(
        self,
        config: StochasticConfig,
        scheme: str = "euler_maruyama",
        chunk_steps: int = _DEFAULT_CHUNK_STEPS,
    ) -> None:
        self._config = config
        self._scheme = scheme
        self._chunk = chunk_steps

    def solve(
        self,
        equation: DifferentialEquation,
        initial: Sequence[float] | np.ndarray,
        paths: int,
        target: Callable[[np.ndarray], np.ndarray],
        horizon: float,
    ) -> TimeSample:
        """Times for ``paths`` paths from ``initial`` to enter ``target``.

        ``target`` maps a ``(k, n)`` array of states to a boolean array of ``k``.
        The starting state must lie outside it.
        """
        start = np.asarray(initial, dtype=float)
        if bool(target(start[None, :])[0]):
            message = "the initial state is already inside the target"
            raise ValueError(message)
        dt = float(self._config.step_size)
        ensemble = StochasticEnsemble(
            self._config.model_copy(
                update={"start_time": 0.0, "end_time": self._chunk * dt},
            ),
            scheme=self._scheme,
        )
        states = np.tile(start, (paths, 1))
        times: list[np.ndarray] = []
        elapsed = 0.0
        while states.shape[0] > 0 and elapsed < horizon:
            result = ensemble.solve(equation, states, keep_every=1)
            samples = result.samples[1:]
            steps, count, dimension = samples.shape
            hits = target(samples.reshape(-1, dimension)).reshape(steps, count)
            first = _first_true(hits)
            done = first >= 0
            times.append(elapsed + (first[done] + 1) * dt)
            states = result.final[~done]
            elapsed += steps * dt
        return TimeSample(
            times=np.concatenate(times) if times else np.empty(0),
            censored=int(states.shape[0]),
        )


@dataclass(frozen=True)
class ResidenceSample:
    """Residence times in two wells, read from long paths."""

    first: TimeSample
    """Residences in the first well."""

    second: TimeSample
    """Residences in the second well."""

    @property
    def asymmetry(self) -> float:
        """Mean residence in the first well minus that in the second."""
        return self.first.mean - self.second.mean

    def asymmetry_interval(
        self,
        quantile: float = _NORMAL_95,
    ) -> tuple[float, float]:
        """Normal interval for :attr:`asymmetry`; the two samples are independent."""
        half = quantile * math.hypot(
            self.first.standard_error,
            self.second.standard_error,
        )
        return self.asymmetry - half, self.asymmetry + half


class ResidenceTracker:
    """Cut labelled paths into residences, a chunk of the ensemble at a time.

    Each path carries the set it was last inside (``-1`` before it has entered
    either), the time it entered it and the state it entered at. A step that finds
    a path inside the *other* set closes the open residence - its duration is the
    entry time into the other set minus the entry time into this one - and opens a
    new one; a step in neither set changes nothing, so leaving a set without
    reaching the other does not end a residence. Residences still open when the
    paths end are not returned. Working through the run as it is made rather than
    on a stored trajectory is what lets a long run keep the entry *states*, which
    the grid comparison needs: by the strong Markov property the mean residence in
    a well is the mean of the grid's ``tau`` over the states the residences began
    at.
    """

    __slots__ = ["_closing", "_durations", "_entered", "_since", "_starts", "_well"]

    def __init__(self, states: np.ndarray, labels: np.ndarray, time: float) -> None:
        self._well = np.asarray(labels, dtype=np.int8).copy()
        self._well[self._well == _CLOSED] = _UNVISITED
        self._since = np.full(self._well.size, float(time))
        self._entered = np.asarray(states, dtype=float).copy()
        self._durations: tuple[list[float], list[float]] = ([], [])
        self._starts: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])
        self._closing = False

    def step(self, states: np.ndarray, labels: np.ndarray, time: float) -> None:
        """Record one step: every path's state and its label at ``time``."""
        self.chunk(states[None], np.asarray(labels)[None], np.array([time]))

    def chunk(self, states: np.ndarray, labels: np.ndarray, times: np.ndarray) -> None:
        """Record consecutive steps: ``(steps, paths, n)`` states, ``(steps, paths)``.

        The set each path was last inside is carried forward through the chunk by
        a running maximum of the indices of its labelled steps, so the steps at
        which a path enters a set other than its last one - the only steps that
        change anything - are found with array operations, and only those few
        are visited one by one, in time order.
        """
        labels = np.asarray(labels, dtype=np.int8)
        steps, paths = labels.shape
        labelled = labels != _UNVISITED
        latest = np.maximum.accumulate(
            np.where(labelled, np.arange(steps)[:, None], -1),
            axis=0,
        )
        columns = np.arange(paths)[None, :]
        carried = np.where(
            latest >= 0,
            labels[np.maximum(latest, 0), columns],
            self._well[None, :],
        )
        before = np.vstack([self._well[None, :], carried[:-1]])
        for row, path in np.argwhere(labelled & (labels != before)):
            well = int(self._well[path])
            if well == _CLOSED:
                continue
            if well != _UNVISITED:
                self._durations[well].append(times[row] - self._since[path])
                self._starts[well].append(self._entered[path].copy())
                if self._closing:
                    self._well[path] = _CLOSED
                    continue
            elif self._closing:
                continue
            self._well[path] = labels[row, path]
            self._since[path] = times[row]
            self._entered[path] = states[row, path]

    def stop_opening(self) -> None:
        """Let the open residences finish, and open no new ones.

        Dropping the residences still open at the end of a run biases the mean
        short - a long residence is more likely to be the one cut off - by a
        fraction of the order of ``tau / horizon``: on the three-cell ring, 131
        against the grid's 144 at ``tau / horizon = 0.07``, and 416 against 589
        at 0.3; letting them finish gives 140 +- 2.4 and 571 +- 14.
        Every residence that *began* before the horizon is an unbiased draw, by
        the strong Markov property at its entry time, so after the horizon the
        run continues until those have closed, and no others are counted.
        """
        self._closing = True
        self._well[self._well == _UNVISITED] = _CLOSED

    @property
    def closing(self) -> bool:
        """Whether :meth:`stop_opening` has been called."""
        return self._closing

    @property
    def open_paths(self) -> np.ndarray:
        """Indices of the paths with a residence still open."""
        return np.flatnonzero((self._well == _FIRST) | (self._well == _SECOND))

    def sample(self) -> "ResidenceSample":
        """The completed residences in both wells; open ones count as censored."""
        dimension = self._entered.shape[1]
        first, second = (
            TimeSample(
                times=np.array(self._durations[well], dtype=float),
                censored=int(np.count_nonzero(self._well == well)),
                starts=(
                    np.array(self._starts[well])
                    if self._starts[well]
                    else np.empty((0, dimension))
                ),
            )
            for well in (_FIRST, _SECOND)
        )
        return ResidenceSample(first, second)


class EnsembleResidence:
    """Residence times in two wells from long ensemble paths.

    The paths are run for ``horizon`` in chunks, labelled at every step by which set
    they are in, and cut by a :class:`ResidenceTracker`. Half the paths start at
    each attractor so that neither well is short of samples at a strong field;
    residences are measured from entry, so where a path started does not bias them.
    At the horizon no new residences are opened, and only the paths with one still
    open go on, until it closes or ten more horizons have passed; any still open
    then are reported as ``censored``.
    """

    __slots__ = ["_chunk", "_config", "_scheme"]

    def __init__(
        self,
        config: StochasticConfig,
        scheme: str = "euler_maruyama",
        chunk_steps: int = _DEFAULT_CHUNK_STEPS,
    ) -> None:
        self._config = config
        self._scheme = scheme
        self._chunk = chunk_steps

    def solve(
        self,
        equation: DifferentialEquation,
        starts: tuple[Sequence[float], Sequence[float]],
        paths: int,
        sets: tuple[
            Callable[[np.ndarray], np.ndarray],
            Callable[[np.ndarray], np.ndarray],
        ],
        horizon: float,
    ) -> ResidenceSample:
        """Residences of ``paths`` paths over ``[0, horizon]``."""
        dt = float(self._config.step_size)
        ensemble = StochasticEnsemble(
            self._config.model_copy(
                update={"start_time": 0.0, "end_time": self._chunk * dt},
            ),
            scheme=self._scheme,
        )
        half = paths // 2
        states = np.vstack(
            [
                np.tile(np.asarray(starts[0], dtype=float), (half, 1)),
                np.tile(np.asarray(starts[1], dtype=float), (paths - half, 1)),
            ],
        )
        run = _ResidenceRun(ensemble, equation, sets)
        tracker = ResidenceTracker(states, run.labels(states), 0.0)
        elapsed = 0.0
        while elapsed < horizon:
            states = run.advance(states, tracker, elapsed)
            elapsed += self._chunk * dt
        tracker.stop_opening()
        limit = elapsed + _DRAIN_FACTOR * horizon
        while tracker.open_paths.size and elapsed < limit:
            states = run.advance(states, tracker, elapsed)
            elapsed += self._chunk * dt
        return tracker.sample()


_Sets = tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]


class _ResidenceRun:
    """One residence run's ensemble, equation and sets, advanced a chunk at a time."""

    __slots__ = ["_ensemble", "_equation", "_sets"]

    def __init__(
        self,
        ensemble: StochasticEnsemble,
        equation: DifferentialEquation,
        sets: _Sets,
    ) -> None:
        self._ensemble = ensemble
        self._equation = equation
        self._sets = sets

    def labels(self, states: np.ndarray) -> np.ndarray:
        """The set each state is in."""
        return _label(states, self._sets)

    def advance(
        self,
        states: np.ndarray,
        tracker: ResidenceTracker,
        elapsed: float,
    ) -> np.ndarray:
        """Step the paths with an open residence one chunk; the rest stay put."""
        active = tracker.open_paths if tracker.closing else np.arange(len(states))
        result = self._ensemble.solve(self._equation, states[active], keep_every=1)
        steps = result.samples.shape[0] - 1
        samples = np.zeros((steps, *states.shape))
        samples[:, active] = result.samples[1:]
        labels = np.full((steps, states.shape[0]), _UNVISITED, dtype=np.int8)
        labels[:, active] = self.labels(
            result.samples[1:].reshape(-1, states.shape[1]),
        ).reshape(steps, active.size)
        dt = float(result.times[1] - result.times[0])
        tracker.chunk(samples, labels, elapsed + dt * np.arange(1, steps + 1))
        moved = states.copy()
        moved[active] = result.final
        return moved


def _label(states: np.ndarray, sets: _Sets) -> np.ndarray:
    """Label every state ``0``, ``1`` or ``-1`` by the set it lies in."""
    labels = np.full(states.shape[0], _UNVISITED, dtype=np.int8)
    labels[sets[0](states)] = _FIRST
    labels[sets[1](states)] = _SECOND
    return labels
