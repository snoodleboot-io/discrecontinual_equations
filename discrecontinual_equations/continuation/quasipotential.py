"""The Freidlin-Wentzell quasipotential of a planar flow, and its minimum action paths.

The stationary density of :mod:`.fokker_planck` says where the mass sits and the
exponent of :mod:`.lyapunov` whether paths escape; neither says how a path gets from
one attractor to another. At small noise that is answered by the quasipotential.
For ``dx = f dt + sqrt(eps) G dW`` with ``D = G G^T``, the probability of a path
``psi`` is ``exp(-S / eps)`` with the action ``S = (1/2) integral (psi' - f)^T D^{-1}
(psi' - f) dt``, and the quasipotential ``U(x)`` of an attractor is the least action
over all paths that leave it and reach ``x`` in any time. Escape rates go like
``exp(-U / eps)``, the stationary density like ``exp(-U / eps)`` to leading order, and
the minimiser is the most probable transition path.

*Normalisation.* The rest of the library writes ``dx = f dt + G dW`` with no ``eps``,
so ``U`` is defined here with ``D = G G^T`` exactly as :mod:`.fokker_planck` builds it
- the ``eps = 1`` reading - and solves the Hamilton-Jacobi equation

    ``H(x, grad U) = grad U . f + (1/2) grad U^T D grad U = 0``.

In this convention a gradient drift ``f = -grad V`` with ``G = I`` has ``U = 2 V``,
the linear system ``dx = A x dt + G dW`` has ``U = (1/2) x^T S^{-1} x`` with
``A S + S A^T + D = 0``, and in both the density the grid solve returns is
``exp(-U)`` up to a constant; the tests assert all three. (The ticket wrote the
linear case as ``x^T S^{-1} x``. Under the Hamiltonian it also states that is twice
too large: substituting ``U = x^T M x`` gives ``M A + A^T M + 2 M D M = 0``, solved
by ``M = S^{-1} / 2``.) The Ito-Stratonovich correction to the drift is of order
``eps`` and does not enter ``U``: the drift here is ``f`` as written.

*Method.* ``U`` is computed on the grid of :class:`~.fokker_planck.DensityGrid` by
the ordered line integral method of Dahiya and Cameron (J. Sci. Comput. 2018,
arXiv 1706.07509), the midpoint-quadrature variant OLIM-MID. Grid points are
accepted in increasing order of ``U``, as in Dijkstra's method; when a point ``x0``
is accepted every unaccepted point ``x`` within ``update_radius`` cells of it is
offered

* a one-point update ``U(x0) + Q(x0, x)``, and
* a triangle update for every accepted neighbour ``x1`` of ``x0``: the minimum over
  ``x_s = x0 + s (x1 - x0)``, ``s`` in ``[0, 1]``, of the linearly interpolated
  ``U(x_s)`` plus ``Q(x_s, x)``,

where ``Q(a, x) = |x - a|_A |f_m|_A - <x - a, f_m>_A`` is the geometric action of the
straight segment by the midpoint rule, in the metric ``A = D^{-1}`` taken at the
midpoint. Midpoints of two grid points always lie on the grid of half the spacing,
so ``f`` and ``A`` are evaluated once there and looked up; along a triangle's
segment they are interpolated linearly between its two end midpoints, which makes
the one-dimensional minimisation over ``s`` a quadratic with a closed-form root,
refined twice from the midpoint of the segment. The update radius is what lets the
method see around an anisotropic or rotating drift: a minimum action path can meet
the accepted front at a shallow angle, and only a stencil wide enough to reach back
along it finds the right predecessor. Every triangle is tried rather than the
hierarchical subset of the paper (triangles only round the best one-point
predecessor): the subset was measured here and doubled the rotating
Ornstein-Uhlenbeck error below at 161 points (1.06e-3 against 5.0e-4).

The attractor is initialised exactly where it can be: around a stable equilibrium
``x*`` the points within a couple of cells get the quadratic quasipotential of the
linearisation, ``(1/2) (x - x*)^T S^{-1} (x - x*)`` with ``J S + S J^T + D(x*) = 0``;
around a limit cycle the points within a cell and a half get the paper's
Simpson-rule start from the nearest point of the cycle (see
:class:`CycleAttractor`), with ``grad U`` from the split of the drift into a
gradient and a rotational part.

*Measured.* The error in ``U`` against the closed forms, by doubling the grid on a
fixed box (the tests repeat these):

  - double well ``V = x^4/4 - x^2/2 + y^2/2``, ``D = I``, max error of ``U - 2 (V -
    V_min)`` on the basin of ``(-1, 0)`` within ``{U <= 2}``, box ``[-2, 1] x
    [-1.5, 1.5]``: 0.153, 0.0420, 0.0113 at 41, 81, 161 points per side, observed
    orders 1.87 and 1.89;
  - rotating Ornstein-Uhlenbeck, ``A = [[-1, -2], [2, -1]]``, ``G = [[1, 0.5],
    [0, 0.8]]``, max error against ``(1/2) x^T S^{-1} x`` on ``{U <= 1.5}``, box
    ``[-1.5, 1.5]^2``: 2.27e-3, 9.68e-4, 5.00e-4, observed orders 1.23 and 0.95.

Both at the default update radius of 8 cells, held fixed as the grid is refined.
The error is two terms of different order. Linear interpolation of ``U`` along the
base of a triangle costs ``O(h^2)`` per update over ``O(1 / (K h))`` updates along a
path, which is first order, and is the bound Dahiya and Cameron state for every
OLIM; the rotating case shows it. The midpoint rule on a one-point update of length
up to ``K h`` costs ``O((K h)^3 |f''|)`` per update, second order overall but with a
constant growing like ``K^2``, and since the march keeps the minimum and the rule
underestimates a convex integrand, it is the long updates that are picked. The
double well's ``x - x^3`` is steep enough on this box for that term to dominate:
at 81 points the error is 0.0107, 0.0112 and 0.0420 for radii 2, 4 and 8, while
the rotating case at radius 4 is four times worse than at 8 (4.1e-3). A radius
that grows slowly with the grid, as the paper recommends (its Eq. (28),
``K = 10 + 4 (log2 N - 7)`` for ``2^7 <= N <= 2^12``), trades the two; here it is
left to the caller.

*What the radius must cover.* Below a threshold the radius does not merely cost
accuracy: the error stops converging. Near the attractor the drift is a gradient
part ``-(1/2) D grad U`` plus a rotation ``l`` along the level sets, and a minimum
action path crosses the level sets at a shallow angle - about ``1 / a`` for
``a = |l|_A / |(1/2) D grad U|_A``, ``A = D^{-1}`` - so the accepted point it
comes from lies of the order of ``a`` cells back along the front. A stencil much
narrower than that cannot reach it, and every update is taken from the wrong
place, at every spacing. On the paper's SDE (21), where ``a = 10`` everywhere
(``U = 2 x^2 + y^2`` on ``[-1, 1]^2``, max error on ``{U <= 0.9}``): radius 2
gives 0.381, 0.370 and 0.371 at 129, 257 and 513 points, and radius 4 gives
0.048 and 0.051 at 129 and 257 - neither converges - against 1.6e-3 at radius 8
and 3.1e-4 at the paper's 14 on 257 points. The default 8 converges there, if
slowly (2.28e-3 and 1.34e-3 at 161 and 321 points, order 0.76), and comfortably
on the rotating Ornstein-Uhlenbeck process below (``a = 2.36``). So the radius
should be comparable to the largest rotation-to-gradient ratio on the region
asked about, and a field that does not settle on refining the grid is the
symptom of one that is too small.


*Paths.* :meth:`OrderedLineIntegral.minimum_action_path` follows the computed field
from a chosen point back to the attractor: along a minimum action path
``psi' = f + D grad U`` (the characteristic ``dH / dp``), so the path is traced
backwards along ``-(f + D grad U)`` with ``grad U`` interpolated from the grid. In the
gradient case that is steepest descent of ``V`` run in reverse. Between two chosen
points, :class:`GeometricMinimumAction` relaxes a whole path by the geometric
minimum action method of Heymann and Vanden-Eijnden (Comm. Pure Appl. Math. 2008),
in its semi-implicit form for additive noise - the descent flow as Grafke, Schafer
and Vanden-Eijnden restate it (arXiv 1604.03818, Eq. (19)); the letter usually
cited for that form is Vanden-Eijnden and Heymann, J. Chem. Phys. 128, 061103
(2008). It needs no grid and crosses saddles, which a descent on ``U`` cannot.

*Limits.* The method is planar - a two-dimensional :class:`DensityGrid` only; the
ordered line integral idea extends to three dimensions at a much higher cost and is
not implemented. The drift and noise must be time-independent: ``U`` is a static
object and a periodically forced system needs the quasipotential of its Poincare map
instead. ``D`` must be non-singular at every grid point (scalar noise, or noise on a
subset of the components, is refused, as it is by the density solve); a point
*between* grid points where ``D`` vanishes - the origin of the conformal noise
``sigma z dB`` - is treated as impassable, which is the quasipotential's own limit
there (it diverges logarithmically), but the values in the cells around it are not
claimed to the order above. The geometric minimum action method here assumes a
constant ``D``. ``U`` near the boundary of the box is wrong wherever the true
minimum action path would leave the box, so the box should contain the sublevel set
of ``U`` being asked about: ``U`` increases along every minimum action path, so the
path to ``x`` stays inside ``{U <= U(x)}``.
"""

import heapq
from collections.abc import Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import cho_factor, solve_banded, solve_continuous_lyapunov
from scipy.spatial import cKDTree

from discrecontinual_equations.continuation.fokker_planck import DensityGrid
from discrecontinual_equations.continuation.noise import (
    NoiseMatrix,
    diffusion_matrix,
    field_at,
    jacobian_at,
)
from discrecontinual_equations.function.function import Function

_PLANE = 2
# f_x, f_y, A_xx, A_xy, A_yy per half-lattice point.
_COLUMNS = 5
_DEFAULT_RADIUS = 8
_EQUILIBRIUM_CELLS = 2.0
_CYCLE_CELLS = 1.5
_CYCLE_SAMPLES_PER_CELL = 4
_ELLIPTICITY_FLOOR = 1.0e-10
_REFINEMENTS = 2
_HALF = 0.5
_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_UNKNOWN, _ACCEPTED = 0, 2
_CONSTANT_TOLERANCE = 1.0e-9
_DESCENT_STEP = 0.5
_DESCENT_MAX_STEPS = 20000
_ARRIVAL_CELLS = 2.0


class EquilibriumAttractor:
    """A stable equilibrium, initialised by the quasipotential of its linearisation.

    ``radius`` is how far, in cells, the quadratic form is written before the ordered
    line integrals take over. It is a fixed number of cells, so the region shrinks
    with the spacing and the error it carries (of order ``|x - x*|^3``) shrinks
    faster than the method's own.
    """

    __slots__ = ["_radius", "_state"]

    def __init__(self, state: Sequence[float], radius: float = _EQUILIBRIUM_CELLS):
        self._state = np.asarray(state, dtype=float)
        self._radius = float(radius)

    @property
    def state(self) -> np.ndarray:
        """The equilibrium."""
        return self._state

    def initial(
        self,
        grid: DensityGrid,
        drift: Function,
        noise: NoiseMatrix,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flat indices and values of the points set before the march starts."""
        _check_inside(grid, self._state)
        jacobian = jacobian_at(drift, self._state)
        if np.max(np.linalg.eigvals(jacobian).real) >= 0.0:
            message = (
                f"the equilibrium at {self._state.tolist()} is not stable; a "
                f"quasipotential is measured from an attractor"
            )
            raise ValueError(message)
        diffusion = diffusion_matrix(noise.columns_at(self._state))
        covariance = solve_continuous_lyapunov(jacobian, -diffusion)
        form = _HALF * np.linalg.inv(covariance)
        points = grid.points()
        offsets = points - self._state
        cells = np.linalg.norm(offsets / grid.spacings, axis=1)
        near = np.flatnonzero(cells <= self._radius)
        values = np.einsum("ki,ij,kj->k", offsets[near], form, offsets[near])
        return near, values

    def end(self) -> np.ndarray | None:
        """Where a path traced back to this attractor finishes."""
        return self._state


class CycleAttractor:
    """A stable limit cycle, given as states around it in order.

    ``U`` vanishes on the cycle. The points within ``width`` cells of it start
    from the value Dahiya and Cameron's initialisation gives (their Eqs. (9) to
    (12)), carried over to a matrix ``D``. Near the cycle the drift splits as
    ``f = -(1/2) D grad U + l`` with ``grad U . l = 0``, and ``l`` is close to
    ``f`` on the cycle, where ``grad U`` vanishes; taking ``l`` parallel to ``f``
    at the foot ``x*`` of ``x`` on the cycle gives

        ``grad U(z) = -2 D(z)^{-1} (f(z) - c f(x*))``,
        ``c = <f(z), f(x*)>_A / <f(x*), f(x*)>_A``, ``A = D(z)^{-1}``,

    and ``U(x) = integral of grad U . dx`` from the foot, where ``U`` and
    ``grad U`` vanish, by Simpson's rule:
    ``(4 grad U(x_m) + grad U(x)) . (x - x*) / 6`` with ``x_m`` the midpoint.
    The foot is the projection of ``x`` on the chord from the nearest point of
    the cycle, resampled to a quarter of a cell, to its neighbour on ``x``'s side.

    A straight-segment action from the foot, the obvious alternative, is wrong
    by ``O(h)``: across a rotating flow it pays ``|l|`` per unit length. The
    march accepts the band in the order of these values and cannot repair the
    first ones it accepts, so that error stays. On Dahiya and Cameron's SDE (24)
    it left ``U`` next to the cycle off by 5.45e-3 and 3.27e-3 at 161 and 321
    points (order 0.74); from this start, 1.47e-3 and 3.0e-4 (order 2.3), the
    same as from the exact values.
    """

    __slots__ = ["_states", "_width"]

    def __init__(self, states: Sequence[Sequence[float]], width: float = _CYCLE_CELLS):
        self._states = np.asarray(states, dtype=float)
        self._width = float(width)

    def initial(
        self,
        grid: DensityGrid,
        drift: Function,
        noise: NoiseMatrix,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flat indices and values of the points set before the march starts."""
        for state in self._states:
            _check_inside(grid, state)
        cell = float(np.min(grid.spacings))
        samples = _resample_closed(self._states, cell / _CYCLE_SAMPLES_PER_CELL)
        points = grid.points()
        distance, nearest = cKDTree(samples).query(points)
        near = np.flatnonzero(distance <= self._width * cell)
        values = np.array(
            [
                _cycle_start(drift, noise, samples, int(index), point)
                for index, point in zip(nearest[near], points[near], strict=True)
            ],
        )
        return near, values

    def end(self) -> np.ndarray | None:
        """A cycle has no single end point; a traced path stops on reaching it."""
        return None


def _resample_closed(states: np.ndarray, spacing: float) -> np.ndarray:
    closed = np.vstack([states, states[:1]])
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(lengths)])
    count = max(int(np.ceil(arc[-1] / spacing)), len(states))
    target = np.linspace(0.0, arc[-1], count, endpoint=False)
    return np.column_stack(
        [np.interp(target, arc, closed[:, axis]) for axis in range(_PLANE)],
    )


def _cycle_start(
    drift: Function,
    noise: NoiseMatrix,
    samples: np.ndarray,
    index: int,
    point: np.ndarray,
) -> float:
    """``U`` at ``point`` from the cycle sample ``index`` nearest it (see above)."""
    closest = samples[index]
    neighbours = (samples[index - 1], samples[(index + 1) % len(samples)])
    beside = max(
        neighbours,
        key=lambda other: float((point - closest) @ (other - closest)),
    )
    tangent = (beside - closest) / np.linalg.norm(beside - closest)
    foot = closest + float((point - closest) @ tangent) * tangent
    along = field_at(drift, foot)
    step = point - foot

    def slope(state: np.ndarray) -> np.ndarray:
        metric = np.linalg.inv(diffusion_matrix(noise.columns_at(state)))
        velocity = field_at(drift, state)
        weight = float(along @ metric @ along)
        share = float(velocity @ metric @ along) / weight if weight > 0.0 else 0.0
        return -2.0 * metric @ (velocity - share * along)

    middle = _HALF * (foot + point)
    return float((4.0 * slope(middle) + slope(point)) @ step / 6.0)


def _geometric(step: np.ndarray, velocity: np.ndarray, metric: np.ndarray) -> float:
    return float(
        np.sqrt(step @ metric @ step) * np.sqrt(velocity @ metric @ velocity)
        - step @ metric @ velocity,
    )


def _check_inside(grid: DensityGrid, state: np.ndarray) -> None:
    for axis, value in enumerate(state):
        coordinates = grid.axis(axis)
        if not coordinates[0] <= value <= coordinates[-1]:
            message = f"the attractor state {state.tolist()} lies outside the grid"
            raise ValueError(message)


class GridQuasipotential:
    """``U`` sampled on a planar :class:`DensityGrid`.

    ``values`` is shaped like the grid and is ``inf`` where no path from the attractor
    reached: behind a point where the noise vanishes, or where the grid does not
    connect. ``floor`` is the largest ``U`` on the points the attractor
    initialised, as the march left them: the level below which a state is as close
    to the attractor as the grid resolves.
    """

    __slots__ = ["_floor", "_grid", "_interpolator", "_values"]

    def __init__(self, grid: DensityGrid, values: np.ndarray, floor: float) -> None:
        self._grid = grid
        self._values = np.asarray(values, dtype=float).reshape(grid.shape)
        self._floor = float(floor)
        # Built once: a traced path asks for U at every step, and building the
        # interpolator per call was most of what tracing a path cost.
        self._interpolator = RegularGridInterpolator(
            (grid.axis(0), grid.axis(1)),
            self._values,
            bounds_error=False,
            fill_value=np.inf,
        )

    @property
    def grid(self) -> DensityGrid:
        """The grid ``U`` lives on."""
        return self._grid

    @property
    def values(self) -> np.ndarray:
        """``U``, shaped like the grid."""
        return self._values

    @property
    def floor(self) -> float:
        """The largest ``U`` the march left on the initialised points."""
        return self._floor

    def at(self, state: Sequence[float] | np.ndarray) -> float:
        """``U`` at a state, bilinearly interpolated; ``inf`` off the grid."""
        point = np.asarray(state, dtype=float)[None, :]
        return float(self._interpolator(point)[0])

    def gradient(self) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
        """Interpolators for the two components of ``grad U``, central differences."""
        first, second = np.gradient(
            self._values,
            self._grid.axis(0),
            self._grid.axis(1),
        )
        axes = (self._grid.axis(0), self._grid.axis(1))
        return (
            RegularGridInterpolator(
                axes,
                first,
                bounds_error=False,
                fill_value=np.nan,
            ),
            RegularGridInterpolator(
                axes,
                second,
                bounds_error=False,
                fill_value=np.nan,
            ),
        )


class MinimumActionPath:
    """A path from an attractor, and the geometric action spent along it.

    ``states`` runs as the path was produced: from the chosen point back to the
    attractor for a traced path, from the first end to the second for a relaxed
    one. ``action`` is the action of going the uphill way, attractor to point.

    The two flags say whether to believe it, each for the method it belongs to,
    in the way :class:`~.corrector.CorrectionResult` reports a corrector that ran
    out of iterations: ``arrived`` is whether a traced path reached its attractor
    rather than stalling short of it - at a saddle, say - and ``converged`` is
    whether a relaxed chain stopped moving before its iteration limit, with
    ``iterations`` the number it took. Each is ``True`` for the method that does
    not set it.
    """

    __slots__ = ["action", "arrived", "converged", "iterations", "states"]

    def __init__(
        self,
        states: np.ndarray,
        action: float,
        *,
        arrived: bool = True,
        converged: bool = True,
        iterations: int = 0,
    ) -> None:
        self.states = np.asarray(states, dtype=float)
        self.action = float(action)
        self.arrived = arrived
        self.converged = converged
        self.iterations = iterations


class _HalfLattice:
    """``f`` and ``A = D^{-1}`` on the grid of half the spacing.

    The midpoint of any two grid points ``i`` and ``j`` is the half-lattice point
    ``i + j``, so every quadrature node the method ever needs is a lookup. The five
    numbers per point - ``f_x, f_y, A_xx, A_xy, A_yy`` - are kept side by side in one
    table so that one gather fetches all of them, and the march does its algebra on
    their columns: the einsum over ``2 x 2`` blocks it replaces cost four times as
    much at the sizes one accepted point updates.
    """

    __slots__ = ["table"]

    def __init__(self, grid: DensityGrid, drift: Function, noise: NoiseMatrix):
        axes = [
            np.linspace(grid.axis(a)[0], grid.axis(a)[-1], 2 * grid.shape[a] - 1)
            for a in range(_PLANE)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.stack([block.ravel() for block in mesh], axis=1)
        velocity = np.array([field_at(drift, state) for state in points])
        cells = np.array(
            [diffusion_matrix(noise.columns_at(state)) for state in points],
        )
        _check_nodes(cells.reshape((*mesh[0].shape, _PLANE, _PLANE))[::2, ::2])
        inverse = _inverse(cells)
        self.table = np.column_stack(
            [velocity, inverse[:, 0, 0], inverse[:, 0, 1], inverse[:, 1, 1]],
        ).reshape((*mesh[0].shape, _COLUMNS))


def _check_nodes(cells: np.ndarray) -> None:
    eigenvalues = np.linalg.eigvalsh(cells.reshape(-1, _PLANE, _PLANE))
    scale = float(np.max(eigenvalues))
    if float(np.min(eigenvalues)) <= _ELLIPTICITY_FLOOR * max(scale, 1.0):
        message = (
            "the diffusion matrix G G^T is singular at a grid point; the "
            "quasipotential needs a non-degenerate diffusion"
        )
        raise ValueError(message)


def _inverse(cells: np.ndarray) -> np.ndarray:
    """``D^{-1}`` per point, ``nan`` where ``D`` is singular: such a point is a wall."""
    eigenvalues = np.linalg.eigvalsh(cells)
    scale = max(float(np.max(eigenvalues)), 1.0)
    singular = eigenvalues[:, 0] <= _ELLIPTICITY_FLOOR * scale
    safe = cells.copy()
    safe[singular] = np.eye(_PLANE)
    inverse = np.linalg.inv(safe)
    inverse[singular] = np.nan
    return inverse


def _form(first_x, first_y, metric, second_x, second_y):
    """``u^T A v`` for columns of vectors, ``A`` read from half-lattice rows."""
    return (
        metric[:, 2] * first_x * second_x
        + metric[:, 3] * (first_x * second_y + first_y * second_x)
        + metric[:, 4] * first_y * second_y
    )


def _action(step_x, step_y, coefficients):
    """``Q`` of many segments at once; ``inf`` across a point where ``D`` vanishes.

    ``coefficients`` holds ``f`` and ``A`` at each segment's midpoint, one row per
    segment, in the half lattice's column order.
    """
    velocity_x, velocity_y = coefficients[:, 0], coefficients[:, 1]
    values = np.sqrt(
        _form(step_x, step_y, coefficients, step_x, step_y)
        * _form(velocity_x, velocity_y, coefficients, velocity_x, velocity_y),
    ) - _form(step_x, step_y, coefficients, velocity_x, velocity_y)
    return np.where(np.isnan(values), np.inf, values)


class OrderedLineIntegral:
    """The quasipotential of a planar ``dx = f dt + G dW`` by OLIM-MID.

    ``update_radius`` is the stencil radius in cells (``K`` in the paper). It must
    be comparable to the drift's rotation-to-gradient ratio on the region asked
    about, or the error does not converge as the grid is refined; see the module
    docstring for that, the method, the normalisation, and its limits.
    """

    __slots__ = ["_drift", "_grid", "_noise", "_offsets"]

    def __init__(
        self,
        drift: Function,
        noise: NoiseMatrix,
        grid: DensityGrid,
        update_radius: int = _DEFAULT_RADIUS,
    ) -> None:
        if grid.dimension != _PLANE:
            message = (
                f"the ordered line integral method here is planar; the grid has "
                f"{grid.dimension} axes"
            )
            raise ValueError(message)
        self._drift = drift
        self._noise = noise
        self._grid = grid
        reach = np.arange(-update_radius, update_radius + 1)
        first, second = np.meshgrid(reach, reach, indexing="ij")
        keep = (first * first + second * second <= update_radius * update_radius) & (
            (first != 0) | (second != 0)
        )
        self._offsets = (first[keep], second[keep])

    def solve(
        self,
        attractor: EquilibriumAttractor | CycleAttractor,
    ) -> GridQuasipotential:
        """March ``U`` out from ``attractor`` over the whole grid."""
        lattice = _HalfLattice(self._grid, self._drift, self._noise)
        march = _March(self._grid, lattice, self._offsets)
        indices, values = attractor.initial(self._grid, self._drift, self._noise)
        march.run(indices, values)
        # The floor is read after the march, not from what the attractor wrote.
        # The march still lowers some of the starting values as the earlier ones
        # are accepted - 173 of 374 at mu = 1 on the noisy Hopf film's grid, the
        # largest from 0.0230 to 0.0219, and at mu = 0.025, where the cycle is
        # three cells from the singular origin, 31 of 56, from 0.0208 to 0.0156 -
        # and what it leaves is the level the grid resolves next to the attractor.
        floor = float(np.max(march.values.ravel()[indices]))
        return GridQuasipotential(self._grid, march.values, floor)

    def minimum_action_path(
        self,
        field: GridQuasipotential,
        start: Sequence[float] | np.ndarray,
        end: Sequence[float] | np.ndarray | None = None,
        step: float | None = None,
    ) -> MinimumActionPath:
        """Trace the minimum action path from ``start`` back to the attractor.

        The path is followed along ``-(f + D grad U)`` in steps of ``step`` in arc
        length (half a cell by default) by the midpoint rule, and stops when a step
        would raise ``U``. That happens at the attractor, where ``U`` is least, but
        not only there: from a state in another basin the path runs down to the
        saddle between the basins and stalls beside it, since ``-(f + D grad U)``
        vanishes there and the grid cannot say which way to fall; it also stalls
        where the grid does not resolve ``U``, as beside a point where the noise
        vanishes. ``arrived`` on the result says which. A path is counted as
        arrived when it stopped within two cells of ``end`` - an equilibrium's own
        state, which is then appended as the last state - or, with no ``end``, when
        ``U`` where it stopped is no more than the field's ``floor``, the largest
        ``U`` the march left on the points the attractor initialised: no lower than
        the grid resolves next to the attractor.
        A path that has not arrived is returned as far as it got, without ``end``.
        Paths that must cross a saddle are what :class:`GeometricMinimumAction` is
        for.

        The action is accumulated along the way by the same midpoint quadrature the
        march uses, plus ``U`` where the path stopped, so for an arrived path it is
        an independent estimate of ``U(start)``.
        """
        spacing = float(np.min(self._grid.spacings))
        length = step or _DESCENT_STEP * spacing
        gradient = field.gradient()
        current = np.asarray(start, dtype=float)
        level = field.at(current)
        states = [current]
        action = 0.0
        for _ in range(_DESCENT_MAX_STEPS):
            half = current + _HALF * length * self._backward(current, gradient)
            direction = self._backward(half, gradient)
            if not np.all(np.isfinite(direction)):
                break
            following = current + length * direction
            # U falls along every minimum action path, so the path has arrived
            # when a step would raise it: at the attractor, where U is least, or
            # in a region the grid does not resolve - beside a point where the
            # noise vanishes, say - where going on would only circle.
            following_level = field.at(following)
            if not following_level < level:
                break
            level = following_level
            metric = np.linalg.inv(diffusion_matrix(self._noise.columns_at(half)))
            velocity = field_at(self._drift, half)
            action += _geometric(current - following, velocity, metric)
            current = following
            states.append(current)
        remaining = field.at(current)
        if end is None:
            arrived = remaining <= field.floor
        else:
            target = np.asarray(end, dtype=float)
            arrived = bool(
                np.linalg.norm(current - target) <= _ARRIVAL_CELLS * spacing,
            )
            if arrived:
                states.append(target)
        return MinimumActionPath(np.array(states), action + remaining, arrived=arrived)

    def _backward(self, state: np.ndarray, gradient) -> np.ndarray:
        slope = np.array([component(state[None, :])[0] for component in gradient])
        velocity = (
            field_at(self._drift, state)
            + diffusion_matrix(
                self._noise.columns_at(state),
            )
            @ slope
        )
        size = float(np.linalg.norm(velocity))
        if size == 0.0:
            return np.full(_PLANE, np.nan)
        return -velocity / size


class _March:
    """Dijkstra's ordering: accept the smallest tentative ``U``, update around it."""

    __slots__ = [
        "_coordinates",
        "_offsets",
        "_shape",
        "_status",
        "_table",
        "values",
    ]

    def __init__(
        self,
        grid: DensityGrid,
        lattice: _HalfLattice,
        offsets: tuple[np.ndarray, np.ndarray],
    ) -> None:
        self._shape = grid.shape
        self._table = lattice.table
        self._offsets = offsets
        self._coordinates = (grid.axis(0), grid.axis(1))
        self._status = np.full(self._shape, _UNKNOWN, dtype=np.int8)
        self.values = np.full(self._shape, np.inf)

    def run(self, indices: np.ndarray, values: np.ndarray) -> None:
        """Accept the initialised points in order, then march until none is left."""
        rows, columns = np.unravel_index(indices, self._shape)
        self.values[rows, columns] = values
        heap: list[tuple[float, int, int]] = []
        for order in np.argsort(values):
            self._accept(int(rows[order]), int(columns[order]), heap)
        while heap:
            value, row, column = heapq.heappop(heap)
            if self._status[row, column] == _ACCEPTED:
                continue
            if value > self.values[row, column]:
                continue
            self._accept(row, column, heap)

    def _accept(self, row: int, column: int, heap: list) -> None:
        self._status[row, column] = _ACCEPTED
        targets_x, targets_y = self._open_targets(row, column)
        if targets_x.size == 0:
            return
        first, second = self._coordinates
        reach_x = first[targets_x] - first[row]
        reach_y = second[targets_y] - second[column]
        near = self._table[row + targets_x, column + targets_y]
        own = float(self.values[row, column])
        best = own + _action(reach_x, reach_y, near)
        neighbours = self._accepted_neighbours(row, column)
        if neighbours:
            count = targets_x.size
            pairs = np.tile(np.arange(count), len(neighbours))
            others_x = np.repeat([x for x, _ in neighbours], count)
            others_y = np.repeat([y for _, y in neighbours], count)
            triangle = _triangle(
                own,
                self.values[others_x, others_y],
                (first[others_x] - first[row], second[others_y] - second[column]),
                (reach_x[pairs], reach_y[pairs]),
                (
                    near[pairs],
                    self._table[
                        others_x + targets_x[pairs],
                        others_y + targets_y[pairs],
                    ],
                ),
            )
            nearest = triangle.reshape(len(neighbours), count).min(axis=0)
            best = np.minimum(best, nearest)
        better = best < self.values[targets_x, targets_y]
        for x, y, value in zip(
            targets_x[better].tolist(),
            targets_y[better].tolist(),
            best[better].tolist(),
            strict=True,
        ):
            self.values[x, y] = value
            heapq.heappush(heap, (value, x, y))

    def _open_targets(self, row: int, column: int) -> tuple[np.ndarray, np.ndarray]:
        """Every unaccepted point within the update radius of ``(row, column)``."""
        size_x, size_y = self._shape
        targets_x = row + self._offsets[0]
        targets_y = column + self._offsets[1]
        inside = (
            (targets_x >= 0)
            & (targets_x < size_x)
            & (targets_y >= 0)
            & (targets_y < size_y)
        )
        targets_x, targets_y = targets_x[inside], targets_y[inside]
        unaccepted = self._status[targets_x, targets_y] != _ACCEPTED
        return targets_x[unaccepted], targets_y[unaccepted]

    def _accepted_neighbours(self, row: int, column: int) -> list[tuple[int, int]]:
        size_x, size_y = self._shape
        return [
            (row + step_x, column + step_y)
            for step_x, step_y in _NEIGHBOURS
            if 0 <= row + step_x < size_x
            and 0 <= column + step_y < size_y
            and self._status[row + step_x, column + step_y] == _ACCEPTED
        ]


def _triangle(own, other, edge, reach, midpoints):
    """The triangle update from the segment ``x0``-``x1`` to every target.

    ``edge`` is ``x1 - x0`` and ``reach`` is ``x - x0``, as component pairs;
    ``midpoints`` holds ``f`` and ``A`` at the midpoints of ``x0``-``x`` and of
    ``x1``-``x``, one half-lattice row per target each.
    With ``f`` and ``A`` frozen, ``F(s) = U(x_s) + Q(x_s, x)`` is convex in ``s`` and
    ``F'(s) = 0`` is a quadratic, so the minimiser is closed-form; ``f`` and ``A``
    are then re-read at that ``s`` and the root taken again. Two refinements from the
    middle of the segment are enough because the two end midpoints are half a cell
    apart, so ``f`` and ``A`` barely move along it.
    """
    near, far = midpoints
    rise = other - own
    fraction = np.full(rise.shape[0], _HALF)
    for _ in range(_REFINEMENTS):
        coefficients = near + fraction[:, None] * (far - near)
        fraction = _stationary_fraction(edge, reach, rise, coefficients)
    coefficients = near + fraction[:, None] * (far - near)
    step_x = reach[0] - fraction * edge[0]
    step_y = reach[1] - fraction * edge[1]
    return own + fraction * rise + _action(step_x, step_y, coefficients)


def _stationary_fraction(edge, reach, rise, coefficients):
    """Root of ``F'(s)`` in ``[0, 1]`` for frozen ``f`` and ``A``.

    With ``d(s) = reach - s edge``, ``F'(s) = rise + <edge, f>_A - |f|_A <edge, d>_A /
    |d|_A``, so the root solves ``<edge, d>_A / |d|_A = r`` with ``r = (rise +
    <edge, f>_A) / |f|_A``; squaring gives a quadratic in ``s`` whose admissible root
    is the one below. Where ``r^2`` is at least ``|edge|_A^2`` the derivative has one
    sign on the whole segment and the minimum is an end.
    """
    edge_x, edge_y = edge
    reach_x, reach_y = reach
    velocity_x, velocity_y = coefficients[:, 0], coefficients[:, 1]
    length = _form(edge_x, edge_y, coefficients, edge_x, edge_y)
    projection = _form(edge_x, edge_y, coefficients, reach_x, reach_y)
    squared = _form(reach_x, reach_y, coefficients, reach_x, reach_y)
    speed = np.sqrt(_form(velocity_x, velocity_y, coefficients, velocity_x, velocity_y))
    slope = rise + _form(edge_x, edge_y, coefficients, velocity_x, velocity_y)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = slope / speed
        room = length - ratio * ratio
        root = projection / length - ratio * np.sqrt(
            np.maximum(length * squared - projection * projection, 0.0),
        ) / (length * np.sqrt(room))
    fraction = np.where(slope > 0.0, 0.0, 1.0)
    fraction = np.where(room > 0.0, root, fraction)
    fraction = np.where(np.isfinite(fraction), fraction, _HALF)
    return np.clip(fraction, 0.0, 1.0)


class GeometricMinimumAction:
    """The minimum action path between two points, by the simplified gMAM.

    The path is a chain of states with its ends fixed, parametrised by normalised arc
    length ``alpha`` in ``[0, 1]``. In coordinates ``z = L^{-1} x`` with
    ``D = L L^T`` the noise is the identity, and with ``b`` the drift and ``J`` its
    Jacobian there the steepest descent of the geometric action, preconditioned by
    ``lambda = |b| / |z'|``, is

        ``z_tau = lambda^2 z'' - lambda (J - J^T) z' - J^T b``

    up to a tangential part. Only the component of the explicit terms normal to the
    path is applied, and the reparametrisation to equal arc length after every step
    redistributes the nodes; see the comment in ``_relax`` for why the tangential
    part cannot simply be left to the reparametrisation. On the rotating
    Ornstein-Uhlenbeck process from the origin to ``(0.8, -0.3)`` the action is
    0.34%, 0.16% and 0.11% above ``(1/2) x^T S^{-1} x`` with 41, 81 and 161
    nodes, the same at steps 0.05 and 0.01; what remains is the infinitely many
    turns the exact path makes as it spirals out of the origin, which a chain of
    nodes cannot follow. The ``z''`` term is taken implicitly - a tridiagonal
    solve per component - so the step is not limited by the node spacing. The
    action reported is ``integral |b| |z'| - <b, z'> d alpha`` by the midpoint rule.

    ``D`` must be constant; a state-dependent ``D`` needs the metric's own
    derivatives in the flow and is refused rather than silently ignored.
    """

    __slots__ = ["_drift", "_iterations", "_noise", "_step", "_tolerance"]

    def __init__(
        self,
        drift: Function,
        noise: NoiseMatrix,
        step: float = 0.05,
        iterations: int = 20000,
        tolerance: float = 1.0e-10,
    ) -> None:
        self._drift = drift
        self._noise = noise
        self._step = step
        self._iterations = iterations
        self._tolerance = tolerance

    def path(self, initial: Sequence[Sequence[float]]) -> MinimumActionPath:
        """Relax ``initial`` (states from one end to the other) to the minimiser.

        The chain is relaxed until no node moves by more than ``tolerance`` in a
        step, or for ``iterations`` steps; the result's ``converged`` says which,
        and a chain still moving is returned as it stands rather than refused.
        """
        states = np.asarray(initial, dtype=float)
        factor = self._constant_factor(states)
        inverse = np.linalg.inv(factor)
        chain = states @ inverse.T
        nodes = chain.shape[0]
        spacing = 1.0 / (nodes - 1)
        converged = False
        taken = 0
        for taken in range(1, self._iterations + 1):  # noqa: B007 (read after)
            following = self._relax(chain, factor, inverse, spacing)
            following = _equal_arc_length(following)
            change = float(np.max(np.abs(following - chain)))
            chain = following
            if change < self._tolerance:
                converged = True
                break
        result = chain @ factor.T
        return MinimumActionPath(
            result,
            self._action(chain, factor, inverse),
            converged=converged,
            iterations=taken,
        )

    def _constant_factor(self, states: np.ndarray) -> np.ndarray:
        cells = np.array(
            [diffusion_matrix(self._noise.columns_at(state)) for state in states],
        )
        deviation = float(np.max(np.abs(cells - cells[0])))
        if deviation > _CONSTANT_TOLERANCE * max(float(np.max(np.abs(cells[0]))), 1.0):
            message = (
                "the geometric minimum action method here needs a constant "
                "diffusion matrix G G^T"
            )
            raise ValueError(message)
        _check_nodes(cells[:1])
        lower, _ = cho_factor(cells[0], lower=True)
        return np.tril(lower)

    def _drift_at(self, chain, factor, inverse):
        velocity = np.array(
            [inverse @ field_at(self._drift, factor @ state) for state in chain],
        )
        jacobians = np.array(
            [
                inverse @ jacobian_at(self._drift, factor @ state) @ factor
                for state in chain
            ],
        )
        return velocity, jacobians

    def _relax(self, chain, factor, inverse, spacing):
        velocity, jacobians = self._drift_at(chain, factor, inverse)
        tangent = np.gradient(chain, spacing, axis=0)
        speed = np.linalg.norm(tangent, axis=1)
        scale = np.linalg.norm(velocity, axis=1) / np.where(speed > 0.0, speed, 1.0)
        skew = jacobians - np.transpose(jacobians, (0, 2, 1))
        force = -scale[:, None] * np.einsum("kij,kj->ki", skew, tangent) - np.einsum(
            "kji,kj->ki",
            jacobians,
            velocity,
        )
        # Only the normal part moves the curve; the tangential part only slides
        # nodes along it, and the reparametrisation's linear interpolation then
        # cuts the corners it slid them round. Left in, that bias balanced the
        # flow at a fixed point that depended on the step size: 0.63% and 1.02%
        # above the exact action at steps 0.05 and 0.01, and growing with the
        # node count. Projected out, the answer is the same at both steps and
        # converges as nodes are added.
        unit = tangent / np.where(speed > 0.0, speed, 1.0)[:, None]
        force -= np.sum(force * unit, axis=1)[:, None] * unit
        explicit = chain + self._step * force
        weight = self._step * scale * scale / (spacing * spacing)
        nodes = chain.shape[0]
        bands = np.zeros((3, nodes))
        bands[1] = 1.0 + 2.0 * weight
        bands[0, 1:] = -weight[:-1]
        bands[2, :-1] = -weight[1:]
        bands[1, 0] = bands[1, -1] = 1.0
        bands[0, 1] = 0.0
        bands[2, -2] = 0.0
        explicit[0], explicit[-1] = chain[0], chain[-1]
        return np.column_stack(
            [solve_banded((1, 1), bands, explicit[:, axis]) for axis in range(_PLANE)],
        )

    def _action(self, chain, factor, inverse) -> float:
        middle = _HALF * (chain[1:] + chain[:-1])
        steps = np.diff(chain, axis=0)
        velocity = np.array(
            [inverse @ field_at(self._drift, factor @ state) for state in middle],
        )
        return float(
            np.sum(
                np.linalg.norm(velocity, axis=1) * np.linalg.norm(steps, axis=1)
                - np.sum(velocity * steps, axis=1),
            ),
        )


def _equal_arc_length(chain: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(lengths)])
    target = np.linspace(0.0, arc[-1], chain.shape[0])
    return np.column_stack(
        [np.interp(target, arc, chain[:, axis]) for axis in range(_PLANE)],
    )
